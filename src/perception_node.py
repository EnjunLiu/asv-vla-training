"""订阅相机与 task_embedding，发布结构化实体。"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from interfaces.msg import CameraFrame, EntityState, EntityStateArray, TaskEmbedding

try:
    from .entity_contract import ENTITY_EMBEDDING_DIM
except ImportError:
    from entity_contract import ENTITY_EMBEDDING_DIM
from .entity_embedding import EntityEmbeddingHead, EntityEmbeddingRuntime
from .perception import (
    CameraProfile,
    FrameMetadata,
    GeometryObservation,
    ImageEntityModel,
    ImageEntityPerceptionError,
    TemporalEntityTracker,
    decode_camera_image,
    project_target_to_pixel,
    validate_task_embedding,
)

RELIABLE_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
)
EMBEDDING_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)
SENSOR_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=2,
    reliability=ReliabilityPolicy.BEST_EFFORT,
)
_ZERO_EMBEDDING = [0.0] * ENTITY_EMBEDDING_DIM


def _embedding_list(values: Sequence[float] | np.ndarray) -> list[float]:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if array.size != ENTITY_EMBEDDING_DIM:
        return list(_ZERO_EMBEDDING)
    return [float(value) for value in array]


class PerceptionNode(Node):

    def __init__(self) -> None:
        super().__init__("perception")
        self.device = str(self.declare_parameter("device", "cuda").value).strip() or "cuda"
        model_path = str(self.declare_parameter("model_path", "").value) or str(
            Path.home() / "jetson_asv_ws" / "models" / "perception.npz"
        )
        embedding_path = str(self.declare_parameter("entity_embedding_path", "").value).strip()
        if not embedding_path:
            default_embedding = Path.home() / "jetson_asv_ws" / "models" / "entity_embedding.pt"
            if default_embedding.is_file():
                embedding_path = str(default_embedding)
        self.profile = CameraProfile()
        self.publisher = self.create_publisher(EntityStateArray, "/vla/entities", RELIABLE_QOS)
        self.create_subscription(CameraFrame, "/ue/camera_frame", self.on_frame, SENSOR_QOS)
        self.create_subscription(TaskEmbedding, "/vla/task_embedding", self.on_embedding, EMBEDDING_QOS)
        self.task_embedding = None
        self.task_instruction = ""
        self.task_instruction_id = "task_embedding"
        self.tracker = TemporalEntityTracker(velocity_filter="ema", alpha=0.35)
        self.model = ImageEntityModel.load(model_path)
        self.model.validate_device(self.device)
        self.entity_embedding = None
        if embedding_path:
            self.entity_embedding = EntityEmbeddingHead.load(
                embedding_path, device=self.device
            )
        self.get_logger().info(
            f"ready model={self.model.model_version} device={self.device} "
            f"entity_embedding={'on' if self.entity_embedding else 'off'}"
        )

    def on_embedding(self, message: TaskEmbedding) -> None:
        if not message.valid:
            self.task_embedding = None
            self.task_instruction = ""
            return
        self.task_instruction = str(message.instruction).strip()
        self.task_embedding = validate_task_embedding(
            message.embedding, expected_dim=self.model.task_embedding_dim
        )

    def on_frame(self, frame: CameraFrame) -> None:
        out = EntityStateArray()
        out.stamp_us = int(frame.stamp_us)
        out.run_id = str(frame.run_id)
        out.scene_seed = int(frame.scene_seed)
        out.frame_index = int(frame.frame_index)
        out.frame_id = "base_link"
        out.instruction_id = self.task_instruction_id
        out.instruction = self.task_instruction
        out.source = "perception"

        if self.task_embedding is None or not self.task_instruction or not frame.valid or not frame.data:
            out.valid = False
            out.detail = "NOT_READY"
            self.publisher.publish(out)
            return

        try:
            image = decode_camera_image(frame.data, frame.encoding)
            predictions = self.model.predict(
                image, device=self.device, task_embedding=self.task_embedding
            )
            entity_embeddings = {}
            if self.entity_embedding is not None:
                entity_embeddings = self.entity_embedding.encode_entities(
                    image,
                    (
                        {
                            "entity_id": pred.entity_id,
                            "visible": pred.visible,
                            "valid": pred.valid,
                            "relative_position_m": (
                                pred.relative_x,
                                pred.relative_y,
                                pred.relative_z,
                            ),
                        }
                        for pred in predictions
                        if pred.visible
                    ),
                    self.profile,
                )
        except ImageEntityPerceptionError as exc:
            out.valid = False
            out.detail = str(exc)
            self.publisher.publish(out)
            return

        for pred in predictions:
            if not pred.visible:
                continue
            if not all(
                math.isfinite(float(value))
                for value in (pred.relative_x, pred.relative_y, pred.relative_z)
            ):
                continue
            entity = EntityState()
            entity.entity_id = pred.entity_id
            entity.class_name = "boat"
            entity.color = ""
            entity.is_target = False
            entity.relative_x = pred.relative_x
            entity.relative_y = pred.relative_y
            entity.relative_z = pred.relative_z
            entity.confidence = float(pred.confidence)
            entity.entity_embedding = _embedding_list(
                entity_embeddings.get(pred.entity_id, _ZERO_EMBEDDING)
            )
            entity.valid = True
            entity.visible = True
            try:
                px, py, _ = project_target_to_pixel(
                    entity.relative_x, entity.relative_y, entity.relative_z, self.profile
                )
                half_w, half_h = 32.0, 16.0
                entity.bbox_x_min = max(0.0, px - half_w)
                entity.bbox_y_min = max(0.0, py - half_h)
                entity.bbox_x_max = min(self.profile.width - 1.0, px + half_w)
                entity.bbox_y_max = min(self.profile.height - 1.0, py + half_h)
                entity.bbox_valid = True
            except Exception:
                entity.bbox_valid = False
            out.entities.append(entity)

        out.valid = bool(out.entities)
        out.detail = f"entities={len(out.entities)}"
        self._track(out)
        self.publisher.publish(out)

    def _track(self, message: EntityStateArray) -> None:
        if not message.valid:
            self.tracker.reset()
            return
        frame = FrameMetadata(
            run_id=message.run_id,
            scene_seed=message.scene_seed,
            frame_index=message.frame_index,
            stamp_us=message.stamp_us,
        )
        observations = [
            GeometryObservation(
                entity_id=e.entity_id,
                relative_x=e.relative_x,
                relative_y=e.relative_y,
                relative_z=e.relative_z,
                class_name=e.class_name,
                color=e.color,
                is_target=False,
                visible=e.visible,
                confidence=e.confidence,
                run_id=message.run_id,
                scene_seed=message.scene_seed,
                frame_index=message.frame_index,
                stamp_us=message.stamp_us,
            )
            for e in message.entities
            if e.valid
        ]
        tracked = self.tracker.update(observations, frame=frame)
        by_id = {}
        for item in tracked:
            entity = EntityState()
            for field, value in item.as_entity_kwargs().items():
                setattr(entity, field, value)
            source_entity = next(
                (candidate for candidate in message.entities if candidate.entity_id == entity.entity_id),
                None,
            )
            if source_entity is not None:
                entity.entity_embedding = _embedding_list(source_entity.entity_embedding)
            else:
                entity.entity_embedding = list(_ZERO_EMBEDDING)
            by_id[entity.entity_id] = entity
        message.entities = [by_id.get(e.entity_id, e) for e in message.entities]


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
