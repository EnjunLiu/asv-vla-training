"""订阅 ``/task/task_text`` 并发布 ``/vla/task_embedding``。"""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from interfaces.msg import TaskEmbedding

from .language import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_PATH,
    DEFAULT_TASK_DESCRIPTION,
    EMBEDDING_DIM,
    TaskEncoder,
    TaskEncoderError,
    TaskEmbeddingState,
    embedding_tuple,
    state_payload,
)

TASK_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)
LANG_QOS = TASK_QOS


class TaskNode(Node):

    def __init__(self) -> None:
        super().__init__("task")
        self.model_path = str(self.declare_parameter("model_path", DEFAULT_MODEL_PATH).value)
        self.device = str(self.declare_parameter("device", "cpu").value).strip()
        self.quantize_int8 = bool(self.declare_parameter("quantize_int8", True).value)
        self.model_id = str(self.declare_parameter("model_id", DEFAULT_MODEL_ID).value).strip()
        self.task_description = str(
            self.declare_parameter("task_description", DEFAULT_TASK_DESCRIPTION).value
        )
        self.run_id = str(self.declare_parameter("run_id", "task-qwen").value)

        self._pub = self.create_publisher(TaskEmbedding, "/vla/task_embedding", TASK_QOS)
        self.create_subscription(String, "/task/task_text", self.on_task, TASK_QOS)
        self._encoder: TaskEncoder | None = None
        self._state = TaskEmbeddingState(model_id=self.model_id, detail="STARTING")

        try:
            self._encoder = TaskEncoder(
                self.model_path,
                device=self.device,
                quantize_int8=self.quantize_int8,
                task_description=self.task_description,
            )
            self.get_logger().info(
                f"READY model={self.model_id} device={self._encoder.device} "
                f"quantize_int8={self.quantize_int8}"
            )
        except Exception as exc:
            self._state = TaskEmbeddingState(
                model_id=self.model_id, valid=False, detail=f"MODEL_UNAVAILABLE:{exc}"
            )
            self.get_logger().error(self._state.detail)

        self.create_timer(1.0, self._publish)
        if self.task_description.strip():
            self.on_task(String(data=self.task_description))

    def on_task(self, message: String) -> None:
        instruction = str(message.data).strip()
        if not instruction:
            self._state = TaskEmbeddingState(model_id=self.model_id, valid=False, detail="EMPTY")
            self._publish()
            return
        if self._encoder is None:
            self._state = TaskEmbeddingState(
                instruction=instruction, model_id=self.model_id, valid=False, detail="NO_ENCODER"
            )
            self._publish()
            return
        try:
            result = self._encoder.encode_with_metadata(instruction)
            self._state = TaskEmbeddingState(
                instruction=instruction,
                embedding=embedding_tuple(result.embedding),
                model_id=self.model_id,
                cached=result.cached,
                valid=True,
                detail="OK",
            )
            self.get_logger().info(f"TASK_READY_VALID instruction={instruction}")
        except TaskEncoderError as exc:
            self._state = TaskEmbeddingState(
                instruction=instruction, model_id=self.model_id, valid=False, detail=str(exc)
            )
        self._publish()

    def _publish(self) -> None:
        msg = TaskEmbedding()
        for key, value in state_payload(
            self._state, run_id=self.run_id, stamp_us=self.get_clock().now().nanoseconds // 1000
        ).items():
            setattr(msg, key, value)
        self._pub.publish(msg)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = TaskNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
