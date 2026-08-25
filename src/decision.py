from __future__ import annotations

"""输出单条有界机体动作的轻量多模态策略。"""

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch import Tensor, nn

ENTITY_KINEMATIC_DIM = 4
ENTITY_EMBEDDING_DIM = 64
ENTITY_FEATURE_DIM = ENTITY_KINEMATIC_DIM + ENTITY_EMBEDDING_DIM
POSITION_SCALE_M = 20.0
VELOCITY_SCALE_MPS = 5.0


@dataclass(frozen=True)
class PolicyConfig:
    task_embedding_dim: int = 256
    entity_count: int = 16
    entity_kinematic_dim: int = 4
    entity_embedding_dim: int = 64
    entity_feature_dim: int = 68
    action_dim: int = 2
    asv_state_dim: int = 2
    task_embedding_hidden: int = 128
    entity_feature_hidden: int = 64
    entity_hidden: int = 192
    asv_state_hidden: int = 64
    fusion_hidden: int = 256
    maximum_action_m: float = 0.55
    invalid_stop_logit: float = 20.0
    maximum_trainable_parameters: int = 2_000_000
    task_embedding_conditioned_entity_attention: bool = True
    entity_attention_mode: str = "task_embedding_only"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PolicyConfig":
        model = value.get("model", value)
        if not isinstance(model, Mapping):
            raise ValueError("model configuration must be a mapping")
        model = dict(model)
        if "horizon" in model:
            raise ValueError(
                "legacy trajectory policy config is unsupported; remove "
                "horizon and retrain with a single [B, 2] action"
            )
        known = {field.name for field in cls.__dataclass_fields__.values()}
        unknown = set(model) - known
        if unknown:
            raise ValueError(f"unknown model configuration keys: {sorted(unknown)}")
        return cls(**dict(model))

    def __post_init__(self) -> None:
        positive_ints = (
            self.task_embedding_dim,
            self.entity_count,
            self.entity_kinematic_dim,
            self.entity_embedding_dim,
            self.entity_feature_dim,
            self.action_dim,
            self.asv_state_dim,
            self.task_embedding_hidden,
            self.entity_feature_hidden,
            self.entity_hidden,
            self.asv_state_hidden,
            self.fusion_hidden,
            self.maximum_trainable_parameters,
        )
        if any(value <= 0 for value in positive_ints):
            raise ValueError("all dimensions and parameter limits must be positive")
        if self.entity_kinematic_dim + self.entity_embedding_dim != self.entity_feature_dim:
            raise ValueError(
                "entity_feature_dim must equal "
                "entity_kinematic_dim + entity_embedding_dim"
            )
        if self.action_dim != 2:
            raise ValueError("single-step body-frame action_dim must be 2")
        if self.asv_state_dim != 2:
            raise ValueError("asv_state_dim must be 2: surge_velocity and yaw_rate")
        if self.maximum_action_m <= 0.0:
            raise ValueError("maximum_action_m must be positive")
        if not torch.isfinite(torch.tensor(self.invalid_stop_logit)):
            raise ValueError("invalid_stop_logit must be finite")
        if self.entity_attention_mode not in {
            "task_embedding_additive",
            "task_embedding_only",
        }:
            raise ValueError(
                f"unsupported entity_attention_mode={self.entity_attention_mode!r}"
            )
        if (
            not self.task_embedding_conditioned_entity_attention
        ):
            raise ValueError(
                "entity attention mode requires "
                "task_embedding_conditioned_entity_attention=true"
            )

@dataclass(frozen=True)
class PolicyOutput:
    """策略张量及显式安全关闭样本有效掩码。"""

    action: Tensor
    stop_logit: Tensor
    valid_mask: Tensor


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
        nn.GELU(),
    )


class ActionPolicy(nn.Module):
    """融合语言和结构化实体几何，输出一条有界动作。

    The decision head receives task language, structured entity geometry and
    the current vessel dynamics. It does not receive the previous action or
    image/bbox tensors; temporal smoothness is learned from current dynamics.
    """

    def __init__(self, config: PolicyConfig | None = None) -> None:
        super().__init__()
        self.config = config or PolicyConfig()
        cfg = self.config

        self.task_embedding_encoder = _mlp(
            cfg.task_embedding_dim, cfg.task_embedding_hidden, cfg.task_embedding_hidden
        )
        self.entity_kinematic_encoder = _mlp(
            cfg.entity_kinematic_dim,
            cfg.entity_feature_hidden,
            cfg.entity_feature_hidden,
        )
        self.entity_embedding_encoder = _mlp(
            cfg.entity_embedding_dim,
            cfg.entity_feature_hidden,
            cfg.entity_feature_hidden,
        )
        self.entity_fusion = nn.Sequential(
            nn.Linear(cfg.entity_feature_hidden * 2, cfg.entity_hidden),
            nn.GELU(),
        )
        self.asv_state_encoder = _mlp(
            cfg.asv_state_dim,
            cfg.asv_state_hidden,
            cfg.asv_state_hidden,
        )
        self.entity_attention = nn.Linear(cfg.entity_hidden, 1)
        if cfg.task_embedding_conditioned_entity_attention:
            self.entity_task_embedding_query: nn.Linear | None = nn.Linear(
                cfg.task_embedding_hidden,
                cfg.entity_hidden,
                bias=False,
            )
        else:
            # 保持旧版 state_dict 可加载。
            self.entity_task_embedding_query = None
        # 有效位属于决策输入，用于区分真实零动作与回合初始零哨兵。
        fusion_input_dim = (
            cfg.task_embedding_hidden + cfg.entity_hidden + cfg.asv_state_hidden + 1
        )
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, cfg.fusion_hidden),
            nn.GELU(),
            nn.LayerNorm(cfg.fusion_hidden),
            nn.Linear(cfg.fusion_hidden, cfg.fusion_hidden),
            nn.GELU(),
        )
        self.action_head = nn.Linear(cfg.fusion_hidden, cfg.action_dim)
        self.stop_head = nn.Linear(cfg.fusion_hidden, 1)

        parameter_count = self.trainable_parameter_count()
        if parameter_count > cfg.maximum_trainable_parameters:
            raise ValueError(
                f"policy has {parameter_count} trainable parameters; "
                f"limit is {cfg.maximum_trainable_parameters}"
            )

    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    @staticmethod
    def _expect_shape(tensor: Tensor, shape: tuple[int, ...], name: str) -> None:
        if tuple(tensor.shape) != shape:
            raise ValueError(
                f"{name} shape {tuple(tensor.shape)} does not match {shape}"
            )

    @staticmethod
    def _as_mask(
        mask: Tensor | None,
        *,
        batch_size: int,
        device: torch.device,
        name: str,
    ) -> Tensor:
        if mask is None:
            return torch.ones(batch_size, dtype=torch.bool, device=device)
        if tuple(mask.shape) != (batch_size,):
            raise ValueError(
                f"{name} shape {tuple(mask.shape)} does not match "
                f"{(batch_size,)}"
            )
        return mask.to(device=device, dtype=torch.bool)

    @staticmethod
    def _sanitize_masked(values: Tensor, mask: Tensor, name: str) -> Tensor:
        expanded = mask
        while expanded.ndim < values.ndim:
            expanded = expanded.unsqueeze(-1)
        expanded = expanded.expand_as(values)
        active_values = values[expanded]
        if active_values.numel() and not torch.isfinite(active_values).all():
            raise ValueError(f"{name} contains NaN or Inf in an active position")
        return torch.where(expanded, values, torch.zeros_like(values))

    def forward(
        self,
        *,
        task_embedding: Tensor,
        entity_states: Tensor,
        asv_state: Tensor,
        task_embedding_valid: Tensor | None = None,
        entity_states_mask: Tensor | None = None,
        asv_state_valid: Tensor | None = None,
        policy_input_valid: Tensor | None = None,
    ) -> PolicyOutput:
        cfg = self.config
        if task_embedding.ndim != 2:
            raise ValueError("task_embedding must have shape [B, task_embedding_dim]")
        batch_size = int(task_embedding.shape[0])
        device = task_embedding.device
        dtype = task_embedding.dtype
        if (
            entity_states.device != device
            or asv_state.device != device
        ):
            raise ValueError("all policy inputs must be on the same device")
        if entity_states.dtype != dtype or asv_state.dtype != dtype:
            raise ValueError("all floating policy inputs must share one dtype")

        self._expect_shape(
            task_embedding, (batch_size, cfg.task_embedding_dim), "task_embedding"
        )
        self._expect_shape(
            entity_states,
            (
                batch_size,
                cfg.entity_count,
                cfg.entity_feature_dim,
            ),
            "entity_states",
        )
        self._expect_shape(asv_state, (batch_size, cfg.asv_state_dim), "asv_state")
        task_embedding_mask = self._as_mask(
            task_embedding_valid,
            batch_size=batch_size,
            device=device,
            name="task_embedding_valid",
        )
        input_mask = self._as_mask(
            policy_input_valid,
            batch_size=batch_size,
            device=device,
            name="policy_input_valid",
        )

        if entity_states_mask is None:
            entity_states_mask = torch.ones(
                batch_size,
                cfg.entity_count,
                dtype=torch.bool,
                device=device,
            )
        self._expect_shape(
            entity_states_mask,
            (batch_size, cfg.entity_count),
            "entity_states_mask",
        )
        active_entity_mask = entity_states_mask.to(
            device=device, dtype=torch.bool
        )
        asv_mask = self._as_mask(
            asv_state_valid,
            batch_size=batch_size,
            device=device,
            name="asv_state_valid",
        )

        valid_mask = task_embedding_mask & input_mask
        active_entity_mask = active_entity_mask & valid_mask.unsqueeze(1)

        task_embedding_clean = self._sanitize_masked(
            task_embedding.detach(), task_embedding_mask & valid_mask, "task_embedding"
        )
        entity_states_clean = self._sanitize_masked(
            entity_states, active_entity_mask, "entity_states"
        )
        asv_state_clean = self._sanitize_masked(
            asv_state.detach(), asv_mask & valid_mask, "asv_state"
        )
        task_embedding_token = self.task_embedding_encoder(task_embedding_clean)
        kinematic = entity_states_clean[..., : cfg.entity_kinematic_dim]
        embedding = entity_states_clean[..., cfg.entity_kinematic_dim :]
        entity_token = self.entity_fusion(
            torch.cat(
                (
                    self.entity_kinematic_encoder(kinematic),
                    self.entity_embedding_encoder(embedding),
                ),
                dim=-1,
            )
        )
        asv_state_token = self.asv_state_encoder(asv_state_clean)

        attention_score = self.entity_attention(entity_token).squeeze(-1)
        if self.entity_task_embedding_query is not None:
            task_embedding_query = self.entity_task_embedding_query(task_embedding_token)
            task_embedding_score = torch.sum(
                entity_token * task_embedding_query.unsqueeze(1), dim=-1
            ) / (cfg.entity_hidden**0.5)
            mode = cfg.entity_attention_mode
            # 旧检查点没有显式 mode 字段，使用加法形式；保留其布尔标志行为。
            if mode == "task_embedding_additive":
                attention_score = attention_score + task_embedding_score
            else:
                attention_score = task_embedding_score
        attention_weight = torch.exp(attention_score.clamp(-20.0, 20.0))
        attention_weight = attention_weight * active_entity_mask.to(
            dtype=dtype
        )
        attention_weight = attention_weight / attention_weight.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0e-12)
        pooled_entity = torch.sum(
            entity_token * attention_weight.unsqueeze(-1), dim=1
        )
        fused = self.fusion(
            torch.cat(
                (
                    task_embedding_token,
                    pooled_entity,
                    asv_state_token,
                    asv_mask.to(dtype=dtype).unsqueeze(-1),
                ),
                dim=-1,
            )
        )
        stop_logit = self.stop_head(fused)
        raw_action = self.action_head(fused)
        raw_norm = torch.linalg.vector_norm(raw_action, dim=-1, keepdim=True)
        radial_scale = torch.where(
            raw_norm > 1.0e-6,
            torch.tanh(raw_norm) / raw_norm.clamp_min(1.0e-6),
            torch.ones_like(raw_norm),
        )
        movement_gate = torch.sigmoid(-stop_logit)
        action = raw_action * radial_scale * cfg.maximum_action_m * movement_gate
        # STOP 是执行合同，不只是少移动；发布时分类器选择 STOP 必须完全静止。
        movement_selected = (stop_logit < 0.0).view(batch_size, 1)
        action = torch.where(movement_selected, action, torch.zeros_like(action))

        action = torch.where(
            valid_mask.view(batch_size, 1), action, torch.zeros_like(action)
        )
        stop_logit = torch.where(
            valid_mask.view(batch_size, 1),
            stop_logit,
            torch.full_like(stop_logit, cfg.invalid_stop_logit),
        )
        return PolicyOutput(
            action=action,
            stop_logit=stop_logit,
            valid_mask=valid_mask,
        )

class EntityFeaturesError(RuntimeError):
    pass


@dataclass(frozen=True)
class EntityFeaturesResult:
    features: np.ndarray
    mask: np.ndarray
    entity_ids: tuple[str, ...]
    entity_count: int
    dropped_count: int


@dataclass
class DecisionEntityFrame:
    """内部结构化实体张量，不作为 ROS 消息暴露。"""

    stamp_us: int
    run_id: str
    scene_seed: int
    frame_index: int
    frame_id: str
    max_entities: int
    feature_dim: int
    entity_count: int
    entity_ids: list[str]
    features: list[float]
    mask: list[bool]
    valid: bool
    detail: str
    safety_entities: tuple[EntityPhysicalState, ...] = ()


def _clip_feature(value: float, scale: float, low: float = -1.0) -> float:
    return float(np.clip(value / scale, low, 1.0))


def _entity_distance(entity: Any) -> float:
    return math.hypot(float(entity.relative_x), float(entity.relative_y))


def _read_entity_embedding(entity: Any) -> np.ndarray:
    raw = getattr(entity, "entity_embedding", None)
    if raw is None:
        return np.zeros(ENTITY_EMBEDDING_DIM, dtype=np.float32)
    values = np.asarray(raw, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return np.zeros(ENTITY_EMBEDDING_DIM, dtype=np.float32)
    if values.size < ENTITY_EMBEDDING_DIM:
        padded = np.zeros(ENTITY_EMBEDDING_DIM, dtype=np.float32)
        padded[: values.size] = values
        return padded
    return values[:ENTITY_EMBEDDING_DIM].astype(np.float32, copy=False)


def _entity_candidate(entity: Any) -> tuple[Any, str]:
    entity_id = str(entity.entity_id).strip()
    values = (
        entity.relative_x,
        entity.relative_y,
        entity.relative_velocity_x,
        entity.relative_velocity_y,
    )
    if not entity_id:
        raise EntityFeaturesError("a valid visible entity has an empty entity_id")
    if not all(math.isfinite(float(value)) for value in values):
        raise EntityFeaturesError(f"entity {entity_id!r} contains NaN or Inf")
    embedding = _read_entity_embedding(entity)
    if not np.all(np.isfinite(embedding)):
        raise EntityFeaturesError(f"entity {entity_id!r} embedding is non-finite")
    return entity, entity_id


def _entity_sort_key(candidate: tuple[Any, str]) -> tuple[Any, ...]:
    entity, entity_id = candidate
    return _entity_distance(entity), entity_id


def _entity_row(entity: Any) -> np.ndarray:
    row = np.zeros(ENTITY_FEATURE_DIM, dtype=np.float32)
    row[0] = _clip_feature(float(entity.relative_x), POSITION_SCALE_M)
    row[1] = _clip_feature(float(entity.relative_y), POSITION_SCALE_M)
    velocity_valid = bool(getattr(entity, "velocity_valid", True))
    vx = float(entity.relative_velocity_x) if velocity_valid else 0.0
    vy = float(entity.relative_velocity_y) if velocity_valid else 0.0
    row[2] = _clip_feature(vx, VELOCITY_SCALE_MPS)
    row[3] = _clip_feature(vy, VELOCITY_SCALE_MPS)
    row[ENTITY_KINEMATIC_DIM:] = _read_entity_embedding(entity)
    return row


def build_entity_states(
    entities: Iterable[Any],
    *,
    max_entities: int,
) -> EntityFeaturesResult:
    if int(max_entities) <= 0:
        raise EntityFeaturesError(f"max_entities must be positive, got {max_entities}")
    candidates, seen_ids = [], set()
    for entity in entities:
        if not bool(entity.valid) or not bool(entity.visible):
            continue
        candidate = _entity_candidate(entity)
        if candidate[1] in seen_ids:
            raise EntityFeaturesError(f"duplicate valid visible entity_id {candidate[1]!r}")
        seen_ids.add(candidate[1])
        candidates.append(candidate)
    selected = sorted(candidates, key=_entity_sort_key)[: int(max_entities)]
    features = np.zeros((int(max_entities), ENTITY_FEATURE_DIM), dtype=np.float32)
    mask = np.zeros(int(max_entities), dtype=np.bool_)
    entity_ids = [""] * int(max_entities)
    for index, (entity, entity_id) in enumerate(selected):
        features[index] = _entity_row(entity)
        mask[index] = True
        entity_ids[index] = entity_id
    if not np.all(np.isfinite(features)):
        raise EntityFeaturesError("entity features contain NaN or Inf")
    return EntityFeaturesResult(
        features,
        mask,
        tuple(entity_ids),
        len(selected),
        max(0, len(candidates) - len(selected)),
    )
