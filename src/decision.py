from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
import time
from typing import Any, Iterable, Protocol, Sequence

import numpy as np
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from interfaces.msg import ASVState, DesiredDisplacement, EntityStateArray, TaskEmbedding
except ModuleNotFoundError:  # 允许算法部分在离线测试中运行。
    rclpy = None

    class Node:
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("ROS 2 dependencies are required to instantiate DecisionNode")

    class _QoS:
        RELIABLE = 1
        TRANSIENT_LOCAL = 1

    DurabilityPolicy = ReliabilityPolicy = _QoS

    class QoSProfile:
        def __init__(self, **kwargs) -> None:
            self.settings = kwargs

    ASVState = DesiredDisplacement = EntityStateArray = TaskEmbedding = Any


# ``HORIZON`` 仅是离线/模型常量，不属于在线 ROS 控制合同。
HORIZON = 20
ACTION_DIM = 2
DT_SEC = 0.5
FRAME_ID = "base_link"
# 训练决策头与在线 desired_x/desired_y 共用 0.5 秒周期的有界单步位移。
MAX_DISPLACEMENT_M = 0.50
SAFE_STOP_SOURCE = "safe_stop"
FLOAT_TOLERANCE = 1.0e-6


class DesiredDisplacementLike(Protocol):
    stamp_us: int
    run_id: str
    scene_seed: int
    frame_index: int
    frame_id: str
    source: str
    step_dt: float
    desired_x: float
    desired_y: float
    safe_stop: bool
    valid: bool


def finite_zero(value: float, tolerance: float = FLOAT_TOLERANCE) -> bool:
    return math.isfinite(value) and abs(value) <= tolerance


def is_safe_stop(message: DesiredDisplacementLike) -> bool:
    """校验不可执行的单点安全停止标记。"""

    return (
        int(message.stamp_us) > 0
        and bool(str(message.run_id).strip())
        and message.frame_id == FRAME_ID
        and message.source == SAFE_STOP_SOURCE
        and math.isfinite(float(message.step_dt))
        and abs(float(message.step_dt) - DT_SEC) <= FLOAT_TOLERANCE
        and finite_zero(float(message.desired_x))
        and finite_zero(float(message.desired_y))
        and bool(message.safe_stop)
        and not bool(message.valid)
    )



POLICY_DRIVEN = "policy_driven"
GUARD_FAIL_CLOSED = "fail_closed"


"""单条在线机体位移的确定性安全门。

安全门是 CUDA 策略与运动学控制器之间的唯一组件；校验当前 ``(desired_x, desired_y)``、
检查单步碰撞包络，任何拒绝都安全关闭；不接受模型离线 [20, 2] 输出。
"""

PASS = "PASS"
POLICY_STOP = "POLICY_STOP"
STALE_INPUT = "STALE_INPUT"
INVALID_MODALITY = "INVALID_MODALITY"
INVALID_SHAPE = "INVALID_SHAPE"
NONFINITE = "NONFINITE"
SPEED_LIMIT = "SPEED_LIMIT"
COLLISION_RISK = "COLLISION_RISK"
CONTROL_UNREACHABLE = "CONTROL_UNREACHABLE"
ESTOP = "ESTOP"

REJECTION_CODES = frozenset(
    {
        STALE_INPUT,
        INVALID_MODALITY,
        INVALID_SHAPE,
        NONFINITE,
        SPEED_LIMIT,
        COLLISION_RISK,
        CONTROL_UNREACHABLE,
        ESTOP,
    }
)

DEFAULT_MAX_STEP_M = MAX_DISPLACEMENT_M
DEFAULT_MAX_SPEED_MPS = DEFAULT_MAX_STEP_M / DT_SEC
DEFAULT_STALE_TIMEOUT_SEC = 1.0
DEFAULT_ESTOP_TIMEOUT_SEC = 2.0
DEFAULT_COLLISION_MARGIN_M = 0.5


@dataclass(frozen=True)
class SafetyGateConfig:
    """单条机体命令在 ``dt`` 内的限制。"""

    max_step_m: float = DEFAULT_MAX_STEP_M
    stale_timeout_sec: float = DEFAULT_STALE_TIMEOUT_SEC
    estop_timeout_sec: float = DEFAULT_ESTOP_TIMEOUT_SEC
    collision_margin_m: float = DEFAULT_COLLISION_MARGIN_M

    def __post_init__(self) -> None:
        if not math.isfinite(self.max_step_m) or self.max_step_m <= 0.0:
            raise ValueError("max_step_m must be positive and finite")
        if not math.isfinite(self.stale_timeout_sec) or self.stale_timeout_sec <= 0.0:
            raise ValueError("stale_timeout_sec must be positive and finite")
        if not math.isfinite(self.estop_timeout_sec) or self.estop_timeout_sec <= 0.0:
            raise ValueError("estop_timeout_sec must be positive and finite")
        if self.estop_timeout_sec < self.stale_timeout_sec:
            raise ValueError("estop_timeout_sec must be >= stale_timeout_sec")
        if not math.isfinite(self.collision_margin_m) or self.collision_margin_m <= 0.0:
            raise ValueError("collision_margin_m must be positive and finite")


@dataclass(frozen=True)
class SafetyGateResult:
    desired_x: float
    desired_y: float
    safe_stop: bool
    valid: bool
    reason: str
    detail: str = ""


def limit_displacement_rate(
    result: SafetyGateResult,
    previous: tuple[float, float] | None,
    *,
    max_delta_m: float = DEFAULT_MAX_STEP_M,
) -> SafetyGateResult:
    """限制连续可执行位移之间的变化。"""

    if not result.valid or previous is None:
        return result
    if not math.isfinite(max_delta_m) or max_delta_m <= 0.0:
        return _reject(CONTROL_UNREACHABLE, "invalid rate limit")
    previous_x, previous_y = map(float, previous)
    if not (math.isfinite(previous_x) and math.isfinite(previous_y)):
        return _reject(CONTROL_UNREACHABLE, "invalid previous displacement")
    delta_x = float(result.desired_x) - previous_x
    delta_y = float(result.desired_y) - previous_y
    delta_norm = math.hypot(delta_x, delta_y)
    if delta_norm <= max_delta_m + FLOAT_TOLERANCE:
        return result
    scale = max_delta_m / delta_norm
    return SafetyGateResult(
        desired_x=previous_x + delta_x * scale,
        desired_y=previous_y + delta_y * scale,
        safe_stop=False,
        valid=True,
        reason=result.reason,
        detail="RATE_LIMITED",
    )


@dataclass(frozen=True)
class _Entity:
    entity_id: str
    relative_x: float
    relative_y: float
    relative_vx: float
    relative_vy: float


def _reject(code: str, detail: str = "") -> SafetyGateResult:
    """对每个拒绝输入返回不可执行的零命令。"""

    return SafetyGateResult(
        desired_x=0.0,
        desired_y=0.0,
        safe_stop=True,
        valid=False,
        reason=code,
        detail=detail,
    )


def _check_modality_and_shape(
    *,
    stamp_us: int,
    run_id: str,
    frame_id: str,
    model_version: str,
    dt: float,
    desired_x: float,
    desired_y: float,
    policy_valid: bool,
    language_valid: bool,
    entity_valid: bool,
    last_valid_stamp_us: int,
) -> str | None:
    """在执行命令前返回拒绝代码。"""

    if int(stamp_us) <= 0 or int(stamp_us) <= int(last_valid_stamp_us):
        return STALE_INPUT
    if not all((language_valid, entity_valid)):
        return INVALID_MODALITY
    if not policy_valid:
        return INVALID_MODALITY
    if not str(run_id).strip() or not str(model_version).strip():
        return INVALID_MODALITY
    if frame_id != FRAME_ID:
        return INVALID_MODALITY
    if not math.isfinite(float(dt)) or abs(float(dt) - DT_SEC) > FLOAT_TOLERANCE:
        return INVALID_SHAPE
    try:
        x = float(desired_x)
        y = float(desired_y)
    except (TypeError, ValueError):
        return INVALID_SHAPE
    if not (math.isfinite(x) and math.isfinite(y)):
        return NONFINITE
    return None


def _check_kinematics(
    desired_x: float,
    desired_y: float,
    config: SafetyGateConfig,
) -> str | None:
    """检查单步位移范数。"""

    norm = math.hypot(float(desired_x), float(desired_y))
    if not math.isfinite(norm):
        return NONFINITE
    if norm > config.max_step_m + FLOAT_TOLERANCE:
        return SPEED_LIMIT
    return None


def _check_entity_finiteness(entities: Sequence[_Entity]) -> str | None:
    for entity in entities:
        values = (
            entity.relative_x,
            entity.relative_y,
            entity.relative_vx,
            entity.relative_vy,
        )
        if not all(math.isfinite(float(value)) for value in values):
            return NONFINITE
    return None


def _check_collision(
    desired_x: float,
    desired_y: float,
    entities: Sequence[_Entity],
    config: SafetyGateConfig,
    *,
    dt: float = DT_SEC,
) -> str | None:
    """将执行位移与恒速实体进行碰撞检查。"""

    for entity in entities:
        predicted_x = float(entity.relative_x) + float(entity.relative_vx) * float(dt)
        predicted_y = float(entity.relative_y) + float(entity.relative_vy) * float(dt)
        distance = math.hypot(
            float(desired_x) - predicted_x,
            float(desired_y) - predicted_y,
        )
        if not math.isfinite(distance):
            return NONFINITE
        if distance < config.collision_margin_m:
            return COLLISION_RISK
    return None


def evaluate_safety_gate(
    *,
    stamp_us: int,
    run_id: str,
    frame_id: str,
    model_version: str,
    dt: float,
    desired_x: float,
    desired_y: float,
    safe_stop: bool,
    valid: bool,
    reason: str,
    language_valid: bool = True,
    entity_valid: bool = True,
    entities: Sequence[_Entity] | None = None,
    last_valid_stamp_us: int = 0,
    time_since_last_valid_sec: float = 0.0,
    config: SafetyGateConfig | None = None,
) -> SafetyGateResult:
    """通过安全门评估一条机体位移。"""

    cfg = config or SafetyGateConfig()
    rejection = _check_modality_and_shape(
        stamp_us=stamp_us,
        run_id=run_id,
        frame_id=frame_id,
        model_version=model_version,
        dt=dt,
        desired_x=desired_x,
        desired_y=desired_y,
        policy_valid=valid,
        language_valid=language_valid,
        entity_valid=entity_valid,
        last_valid_stamp_us=last_valid_stamp_us,
    )
    if rejection is not None:
        return _reject(rejection, reason)

    # stop 是无效保持标记，不是有效零动作。
    if safe_stop:
        return _reject(POLICY_STOP, reason or "policy requested stop")

    try:
        age = float(time_since_last_valid_sec)
    except (TypeError, ValueError):
        return _reject(NONFINITE, "invalid policy age")
    if not math.isfinite(age) or age < 0.0:
        return _reject(NONFINITE, "invalid policy age")
    if age > cfg.estop_timeout_sec:
        return _reject(ESTOP, f"no valid policy for {age:.2f}s")
    if age > cfg.stale_timeout_sec:
        return _reject(STALE_INPUT, f"last valid policy is {age:.2f}s old")

    rejection = _check_kinematics(desired_x, desired_y, cfg)
    if rejection is not None:
        return _reject(rejection)

    checked_entities = tuple(entities or ())
    rejection = _check_entity_finiteness(checked_entities)
    if rejection is not None:
        return _reject(rejection)
    if checked_entities:
        rejection = _check_collision(
            desired_x,
            desired_y,
            checked_entities,
            cfg,
            dt=float(dt),
        )
        if rejection is not None:
            return _reject(rejection)

    return SafetyGateResult(
        desired_x=float(desired_x),
        desired_y=float(desired_y),
        safe_stop=False,
        valid=True,
        reason=PASS,
    )




"""输出单条有界机体动作的轻量多模态策略。"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class SmallPolicyConfig:
    language_dim: int = 256
    entity_count: int = 16
    entity_kinematic_dim: int = 4
    entity_embedding_dim: int = 64
    entity_geometry_dim: int = 68
    action_dim: int = 2
    ego_state_dim: int = 2
    language_hidden: int = 128
    entity_geometry_hidden: int = 64
    entity_hidden: int = 192
    ego_state_hidden: int = 64
    fusion_hidden: int = 256
    maximum_action_m: float = 0.5
    invalid_stop_logit: float = 20.0
    maximum_trainable_parameters: int = 2_000_000
    language_conditioned_entity_attention: bool = True
    entity_attention_mode: str = "language_only"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SmallPolicyConfig":
        model = value.get("model", value)
        if not isinstance(model, Mapping):
            raise ValueError("model configuration must be a mapping")
        model = dict(model)
        if "horizon" in model:
            raise ValueError(
                "legacy trajectory policy config is unsupported; remove "
                "horizon and retrain with a single [B, 2] action"
            )
        if "maximum_step_m" in model and "maximum_action_m" not in model:
            model["maximum_action_m"] = model.pop("maximum_step_m")
        known = {field.name for field in cls.__dataclass_fields__.values()}
        unknown = set(model) - known
        if unknown:
            raise ValueError(f"unknown model configuration keys: {sorted(unknown)}")
        return cls(**dict(model))

    def __post_init__(self) -> None:
        positive_ints = (
            self.language_dim,
            self.entity_count,
            self.entity_kinematic_dim,
            self.entity_embedding_dim,
            self.entity_geometry_dim,
            self.action_dim,
            self.ego_state_dim,
            self.language_hidden,
            self.entity_geometry_hidden,
            self.entity_hidden,
            self.ego_state_hidden,
            self.fusion_hidden,
            self.maximum_trainable_parameters,
        )
        if any(value <= 0 for value in positive_ints):
            raise ValueError("all dimensions and parameter limits must be positive")
        if self.entity_kinematic_dim + self.entity_embedding_dim != self.entity_geometry_dim:
            raise ValueError(
                "entity_geometry_dim must equal "
                "entity_kinematic_dim + entity_embedding_dim"
            )
        if self.action_dim != 2:
            raise ValueError("single-step body-frame action_dim must be 2")
        if self.ego_state_dim != 2:
            raise ValueError("ego_state_dim must be 2: surge_velocity and yaw_rate")
        if self.maximum_action_m <= 0.0:
            raise ValueError("maximum_action_m must be positive")
        if not torch.isfinite(torch.tensor(self.invalid_stop_logit)):
            raise ValueError("invalid_stop_logit must be finite")
        if self.entity_attention_mode not in {
            "legacy",
            "language_additive",
            "language_only",
        }:
            raise ValueError(
                f"unsupported entity_attention_mode={self.entity_attention_mode!r}"
            )
        if (
            self.entity_attention_mode != "legacy"
            and not self.language_conditioned_entity_attention
        ):
            raise ValueError(
                "language attention mode requires "
                "language_conditioned_entity_attention=true"
            )

    @property
    def maximum_step_m(self) -> float:
        """兼容旧上限名称。"""
        return self.maximum_action_m


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


class SmallActionPolicy(nn.Module):
    """融合语言和结构化实体几何，输出一条有界动作。

    The decision head receives task language, structured entity geometry and
    the current vessel dynamics. It does not receive the previous action or
    image/bbox tensors; temporal smoothness is learned from current dynamics.
    """

    def __init__(self, config: SmallPolicyConfig | None = None) -> None:
        super().__init__()
        self.config = config or SmallPolicyConfig()
        cfg = self.config

        self.language_encoder = _mlp(
            cfg.language_dim, cfg.language_hidden, cfg.language_hidden
        )
        self.entity_kinematic_encoder = _mlp(
            cfg.entity_kinematic_dim,
            cfg.entity_geometry_hidden,
            cfg.entity_geometry_hidden,
        )
        self.entity_embedding_encoder = _mlp(
            cfg.entity_embedding_dim,
            cfg.entity_geometry_hidden,
            cfg.entity_geometry_hidden,
        )
        self.entity_fusion = nn.Sequential(
            nn.Linear(cfg.entity_geometry_hidden * 2, cfg.entity_hidden),
            nn.GELU(),
        )
        self.ego_state_encoder = _mlp(
            cfg.ego_state_dim,
            cfg.ego_state_hidden,
            cfg.ego_state_hidden,
        )
        self.entity_attention = nn.Linear(cfg.entity_hidden, 1)
        if cfg.language_conditioned_entity_attention:
            self.entity_language_query: nn.Linear | None = nn.Linear(
                cfg.language_hidden,
                cfg.entity_hidden,
                bias=False,
            )
        else:
            # 保持旧版 state_dict 可加载。
            self.entity_language_query = None
        # 有效位属于决策输入，用于区分真实零动作与回合初始零哨兵。
        fusion_input_dim = (
            cfg.language_hidden + cfg.entity_hidden + cfg.ego_state_hidden + 1
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
        language: Tensor,
        entity_geometry: Tensor,
        ego_state: Tensor,
        language_valid: Tensor | None = None,
        entity_geometry_mask: Tensor | None = None,
        ego_state_valid: Tensor | None = None,
        policy_input_valid: Tensor | None = None,
        **legacy_inputs: Tensor,
    ) -> PolicyOutput:
        if legacy_inputs:
            raise ValueError(
                "decision policy accepts language, entity_geometry, "
                "ego_state, and their validity masks; "
                f"legacy inputs are unsupported: {sorted(legacy_inputs)}"
            )
        cfg = self.config
        if language.ndim != 2:
            raise ValueError("language must have shape [B, language_dim]")
        batch_size = int(language.shape[0])
        device = language.device
        dtype = language.dtype
        if (
            entity_geometry.device != device
            or ego_state.device != device
        ):
            raise ValueError("all policy inputs must be on the same device")
        if entity_geometry.dtype != dtype or ego_state.dtype != dtype:
            raise ValueError("all floating policy inputs must share one dtype")

        self._expect_shape(
            language, (batch_size, cfg.language_dim), "language"
        )
        self._expect_shape(
            entity_geometry,
            (
                batch_size,
                cfg.entity_count,
                cfg.entity_geometry_dim,
            ),
            "entity_geometry",
        )
        self._expect_shape(ego_state, (batch_size, cfg.ego_state_dim), "ego_state")
        language_mask = self._as_mask(
            language_valid,
            batch_size=batch_size,
            device=device,
            name="language_valid",
        )
        input_mask = self._as_mask(
            policy_input_valid,
            batch_size=batch_size,
            device=device,
            name="policy_input_valid",
        )

        if entity_geometry_mask is None:
            entity_geometry_mask = torch.ones(
                batch_size,
                cfg.entity_count,
                dtype=torch.bool,
                device=device,
            )
        self._expect_shape(
            entity_geometry_mask,
            (batch_size, cfg.entity_count),
            "entity_geometry_mask",
        )
        geometry_entity_mask = entity_geometry_mask.to(
            device=device, dtype=torch.bool
        )
        ego_mask = self._as_mask(
            ego_state_valid,
            batch_size=batch_size,
            device=device,
            name="ego_state_valid",
        )

        valid_mask = language_mask & input_mask
        geometry_entity_mask = geometry_entity_mask & valid_mask.unsqueeze(1)

        language_clean = self._sanitize_masked(
            language.detach(), language_mask & valid_mask, "language"
        )
        entity_geometry_clean = self._sanitize_masked(
            entity_geometry, geometry_entity_mask, "entity_geometry"
        )
        ego_state_clean = self._sanitize_masked(
            ego_state.detach(), ego_mask & valid_mask, "ego_state"
        )
        language_token = self.language_encoder(language_clean)
        kinematic = entity_geometry_clean[..., : cfg.entity_kinematic_dim]
        embedding = entity_geometry_clean[..., cfg.entity_kinematic_dim :]
        entity_token = self.entity_fusion(
            torch.cat(
                (
                    self.entity_kinematic_encoder(kinematic),
                    self.entity_embedding_encoder(embedding),
                ),
                dim=-1,
            )
        )
        ego_state_token = self.ego_state_encoder(ego_state_clean)

        attention_score = self.entity_attention(entity_token).squeeze(-1)
        if self.entity_language_query is not None:
            language_query = self.entity_language_query(language_token)
            language_score = torch.sum(
                entity_token * language_query.unsqueeze(1), dim=-1
            ) / (cfg.entity_hidden**0.5)
            mode = cfg.entity_attention_mode
            # 旧检查点没有显式 mode 字段，使用加法形式；保留其布尔标志行为。
            if mode in {"legacy", "language_additive"}:
                attention_score = attention_score + language_score
            else:
                attention_score = language_score
        attention_weight = torch.exp(attention_score.clamp(-20.0, 20.0))
        attention_weight = attention_weight * geometry_entity_mask.to(
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
                    language_token,
                    pooled_entity,
                    ego_state_token,
                    ego_mask.to(dtype=dtype).unsqueeze(-1),
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


DEFAULT_POLICY_MODEL_PATH = (
    "models/"
    "policy.pt"
)

_FLOAT_INPUT_NAMES = (
    "language",
    "entity_geometry",
    "ego_state",
)
_MASK_INPUT_NAMES = (
    "language_valid",
    "entity_geometry_mask",
    "ego_state_valid",
    "policy_input_valid",
)


class PolicyRuntimeError(RuntimeError):
    """策略检查点不满足 CUDA 合同时抛出。"""


@dataclass
class TorchPolicyRunner:
    """经过校验、仅用于推理的动作策略实例。"""

    model: SmallActionPolicy
    config: SmallPolicyConfig
    device: torch.device
    model_path: str

    @classmethod
    def load(
        cls, model_path: str | Path, *, device: str = "cuda"
    ) -> "TorchPolicyRunner":
        """将训练检查点加载到 CUDA，不回退 CPU。"""

        requested = str(device).strip().lower()
        if requested != "cuda":
            raise PolicyRuntimeError(
                f"torch policy requires device='cuda', got {device!r}"
            )
        if not torch.cuda.is_available():
            raise PolicyRuntimeError("CUDA is unavailable for the torch policy")

        path = Path(model_path).expanduser()
        if not path.is_file():
            raise PolicyRuntimeError(f"policy checkpoint not found: {path}")

        try:
            try:
                checkpoint = torch.load(
                    path, map_location="cpu", weights_only=True
                )
            except TypeError:
                # 旧版 Torch 只接受 map-location 参数；复制到 CUDA 前先校验。
                checkpoint = torch.load(path, map_location="cpu")
            if not isinstance(checkpoint, Mapping):
                raise PolicyRuntimeError("policy checkpoint must be a mapping")
            raw_config = checkpoint.get("model_config")
            config = (
                SmallPolicyConfig()
                if raw_config is None
                else SmallPolicyConfig.from_mapping(raw_config)
            )
            state_dict = checkpoint.get("model_state_dict")
            if not isinstance(state_dict, Mapping):
                raise PolicyRuntimeError(
                    "policy checkpoint is missing model_state_dict"
                )
            model = SmallActionPolicy(config)
            model.load_state_dict(state_dict, strict=True)
            target = torch.device("cuda")
            model.to(device=target)
            model.eval()
        except PolicyRuntimeError:
            raise
        except Exception as exc:
            raise PolicyRuntimeError(
                f"failed to load torch policy checkpoint {path}: {exc}"
            ) from exc

        try:
            parameter_device = next(model.parameters()).device
        except StopIteration as exc:
            raise PolicyRuntimeError("torch policy has no parameters") from exc
        if parameter_device.type != "cuda":
            raise PolicyRuntimeError(
                f"torch policy loaded on {parameter_device}, expected CUDA"
            )
        return cls(
            model=model,
            config=config,
            device=target,
            model_path=str(path),
        )

    def run(
        self, inputs: Mapping[str, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """执行一个或多个策略样本并返回 NumPy 输出。"""

        missing = [
            name for name in (*_FLOAT_INPUT_NAMES, *_MASK_INPUT_NAMES)
            if name not in inputs
        ]
        if missing:
            raise PolicyRuntimeError(
                f"policy inputs missing required fields: {', '.join(missing)}"
            )

        try:
            tensors: dict[str, torch.Tensor] = {}
            for name in _FLOAT_INPUT_NAMES:
                tensors[name] = torch.as_tensor(
                    np.asarray(inputs[name]),
                    dtype=torch.float32,
                    device=self.device,
                )
            for name in _MASK_INPUT_NAMES:
                tensors[name] = torch.as_tensor(
                    np.asarray(inputs[name]),
                    dtype=torch.bool,
                    device=self.device,
                )
            with torch.inference_mode():
                output: PolicyOutput = self.model(**tensors)
            action = output.action.detach().to("cpu").numpy()
            stop_logit = output.stop_logit.detach().to("cpu").numpy()
            valid_mask = output.valid_mask.detach().to("cpu").numpy()
        except Exception as exc:
            raise PolicyRuntimeError(f"torch policy inference failed: {exc}") from exc

        if (
            action.ndim != 2
            or action.shape[1:] != (self.config.action_dim,)
            or stop_logit.ndim != 2
            or stop_logit.shape[1:] != (1,)
            or valid_mask.ndim != 1
            or valid_mask.shape[0] != action.shape[0]
        ):
            raise PolicyRuntimeError(
                "torch policy returned an invalid output shape: "
                f"action={action.shape}, stop_logit={stop_logit.shape}, "
                f"valid_mask={valid_mask.shape}"
            )
        if not (
            np.all(np.isfinite(action))
            and np.all(np.isfinite(stop_logit))
        ):
            raise PolicyRuntimeError("torch policy returned NaN or Inf")
        return action, stop_logit, valid_mask.astype(bool, copy=False)


__all__ = [
    "DEFAULT_POLICY_MODEL_PATH",
    "PolicyOutput",
    "PolicyRuntimeError",
    "SmallActionPolicy",
    "SmallPolicyConfig",
    "TorchPolicyRunner",
]



DEFAULT_POLICY_BACKEND = "torch_cuda"
POLICY_MODEL_VERSION = "vla_torch_cuda_ego_dynamics"
LANGUAGE_DIM = 256
EGO_STATE_DIM = 2
STALE_SEC = 1.0
try:
    from .entity_contract import (
        ENTITY_COUNT,
        ENTITY_EMBEDDING_DIM,
        ENTITY_FEATURE_DIM,
        ENTITY_GEOMETRY_DIM,
        ENTITY_KINEMATIC_DIM,
        POSITION_SCALE_M,
        VELOCITY_SCALE_MPS,
    )
except ImportError:
    from entity_contract import (
        ENTITY_COUNT,
        ENTITY_EMBEDDING_DIM,
        ENTITY_FEATURE_DIM,
        ENTITY_GEOMETRY_DIM,
        ENTITY_KINEMATIC_DIM,
        POSITION_SCALE_M,
        VELOCITY_SCALE_MPS,
    )
MIN_INFERENCE_INTERVAL_SEC = DT_SEC
SYNC_CACHE_SIZE = 256
SYNC_CACHE_TTL_SEC = 5.0
POLICY_MAX_STEP_M = MAX_DISPLACEMENT_M
POLICY_MAX_ACTION_DELTA_M = 0.05


LANG_QOS = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)

FrameKey = tuple[str, int, int]
IDENTITY_FIELDS = ("run_id", "scene_seed", "frame_index")



@dataclass(frozen=True)
class _SyncEntry:
    message: Any
    received_at: float


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
    instruction_id: str
    instruction: str
    detail: str
    safety_entities: tuple[_Entity, ...] = ()


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


def build_entity_features(entities: Iterable[Any]) -> EntityFeaturesResult:
    candidates, seen_ids = [], set()
    for entity in entities:
        if not bool(entity.valid) or not bool(entity.visible):
            continue
        candidate = _entity_candidate(entity)
        if candidate[1] in seen_ids:
            raise EntityFeaturesError(f"duplicate valid visible entity_id {candidate[1]!r}")
        seen_ids.add(candidate[1])
        candidates.append(candidate)
    selected = sorted(candidates, key=_entity_sort_key)[:ENTITY_COUNT]
    features = np.zeros((ENTITY_COUNT, ENTITY_FEATURE_DIM), dtype=np.float32)
    mask = np.zeros(ENTITY_COUNT, dtype=np.bool_)
    entity_ids = [""] * ENTITY_COUNT
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


def _identity_tuple(message: Any) -> tuple[str, int, int] | None:
    """返回完整帧身份；无法信任时返回 None。"""

    try:
        run_id = str(getattr(message, "run_id")).strip()
        scene_seed = int(getattr(message, "scene_seed"))
        frame_index = int(getattr(message, "frame_index"))
    except (AttributeError, TypeError, ValueError):
        return None
    if not run_id or scene_seed <= 0 or frame_index < 0:
        return None
    return run_id, scene_seed, frame_index


def identity_mismatch_reason(
    language: Any,
    entities: Any,
) -> str | None:
    """校验任务身份，不把语言消息当作相机帧。

    ``TaskEmbedding`` is task-level: its ``run_id`` identifies the encoder
    provenance and its ``stamp_us`` is publication time.  Neither is compared
    with the camera-frame identity.  The task text, and ``instruction_id``
    when both message interfaces provide it, are the synchronization keys.
    """

    language_instruction = str(getattr(language, "instruction", "")).strip()
    entity_instruction = str(getattr(entities, "instruction", "")).strip()
    if not language_instruction or not entity_instruction:
        return "IDENTITY_MISMATCH"
    if language_instruction != entity_instruction:
        return "IDENTITY_MISMATCH"
    language_instruction_id = str(
        getattr(language, "instruction_id", "")
    ).strip()
    entity_instruction_id = str(getattr(entities, "instruction_id", "")).strip()
    if (
        language_instruction_id
        and entity_instruction_id
        and language_instruction_id != entity_instruction_id
    ):
        return "IDENTITY_MISMATCH"
    return None



def bound_policy_displacement(
    displacement: Sequence[float] | np.ndarray,
    *,
    safe_stop: bool = False,
    valid: bool = True,
    max_step_m: float = POLICY_MAX_STEP_M,
) -> tuple[float, float] | None:
    """校验并按范数限制一条 ``[desired_x, desired_y]`` 动作。"""

    if safe_stop or not valid:
        return None
    try:
        maximum = float(max_step_m)
        values = np.asarray(displacement, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(maximum) or maximum < 0.0:
        return None
    if values.size != ACTION_DIM or not np.all(np.isfinite(values)):
        return None
    norm = float(np.linalg.norm(values))
    if not math.isfinite(norm):
        return None
    if norm > maximum and norm > 0.0:
        values = values * (maximum / norm)
    return float(values[0]), float(values[1])


def smooth_policy_displacement(
    displacement: Sequence[float] | np.ndarray,
    *,
    previous_action: Sequence[float] | np.ndarray | None = None,
    max_step_m: float = POLICY_MAX_STEP_M,
    max_delta_m: float = POLICY_MAX_ACTION_DELTA_M,
) -> tuple[float, float] | None:
    """围绕上一条命令限制每帧动作变化。

    Delta smoothing is only meaningful when a previous command is available.
    Without one, return the bounded policy action directly.
    """

    current = bound_policy_displacement(displacement, max_step_m=max_step_m)
    if current is None:
        return None
    try:
        maximum_delta = float(max_delta_m)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(maximum_delta) or maximum_delta < 0.0:
        return None
    if previous_action is None:
        return current
    else:
        try:
            previous = np.asarray(previous_action, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            return None
        if previous.size != ACTION_DIM or not np.all(np.isfinite(previous)):
            return None
    current_array = np.asarray(current, dtype=np.float64)
    delta = current_array - previous
    delta_norm = float(np.linalg.norm(delta))
    if not math.isfinite(delta_norm):
        return None
    if delta_norm > maximum_delta and delta_norm > 0.0:
        current_array = previous + delta * (maximum_delta / delta_norm)
    return bound_policy_displacement(current_array, max_step_m=max_step_m)


class FrameSyncCache:
    """按场景隔离的有界结构化实体帧缓存。"""

    def __init__(
        self,
        *,
        cache_size: int = SYNC_CACHE_SIZE,
        ttl_sec: float = SYNC_CACHE_TTL_SEC,
    ) -> None:
        if int(cache_size) <= 0:
            raise ValueError("cache_size must be positive")
        if not math.isfinite(float(ttl_sec)) or float(ttl_sec) <= 0.0:
            raise ValueError("ttl_sec must be finite and positive")
        self.cache_size = int(cache_size)
        self.ttl_sec = float(ttl_sec)
        self._entities: OrderedDict[FrameKey, _SyncEntry] = OrderedDict()
        self._ego_states: OrderedDict[FrameKey, _SyncEntry] = OrderedDict()
        self._active_run: tuple[str, int] | None = None

    @staticmethod
    def key_for(message: Any) -> FrameKey:
        return (
            str(message.run_id),
            int(message.scene_seed),
            int(message.frame_index),
        )

    @property
    def active_run(self) -> tuple[str, int] | None:
        return self._active_run

    @property
    def entity_size(self) -> int:
        return len(self._entities)

    @property
    def ego_size(self) -> int:
        return len(self._ego_states)

    def keys(self) -> tuple[FrameKey, ...]:
        return tuple(self._entities.keys())

    def clear(self) -> None:
        self._entities.clear()
        self._ego_states.clear()

    def put_entities(
        self, message: Any, received_at: float | None = None
    ) -> tuple[FrameKey, bool]:
        key = self.key_for(message)
        run = (key[0], key[1])
        switched = self._active_run is not None and run != self._active_run
        if switched:
            self.clear()
        self._active_run = run
        self._entities[key] = _SyncEntry(
            message=message,
            received_at=time.monotonic() if received_at is None else float(received_at),
        )
        self._entities.move_to_end(key)
        while len(self._entities) > self.cache_size:
            self._entities.popitem(last=False)
        return key, switched

    def entity_for(self, key: FrameKey) -> Any | None:
        entry = self._entities.get(key)
        return entry.message if entry is not None else None

    def put_ego(
        self, message: Any, received_at: float | None = None
    ) -> tuple[FrameKey, bool]:
        key = self.key_for(message)
        run = (key[0], key[1])
        switched = self._active_run is not None and run != self._active_run
        if switched:
            self.clear()
        self._active_run = run
        self._ego_states[key] = _SyncEntry(
            message=message,
            received_at=time.monotonic() if received_at is None else float(received_at),
        )
        self._ego_states.move_to_end(key)
        while len(self._ego_states) > self.cache_size:
            self._ego_states.popitem(last=False)
        return key, switched

    def ego_for(self, key: FrameKey) -> Any | None:
        entry = self._ego_states.get(key)
        return entry.message if entry is not None else None

    def expire(self, now: float | None = None) -> int:
        current = time.monotonic() if now is None else float(now)
        removed = 0
        for key, entry in tuple(self._entities.items()):
            if current - entry.received_at > self.ttl_sec:
                del self._entities[key]
                removed += 1
        for key, entry in tuple(self._ego_states.items()):
            if current - entry.received_at > self.ttl_sec:
                del self._ego_states[key]
                removed += 1
        return removed

    def consume(self, key: FrameKey) -> None:
        self._entities.pop(key, None)


class DecisionNode(Node):
    """按相机节拍运行语言条件决策头。"""

    def __init__(self, model_path: str = "") -> None:
        super().__init__("decision")
        self.declare_parameter("model_path", model_path)
        self.declare_parameter("device", "cuda")
        
        self._backend = DEFAULT_POLICY_BACKEND
        policy_device = str(self.get_parameter("device").value).strip() or "cuda"
        self._frame_sync = FrameSyncCache(
            cache_size=int(self.declare_parameter("sync_cache_size", SYNC_CACHE_SIZE).value),
            ttl_sec=float(self.declare_parameter("sync_cache_ttl_sec", SYNC_CACHE_TTL_SEC).value),
        )
        self._language: TaskEmbedding | None = None
        self._language_stamp = 0.0
        self._language_task_key: str | None = None
        self._entities: DecisionEntityFrame | None = None
        self._last_out_stamp_us = 0
        self._last_inference_time = 0.0
        self._active_run: tuple[str, int] | None = None
        self._last_entity_identity: FrameKey | None = None
        self._last_entity_frame_index = -1
        self._last_inferred_frame_index = -1
        self._last_gate_frame_index = -1
        self._safety_config = SafetyGateConfig(
            max_step_m=float(
                self.declare_parameter(
                    "safety_max_step_m", MAX_DISPLACEMENT_M
                ).value
            ),
            stale_timeout_sec=float(
                self.declare_parameter("safety_stale_timeout_sec", 1.0).value
            ),
            estop_timeout_sec=float(
                self.declare_parameter("safety_estop_timeout_sec", 2.0).value
            ),
            collision_margin_m=float(
                self.declare_parameter("safety_collision_margin_m", 0.5).value
            ),
        )

        self._lang_sub = self.create_subscription(
            TaskEmbedding, "/vla/task_embedding", self._on_language, LANG_QOS
        )
        self._ent_sub = self.create_subscription(
            EntityStateArray, "/vla/entities", self._on_entities, 10
        )
        self._ego_sub = self.create_subscription(
            ASVState, "/ue/asv_state", self._on_ego_state, 10
        )
        self._pub = self.create_publisher(
            DesiredDisplacement, "/control/desired_displacement", 10
        )
        self.create_timer(1.0, self._expire_cache)

        self._torch_runner = None
        self._policy_load_error = ""
        self._model_version = POLICY_MODEL_VERSION
        try:
            requested_path = str(self.get_parameter("model_path").value) or model_path
            self._torch_runner = TorchPolicyRunner.load(
                requested_path or DEFAULT_POLICY_MODEL_PATH,
                device=policy_device,
            )
            self.get_logger().info(
                f"POLICY_READY backend={self._backend} device={policy_device} "
                f"model={requested_path or DEFAULT_POLICY_MODEL_PATH} "
                "inputs=task_embedding+structured_entities+ego_dynamics "
                "output=[desired_x,desired_y]"
            )
        except Exception as exc:
            self._policy_load_error = f"MODEL_LOAD_ERROR:{exc}"
            self.get_logger().error(
                f"POLICY_LOAD_ERROR backend={self._backend} error={self._policy_load_error}"
            )

    def _clear_control_history(self) -> None:
        self._last_gate_frame_index = -1

    def _on_ego_state(self, message: ASVState) -> None:
        """订阅当前船体动力学，作为决策输入。"""

        self._frame_sync.put_ego(message)

    def _expire_cache(self) -> None:
        self._frame_sync.expire()

    def _on_language(self, message: TaskEmbedding) -> None:
        task_key = str(getattr(message, "instruction", "")).strip()
        if self._language_task_key != task_key:
            # 新指令会使排队帧和保护历史失效；下一条匹配实体帧从冷状态开始。
            self._frame_sync.clear()
            self._last_entity_identity = None
            self._last_entity_frame_index = -1
            self._last_inferred_frame_index = -1
            self._clear_control_history()
        self._language_task_key = task_key
        self._language = message
        self._language_stamp = time.monotonic()
        for key in self._frame_sync.keys():
            self._maybe_infer(key)

    def _new_entity_features(self, source: EntityStateArray) -> DecisionEntityFrame:
        safety_entities = tuple(
            _Entity(
                entity_id=str(entity.entity_id),
                relative_x=float(entity.relative_x),
                relative_y=float(entity.relative_y),
                relative_vx=float(entity.relative_velocity_x),
                relative_vy=float(entity.relative_velocity_y),
            )
            for entity in source.entities
            if bool(entity.valid) and bool(entity.visible)
        )
        return DecisionEntityFrame(
            stamp_us=int(source.stamp_us),
            run_id=str(source.run_id),
            scene_seed=int(source.scene_seed),
            frame_index=int(source.frame_index),
            frame_id=str(source.frame_id),
            max_entities=ENTITY_COUNT,
            feature_dim=ENTITY_GEOMETRY_DIM,
            entity_count=0,
            entity_ids=[""] * ENTITY_COUNT,
            features=[0.0] * (ENTITY_COUNT * ENTITY_GEOMETRY_DIM),
            mask=[False] * ENTITY_COUNT,
            valid=False,
            instruction_id=str(source.instruction_id),
            instruction=str(source.instruction),
            detail="UNINITIALIZED",
            safety_entities=safety_entities,
        )

    def _on_entities(self, source: EntityStateArray) -> None:
        message = self._new_entity_features(source)
        if not source.valid:
            message.detail = f"INVALID_SOURCE:{source.detail}"
        elif not message.run_id:
            message.detail = "INVALID_RUN_ID: run_id is empty"
        elif message.frame_id != FRAME_ID:
            message.detail = f"INVALID_FRAME: expected {FRAME_ID}, got {message.frame_id!r}"
        else:
            try:
                result = build_entity_features(source.entities)
                message.entity_count = result.entity_count
                message.entity_ids = list(result.entity_ids)
                message.features = result.features.reshape(-1).tolist()
                message.mask = result.mask.tolist()
                message.valid = True
                message.detail = (
                    f"OK:selected={result.entity_count};dropped={result.dropped_count}"
                )
            except (EntityFeaturesError, ValueError) as exc:
                message.detail = f"{type(exc).__name__.upper()}:{exc}"
            except Exception as exc:
                message.detail = f"UNEXPECTED_ENTITY_TENSOR_ERROR:{type(exc).__name__}:{exc}"
        self._on_feature_message(message)

    def _on_feature_message(self, message: DecisionEntityFrame) -> None:
        self._entities = message
        now = time.monotonic()
        identity = _identity_tuple(message)
        if identity is None:
            self._publish_fail_closed(message, "IDENTITY_MISMATCH")
            return
        if (
            self._last_entity_identity is not None
            and identity[:2] == self._last_entity_identity[:2]
            and identity[2] <= self._last_entity_identity[2]
        ):
            self._publish_fail_closed(message, "IDENTITY_MISMATCH")
            return

        if (
            self._language is not None
            and identity_mismatch_reason(self._language, message) is not None
        ):
            self._frame_sync.clear()
            self._clear_control_history()
            self._last_entity_identity = identity
            self._last_entity_frame_index = identity[2]
            self._last_inferred_frame_index = -1
            self._publish_fail_closed(message, "IDENTITY_MISMATCH")
            return

        key, switched = self._frame_sync.put_entities(message, received_at=now)
        run = identity[:2]
        if switched or self._active_run != run:
            self._active_run = run
            self._last_entity_frame_index = -1
            self._last_inferred_frame_index = -1
            self._last_entity_identity = None
            self._clear_control_history()
        elif (
            self._last_entity_frame_index >= 0
            and identity[2] <= self._last_entity_frame_index
        ):
            self._clear_control_history()
        self._last_entity_identity = identity
        self._last_entity_frame_index = identity[2]
        self._maybe_infer(key, now=now)

    def _new_output(self, ent: DecisionEntityFrame) -> DesiredDisplacement:
        message = DesiredDisplacement()
        stamp = int(ent.stamp_us)
        self._last_out_stamp_us = max(stamp, self._last_out_stamp_us + 1)
        message.stamp_us = self._last_out_stamp_us
        message.run_id = str(ent.run_id)
        message.scene_seed = int(ent.scene_seed)
        message.frame_index = int(ent.frame_index)
        message.frame_id = FRAME_ID
        message.source = self._model_version
        message.step_dt = DT_SEC
        return message

    def _publish_fail_closed(
        self,
        ent: DecisionEntityFrame,
        reason: str,
        *,
        raw_action: Sequence[float] | np.ndarray | None = None,
    ) -> None:
        self._clear_control_history()
        message = self._new_output(ent)
        message.desired_x = 0.0
        message.desired_y = 0.0
        message.safe_stop = True
        message.valid = False
        message.reason = str(reason)
        self._pub.publish(message)

    def _maybe_infer(
        self,
        key: FrameKey,
        *,
        now: float | None = None,
    ) -> None:
        current = time.monotonic() if now is None else float(now)
        if current - self._last_inference_time < MIN_INFERENCE_INTERVAL_SEC:
            return
        ent = self._frame_sync.entity_for(key)
        if ent is None:
            return
        if self._language is None:
            return
        identity_reason = identity_mismatch_reason(self._language, ent)
        if identity_reason is not None:
            self._frame_sync.consume(key)
            self._clear_control_history()
            self._last_inference_time = current
            self._last_inferred_frame_index = int(ent.frame_index)
            self._publish_fail_closed(ent, identity_reason)
            return
        self._frame_sync.consume(key)
        self._last_inference_time = current
        message = self._new_output(ent)
        self._last_inferred_frame_index = int(ent.frame_index)

        ego = self._frame_sync.ego_for(key)
        if ego is None or _identity_tuple(ego) != key or not bool(ego.valid):
            self._publish_fail_closed(ent, "MISSING_OR_MISMATCHED_EGO_STATE")
            return

        if self._torch_runner is None:
            self._publish_fail_closed(ent, self._policy_load_error or "NO_MODEL_LOADED")
            return

        try:
            inputs = self._build_inputs(
                self._language,
                ent,
                ego_state=ego,
            )
            action, stop_logit, valid_mask = self._torch_runner.run(inputs)
            action = np.asarray(action, dtype=np.float32)
            stop_logit = np.asarray(stop_logit, dtype=np.float32)
            valid_mask = np.asarray(valid_mask, dtype=bool)
            if (
                action.shape != (1, ACTION_DIM)
                or stop_logit.shape != (1, 1)
                or valid_mask.shape != (1,)
                or not np.all(np.isfinite(action))
                or not np.all(np.isfinite(stop_logit))
            ):
                raise ValueError(
                    f"invalid direct policy shapes action={action.shape} "
                    f"stop_logit={stop_logit.shape} valid_mask={valid_mask.shape}"
                )
        except Exception as exc:
            self._publish_fail_closed(ent, f"INFERENCE_ERROR:{exc}")
            return

        policy_valid = bool(valid_mask[0]) and bool(self._language.valid) and bool(ent.valid)
        stop = float(stop_logit[0, 0]) >= 0.0
        if stop or not policy_valid:
            self._publish_fail_closed(ent, "POLICY_STOP" if stop else "POLICY_INVALID")
            return

        displacement = bound_policy_displacement(action[0], valid=True)
        if displacement is None:
            self._publish_fail_closed(ent, "POLICY_ACTION_INVALID")
            return

        safety = evaluate_safety_gate(
            stamp_us=int(message.stamp_us),
            run_id=str(ent.run_id),
            frame_id=FRAME_ID,
            model_version=self._model_version,
            dt=DT_SEC,
            desired_x=float(displacement[0]),
            desired_y=float(displacement[1]),
            safe_stop=False,
            valid=True,
            reason="POLICY_INFERRED_SMOOTHED",
            language_valid=bool(self._language.valid),
            entity_valid=bool(ent.valid),
            entities=ent.safety_entities,
            last_valid_stamp_us=0,
            time_since_last_valid_sec=0.0,
            config=self._safety_config,
        )
        if not safety.valid or safety.safe_stop:
            self._publish_fail_closed(ent, safety.reason, raw_action=displacement)
            return

        message.desired_x = float(displacement[0])
        message.desired_y = float(displacement[1])
        message.safe_stop = False
        message.valid = True
        message.reason = "POLICY_INFERRED_SMOOTHED"
        self._pub.publish(message)

    @staticmethod
    def _build_inputs(
        language: TaskEmbedding,
        entities: DecisionEntityFrame,
        *,
        ego_state: ASVState,
    ) -> dict[str, np.ndarray]:
        """构建语言、实体和当前动力学策略输入。"""

        identity_reason = identity_mismatch_reason(language, entities)
        if identity_reason is not None:
            raise ValueError(identity_reason)
        embedding = np.asarray(language.embedding, dtype=np.float32).reshape(-1)
        if embedding.size != LANGUAGE_DIM or not np.all(np.isfinite(embedding)):
            raise ValueError("task embedding must be finite float32[256]")
        if int(entities.max_entities) != ENTITY_COUNT:
            raise ValueError("EntityFeatures max_entities does not match policy contract")
        if int(entities.feature_dim) != ENTITY_GEOMETRY_DIM:
            raise ValueError("EntityFeatures feature_dim does not match policy contract")
        geometry = np.asarray(entities.features, dtype=np.float32).reshape(
            ENTITY_COUNT, ENTITY_GEOMETRY_DIM
        )
        mask = np.asarray(entities.mask, dtype=bool).reshape(ENTITY_COUNT)
        if not np.all(np.isfinite(geometry[mask])):
            raise ValueError("active structured entity features are non-finite")
        ego_values = np.asarray(
            [
                np.clip(float(ego_state.surge_velocity) / 5.0, -1.0, 1.0),
                np.clip(float(ego_state.yaw_rate) / 1.0, -1.0, 1.0),
            ],
            dtype=np.float32,
        )
        ego_valid = bool(ego_state.valid) and bool(np.all(np.isfinite(ego_values)))
        policy_valid = bool(language.valid) and bool(entities.valid) and bool(np.any(mask)) and ego_valid
        return {
            "language": embedding.reshape(1, LANGUAGE_DIM),
            "entity_geometry": geometry.reshape(1, ENTITY_COUNT, ENTITY_GEOMETRY_DIM),
            "ego_state": ego_values.reshape(1, EGO_STATE_DIM),
            "language_valid": np.asarray([bool(language.valid)], dtype=bool),
            "entity_geometry_mask": mask.reshape(1, ENTITY_COUNT),
            "ego_state_valid": np.asarray([ego_valid], dtype=bool),
            "policy_input_valid": np.asarray([policy_valid], dtype=bool),
        }



def main(args=None) -> None:
    rclpy.init(args=args)
    node = DecisionNode()
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
