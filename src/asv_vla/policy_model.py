"""Small multimodal policy with one bounded body-frame action output."""

from __future__ import annotations

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
    entity_geometry_dim: int = 16
    action_dim: int = 2
    previous_action_dim: int = 2
    language_hidden: int = 128
    entity_geometry_hidden: int = 64
    entity_hidden: int = 192
    previous_action_hidden: int = 64
    fusion_hidden: int = 256
    maximum_action_m: float = 0.3
    invalid_stop_logit: float = 20.0
    maximum_trainable_parameters: int = 2_000_000
    language_conditioned_entity_attention: bool = False
    entity_attention_mode: str = "legacy"

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
            self.entity_geometry_dim,
            self.action_dim,
            self.previous_action_dim,
            self.language_hidden,
            self.entity_geometry_hidden,
            self.entity_hidden,
            self.previous_action_hidden,
            self.fusion_hidden,
            self.maximum_trainable_parameters,
        )
        if any(value <= 0 for value in positive_ints):
            raise ValueError("all dimensions and parameter limits must be positive")
        if self.action_dim != 2:
            raise ValueError("single-step body-frame action_dim must be 2")
        if self.previous_action_dim != 2:
            raise ValueError("previous_action_dim must be 2")
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
        """Compatibility spelling for callers that used the old bound."""
        return self.maximum_action_m


@dataclass(frozen=True)
class PolicyOutput:
    """Policy tensors plus an explicit fail-closed sample-validity mask."""

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
    """Fuse language and structured entity geometry into one bounded action.

    The decision head receives the task embedding, the structured output of the
    perception/tracking head, and the previous single-step action. Image and
    ego tensors are deliberately not accepted here; the entity tensor already
    contains image-derived color, relative geometry, and tracked relative
    velocity.
    """

    def __init__(self, config: SmallPolicyConfig | None = None) -> None:
        super().__init__()
        self.config = config or SmallPolicyConfig()
        cfg = self.config

        self.language_encoder = _mlp(
            cfg.language_dim, cfg.language_hidden, cfg.language_hidden
        )
        self.entity_geometry_encoder = _mlp(
            cfg.entity_geometry_dim,
            cfg.entity_geometry_hidden,
            cfg.entity_geometry_hidden,
        )
        self.entity_fusion = nn.Sequential(
            nn.Linear(
                cfg.entity_geometry_hidden,
                cfg.entity_hidden,
            ),
            nn.GELU(),
        )
        self.previous_action_encoder = _mlp(
            cfg.previous_action_dim,
            cfg.previous_action_hidden,
            cfg.previous_action_hidden,
        )
        self.entity_attention = nn.Linear(cfg.entity_hidden, 1)
        if cfg.language_conditioned_entity_attention:
            self.entity_language_query: nn.Linear | None = nn.Linear(
                cfg.language_hidden,
                cfg.entity_hidden,
                bias=False,
            )
        else:
            # Keep the Day 14/15 v1 state_dict exactly loadable.
            self.entity_language_query = None
        # The validity bit is part of the decision input so a real zero action
        # is distinguishable from the zero sentinel used at episode start.
        fusion_input_dim = (
            cfg.language_hidden + cfg.entity_hidden + cfg.previous_action_hidden + 1
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
        previous_action: Tensor,
        language_valid: Tensor | None = None,
        entity_geometry_mask: Tensor | None = None,
        previous_action_valid: Tensor | None = None,
        policy_input_valid: Tensor | None = None,
        **legacy_inputs: Tensor,
    ) -> PolicyOutput:
        if legacy_inputs:
            raise ValueError(
                "decision policy accepts language, entity_geometry, "
                "previous_action, and their validity masks; "
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
            or previous_action.device != device
        ):
            raise ValueError("all policy inputs must be on the same device")
        if entity_geometry.dtype != dtype or previous_action.dtype != dtype:
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
        self._expect_shape(
            previous_action,
            (batch_size, cfg.previous_action_dim),
            "previous_action",
        )
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
        previous_mask = self._as_mask(
            previous_action_valid,
            batch_size=batch_size,
            device=device,
            name="previous_action_valid",
        )

        valid_mask = language_mask & input_mask
        geometry_entity_mask = geometry_entity_mask & valid_mask.unsqueeze(1)

        language_clean = self._sanitize_masked(
            language.detach(), language_mask & valid_mask, "language"
        )
        entity_geometry_clean = self._sanitize_masked(
            entity_geometry, geometry_entity_mask, "entity_geometry"
        )
        previous_action_clean = self._sanitize_masked(
            previous_action.detach(), previous_mask & valid_mask, "previous_action"
        )
        language_token = self.language_encoder(language_clean)
        entity_geometry_token = self.entity_geometry_encoder(
            entity_geometry_clean
        )
        entity_token = self.entity_fusion(
            entity_geometry_token
        )
        previous_action_token = self.previous_action_encoder(previous_action_clean)

        attention_score = self.entity_attention(entity_token).squeeze(-1)
        if self.entity_language_query is not None:
            language_query = self.entity_language_query(language_token)
            language_score = torch.sum(
                entity_token * language_query.unsqueeze(1), dim=-1
            ) / (cfg.entity_hidden**0.5)
            mode = cfg.entity_attention_mode
            # The first v2 checkpoint predates the explicit mode field and
            # used the additive form.  Preserve that behavior when its boolean
            # flag is true and the new field retains the legacy default.
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
                    previous_action_token,
                    previous_mask.to(dtype=dtype).unsqueeze(-1),
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
        # A positive STOP decision is an execution contract, not merely a
        # request to move less. Keep the continuous gate above for useful
        # training gradients, then make the published policy output exactly
        # stationary whenever the classifier selects STOP.
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
    "policy_single_point_v3_full_seed17.pt"
)

_FLOAT_INPUT_NAMES = (
    "language",
    "entity_geometry",
    "previous_action",
)
_MASK_INPUT_NAMES = (
    "language_valid",
    "entity_geometry_mask",
    "previous_action_valid",
    "policy_input_valid",
)


class PolicyRuntimeError(RuntimeError):
    """Raised when a policy checkpoint cannot satisfy the CUDA contract."""


@dataclass
class TorchPolicyRunner:
    """A checked, inference-only instance of the action policy."""

    model: SmallActionPolicy
    config: SmallPolicyConfig
    device: torch.device
    model_path: str

    @classmethod
    def load(
        cls, model_path: str | Path, *, device: str = "cuda"
    ) -> "TorchPolicyRunner":
        """Load a training checkpoint onto CUDA without a CPU fallback."""

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
                # Torch versions before ``weights_only`` accepted only the
                # map-location argument.  Validate before copying to CUDA.
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
        """Run one or more policy samples and return NumPy outputs."""

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


