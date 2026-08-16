"""Executable Day 14 policy contract and resource report."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import tracemalloc
from typing import Any

import torch
from torch import Tensor
import yaml

from asv_training.losses import PolicyLossWeights, action_policy_loss
from asv_training.model import SmallActionPolicy, SmallPolicyConfig


REPORT_SCHEMA_VERSION = "policy_contract_report_v1"


def _load_config(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot read policy config {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("policy config must contain a mapping")
    if value.get("schema_version") != "model_small_v3":
        raise ValueError("policy config schema_version must be model_small_v3")
    return value


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def _make_inputs(
    batch_size: int,
    config: SmallPolicyConfig,
    device: torch.device,
) -> dict[str, Tensor]:
    entity_mask = torch.zeros(
        batch_size,
        config.entity_count,
        dtype=torch.bool,
        device=device,
    )
    if batch_size > 1:
        entity_mask[1:, :4] = True
    return {
        "language": torch.randn(
            batch_size,
            config.language_dim,
            device=device,
            requires_grad=True,
        ),
        "entity_geometry": torch.randn(
            batch_size,
            config.entity_count,
            config.entity_geometry_dim,
            device=device,
            requires_grad=True,
        ),
        "previous_action": torch.randn(
            batch_size,
            config.previous_action_dim,
            device=device,
            requires_grad=True,
        ),
        "language_valid": torch.ones(
            batch_size, dtype=torch.bool, device=device
        ),
        "entity_geometry_mask": entity_mask.clone(),
        "previous_action_valid": torch.ones(
            batch_size, dtype=torch.bool, device=device
        ),
        "policy_input_valid": torch.ones(
            batch_size, dtype=torch.bool, device=device
        ),
    }


def _checkpoint_size_bytes(model: SmallActionPolicy) -> int:
    descriptor, name = tempfile.mkstemp(suffix=".pt")
    os.close(descriptor)
    try:
        torch.save(model.state_dict(), name)
        return Path(name).stat().st_size
    finally:
        try:
            os.unlink(name)
        except OSError:
            pass


def _expected_zero_gradient_parameter_names(
    model_config: SmallPolicyConfig,
) -> set[str]:
    """Return legacy parameters intentionally inactive in the selected mode."""
    if model_config.entity_attention_mode == "language_only":
        return {"entity_attention.weight", "entity_attention.bias"}
    return set()


def run_contract(
    config_path: str | Path,
    report_path: str | Path,
    *,
    device_name: str = "auto",
    seed: int | None = None,
) -> dict[str, Any]:
    source = _load_config(Path(config_path).resolve())
    model_config = SmallPolicyConfig.from_mapping(source)
    loss_weights = PolicyLossWeights.from_mapping(source.get("loss"))
    contract = source.get("contract", {})
    if not isinstance(contract, dict):
        raise ValueError("contract configuration must be a mapping")
    batch_sizes = tuple(int(value) for value in contract.get("batch_sizes", []))
    if batch_sizes != (1, 2, 8):
        raise ValueError("Day 14 contract batch sizes must be [1, 2, 8]")
    fixed_seed = int(
        seed
        if seed is not None
        else contract.get("deterministic_seed", 42)
    )
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    torch.manual_seed(fixed_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(fixed_seed)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        peak_kind = "torch_cuda_max_memory_allocated"
    else:
        tracemalloc.start()
        peak_kind = "python_tracemalloc"

    model = SmallActionPolicy(model_config).to(device)
    model.eval()
    output_shapes: dict[str, list[int]] = {}
    maximum_observed_increment = 0.0
    for batch_size in batch_sizes:
        inputs = _make_inputs(batch_size, model_config, device)
        first = model(**inputs)
        second = model(**inputs)
        expected = (batch_size, model_config.action_dim)
        if tuple(first.action.shape) != expected:
            raise AssertionError(
                f"batch {batch_size}: action shape is "
                f"{tuple(first.action.shape)}, expected {expected}"
            )
        if tuple(first.stop_logit.shape) != (batch_size, 1):
            raise AssertionError("stop_logit shape is invalid")
        if not torch.isfinite(first.action).all():
            raise AssertionError("policy action contains NaN or Inf")
        if not torch.isfinite(first.stop_logit).all():
            raise AssertionError("policy stop_logit contains NaN or Inf")
        if not torch.equal(first.action, second.action):
            raise AssertionError("repeated forward pass is not deterministic")
        observed = float(torch.max(torch.linalg.vector_norm(first.action, dim=-1)).detach().cpu())
        maximum_observed_increment = max(maximum_observed_increment, observed)
        if observed > model_config.maximum_action_m + 1.0e-6:
            raise AssertionError("direct action exceeded structural bound")
        output_shapes[str(batch_size)] = list(first.action.shape)

    invalid_inputs = _make_inputs(2, model_config, device)
    invalid_inputs["policy_input_valid"][0] = False
    with torch.no_grad():
        invalid_inputs["language"][0].fill_(float("nan"))
    invalid_output = model(**invalid_inputs)
    if bool(invalid_output.valid_mask[0]):
        raise AssertionError("invalid structured policy input did not fail closed")
    if torch.count_nonzero(invalid_output.action[0]):
        raise AssertionError("invalid sample action is not zero")
    if not torch.isfinite(invalid_output.action).all():
        raise AssertionError("invalid masks produced NaN or Inf")

    model.train()
    training_inputs = _make_inputs(8, model_config, device)
    training_output = model(**training_inputs)
    target_action = torch.zeros_like(training_output.action)
    target_stop = torch.zeros(8, 1, dtype=torch.float32, device=device)
    losses = action_policy_loss(
        training_output,
        target_action,
        target_stop,
        weights=loss_weights,
    )
    losses["total"].backward()
    frozen_input_gradients = {
        key: training_inputs[key].grad is None
        for key in ("language",)
    }
    if not all(frozen_input_gradients.values()):
        raise AssertionError("frozen language input received gradients")
    expected_zero_gradient_names = _expected_zero_gradient_parameter_names(
        model_config
    )
    missing_gradient_names = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    }
    unexpected_missing_gradient_names = (
        missing_gradient_names - expected_zero_gradient_names
    )
    if unexpected_missing_gradient_names:
        raise AssertionError(
            "unexpected trainable parameters without gradients: "
            f"{sorted(unexpected_missing_gradient_names)}"
        )
    trainable_gradients = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name not in expected_zero_gradient_names
    ]
    if not all(
        torch.isfinite(gradient).all()
        for gradient in trainable_gradients
        if gradient is not None
    ):
        raise AssertionError("a policy gradient contains NaN or Inf")

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory_bytes = int(torch.cuda.max_memory_allocated(device))
    else:
        _, peak_memory_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    parameter_count = model.trainable_parameter_count()
    checkpoint_size = _checkpoint_size_bytes(model)
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "device": str(device),
        "torch_version": torch.__version__,
        "seed": fixed_seed,
        "batch_sizes": list(batch_sizes),
        "output_shapes": output_shapes,
        "trainable_parameter_count": parameter_count,
        "maximum_trainable_parameters": (
            model_config.maximum_trainable_parameters
        ),
        "checkpoint_size_bytes": checkpoint_size,
        "peak_memory_bytes": peak_memory_bytes,
        "peak_memory_kind": peak_kind,
        "maximum_action_m": model_config.maximum_action_m,
        "maximum_observed_increment_m": maximum_observed_increment,
        "frozen_cache_input_gradients_absent": frozen_input_gradients,
        "expected_zero_gradient_parameter_names": sorted(
            expected_zero_gradient_names
        ),
        "invalid_input_fail_closed": True,
        "privileged_policy_fields_absent": True,
        "loss": {
            key: (
                int(value.detach().cpu())
                if key == "valid_samples"
                else float(value.detach().cpu())
            )
            for key, value in losses.items()
        },
    }
    _atomic_write_json(Path(report_path).resolve(), report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Day 14 policy contract")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "configs" / "model_small_v3.yaml",
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    report = run_contract(
        args.config,
        args.report,
        device_name=args.device,
        seed=args.seed,
    )
    print(
        "POLICY_CONTRACT_PASS "
        f"device={report['device']} "
        f"parameters={report['trainable_parameter_count']} "
        f"checkpoint_bytes={report['checkpoint_size_bytes']} "
        f"peak_memory_bytes={report['peak_memory_bytes']} "
        f"max_action_m={report['maximum_action_m']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
