from __future__ import annotations

from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("USE_TORCH", "1")

import numpy as np

DEFAULT_TASK_DESCRIPTION = (
    "Encode an instruction for a twin-thruster unmanned surface vessel "
    "performing follow or stop tasks."
)
EMBEDDING_DIM = 256
DEFAULT_MODEL_PATH = "models/Qwen3-Embedding-0.6B"
DEFAULT_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"


class TaskEncoderError(RuntimeError):
    pass


@dataclass(frozen=True)
class TaskEmbeddingState:
    instruction: str = ""
    embedding: tuple[float, ...] = field(default_factory=lambda: (0.0,) * EMBEDDING_DIM)
    model_id: str = DEFAULT_MODEL_ID
    cached: bool = False
    valid: bool = False
    detail: str = ""


def state_payload(state: TaskEmbeddingState, *, run_id: str, stamp_us: int) -> dict[str, Any]:
    return {
        "stamp_us": int(stamp_us),
        "run_id": str(run_id),
        "instruction": state.instruction,
        "model_id": state.model_id,
        "embedding_dim": EMBEDDING_DIM,
        "embedding": list(state.embedding),
        "cached": state.cached,
        "valid": state.valid,
        "detail": str(state.detail)[:240],
    }


def embedding_tuple(values: object) -> tuple[float, ...]:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if array.size != EMBEDDING_DIM:
        raise ValueError("invalid embedding size")
    return tuple(float(v) for v in array)


@dataclass(frozen=True)
class EncodingResult:
    embedding: np.ndarray
    cached: bool


def _apply_int8_quantization(st_model: Any) -> None:
    import torch
    from torch.ao.quantization import quantize_dynamic

    torch.backends.quantized.engine = "qnnpack"
    auto_model = st_model[0].auto_model.cpu().float()
    st_model[0].auto_model = quantize_dynamic(
        auto_model, {torch.nn.Linear}, dtype=torch.qint8
    )
    st_model.to("cpu")


class TaskEncoder:
    def __init__(
        self,
        model_path: str,
        *,
        task_description: str = DEFAULT_TASK_DESCRIPTION,
        device: str = "cpu",
        quantize_int8: bool = True,
        model: Any | None = None,
    ) -> None:
        self.model_path = str(model_path)
        self.task_description = task_description.strip()
        self.quantize_int8 = bool(quantize_int8)
        self.device = "cpu" if self.quantize_int8 else device
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._model = model if model is not None else self._load_model()
        if hasattr(self._model, "eval"):
            self._model.eval()

    def _load_model(self) -> Any:
        from sentence_transformers import SentenceTransformer
        import torch

        kwargs: dict[str, Any] = {"low_cpu_mem_usage": True}
        load_device = "cpu" if self.quantize_int8 else self.device
        if not self.quantize_int8 and self.device.startswith("cuda"):
            kwargs["torch_dtype"] = torch.float16
        model = SentenceTransformer(
            str(Path(self.model_path).expanduser()),
            device=load_device,
            local_files_only=True,
            model_kwargs=kwargs,
        )
        if self.quantize_int8:
            _apply_int8_quantization(model)
        return model

    def _prompt(self, instruction: str) -> str:
        return f"Instruct: {self.task_description}\nQuery: {instruction.strip()}"

    def encode_with_metadata(self, text: str) -> EncodingResult:
        instruction = text.strip()
        if not instruction:
            raise TaskEncoderError("empty instruction")
        if instruction in self._cache:
            return EncodingResult(self._cache[instruction].copy(), True)
        with torch_inference():
            raw = self._model.encode(
                [self._prompt(instruction)],
                batch_size=1,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )
        vector = np.asarray(raw[0], dtype=np.float32)[:EMBEDDING_DIM]
        vector /= max(float(np.linalg.norm(vector)), 1e-12)
        self._cache[instruction] = vector.copy()
        return EncodingResult(vector, False)

    def encode(self, text: str) -> np.ndarray:
        return self.encode_with_metadata(text).embedding


def torch_inference():
    try:
        import torch

        return torch.inference_mode()
    except ImportError:
        return nullcontext()
