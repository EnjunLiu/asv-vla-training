from __future__ import annotations

from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass
import os
from pathlib import Path
from threading import RLock
from typing import Any, Callable

# The Jetson system environment may contain an incompatible TensorFlow/tf-keras
# pair. This component is deliberately PyTorch-only, so prevent Transformers
# from probing optional backends before sentence-transformers is imported.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("USE_TORCH", "1")

import numpy as np


DEFAULT_TASK_DESCRIPTION = (
    "Encode an instruction for a twin-thruster unmanned surface vessel "
    "performing follow or stop tasks."
)


class LanguageEncoderError(RuntimeError):
    """Base class for deterministic language-encoder failures."""


class LanguageEncoderMemoryError(LanguageEncoderError):
    """Raised when the backend reports an allocation/low-memory failure."""


class EmptyInstructionError(LanguageEncoderError):
    """Raised when the task instruction is empty."""


class InstructionTooLongError(LanguageEncoderError):
    """Raised when the task instruction exceeds the application limit."""


class InvalidEmbeddingError(LanguageEncoderError):
    """Raised when the backend returns an unusable embedding."""


@dataclass(frozen=True)
class EncodingResult:
    embedding: np.ndarray
    cached: bool


class USVLanguageEncoder:
    """Frozen, cached Qwen embedding wrapper with a fixed 256-D contract."""

    def __init__(
        self,
        model_path: str,
        *,
        output_dim: int = 256,
        max_chars: int = 512,
        task_description: str = DEFAULT_TASK_DESCRIPTION,
        device: str = "cuda",
        cache_size: int = 32,
        model: Any | None = None,
        inference_context: Callable[[], Any] | None = None,
    ) -> None:
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        if cache_size <= 0:
            raise ValueError("cache_size must be positive")
        if not task_description.strip():
            raise ValueError("task_description must not be empty")

        self.model_path = str(model_path)
        self.output_dim = int(output_dim)
        self.max_chars = int(max_chars)
        self.task_description = task_description.strip()
        self.device = device
        self.cache_size = int(cache_size)
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_lock = RLock()

        self._model = model if model is not None else self._load_model()
        self._inference_context = (
            inference_context
            if inference_context is not None
            else self._make_inference_context()
        )
        self._freeze_model()
        self._validate_native_dimension()

    def _load_model(self) -> Any:
        model_dir = Path(self.model_path).expanduser().resolve()
        if not model_dir.is_dir():
            raise LanguageEncoderError(f"model directory does not exist: {model_dir}")

        try:
            import torch
        except ImportError as exc:
            raise LanguageEncoderError("PyTorch is not installed") from exc

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise LanguageEncoderError("CUDA was requested but is not available")

        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            raise LanguageEncoderError(
                "failed to import sentence-transformers; verify the pinned "
                "PyTorch-only dependencies and USE_TF=0: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        model_kwargs = {}
        if self.device.startswith("cuda"):
            # Avoid a temporary/default FP32 load exhausting Orin unified memory.
            model_kwargs["torch_dtype"] = torch.float16
            # Materialize weights with a lower CPU-side peak before moving them
            # to CUDA; Jetson CPU and GPU share the same physical memory.
            model_kwargs["low_cpu_mem_usage"] = True

        try:
            return SentenceTransformer(
                str(model_dir),
                device=self.device,
                local_files_only=True,
                model_kwargs=model_kwargs,
            )
        except Exception as exc:
            raise LanguageEncoderError(
                f"failed to load language model from {model_dir}: {exc}"
            ) from exc

    @staticmethod
    def _make_inference_context() -> Callable[[], Any]:
        try:
            import torch
        except ImportError:
            return nullcontext
        return torch.inference_mode

    def _freeze_model(self) -> None:
        eval_method = getattr(self._model, "eval", None)
        if callable(eval_method):
            eval_method()

        parameters_method = getattr(self._model, "parameters", None)
        if callable(parameters_method):
            for parameter in parameters_method():
                requires_grad_method = getattr(parameter, "requires_grad_", None)
                if callable(requires_grad_method):
                    requires_grad_method(False)

    def _validate_native_dimension(self) -> None:
        dimension_method = getattr(
            self._model, "get_sentence_embedding_dimension", None
        )
        if not callable(dimension_method):
            return
        native_dim = dimension_method()
        if native_dim is not None and int(native_dim) < self.output_dim:
            raise InvalidEmbeddingError(
                f"native embedding dimension {native_dim} is smaller than "
                f"requested output dimension {self.output_dim}"
            )

    def _normalize_instruction(self, text: str) -> str:
        if not isinstance(text, str):
            raise LanguageEncoderError("instruction must be a string")
        normalized = text.strip()
        if not normalized:
            raise EmptyInstructionError("instruction is empty")
        if len(normalized) > self.max_chars:
            raise InstructionTooLongError(
                f"instruction length {len(normalized)} exceeds "
                f"max_chars={self.max_chars}"
            )
        return normalized

    def _prompt(self, instruction: str) -> str:
        return (
            f"Instruct: {self.task_description}\n"
            f"Query: {instruction}"
        )

    def _cache_key(self, instruction: str) -> str:
        return (
            f"{self.output_dim}\0{self.task_description}\0{instruction}"
        )

    def _compute(self, instruction: str) -> np.ndarray:
        prompt = self._prompt(instruction)
        try:
            with self._inference_context():
                raw = self._model.encode(
                    [prompt],
                    # Online ROS callbacks encode one changed instruction at a
                    # time.  SentenceTransformer's default batch_size=32 can
                    # create a large temporary peak on Jetson unified memory.
                    batch_size=1,
                    convert_to_numpy=True,
                    normalize_embeddings=False,
                    show_progress_bar=False,
                )
        except Exception as exc:
            error_text = str(exc).lower()
            if isinstance(exc, MemoryError) or any(
                marker in error_text
                for marker in (
                    "out of memory",
                    "cuda oom",
                    "nvmapmemalloc",
                    "nvml_success",
                    "memory allocation",
                )
            ):
                error_kind = (
                    "CUDA_MEMORY_ERROR"
                    if self.device.startswith("cuda")
                    else "MEMORY_ERROR"
                )
                raise LanguageEncoderMemoryError(
                    f"{error_kind}: batch_size=1; "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            raise LanguageEncoderError(
                f"embedding inference failed; batch_size=1; "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        array = np.asarray(raw)
        if array.ndim == 2 and array.shape[0] == 1:
            array = array[0]
        vector = np.asarray(array, dtype=np.float32).reshape(-1)
        if vector.size < self.output_dim:
            raise InvalidEmbeddingError(
                f"backend returned {vector.size} values; "
                f"{self.output_dim} are required"
            )

        # Qwen3-Embedding supports Matryoshka Representation Learning.
        # Truncate first, then normalize the retained dimensions.
        vector = np.ascontiguousarray(vector[: self.output_dim], dtype=np.float32)
        if not np.all(np.isfinite(vector)):
            raise InvalidEmbeddingError("embedding contains NaN or Inf")
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= 1.0e-12:
            raise InvalidEmbeddingError("embedding norm is zero or invalid")
        return np.ascontiguousarray(vector / norm, dtype=np.float32)

    def encode_with_metadata(self, text: str) -> EncodingResult:
        instruction = self._normalize_instruction(text)
        key = self._cache_key(instruction)

        with self._cache_lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                return EncodingResult(cached.copy(), True)

        embedding = self._compute(instruction)
        with self._cache_lock:
            self._cache[key] = embedding.copy()
            self._cache.move_to_end(key)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return EncodingResult(embedding.copy(), False)

    def encode(self, text: str) -> np.ndarray:
        return self.encode_with_metadata(text).embedding

    @property
    def cache_entries(self) -> int:
        with self._cache_lock:
            return len(self._cache)

    def clear_cache(self) -> None:
        with self._cache_lock:
            self._cache.clear()


