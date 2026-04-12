"""KV cache quantization — reduce cache memory via lower precision.

Provides hooks for quantizing KV cache tensors in compatible runtimes.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class KVCacheQuantizer:
    """Quantize KV cache tensors to reduce memory footprint."""

    def __init__(self, bits: int = 8) -> None:
        self.bits = bits
        self._torch_available = False
        self._check_torch()

    def _check_torch(self) -> None:
        try:
            import torch  # noqa: F401
            self._torch_available = True
        except ImportError:
            pass

    def quantize_tensor(self, tensor: Any, bits: int | None = None) -> Any:
        """Quantize a single tensor to the specified bit width."""
        if not self._torch_available:
            logger.debug("Torch not available, skipping quantization")
            return tensor

        import torch

        target_bits = bits or self.bits

        if not isinstance(tensor, torch.Tensor):
            return tensor

        if target_bits == 8:
            return self._quantize_int8(tensor)
        elif target_bits == 4:
            return self._quantize_int4(tensor)
        else:
            logger.warning("Unsupported bit width: %d, returning original", target_bits)
            return tensor

    def _quantize_int8(self, tensor: Any) -> Any:
        """Quantize to INT8."""
        import torch

        # Per-tensor symmetric quantization
        abs_max = tensor.abs().max()
        if abs_max == 0:
            return tensor.to(torch.int8), torch.tensor(1.0)

        scale = abs_max / 127.0
        quantized = (tensor / scale).round().clamp(-128, 127).to(torch.int8)
        return quantized, scale

    def _quantize_int4(self, tensor: Any) -> Any:
        """Quantize to INT4 (packed into INT8)."""
        import torch

        abs_max = tensor.abs().max()
        if abs_max == 0:
            return tensor.to(torch.int8), torch.tensor(1.0)

        scale = abs_max / 7.0
        quantized = (tensor / scale).round().clamp(-8, 7).to(torch.int8)
        return quantized, scale

    def dequantize_tensor(self, quantized: Any, scale: Any) -> Any:
        """Dequantize a tensor back to float."""
        if not self._torch_available:
            return quantized

        import torch

        if isinstance(quantized, torch.Tensor):
            return quantized.float() * scale
        return quantized

    def estimate_memory_reduction(self, original_dtype: str = "float16") -> float:
        """Estimate memory reduction ratio."""
        dtype_bits = {"float32": 32, "float16": 16, "bfloat16": 16}
        original_bits = dtype_bits.get(original_dtype, 16)
        return 1 - (self.bits / original_bits)
