"""The roofline model, and the LLM serving arithmetic built on it.

Peak numbers come from config/hardware.yaml (with their sources). Bandwidths there are in
GB/s (10^9 bytes); compute in TFLOP/s (10^12)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import config


@dataclass(frozen=True)
class Chip:
    name: str
    flops: float  # FLOP/s, bf16 dense
    bandwidth: float  # bytes/s, HBM
    hbm_bytes: float

    @property
    def ridge(self) -> float:
        """Arithmetic intensity (FLOP/byte) where memory time equals compute time."""
        return self.flops / self.bandwidth


def chip(name: str) -> Chip:
    c: dict[str, Any] = config.load("hardware")["accelerators"][name]
    return Chip(name, float(c["bf16_tflops"]) * 1e12, float(c["hbm_gbps"]) * 1e9, float(c["hbm_gb"]) * 1e9)


def attainable(c: Chip, intensity: float) -> float:
    return min(c.flops, intensity * c.bandwidth)


def bound(c: Chip, flops: float, nbytes: float) -> str:
    return "memory" if flops / nbytes < c.ridge else "compute"


def roof_fraction(c: Chip, flops: float, nbytes: float, seconds: float) -> float:
    """Achieved FLOP/s as a share of what the roofline allows at this intensity."""
    return (flops / seconds) / attainable(c, flops / nbytes)


@dataclass(frozen=True)
class Model:
    """A dense decoder-only transformer, as far as serving arithmetic needs."""

    params: float
    layers: int
    kv_heads: int
    head_dim: int
    weight_bytes: float = 2.0  # bf16
    kv_bytes: float = 2.0

    def kv_per_token(self) -> float:
        """Bytes of KV cache per token: K and V, every layer, every KV head."""
        return 2 * self.layers * self.kv_heads * self.head_dim * self.kv_bytes


def decode_step_seconds(c: Chip, m: Model, batch: int, context: int) -> float:
    """One decode step for `batch` sequences of `context` tokens: read the weights once and every
    sequence's KV cache; 2 FLOPs per parameter per token (attention FLOPs are small beside the
    reads and are left out). The step takes the longer of memory and compute time."""
    mem = m.params * m.weight_bytes + batch * context * m.kv_per_token()
    flops = 2 * m.params * batch
    return max(mem / c.bandwidth, flops / c.flops)


def crossover_batch(c: Chip, m: Model, context: int) -> float:
    """The batch at which decode turns compute-bound: weights + KV reads time = compute time."""
    # params*wb/bw + b*ctx*kv/bw = 2*params*b/flops  ->  b = params*wb/bw / (2*params/flops - ctx*kv/bw)
    denom = 2 * m.params / c.flops - context * m.kv_per_token() / c.bandwidth
    return float("inf") if denom <= 0 else (m.params * m.weight_bytes / c.bandwidth) / denom
