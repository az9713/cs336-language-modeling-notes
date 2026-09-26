"""Transparent resource accounting; rates are inputs, never benchmarks."""
from dataclasses import dataclass
import math


def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class MatmulCost:
    flops: int
    compulsory_bytes: int

    @property
    def intensity(self):
        return self.flops / self.compulsory_bytes


def matmul_cost(rows, inner, columns, bytes_per_element=2):
    for name, value in locals().copy().items():
        positive_integer(value, name)
    flops = 2 * rows * inner * columns
    traffic = bytes_per_element * (
        rows * inner + inner * columns + rows * columns
    )
    return MatmulCost(flops, traffic)


def roofline_seconds(flops, traffic_bytes, peak_flops, bandwidth_bytes):
    values = (flops, traffic_bytes, peak_flops, bandwidth_bytes)
    if any(not math.isfinite(v) for v in values):
        raise ValueError("all values must be finite")
    if min(flops, traffic_bytes) < 0 or min(peak_flops, bandwidth_bytes) <= 0:
        raise ValueError("work must be nonnegative and rates positive")
    return max(flops / peak_flops, traffic_bytes / bandwidth_bytes)


def model_state_bytes(parameters, weight=2, gradient=2, moments=8, master=0):
    positive_integer(parameters, "parameters")
    if min(weight, gradient, moments, master) < 0:
        raise ValueError("byte counts must be nonnegative")
    return parameters * (weight + gradient + moments + master)


def checkpoint_slots(layers, segment):
    """Simplified maximum: stored boundaries plus one live segment."""
    positive_integer(layers, "layers")
    positive_integer(segment, "segment")
    if segment > layers:
        raise ValueError("segment cannot exceed layers")
    return math.ceil(layers / segment) + segment


def memory_sectors(addresses, element_bytes=4, sector_bytes=32):
    """Count touched aligned sectors; addresses are byte offsets, not indices."""
    positive_integer(element_bytes, "element_bytes")
    positive_integer(sector_bytes, "sector_bytes")
    sectors = set()
    for address in addresses:
        if isinstance(address, bool) or not isinstance(address, int) or address < 0:
            raise ValueError("addresses must be nonnegative integer byte offsets")
        first = address // sector_bytes
        last = (address + element_bytes - 1) // sector_bytes
        sectors.update(range(first, last + 1))
    return len(sectors)
