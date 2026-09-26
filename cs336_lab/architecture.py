"""CPU reference operations for the architecture chapter, not fast kernels."""
import math

from .resources import positive_integer


def rms_normalize(values, epsilon=1e-6):
    """Return unit-gain RMSNorm; reject empty or nonfinite inputs."""
    values = tuple(float(v) for v in values)
    if not values or not all(math.isfinite(v) for v in values):
        raise ValueError("values must be nonempty and finite")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    scale = math.hypot(*values) / math.sqrt(len(values))
    denominator = math.hypot(scale, math.sqrt(epsilon))
    return tuple(v / denominator for v in values)


def rotate2(vector, angle):
    """Rotate one coordinate pair by a finite angle in radians."""
    if len(vector) != 2 or not all(math.isfinite(v) for v in vector):
        raise ValueError("vector must contain two finite numbers")
    if not math.isfinite(angle):
        raise ValueError("angle must be finite")
    x, y = vector
    c, s = math.cos(angle), math.sin(angle)
    return (c * x - s * y, s * x + c * y)


def kv_cache_bytes(layers, batch, tokens, kv_heads, head_width,
                   bytes_per_element=2):
    """Unpadded keys plus values; weights and workspace are excluded."""
    for name, value in locals().copy().items():
        positive_integer(value, name)
    return 2 * layers * batch * tokens * kv_heads * head_width * bytes_per_element
