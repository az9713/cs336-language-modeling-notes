"""CPU forward reference: exact softmax attention with a streaming reduction."""
import math
from .resources import positive_integer


def streaming_attention_row(query, keys, values, chunk_size=1):
    """Return one attention row; finite inputs, no masking or input mutation.

    Pass only allowed keys for a causal row. This is a scalar CPU explanation,
    not a GPU kernel or an automatic-differentiation implementation.
    """
    positive_integer(chunk_size, "chunk_size")
    q = tuple(float(x) for x in query)
    keys = tuple(tuple(float(x) for x in row) for row in keys)
    values = tuple(tuple(float(x) for x in row) for row in values)
    if not q or not keys or len(keys) != len(values) or not values[0]:
        raise ValueError("query and matching keys/values must be nonempty")
    width = len(values[0])
    if any(len(k) != len(q) for k in keys):
        raise ValueError("key width must match query width")
    if any(len(v) != width for v in values):
        raise ValueError("value widths must agree")
    if any(not math.isfinite(x) for row in (q,) + keys + values for x in row):
        raise ValueError("inputs must be finite")
    maximum, total = -math.inf, 0.0
    numerator = [0.0] * width
    for start in range(0, len(keys), chunk_size):
        block_keys = keys[start:start + chunk_size]
        block_values = values[start:start + chunk_size]
        scores = [math.fsum(a * b for a, b in zip(q, k)) / math.sqrt(len(q))
                  for k in block_keys]
        if not all(math.isfinite(s) for s in scores):
            raise ValueError("dot products exceed finite arithmetic")
        new_maximum = max(maximum, max(scores))
        correction = math.exp(maximum - new_maximum)
        weights = [math.exp(s - new_maximum) for s in scores]
        total = correction * total + math.fsum(weights)
        numerator = [correction * old + math.fsum(w * v[j]
                     for w, v in zip(weights, block_values))
                     for j, old in enumerate(numerator)]
        maximum = new_maximum
    return tuple(x / total for x in numerator)
