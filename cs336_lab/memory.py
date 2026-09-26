"""Small differentiable CPU reference for a matrix-valued associative memory."""
import math
import torch


def delta_step(state, key, value, beta=1.0, gamma=1.0):
    """Return a gated delta update; do not mutate inputs or normalize keys."""
    if state.ndim != 2 or key.ndim != 1 or value.ndim != 1:
        raise ValueError("expected matrix state and vector key/value")
    if state.shape != (key.numel(), value.numel()) or 0 in state.shape:
        raise ValueError("state shape must be (key width, value width)")
    for tensor in (state, key, value):
        if tensor.device.type != 'cpu' or not tensor.is_floating_point():
            raise ValueError("reference accepts floating-point CPU tensors")
        if tensor.dtype != state.dtype or not torch.isfinite(tensor).all():
            raise ValueError("tensors must share dtype and be finite")
    if not all(math.isfinite(x) and 0 <= x <= 1 for x in (beta, gamma)):
        raise ValueError("beta and gamma must be finite and in [0, 1]")
    decayed = gamma * state
    error = value - decayed.T @ key
    return decayed + beta * torch.outer(key, error)
