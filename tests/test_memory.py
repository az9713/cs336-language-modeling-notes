"""A write must preserve orthogonal reads and reject invalid contracts."""
import unittest
import torch
from cs336_lab.memory import delta_step


class MemoryCheck(unittest.TestCase):
    def test_write_and_contract(self):
        state = torch.tensor([[3.0], [5.0]], dtype=torch.float64)
        key = torch.tensor([1.0, 0.0], dtype=state.dtype)
        value = torch.tensor([7.0], dtype=state.dtype)
        result = delta_step(state, key, value)
        self.assertEqual(result.tolist(), [[7.0], [5.0]])
        self.assertEqual(state.tolist(), [[3.0], [5.0]])
        self.assertEqual(delta_step(state, key, value, beta=0).tolist(),
                         state.tolist())
        for operation in [lambda: delta_step(state, key, value, beta=2),
                          lambda: delta_step(state, key[:1], value),
                          lambda: delta_step(state, key, value.float()),
                          lambda: delta_step(state, key * torch.nan, value)]:
            with self.assertRaises(ValueError):
                operation()


if __name__ == '__main__':
    unittest.main()
