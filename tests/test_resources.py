import unittest

from cs336_lab.resources import (
    checkpoint_slots,
    matmul_cost,
    model_state_bytes,
    roofline_seconds,
)


class ResourceTests(unittest.TestCase):
    def test_matmul_scalar_loop_count(self):
        rows, inner, columns = 2, 3, 4
        products = sum(1 for _ in range(rows) for _ in range(inner)
                       for _ in range(columns))
        self.assertEqual(matmul_cost(rows, inner, columns).flops, 2 * products)
        self.assertEqual(matmul_cost(1, 4, 4).compulsory_bytes, 48)

    def test_roofline_distinct_bottlenecks(self):
        self.assertAlmostEqual(roofline_seconds(20e9, 2e9, 100e12, 1e12),
                               .002)
        self.assertAlmostEqual(roofline_seconds(20e12, 2e9, 100e12, 1e12),
                               .2)
        with self.assertRaises(ValueError):
            roofline_seconds(1, 1, 0, 1)

    def test_state_conventions(self):
        self.assertEqual(model_state_bytes(1_000_000_000), 12_000_000_000)
        self.assertEqual(model_state_bytes(1_000_000_000, master=4),
                         16_000_000_000)

    def test_segment_choice_has_interior_optimum(self):
        costs = {s: checkpoint_slots(64, s) for s in range(1, 65)}
        self.assertEqual(min(costs.values()), 16)
        self.assertEqual(costs[8], 16)
        self.assertGreater(costs[1], costs[8])
        self.assertGreater(costs[64], costs[8])

    def test_matrix_gradient_and_unequal_accumulation(self):
        import torch

        x = torch.arange(6, dtype=torch.float64).reshape(3, 2)
        w = torch.tensor([[.2], [-.1]], dtype=torch.float64,
                         requires_grad=True)
        y = x @ w
        loss = y.square().sum() / 3
        loss.backward()
        expected = x.T @ (2 * y.detach() / 3)
        self.assertTrue(torch.allclose(w.grad, expected))
        total_gradient = w.grad.clone()
        w.grad = None
        for microbatch in [x[:1], x[1:]]:
            ((microbatch @ w).square().sum() / 3).backward()
        self.assertTrue(torch.allclose(w.grad, total_gradient))
        w.grad = None
        for microbatch in [x[:1], x[1:]]:
            ((microbatch @ w).square().mean() / 2).backward()
        self.assertFalse(torch.allclose(w.grad, total_gradient))


if __name__ == "__main__":
    unittest.main()
