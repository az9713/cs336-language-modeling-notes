"""Compare the streaming algorithm to an independent dense softmax."""
import unittest
import torch
from cs336_lab.attention import streaming_attention_row
from cs336_lab.resources import memory_sectors


class StreamingCheck(unittest.TestCase):
    def test_dense_equivalence_and_boundaries(self):
        q = torch.tensor([1.0], dtype=torch.float64)
        k = torch.tensor([[-1000.0], [2.0], [1000.0]], dtype=q.dtype)
        v = torch.tensor([[2.0, 4.0], [5.0, 7.0], [11.0, 13.0]], dtype=q.dtype)
        for end in range(1, 4):
            expected = torch.softmax(k[:end] @ q, 0) @ v[:end]
            for chunk in [1, 2, 10]:
                actual = streaming_attention_row(q, k[:end], v[:end], chunk)
                torch.testing.assert_close(torch.tensor(actual, dtype=q.dtype), expected)
        self.assertEqual(memory_sectors(range(0, 128, 4)), 4)
        self.assertEqual(memory_sectors(range(4, 132, 4)), 5)
        self.assertEqual(memory_sectors([4 * 256 * i for i in range(32)]), 32)
        for operation in [lambda: streaming_attention_row([], [], []),
                          lambda: streaming_attention_row([1], [[1]], [[1]], 0),
                          lambda: streaming_attention_row([1], [[float('nan')]], [[1]]),
                          lambda: memory_sectors([-1])]:
            with self.assertRaises(ValueError):
                operation()


if __name__ == '__main__':
    unittest.main()
