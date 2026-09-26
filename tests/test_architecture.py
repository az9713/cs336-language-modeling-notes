"""Independent mathematical invariants and failure cases."""
import math
import unittest

from cs336_lab.architecture import kv_cache_bytes, rms_normalize, rotate2


class ArchitectureChecks(unittest.TestCase):
    def test_geometry_and_accounting(self):
        q, k = (2.0, -1.0), (0.5, 3.0)
        dot = lambda a, b: sum(x * y for x, y in zip(a, b))
        for m, n, offset in [(0, 1, 7), (3, 12, -4), (8, 8, 100)]:
            score = dot(rotate2(q, m * 0.3), rotate2(k, n * 0.3))
            shifted = dot(rotate2(q, (m + offset) * 0.3),
                          rotate2(k, (n + offset) * 0.3))
            self.assertAlmostEqual(score, shifted)
            self.assertAlmostEqual(score, dot(q, rotate2(k, (n - m) * 0.3)))
        x = rms_normalize((3, 4))
        self.assertAlmostEqual(sum(v * v for v in x) / 2, 12.5 / 12.500001)
        self.assertEqual(rms_normalize((0, 0)), (0, 0))
        self.assertNotEqual(rms_normalize((3, 4)), rms_normalize((4, 5)))
        self.assertEqual(kv_cache_bytes(12, 1, 8192, 3, 64), 72 * 2**20)
        for operation in [lambda: rms_normalize(()),
                          lambda: rms_normalize((math.inf,)),
                          lambda: rotate2((1, 2), math.nan),
                          lambda: kv_cache_bytes(12, 0, 8, 3, 64)]:
            with self.assertRaises(ValueError):
                operation()


if __name__ == '__main__':
    unittest.main()
