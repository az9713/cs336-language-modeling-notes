import itertools
import unittest

from cs336_lab.tokenization import BytePairTokenizer, bits_per_byte


class TokenizerTests(unittest.TestCase):
    def test_worked_merge_trace(self):
        model = BytePairTokenizer.train(["ababab"], 2)
        self.assertEqual(model.merges, ((97, 98), (256, 256)))
        self.assertEqual(model.encode("ababab"), [257, 256])
        self.assertEqual(model.vocabulary[257], b"abab")

    def test_roundtrip_exhaustive_short_strings(self):
        model = BytePairTokenizer.train(["ababab", "aaaa", "學習"], 12)
        for length in range(6):
            for letters in itertools.product("abc", repeat=length):
                text = "".join(letters)
                self.assertEqual(model.decode(model.encode(text)), text)
        for text in ["學習模型", "café", "e\u0301", "🧪", "new_id_729"]:
            self.assertEqual(model.decode(model.encode(text)), text)

    def test_documents_do_not_merge_across_boundaries(self):
        model = BytePairTokenizer.train(["a", "b"], 100)
        self.assertEqual(model.merges, ())

    def test_overlap_and_deterministic_ties(self):
        model = BytePairTokenizer.train(["aaaaa"], 1)
        self.assertEqual(model.encode("aaaaa"), [256, 256, 97])
        left = BytePairTokenizer.train(["ab", "ba"], 1)
        right = BytePairTokenizer.train(["ba", "ab"], 1)
        self.assertEqual(left, right)
        self.assertEqual(left.merges, ((97, 98),))

    def test_failure_contract(self):
        model = BytePairTokenizer.train([], 10)
        with self.assertRaises(ValueError):
            model.decode([-1])
        with self.assertRaises(ValueError):
            model.decode([256])
        with self.assertRaises(UnicodeDecodeError):
            model.decode([255])
        with self.assertRaises(ValueError):
            BytePairTokenizer.train(["abc"], -1)
        with self.assertRaises(ValueError):
            bits_per_byte(1, 0)

    def test_common_unit_reverses_token_loss_ranking(self):
        self.assertLess(bits_per_byte(80 * 2.3, 400),
                        bits_per_byte(100 * 2.0, 400))


if __name__ == "__main__":
    unittest.main()
