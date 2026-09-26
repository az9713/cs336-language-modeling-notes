"""Deterministic byte-pair encoding with deliberately simple training."""
from collections import Counter
from dataclasses import dataclass
import math


def replace_pair(tokens, pair, replacement):
    """Return a new list using nonoverlapping left-to-right replacements."""
    result = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens) and tuple(tokens[i:i + 2]) == pair:
            result.append(replacement)
            i += 2
        else:
            result.append(tokens[i])
            i += 1
    return result


@dataclass(frozen=True)
class BytePairTokenizer:
    """Immutable vocabulary and ordered merge list; no normalization."""

    vocabulary: tuple[bytes, ...]
    merges: tuple[tuple[int, int], ...]

    @classmethod
    def train(cls, documents, merge_count):
        if isinstance(merge_count, bool) or not isinstance(merge_count, int):
            raise TypeError("merge_count must be an integer")
        if merge_count < 0:
            raise ValueError("merge_count must be nonnegative")
        documents = tuple(documents)
        if any(not isinstance(text, str) for text in documents):
            raise TypeError("documents must contain strings")
        streams = [list(text.encode("utf-8")) for text in documents]
        vocabulary = [bytes([i]) for i in range(256)]
        merges = []
        for _ in range(merge_count):
            counts = Counter()
            for stream in streams:
                counts.update(zip(stream, stream[1:]))
            if not counts:
                break
            pair = min(counts, key=lambda p: (-counts[p], p))
            identifier = len(vocabulary)
            vocabulary.append(vocabulary[pair[0]] + vocabulary[pair[1]])
            streams = [
                replace_pair(stream, pair, identifier) for stream in streams
            ]
            merges.append(pair)
        return cls(tuple(vocabulary), tuple(merges))

    def encode(self, text):
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        tokens = list(text.encode("utf-8"))
        for offset, pair in enumerate(self.merges):
            tokens = replace_pair(tokens, pair, 256 + offset)
        return tokens

    def decode_bytes(self, tokens):
        tokens = tuple(tokens)
        if any(isinstance(i, bool) or not isinstance(i, int) for i in tokens):
            raise TypeError("token identifiers must be integers")
        if any(i < 0 or i >= len(self.vocabulary) for i in tokens):
            raise ValueError("unknown token identifier")
        return b"".join(self.vocabulary[i] for i in tokens)

    def decode(self, tokens):
        return self.decode_bytes(tokens).decode("utf-8", errors="strict")


def bits_per_byte(negative_log_likelihood, byte_count):
    """Convert finite, nonnegative NLL in nats to a nonempty byte rate."""
    if not math.isfinite(negative_log_likelihood):
        raise ValueError("NLL must be finite")
    if negative_log_likelihood < 0 or byte_count <= 0:
        raise ValueError("NLL must be nonnegative and byte_count positive")
    return negative_log_likelihood / (byte_count * math.log(2))
