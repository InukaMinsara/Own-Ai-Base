import json
import re
from collections import Counter
from pathlib import Path


class OwnTokenizerV7:
    """Unicode-aware frequency-ranked BPE tokenizer.

    Unlike the original tokenizer, vocabulary entries are ranked by frequency
    and merge candidates are retained by usefulness instead of alphabetical
    order. It works with Latin and non-Latin text, including Sinhala.
    """

    SPECIAL = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
    PIECE_RE = re.compile(r"\S+|\s+", re.UNICODE)

    def __init__(self, vocab_size=2048):
        self.target_vocab_size = vocab_size
        self.token_to_id = {}
        self.id_to_token = {}
        self.merges = []

    def _pieces(self, text):
        return self.PIECE_RE.findall(text)

    @staticmethod
    def _chars(piece):
        return list(piece)

    def train(self, text):
        sequences = [self._chars(piece) for piece in self._pieces(text)]
        token_freq = Counter()

        for seq in sequences:
            token_freq.update(seq)

        vocab = set(token_freq)

        while len(vocab) + len(self.SPECIAL) < self.target_vocab_size:
            pair_counts = Counter()

            for seq in sequences:
                for i in range(len(seq) - 1):
                    pair_counts[(seq[i], seq[i + 1])] += 1

            if not pair_counts:
                break

            pair, count = pair_counts.most_common(1)[0]
            if count < 2:
                break

            merged = pair[0] + pair[1]
            if merged in vocab:
                break

            self.merges.append(pair)
            vocab.add(merged)

            for seq_index, seq in enumerate(sequences):
                out = []
                i = 0
                while i < len(seq):
                    if (
                        i < len(seq) - 1
                        and seq[i] == pair[0]
                        and seq[i + 1] == pair[1]
                    ):
                        out.append(merged)
                        i += 2
                    else:
                        out.append(seq[i])
                        i += 1
                sequences[seq_index] = out

        final = self.SPECIAL[:]
        ranked = sorted(
            vocab,
            key=lambda token: (
                -token_freq.get(token, 0),
                -len(token),
                token,
            ),
        )

        for token in ranked:
            if token not in self.SPECIAL and len(final) < self.target_vocab_size:
                final.append(token)

        self.token_to_id = {token: i for i, token in enumerate(final)}
        self.id_to_token = {i: token for token, i in self.token_to_id.items()}

    def encode(self, text, add_special_tokens=False):
        ids = []

        if add_special_tokens:
            ids.append(self.token_to_id["<BOS>"])

        for piece in self._pieces(text):
            chars = self._chars(piece)

            # Replay learned merge ordering.
            for left, right in self.merges:
                merged = left + right
                out = []
                i = 0
                while i < len(chars):
                    if (
                        i < len(chars) - 1
                        and chars[i] == left
                        and chars[i + 1] == right
                    ):
                        out.append(merged)
                        i += 2
                    else:
                        out.append(chars[i])
                        i += 1
                chars = out

            for token in chars:
                ids.append(
                    self.token_to_id.get(
                        token,
                        self.token_to_id["<UNK>"],
                    )
                )

        if add_special_tokens:
            ids.append(self.token_to_id["<EOS>"])

        return ids

    def decode(self, ids):
        special = set(self.SPECIAL)
        return "".join(
            self.id_to_token.get(int(i), "<UNK>")
            for i in ids
            if self.id_to_token.get(int(i), "<UNK>") not in special
        )

    @property
    def vocab_size(self):
        return len(self.token_to_id)

    @property
    def bos_id(self):
        return self.token_to_id["<BOS>"]

    @property
    def eos_id(self):
        return self.token_to_id["<EOS>"]

    @property
    def pad_id(self):
        return self.token_to_id["<PAD>"]

    def save(self, path):
        data = {
            "target_vocab_size": self.target_vocab_size,
            "token_to_id": self.token_to_id,
            "id_to_token": {str(k): v for k, v in self.id_to_token.items()},
            "merges": [list(pair) for pair in self.merges],
        }
        Path(path).write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load(self, path):
        data = json.loads(
            Path(path).read_text(encoding="utf-8")
        )
        self.target_vocab_size = data.get("target_vocab_size", 2048)
        self.token_to_id = data["token_to_id"]
        self.id_to_token = {
            int(k): v for k, v in data["id_to_token"].items()
        }
        self.merges = [tuple(pair) for pair in data.get("merges", [])]
