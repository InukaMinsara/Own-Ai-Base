import json
import re
from collections import Counter
from pathlib import Path


class OwnTokenizerV8:
    """Hybrid word/Unicode tokenizer with guaranteed character fallback.

    Frequent complete pieces (words/code chunks) become single tokens, while
    every Unicode character seen during training remains available as a
    fallback. This avoids the aggressive UNK behavior of the old tokenizer.
    """

    SPECIAL = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
    PIECE_RE = re.compile(r"\s+|[^\s]+", re.UNICODE)

    def __init__(self, vocab_size=4096):
        self.target_vocab_size = int(vocab_size)
        self.token_to_id = {}
        self.id_to_token = {}
        self.max_piece_len = 1

    def _pieces(self, text):
        return self.PIECE_RE.findall(text)

    def train(self, text):
        piece_freq = Counter()
        char_freq = Counter()
        whitespace_freq = Counter()

        for piece in self._pieces(text):
            piece_freq[piece] += 1
            if piece.isspace():
                whitespace_freq[piece] += 1
                continue
            for ch in piece:
                char_freq[ch] += 1

        vocab = list(self.SPECIAL)
        used = set(vocab)

        # Keep common complete pieces first. This is especially useful for
        # English words, URLs, common code fragments, and Sinhala words.
        ranked_pieces = sorted(
            piece_freq.items(),
            key=lambda item: (-item[1], -len(item[0]), item[0]),
        )

        for piece, _freq in ranked_pieces:
            if len(vocab) >= self.target_vocab_size:
                break
            if piece not in used:
                vocab.append(piece)
                used.add(piece)

        # Always reserve space for character fallback.
        ranked_chars = sorted(
            char_freq.items(),
            key=lambda item: (-item[1], item[0]),
        )

        for ch, _freq in ranked_chars:
            if len(vocab) >= self.target_vocab_size:
                break
            if ch not in used:
                vocab.append(ch)
                used.add(ch)

        # Add common whitespace forms if capacity remains.
        for ws, _freq in sorted(
            whitespace_freq.items(),
            key=lambda item: (-item[1], -len(item[0])),
        ):
            if len(vocab) >= self.target_vocab_size:
                break
            if ws not in used:
                vocab.append(ws)
                used.add(ws)

        # If the target vocab is too small to include every character, this is
        # still a valid tokenizer, but unseen characters can fall back to UNK.
        self.token_to_id = {token: i for i, token in enumerate(vocab)}
        self.id_to_token = {
            i: token for token, i in self.token_to_id.items()
        }
        self.max_piece_len = max(
            [len(token) for token in self.token_to_id if token not in self.SPECIAL]
            or [1]
        )

    def _encode_nonspace(self, piece):
        ids = []
        i = 0
        while i < len(piece):
            best = None
            upper = min(
                len(piece),
                i + self.max_piece_len,
            )
            for end in range(upper, i, -1):
                candidate = piece[i:end]
                if candidate in self.token_to_id:
                    best = candidate
                    break

            if best is None:
                best = piece[i]
                ids.append(
                    self.token_to_id.get(
                        best,
                        self.token_to_id["<UNK>"],
                    )
                )
                i += 1
            else:
                ids.append(self.token_to_id[best])
                i += len(best)

        return ids

    def encode(self, text, add_special_tokens=False):
        ids = []
        if add_special_tokens:
            ids.append(self.bos_id)

        for piece in self._pieces(text):
            if piece.isspace():
                token_id = self.token_to_id.get(piece)
                if token_id is not None:
                    ids.append(token_id)
                    continue

                # Preserve unknown whitespace exactly through known single
                # whitespace characters when possible.
                for ch in piece:
                    ids.append(
                        self.token_to_id.get(
                            ch,
                            self.token_to_id["<UNK>"],
                        )
                    )
            else:
                ids.extend(self._encode_nonspace(piece))

        if add_special_tokens:
            ids.append(self.eos_id)

        return ids

    def decode(self, ids):
        special = set(self.SPECIAL)
        return "".join(
            self.id_to_token.get(
                int(i),
                "<UNK>",
            )
            for i in ids
            if self.id_to_token.get(
                int(i),
                "<UNK>",
            ) not in special
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
            "id_to_token": {
                str(k): v
                for k, v in self.id_to_token.items()
            },
            "max_piece_len": self.max_piece_len,
            "tokenizer_version": "v8-hybrid",
        }
        Path(path).write_text(
            json.dumps(
                data,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def load(self, path):
        data = json.loads(
            Path(path).read_text(
                encoding="utf-8"
            )
        )
        self.target_vocab_size = data.get(
            "target_vocab_size",
            4096,
        )
        self.token_to_id = data["token_to_id"]
        self.id_to_token = {
            int(k): v
            for k, v in data["id_to_token"].items()
        }
        self.max_piece_len = int(
            data.get("max_piece_len", 1)
        )
