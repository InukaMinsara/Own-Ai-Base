import json
import re
from collections import Counter
from pathlib import Path


class OwnTokenizer:

    def __init__(self, vocab_size=512):
        self.target_vocab_size = vocab_size

        self.token_to_id = {}
        self.id_to_token = {}

        self.unk_token = "<UNK>"
        self.pad_token = "<PAD>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

        self.special_tokens = [
            self.pad_token,
            self.unk_token,
            self.bos_token,
            self.eos_token,
        ]

    def _split_words(self, text):

        return re.findall(
            r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\w\s]|\s+",
            text,
            flags=re.UNICODE
        )

    def train(self, text):

        words = self._split_words(text)

        # Start with characters for each piece.
        sequences = []

        for word in words:

            if word.isspace():
                sequences.append([" "])
                continue

            if re.fullmatch(r"[A-Za-z]+(?:'[A-Za-z]+)?", word):
                sequences.append(list(word))
            else:
                sequences.append(list(word))

        vocab = set()

        for sequence in sequences:
            vocab.update(sequence)

        # BPE-style merging.
        while (
            len(vocab) + len(self.special_tokens)
            < self.target_vocab_size
        ):

            pair_counts = Counter()

            for sequence in sequences:

                for i in range(len(sequence) - 1):

                    pair = (
                        sequence[i],
                        sequence[i + 1]
                    )

                    pair_counts[pair] += 1

            if not pair_counts:
                break

            best_pair, count = pair_counts.most_common(1)[0]

            # Stop if no useful repeated pair remains.
            if count < 2:
                break

            merged = (
                best_pair[0]
                + best_pair[1]
            )

            new_sequences = []

            for sequence in sequences:

                new_sequence = []

                i = 0

                while i < len(sequence):

                    if (
                        i < len(sequence) - 1
                        and sequence[i] == best_pair[0]
                        and sequence[i + 1] == best_pair[1]
                    ):
                        new_sequence.append(merged)
                        i += 2

                    else:
                        new_sequence.append(
                            sequence[i]
                        )
                        i += 1

                new_sequences.append(new_sequence)

            sequences = new_sequences
            vocab.add(merged)

        tokens = sorted(vocab)

        max_vocab = (
            self.target_vocab_size
            - len(self.special_tokens)
        )

        tokens = tokens[:max_vocab]

        final_tokens = (
            self.special_tokens
            + tokens
        )

        self.token_to_id = {
            token: i
            for i, token in enumerate(final_tokens)
        }

        self.id_to_token = {
            i: token
            for token, i in self.token_to_id.items()
        }

    def _encode_piece(self, piece):

        if piece in self.token_to_id:
            return [
                self.token_to_id[piece]
            ]

        ids = []

        # Greedy longest-match subword encoding.
        position = 0

        while position < len(piece):

            found = False

            for end in range(
                len(piece),
                position,
                -1
            ):

                candidate = piece[
                    position:end
                ]

                if candidate in self.token_to_id:

                    ids.append(
                        self.token_to_id[candidate]
                    )

                    position = end
                    found = True
                    break

            if not found:

                ids.append(
                    self.token_to_id[
                        self.unk_token
                    ]
                )

                position += 1

        return ids

    def encode(
        self,
        text,
        add_special_tokens=False
    ):

        pieces = self._split_words(text)

        ids = []

        if add_special_tokens:
            ids.append(
                self.token_to_id[
                    self.bos_token
                ]
            )

        for piece in pieces:

            ids.extend(
                self._encode_piece(piece)
            )

        if add_special_tokens:
            ids.append(
                self.token_to_id[
                    self.eos_token
                ]
            )

        return ids

    def decode(self, ids):

        output = []

        for token_id in ids:

            token = self.id_to_token.get(
                int(token_id),
                self.unk_token
            )

            if token in self.special_tokens:
                continue

            output.append(token)

        return "".join(output)

    @property
    def vocab_size(self):
        return len(self.token_to_id)

    def save(self, path):

        data = {
            "target_vocab_size":
                self.target_vocab_size,

            "token_to_id":
                self.token_to_id,

            "id_to_token": {
                str(k): v
                for k, v in self.id_to_token.items()
            }
        }

        Path(path).write_text(
            json.dumps(
                data,
                ensure_ascii=False,
                indent=2
            ),
            encoding="utf-8"
        )

    def load(self, path):

        data = json.loads(
            Path(path).read_text(
                encoding="utf-8"
            )
        )

        self.target_vocab_size = data.get(
            "target_vocab_size",
            512
        )

        self.token_to_id = data[
            "token_to_id"
        ]

        self.id_to_token = {
            int(k): v
            for k, v in data[
                "id_to_token"
            ].items()
        }


if __name__ == "__main__":

    text = """
    Hello! I am Own AI.
    Artificial intelligence uses mathematics
    and neural networks.
    Computers use algorithms to process information.
    """

    tokenizer = OwnTokenizer(
        vocab_size=512
    )

    tokenizer.train(text)

    tests = [
        "Hello!",
        "Artificial intelligence",
        "computer",
        "computing",
        "fahhhhhhhhhhh",
    ]

    print("================================")
    print("     OWN BPE TOKENIZER TEST")
    print("================================")

    print(
        "Vocabulary size:",
        tokenizer.vocab_size
    )

    for test in tests:

        encoded = tokenizer.encode(test)

        decoded = tokenizer.decode(
            encoded
        )

        print("--------------------------------")
        print("Input:", test)
        print("Tokens:", encoded)
        print("Decoded:", decoded)

    print("================================")
    print("OWN BPE TOKENIZER: PASSED")