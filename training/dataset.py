from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tokenizer.tokenizer import OwnTokenizer


DATA_FILE = ROOT / "data" / "processed" / "train.txt"


def load_training_data():

    text = DATA_FILE.read_text(
        encoding="utf-8"
    )

    tokenizer = OwnTokenizer(
        vocab_size=512
    )

    tokenizer.train(text)

    tokens = tokenizer.encode(
        text,
        add_special_tokens=True
    )

    data = torch.tensor(
        tokens,
        dtype=torch.long
    )

    return text, tokenizer, data


if __name__ == "__main__":

    text, tokenizer, data = load_training_data()

    print("================================")
    print("       TRAINING DATA TEST")
    print("================================")

    print("Characters:", len(text))
    print("Vocabulary:", tokenizer.vocab_size)
    print("Tokens:", len(data))
    print("Data type:", data.dtype)

    print("--------------------------------")
    print("First 30 tokens:")
    print(data[:30].tolist())

    print("--------------------------------")
    print("Decoded preview:")

    preview = tokenizer.decode(
        data[:30].tolist()
    )

    print(repr(preview))

    print("================================")
    print("DATASET: PASSED")