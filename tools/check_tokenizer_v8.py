from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tokenizer.tokenizer_v8 import OwnTokenizerV8

TOKENIZER_FILE = (
    ROOT / "data" / "processed" / "token_shards_v8"
    / "tokenizer_v8.json"
)

PROMPTS = [
    "Hello, who are you?",
    "Explain what a computer program is in simple words.",
    "සිංහලෙන් ශ්‍රී ලංකාව ගැන වාක්‍ය දෙකක් ලියන්න.",
    "What is the difference between RAM and storage?",
    "def add(a, b):",
    "2 + 2 =",
]


def main():
    tokenizer = OwnTokenizerV8()
    tokenizer.load(TOKENIZER_FILE)

    print("=" * 68)
    print("OWN AI v8 TOKENIZER ROUND-TRIP CHECK")
    print("=" * 68)
    print("Tokenizer:", TOKENIZER_FILE)
    print("Vocabulary:", tokenizer.vocab_size)
    print()

    failed = 0

    for prompt in PROMPTS:
        decoded = tokenizer.decode(
            tokenizer.encode(
                prompt,
                add_special_tokens=True,
            )
        )

        ok = decoded == prompt

        print("INPUT :", prompt)
        print("OUTPUT:", decoded)
        print("STATUS:", "OK" if ok else "FAIL")
        print()

        if not ok:
            failed += 1

    if failed:
        raise SystemExit(
            f"{failed} tokenizer round-trip checks failed."
        )

    print("All tokenizer round-trip checks passed.")


if __name__ == "__main__":
    main()
