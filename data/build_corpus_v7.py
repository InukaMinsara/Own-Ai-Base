from pathlib import Path
import json
import re


ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
KNOWLEDGE_DIR = ROOT / "data" / "knowledge"
INSTRUCTIONS = ROOT / "data" / "processed" / "instructions_v7.jsonl"
OUTPUT = ROOT / "data" / "processed" / "corpus_v7.txt"


SKIP_DIR_NAMES = {
    ".git",
    "__pycache__",
    "backup_before_expand",
    "backup_before_expand_v2",
    "processed",
}


def normalize_text(text):
    text = text.replace("\x00", "")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def paragraphs(text):
    for block in re.split(r"\n\s*\n", text):
        block = normalize_text(block)
        if len(block) >= 40:
            yield block


def collect_files(folder):
    if not folder.exists():
        return []

    result = []

    for path in folder.rglob("*"):
        if not path.is_file():
            continue

        if any(
            part in SKIP_DIR_NAMES
            for part in path.parts
        ):
            continue

        if path.suffix.lower() in {
            ".txt",
            ".md",
            ".json",
            ".jsonl",
        }:
            result.append(path)

    return result


def load_instruction_pairs():
    pairs = []

    if not INSTRUCTIONS.exists():
        return pairs

    for line in INSTRUCTIONS.read_text(
        encoding="utf-8"
    ).splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue

        instruction = str(
            item.get("instruction", "")
        ).strip()

        response = str(
            item.get("response", "")
        ).strip()

        if instruction and response:
            pairs.append(
                f"User: {instruction}\nAssistant: {response}"
            )

    return pairs


def main():
    seen = set()
    sections = []

    sources = (
        collect_files(RAW_DIR)
        + collect_files(KNOWLEDGE_DIR)
    )

    for path in sources:
        try:
            text = path.read_text(
                encoding="utf-8",
                errors="ignore",
            )
        except OSError:
            continue

        for block in paragraphs(text):
            key = re.sub(
                r"\s+",
                " ",
                block.lower(),
            )

            if key in seen:
                continue

            seen.add(key)
            sections.append(block)

    sections.extend(load_instruction_pairs())

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT.write_text(
        "\n\n".join(sections) + "\n",
        encoding="utf-8",
    )

    print("=" * 56)
    print("OWN AI v7 TRAINING CORPUS")
    print("=" * 56)
    print("Sources:", len(sources))
    print("Unique sections:", len(sections))
    print("Characters:", len(
        OUTPUT.read_text(
            encoding="utf-8"
        )
    ))
    print("Output:", OUTPUT)
    print("=" * 56)


if __name__ == "__main__":
    main()
