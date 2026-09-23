from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

# Allow both:
#   python data\\build_token_shards_v8.py
# and:
#   python -m data.build_token_shards_v8
# when launched from the project root.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tokenizer.tokenizer_v8 import OwnTokenizerV8


OUT_ROOT = ROOT / "data" / "processed" / "token_shards_v8"
CLEAN_MANIFEST = ROOT / "data" / "processed" / "corpus_v8" / "clean_manifest_v8.jsonl"

VOCAB_SIZE = int(os.getenv("OWN_AI_V8_VOCAB", "4096"))
SHARD_MB = int(os.getenv("OWN_AI_V8_SHARD_MB", "128"))
TOKENIZER_SAMPLE_MB = int(
    os.getenv("OWN_AI_V8_TOKENIZER_SAMPLE_MB", "128")
)

ALLOWED = {
    ".txt",
    ".md",
    ".csv",
    ".json",
    ".jsonl",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".html",
    ".css",
    ".xml",
    ".yaml",
    ".yml",
    ".sql",
}

SKIP_PARTS = {
    ".git",
    "__pycache__",
    ".venv",
    "checkpoints",
    "runtime",
}


def data_roots():
    raw = os.getenv("OWN_AI_V8_DATA_DIRS", "")
    if raw.strip():
        return [
            Path(item.strip())
            for item in raw.split(";")
            if item.strip()
        ]

    return [
        ROOT / "data" / "cache",
        ROOT / "data" / "raw",
        ROOT / "data" / "knowledge",
    ]


def iter_clean_manifest_files():
    if not CLEAN_MANIFEST.exists():
        return []

    files = []
    seen = set()

    with CLEAN_MANIFEST.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            if record.get("status") != "accepted":
                continue

            path = Path(record["path"])
            key = str(path.resolve()).lower()

            if key in seen or not path.is_file():
                continue

            if path.suffix.lower() not in ALLOWED:
                continue

            seen.add(key)
            files.append(path)

    return files


def iter_files(roots):
    seen = set()

    for root in roots:
        root = root.resolve()

        if not root.exists():
            continue

        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue

            if path.suffix.lower() not in ALLOWED:
                continue

            if any(
                part in SKIP_PARTS
                for part in path.parts
            ):
                continue

            try:
                stat = path.stat()
            except OSError:
                continue

            key = (
                str(path).lower(),
                stat.st_size,
                stat.st_mtime_ns,
            )

            if key in seen:
                continue

            seen.add(key)
            yield path


def normalize(text):
    # Remove actual NUL characters without embedding a literal NUL in source.
    text = text.replace(chr(0), "")

    text = " ".join(
        line.strip()
        for line in text.splitlines()
        if line.strip()
    )

    return text.strip()


def iter_file_chunks(path, chunk_chars):
    """Read text incrementally so large files never need to fit in RAM."""

    buffer = ""

    try:
        with path.open(
            "r",
            encoding="utf-8",
            errors="ignore",
        ) as handle:
            while True:
                block = handle.read(
                    max(
                        64_000,
                        min(
                            chunk_chars,
                            4 * 1024 * 1024,
                        ),
                    )
                )

                if not block:
                    break

                buffer += block

                while len(buffer) >= chunk_chars:
                    boundary = buffer.rfind(
                        "\n\n",
                        0,
                        chunk_chars,
                    )

                    if boundary < chunk_chars // 2:
                        boundary = chunk_chars

                    piece = normalize(
                        buffer[:boundary]
                    )

                    buffer = buffer[boundary:]

                    if piece:
                        yield piece

            tail = normalize(buffer)

            if tail:
                yield tail

    except OSError as exc:
        print(
            "SKIP:",
            path,
            "-",
            exc,
        )


def build_tokenizer(files):
    sample_limit = (
        TOKENIZER_SAMPLE_MB
        * 1024
        * 1024
    )

    parts = []
    used = 0

    ordered = sorted(
        files,
        key=lambda path: hashlib.sha1(
            str(path).encode(
                "utf-8",
                errors="ignore",
            )
        ).hexdigest(),
    )

    for path in ordered:
        if used >= sample_limit:
            break

        remaining = sample_limit - used

        try:
            with path.open(
                "r",
                encoding="utf-8",
                errors="ignore",
            ) as handle:
                sample = handle.read(
                    min(
                        remaining,
                        8 * 1024 * 1024,
                    )
                )
        except OSError:
            continue

        if not sample:
            continue

        parts.append(sample)
        used += len(sample.encode("utf-8"))

    if not parts:
        raise ValueError(
            "No readable training text found."
        )

    tokenizer = OwnTokenizerV8(
        vocab_size=VOCAB_SIZE
    )

    tokenizer.train(
        "\n\n".join(parts)
    )

    return tokenizer, used


def split_for_hash(value, ordinal):
    # Guarantee at least one validation and one test chunk on small datasets.
    # For large datasets, the normal deterministic 90/5/5 hash split is used.
    if ordinal == 1:
        return "train"
    if ordinal == 2:
        return "val"
    if ordinal == 3:
        return "test"

    bucket = int(
        value[:8],
        16,
    ) % 100

    if bucket < 90:
        return "train"

    if bucket < 95:
        return "val"

    return "test"


def append_token_ids(path, token_ids):
    import array

    values = array.array(
        "I",
        (
            int(token_id)
            for token_id in token_ids
        ),
    )

    with path.open("ab") as handle:
        values.tofile(handle)

    return len(values)


def reset_output():
    for split in (
        "train",
        "val",
        "test",
    ):
        directory = OUT_ROOT / split
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        for old in directory.glob("*"):
            if old.is_file():
                old.unlink()


def main():
    files = iter_clean_manifest_files()
    source_mode = "clean_manifest"

    if not files:
        roots = data_roots()
        files = list(
            iter_files(roots)
        )
        source_mode = "fallback_roots"

    if not files:
        raise ValueError(
            "No accepted corpus files found. Run "
            "phase2_build_corpus.bat first, or set "
            "OWN_AI_V8_DATA_DIRS for the fallback mode."
        )

    tokenizer, sample_bytes = build_tokenizer(
        files
    )

    reset_output()

    tokenizer_path = (
        OUT_ROOT / "tokenizer_v8.json"
    )

    tokenizer.save(
        tokenizer_path
    )

    shard_token_limit = max(
        1024,
        (SHARD_MB * 1024 * 1024) // 4,
    )

    state = {
        split: {
            "index": 0,
            "tokens": 0,
            "files": 0,
            "bytes": 0,
        }
        for split in (
            "train",
            "val",
            "test",
        )
    }

    seen_chunks = set()
    unique_chunk_ordinal = 0

    chunk_chars = max(
        64_000,
        SHARD_MB * 256,
    )

    for file_index, path in enumerate(
        files,
        1,
    ):
        for chunk in iter_file_chunks(
            path,
            chunk_chars,
        ):
            if len(chunk) < 32:
                continue

            chunk_hash = hashlib.sha256(
                chunk.encode("utf-8")
            ).hexdigest()

            if chunk_hash in seen_chunks:
                continue

            seen_chunks.add(
                chunk_hash
            )
            unique_chunk_ordinal += 1

            split = split_for_hash(
                chunk_hash,
                unique_chunk_ordinal,
            )

            token_ids = tokenizer.encode(
                chunk,
                add_special_tokens=True,
            )

            if len(token_ids) < 16:
                continue

            info = state[split]

            if (
                info["tokens"]
                and info["tokens"]
                + len(token_ids)
                > shard_token_limit
            ):
                info["index"] += 1
                info["tokens"] = 0
                info["files"] = 0
                info["bytes"] = 0

            shard_number = info["index"]

            out_dir = (
                OUT_ROOT / split
            )

            shard_path = (
                out_dir
                / f"shard-{shard_number:06d}.bin"
            )

            added = append_token_ids(
                shard_path,
                token_ids,
            )

            info["tokens"] += added
            info["files"] += 1
            info["bytes"] += len(
                chunk.encode("utf-8")
            )

            metadata = {
                "split": split,
                "shard": shard_number,
                "tokens": info["tokens"],
                "source_file": str(
                    path.relative_to(ROOT)
                    if ROOT in path.parents
                    else path
                ),
            }

            shard_path.with_suffix(
                ".json"
            ).write_text(
                json.dumps(
                    metadata,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

        if file_index % 25 == 0:
            print(
                f"Processed files: "
                f"{file_index}/{len(files)} | "
                f"unique chunks: "
                f"{len(seen_chunks)}"
            )

    summary = {
        "source_mode": source_mode,
        "clean_manifest": str(CLEAN_MANIFEST),
        "files": len(files),
        "unique_chunks": len(
            seen_chunks
        ),
        "tokenizer_sample_bytes": sample_bytes,
        "vocab_size": tokenizer.vocab_size,
        "shard_mb": SHARD_MB,
        "splits": state,
    }

    OUT_ROOT.joinpath(
        "dataset_summary.json"
    ).write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 68)
    print("OWN AI v8 TOKEN SHARD BUILDER")
    print("=" * 68)
    print("Source mode:", source_mode)
    print("Files:", len(files))
    print("Unique chunks:", len(seen_chunks))
    print(
        "Tokenizer sample bytes:",
        sample_bytes,
    )
    print(
        "Vocabulary:",
        tokenizer.vocab_size,
    )
    print(
        "Train:",
        state["train"],
    )
    print(
        "Val:",
        state["val"],
    )
    print(
        "Test:",
        state["test"],
    )
    print(
        "Output:",
        OUT_ROOT,
    )
    print("=" * 68)


if __name__ == "__main__":
    main()
