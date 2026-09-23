from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LAKE_ROOT = Path(
    os.getenv(
        "OWN_AI_V8_DATA_ROOT",
        r"E:\My Drive [Inuka Minsara]\OwnAI_Dataset",
    )
).resolve()

OUT_ROOT = ROOT / "data" / "processed" / "corpus_v8"
MANIFEST = OUT_ROOT / "clean_manifest_v8.jsonl"
REJECTS = OUT_ROOT / "rejected_v8.jsonl"
SUMMARY = OUT_ROOT / "summary_v8.json"
INDEX = OUT_ROOT / "dedup_v8.sqlite"

TEXT_EXTS = {
    ".txt", ".md", ".csv", ".json", ".jsonl",
    ".py", ".js", ".ts", ".tsx", ".jsx",
    ".html", ".css", ".xml", ".yaml", ".yml",
    ".sql", ".sh", ".bat", ".ps1", ".toml",
    ".ini", ".rst", ".tex", ".log",
}

DOCUMENT_EXTS = {".pdf", ".docx"}

ALLOWED = TEXT_EXTS | DOCUMENT_EXTS

SKIP_DIRS = {
    ".git", ".venv", "__pycache__", "node_modules",
    "checkpoints", "runtime", ".cache", ".tmp",
    "trash", "quarantine", "_phase2_processed",
}

SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"AIza[0-9A-Za-z_-]{20,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(
        r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*['\"]?[A-Za-z0-9_./+=-]{12,}"
    ),
]

MIN_CHARS = int(os.getenv("OWN_AI_V8_PHASE2_MIN_CHARS", "256"))
NEAR_DUP_DISTANCE = int(
    os.getenv("OWN_AI_V8_NEAR_DUP_DISTANCE", "3")
)
MINHASH_WORDS = int(
    os.getenv("OWN_AI_V8_SIMHASH_WORDS", "192")
)


def iter_files():
    if not LAKE_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset root not found: {LAKE_ROOT}"
        )

    for path in sorted(
        LAKE_ROOT.rglob("*"),
        key=lambda p: str(p).lower(),
    ):
        if not path.is_file():
            continue
        if path.suffix.lower() not in ALLOWED:
            continue
        if any(
            part.lower() in SKIP_DIRS
            for part in path.parts
        ):
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        yield path, stat


def category_for(path):
    try:
        relative = path.relative_to(LAKE_ROOT)
        if relative.parts:
            return relative.parts[0].lower()
    except ValueError:
        pass
    return "uncategorized"


def source_type_for(path):
    category = category_for(path)
    if category in {
        "owned", "private", "user", "personal",
        "conversations",
    }:
        return "user_supplied"
    if category in {
        "open_license", "open", "public_domain",
    }:
        return "open_license"
    return "unspecified"


def license_status_for(path):
    source_type = source_type_for(path)
    if source_type == "user_supplied":
        return "user_supplied"
    if source_type == "open_license":
        return "open_license_metadata_required"
    return "unknown"


def normalize_text(text):
    text = unicodedata.normalize("NFKC", text)
    text = text.replace(chr(0), "")
    cleaned = []
    for ch in text:
        if ch in "\n\r\t" or ch.isprintable():
            cleaned.append(ch)
    text = "".join(cleaned)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_text(path):
    ext = path.suffix.lower()

    if ext in TEXT_EXTS:
        return path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

    if ext == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError(
                "pypdf is required for PDF extraction"
            ) from exc

        reader = PdfReader(str(path))
        pages = []
        for page in reader.pages:
            value = page.extract_text() or ""
            if value:
                pages.append(value)
        return "\n\n".join(pages)

    if ext == ".docx":
        try:
            from docx import Document
        except ImportError as exc:
            raise RuntimeError(
                "python-docx is required for DOCX extraction"
            ) from exc

        document = Document(str(path))
        blocks = [
            paragraph.text
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        ]
        return "\n\n".join(blocks)

    raise ValueError(
        f"Unsupported extension: {ext}"
    )


def sha256_text(text):
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def language_stats(text):
    chars = [ch for ch in text if not ch.isspace()]
    if not chars:
        return {"language": "unknown", "sinhala_ratio": 0.0}

    sinhala = sum(
        0x0D80 <= ord(ch) <= 0x0DFF
        for ch in chars
    )
    latin = sum(
        ("A" <= ch <= "Z")
        or ("a" <= ch <= "z")
        for ch in chars
    )

    sinhala_ratio = sinhala / len(chars)
    latin_ratio = latin / len(chars)

    if sinhala_ratio >= 0.30:
        language = "si"
    elif latin_ratio >= 0.60:
        language = "en"
    elif sinhala_ratio > 0:
        language = "mixed-si"
    else:
        language = "other"

    return {
        "language": language,
        "sinhala_ratio": round(sinhala_ratio, 4),
    }


def simhash(text):
    words = re.findall(r"\S+", text.lower())
    if len(words) < 4:
        return 0

    features = []
    limit = MINHASH_WORDS
    for i in range(len(words) - 3):
        features.append(
            " ".join(words[i:i + 4])
        )
        if len(features) >= limit:
            break

    if not features:
        return 0

    votes = [0] * 64
    for feature in features:
        digest = hashlib.blake2b(
            feature.encode("utf-8"),
            digest_size=8,
        ).digest()
        value = int.from_bytes(
            digest,
            "big",
        )
        for bit in range(64):
            if value & (1 << bit):
                votes[bit] += 1
            else:
                votes[bit] -= 1

    result = 0
    half = len(features) / 2
    for bit, vote in enumerate(votes):
        if vote >= half:
            result |= 1 << bit

    return result


def hamming_distance(a, b):
    return (a ^ b).bit_count()


def repeated_line_ratio(text):
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]
    if len(lines) < 2:
        return 0.0

    counts = Counter(lines)
    repeated = sum(
        count - 1
        for count in counts.values()
        if count > 1
    )
    return repeated / len(lines)


def quality_report(text):
    chars = len(text)
    if chars == 0:
        return {
            "ok": False,
            "score": 0,
            "reason": "empty",
        }

    non_ws = [
        ch for ch in text
        if not ch.isspace()
    ]
    printable = sum(
        ch.isprintable()
        or ch in "\n\r\t"
        for ch in text
    )
    alnum = sum(
        ch.isalnum()
        for ch in non_ws
    )

    printable_ratio = printable / chars
    alnum_ratio = alnum / max(
        1,
        len(non_ws),
    )
    repeat_ratio = repeated_line_ratio(text)
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    secret_hits = sum(
        bool(pattern.search(text))
        for pattern in SECRET_PATTERNS
    )

    score = 100.0
    score -= max(
        0,
        0.90 - printable_ratio,
    ) * 80
    score -= max(
        0,
        0.20 - alnum_ratio,
    ) * 70
    score -= max(
        0,
        repeat_ratio - 0.10,
    ) * 80
    if chars < MIN_CHARS:
        score -= 35
    if secret_hits:
        score -= 100

    score = max(
        0.0,
        min(100.0, score),
    )

    if secret_hits:
        reason = "possible_secret"
    elif chars < MIN_CHARS:
        reason = "too_short"
    elif printable_ratio < 0.85:
        reason = "low_printable_ratio"
    elif alnum_ratio < 0.08:
        reason = "low_alnum_ratio"
    elif repeat_ratio > 0.50:
        reason = "boilerplate_or_repetition"
    else:
        reason = "accepted"

    return {
        "ok": reason == "accepted",
        "score": round(score, 2),
        "reason": reason,
        "chars": chars,
        "lines": len(lines),
        "words": len(re.findall(r"\S+", text)),
        "printable_ratio": round(
            printable_ratio,
            4,
        ),
        "alnum_ratio": round(
            alnum_ratio,
            4,
        ),
        "repeated_line_ratio": round(
            repeat_ratio,
            4,
        ),
        "secret_hits": secret_hits,
        **language_stats(text),
    }


def init_db():
    OUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )
    if INDEX.exists():
        INDEX.unlink()

    connection = sqlite3.connect(
        INDEX
    )
    connection.execute(
        """
        CREATE TABLE documents (
            id INTEGER PRIMARY KEY,
            normalized_sha256 TEXT UNIQUE NOT NULL,
            source_sha256 TEXT NOT NULL,
            simhash INTEGER NOT NULL,
            simhash_signed INTEGER NOT NULL,
            path TEXT NOT NULL,
            category TEXT NOT NULL,
            extension TEXT NOT NULL,
            chars INTEGER NOT NULL,
            quality REAL NOT NULL,
            split TEXT,
            status TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX idx_documents_band0
        ON documents ((simhash_signed >> 48))
        """
    )
    connection.execute(
        """
        CREATE INDEX idx_documents_band1
        ON documents ((simhash_signed >> 32) & 65535)
        """
    )
    connection.execute(
        """
        CREATE INDEX idx_documents_band2
        ON documents ((simhash_signed >> 16) & 65535)
        """
    )
    connection.execute(
        """
        CREATE INDEX idx_documents_band3
        ON documents (simhash_signed & 65535)
        """
    )
    return connection


def sqlite_signed(value):
    if value >= 2**63:
        return value - 2**64
    return value


def split_for_digest(digest):
    bucket = int(
        digest[:8],
        16,
    ) % 100
    if bucket < 90:
        return "train"
    if bucket < 95:
        return "val"
    return "test"


def near_duplicate(connection, signature):
    signed = sqlite_signed(signature)
    bands = [
        (0, (signed >> 48) & 65535),
        (1, (signed >> 32) & 65535),
        (2, (signed >> 16) & 65535),
        (3, signed & 65535),
    ]

    candidate_ids = set()
    for band, value in bands:
        if band == 0:
            rows = connection.execute(
                "SELECT id FROM documents WHERE (simhash_signed >> 48) = ?",
                (value,),
            ).fetchall()
        elif band == 1:
            rows = connection.execute(
                "SELECT id FROM documents WHERE ((simhash_signed >> 32) & 65535) = ?",
                (value,),
            ).fetchall()
        elif band == 2:
            rows = connection.execute(
                "SELECT id FROM documents WHERE ((simhash_signed >> 16) & 65535) = ?",
                (value,),
            ).fetchall()
        else:
            rows = connection.execute(
                "SELECT id FROM documents WHERE (simhash_signed & 65535) = ?",
                (value,),
            ).fetchall()

        candidate_ids.update(
            row[0]
            for row in rows
        )

    for candidate_id in candidate_ids:
        row = connection.execute(
            """
            SELECT simhash_signed, path, chars
            FROM documents
            WHERE id = ?
            """,
            (candidate_id,),
        ).fetchone()

        if not row:
            continue

        old_hash, old_path, old_chars = row
        old_unsigned = (
            old_hash
            if old_hash >= 0
            else old_hash + 2**64
        )

        size_ratio = min(
            signature and 1.0 or 1.0,
            old_chars / max(
                1,
                old_chars,
            ),
        )
        _ = size_ratio

        if (
            hamming_distance(
                signature,
                old_unsigned,
            )
            <= NEAR_DUP_DISTANCE
        ):
            return {
                "path": old_path,
                "hamming": hamming_distance(
                    signature,
                    old_unsigned,
                ),
            }

    return None


def write_jsonl(handle, record):
    handle.write(
        json.dumps(
            record,
            ensure_ascii=False,
        )
        + "\n"
    )


def main():
    connection = init_db()

    if MANIFEST.exists():
        MANIFEST.unlink()
    if REJECTS.exists():
        REJECTS.unlink()

    stats = Counter()
    category_counts = Counter()
    language_counts = Counter()
    split_counts = Counter()

    manifest = MANIFEST.open(
        "w",
        encoding="utf-8",
    )
    rejects = REJECTS.open(
        "w",
        encoding="utf-8",
    )

    try:
        for index, (path, stat) in enumerate(
            iter_files(),
            1,
        ):
            stats["candidates"] += 1

            source = {
                "path": str(path),
                "relative_path": str(
                    path.relative_to(
                        LAKE_ROOT
                    )
                ),
                "category": category_for(path),
                "extension": path.suffix.lower(),
                "bytes": stat.st_size,
                "source_type": source_type_for(path),
                "license_status": license_status_for(path),
            }

            try:
                raw = read_text(path)
                normalized = normalize_text(raw)
            except Exception as exc:
                stats["rejected"] += 1
                write_jsonl(
                    rejects,
                    {
                        **source,
                        "reason": "extraction_error",
                        "error": str(exc),
                    },
                )
                continue

            quality = quality_report(
                normalized
            )

            if not quality["ok"]:
                stats["rejected"] += 1
                stats[
                    "reject_" + quality["reason"]
                ] += 1
                write_jsonl(
                    rejects,
                    {
                        **source,
                        "reason": quality["reason"],
                        "quality": quality,
                    },
                )
                continue

            normalized_digest = sha256_text(
                normalized
            )

            duplicate = connection.execute(
                """
                SELECT path
                FROM documents
                WHERE normalized_sha256 = ?
                LIMIT 1
                """,
                (normalized_digest,),
            ).fetchone()

            if duplicate:
                stats["exact_duplicates"] += 1
                write_jsonl(
                    rejects,
                    {
                        **source,
                        "reason": "exact_duplicate_normalized",
                        "duplicate_of": duplicate[0],
                        "normalized_sha256": normalized_digest,
                    },
                )
                continue

            signature = simhash(
                normalized
            )
            near = near_duplicate(
                connection,
                signature,
            )

            if near:
                stats["near_duplicates"] += 1
                write_jsonl(
                    rejects,
                    {
                        **source,
                        "reason": "near_duplicate",
                        "duplicate_of": near["path"],
                        "hamming": near["hamming"],
                        "normalized_sha256": normalized_digest,
                    },
                )
                continue

            digest = sha256_file(path)
            split = split_for_digest(
                digest
            )

            record = {
                **source,
                "status": "accepted",
                "pipeline_version": "v8-phase2-corpus-1",
                "source_sha256": digest,
                "normalized_sha256": normalized_digest,
                "simhash": f"{signature:016x}",
                "split": split,
                "quality": quality,
            }

            write_jsonl(
                manifest,
                record,
            )

            connection.execute(
                """
                INSERT INTO documents (
                    normalized_sha256,
                    source_sha256,
                    simhash,
                    simhash_signed,
                    path,
                    category,
                    extension,
                    chars,
                    quality,
                    split,
                    status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    normalized_digest,
                    digest,
                    sqlite_signed(signature),
                    sqlite_signed(signature),
                    str(path),
                    source["category"],
                    source["extension"],
                    quality["chars"],
                    quality["score"],
                    split,
                    "accepted",
                ),
            )

            category_counts[
                source["category"]
            ] += 1
            language_counts[
                quality["language"]
            ] += 1
            split_counts[
                split
            ] += 1
            stats["accepted"] += 1
            stats["accepted_bytes"] += stat.st_size

            if index % 50 == 0:
                print(
                    f"Scanned {index} | "
                    f"Accepted {stats['accepted']} | "
                    f"Rejected {stats['rejected']} | "
                    f"Exact dup {stats['exact_duplicates']} | "
                    f"Near dup {stats['near_duplicates']}"
                )

            connection.commit()
    finally:
        manifest.close()
        rejects.close()
        connection.commit()
        connection.close()

    summary = {
        "pipeline_version": "v8-phase2-corpus-1",
        "dataset_root": str(LAKE_ROOT),
        "candidates": stats["candidates"],
        "accepted": stats["accepted"],
        "rejected": stats["rejected"],
        "exact_duplicates": stats["exact_duplicates"],
        "near_duplicates": stats["near_duplicates"],
        "accepted_bytes": stats["accepted_bytes"],
        "accepted_gib": round(
            stats["accepted_bytes"]
            / (1024**3),
            4,
        ),
        "categories": dict(
            sorted(category_counts.items())
        ),
        "languages": dict(
            sorted(language_counts.items())
        ),
        "splits": dict(
            sorted(split_counts.items())
        ),
        "manifest": str(MANIFEST),
        "rejects": str(REJECTS),
        "dedup_index": str(INDEX),
        "note": (
            "Phase 2 keeps source files in place and writes "
            "only manifests/indexes, avoiding a second full copy "
            "of the cloud dataset."
        ),
    }

    SUMMARY.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 68)
    print("OWN AI v8 PHASE 2 CLEAN CORPUS")
    print("=" * 68)
    print("Dataset root:", LAKE_ROOT)
    print("Candidates:", stats["candidates"])
    print("Accepted:", stats["accepted"])
    print("Rejected:", stats["rejected"])
    print(
        "Exact duplicates:",
        stats["exact_duplicates"],
    )
    print(
        "Near duplicates:",
        stats["near_duplicates"],
    )
    print(
        "Accepted GiB:",
        round(
            stats["accepted_bytes"]
            / (1024**3),
            4,
        ),
    )
    print("Categories:", dict(category_counts))
    print("Languages:", dict(language_counts))
    print("Splits:", dict(split_counts))
    print("Manifest:", MANIFEST)
    print("Rejects:", REJECTS)
    print("Index:", INDEX)
    print("=" * 68)


if __name__ == "__main__":
    main()
