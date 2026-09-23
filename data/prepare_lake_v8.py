from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "data" / "processed" / "data_lake_v8"
MANIFEST = OUT_ROOT / "manifest_v8.jsonl"
REJECTS = OUT_ROOT / "rejected_v8.jsonl"
SUMMARY = OUT_ROOT / "summary_v8.json"

ALLOWED = {
    ".txt", ".md", ".csv", ".json", ".jsonl",
    ".py", ".js", ".ts", ".tsx", ".jsx",
    ".html", ".css", ".xml", ".yaml", ".yml", ".sql",
    ".sh", ".bat", ".ps1", ".toml", ".ini",
    ".rst", ".tex", ".log",
    ".pdf", ".docx",
}

SKIP_DIRS = {
    ".git", ".venv", "__pycache__", "node_modules",
    "checkpoints", "runtime", ".cache", ".tmp",
    "trash", "quarantine",
}

TEXT_EXTS = {
    ".txt", ".md", ".csv", ".json", ".jsonl",
    ".py", ".js", ".ts", ".tsx", ".jsx",
    ".html", ".css", ".xml", ".yaml", ".yml",
    ".sql", ".sh", ".bat", ".ps1", ".toml",
    ".ini", ".rst", ".tex", ".log",
}

SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"AIza[0-9A-Za-z_-]{20,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*['\"]?[A-Za-z0-9_./+=-]{12,}"),
]

DEFAULT_ROOTS = [
    ROOT / "data" / "cache",
    ROOT / "data" / "raw",
    ROOT / "data" / "knowledge",
]


def data_roots():
    raw = os.getenv("OWN_AI_V8_DATA_DIRS", "")
    if raw.strip():
        return [Path(item.strip()) for item in raw.split(";") if item.strip()]
    return DEFAULT_ROOTS


def file_category(path):
    parts = path.parts
    if ROOT in path.parents:
        rel = path.relative_to(ROOT)
    else:
        rel = path
    rel_parts = rel.parts
    if len(rel_parts) >= 2 and rel_parts[0].lower() == "ownai_dataset":
        return rel_parts[1].lower()
    if len(rel_parts) >= 1:
        return rel_parts[0].lower()
    return "uncategorized"


def iter_files(roots):
    seen_paths = set()
    for root in roots:
        root = root.resolve()
        if not root.exists():
            print("MISSING:", root)
            continue
        for path in sorted(root.rglob("*"), key=lambda p: str(p).lower()):
            if not path.is_file():
                continue
            if path.suffix.lower() not in ALLOWED:
                continue
            if any(part.lower() in SKIP_DIRS for part in path.parts):
                continue
            key = str(path).lower()
            if key in seen_paths:
                continue
            seen_paths.add(key)
            try:
                stat = path.stat()
            except OSError:
                continue
            yield path, stat


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def analyze_text(path, min_chars):
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        return {"ok": False, "reason": f"read_error:{exc}"}

    text = text.replace(chr(0), "")
    char_count = len(text)
    if char_count < min_chars:
        return {"ok": False, "reason": "too_short", "chars": char_count}

    non_ws = [ch for ch in text if not ch.isspace()]
    printable = sum(ch.isprintable() or ch in "\n\r\t" for ch in text)
    alpha_num = sum(ch.isalnum() for ch in non_ws)
    printable_ratio = printable / max(1, len(text))
    alnum_ratio = alpha_num / max(1, len(non_ws))

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    line_counter = Counter(lines)
    repeated_lines = sum(count - 1 for count in line_counter.values() if count > 1)
    repeated_ratio = repeated_lines / max(1, len(lines))

    secret_hits = []
    for pattern in SECRET_PATTERNS:
        if pattern.search(text):
            secret_hits.append(pattern.pattern)

    ok = (
        printable_ratio >= float(os.getenv("OWN_AI_V8_MIN_PRINTABLE", "0.85"))
        and alnum_ratio >= float(os.getenv("OWN_AI_V8_MIN_ALNUM", "0.12"))
        and repeated_ratio <= float(os.getenv("OWN_AI_V8_MAX_REPEAT_LINES", "0.35"))
        and not secret_hits
    )

    reason = "accepted"
    if secret_hits:
        reason = "possible_secret"
    elif printable_ratio < float(os.getenv("OWN_AI_V8_MIN_PRINTABLE", "0.85")):
        reason = "low_printable_ratio"
    elif alnum_ratio < float(os.getenv("OWN_AI_V8_MIN_ALNUM", "0.12")):
        reason = "low_alnum_ratio"
    elif repeated_ratio > float(os.getenv("OWN_AI_V8_MAX_REPEAT_LINES", "0.35")):
        reason = "boilerplate_or_repetition"

    return {
        "ok": ok,
        "reason": reason,
        "chars": char_count,
        "lines": len(lines),
        "printable_ratio": round(printable_ratio, 4),
        "alnum_ratio": round(alnum_ratio, 4),
        "repeated_line_ratio": round(repeated_ratio, 4),
        "secret_hits": len(secret_hits),
    }


def split_for_hash(value, ordinal, total):
    # Guarantee non-empty train/val/test for any dataset with >= 3 items.
    # After the first three items, use the deterministic 90/5/5 hash split.
    if total >= 3:
        if ordinal == 1:
            return "train"
        if ordinal == 2:
            return "val"
        if ordinal == 3:
            return "test"
    elif total == 2:
        return "train" if ordinal == 1 else "val"
    else:
        return "train"

    bucket = int(value[:8], 16) % 100
    if bucket < 90:
        return "train"
    if bucket < 95:
        return "val"
    return "test"


def display_path(path):
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main():
    roots = data_roots()
    min_chars = int(os.getenv("OWN_AI_V8_LAKE_MIN_CHARS", "128"))

    files = list(iter_files(roots))
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    if MANIFEST.exists():
        MANIFEST.unlink()
    if REJECTS.exists():
        REJECTS.unlink()

    if not files:
        raise ValueError("No supported dataset files found.")

    # Read/hash every candidate once. Exact duplicates are collapsed by content hash.
    hashed = []
    for index, (path, stat) in enumerate(files, 1):
        try:
            digest = sha256_file(path)
        except OSError as exc:
            record = {"path": display_path(path), "reason": f"hash_error:{exc}"}
            with REJECTS.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            continue

        analysis = (
            analyze_text(path, min_chars)
            if path.suffix.lower() in TEXT_EXTS
            else {"ok": True, "reason": "document_pending_extraction", "chars": None}
        )
        hashed.append((path, stat, digest, analysis))

        if index % 100 == 0:
            print(f"Scanned: {index}/{len(files)}")

    unique_by_hash = {}
    duplicate_count = 0
    for path, stat, digest, analysis in hashed:
        if digest in unique_by_hash:
            duplicate_count += 1
            with REJECTS.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "path": display_path(path),
                    "reason": "exact_duplicate",
                    "duplicate_of": unique_by_hash[digest]["path"],
                    "sha256": digest,
                }, ensure_ascii=False) + "\n")
            continue
        unique_by_hash[digest] = {
            "path": display_path(path),
            "analysis": analysis,
        }

    unique_items = []
    rejected = duplicate_count
    for path, stat, digest, analysis in hashed:
        first = unique_by_hash.get(digest)
        if not first or first["path"] != display_path(path):
            continue
        if not analysis.get("ok", False):
            rejected += 1
            with REJECTS.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "path": display_path(path),
                    "reason": analysis.get("reason", "rejected"),
                    "sha256": digest,
                }, ensure_ascii=False) + "\n")
            continue
        unique_items.append((path, stat, digest, analysis))

    unique_items.sort(key=lambda item: item[2])
    total = len(unique_items)
    category_counts = Counter()
    split_counts = Counter()
    extension_counts = Counter()
    total_bytes = 0

    with MANIFEST.open("w", encoding="utf-8") as manifest:
        for ordinal, (path, stat, digest, analysis) in enumerate(unique_items, 1):
            category = file_category(path)
            split = split_for_hash(digest, ordinal, total)
            record = {
                "path": display_path(path),
                "absolute_path": str(path),
                "category": category,
                "extension": path.suffix.lower(),
                "bytes": stat.st_size,
                "sha256": digest,
                "mtime_ns": stat.st_mtime_ns,
                "split": split,
                "content": analysis,
                "status": "accepted",
                "pipeline_version": "v8-phase1-lake-1",
            }
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
            category_counts[category] += 1
            split_counts[split] += 1
            extension_counts[path.suffix.lower()] += 1
            total_bytes += stat.st_size

    summary = {
        "pipeline_version": "v8-phase1-lake-1",
        "roots": [str(root.resolve()) for root in roots],
        "candidate_files": len(files),
        "hashed_candidates": len(hashed),
        "accepted_unique_files": len(unique_items),
        "exact_duplicates": duplicate_count,
        "rejected": rejected,
        "accepted_bytes": total_bytes,
        "accepted_gib": round(total_bytes / (1024 ** 3), 4),
        "categories": dict(sorted(category_counts.items())),
        "extensions": dict(sorted(extension_counts.items())),
        "splits": dict(sorted(split_counts.items())),
        "manifest": str(MANIFEST),
        "rejects": str(REJECTS),
        "note": "PDF/DOCX are indexed for Phase 1; extraction is a later pipeline stage.",
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 68)
    print("OWN AI v8 PHASE 1 DATA LAKE AUDIT")
    print("=" * 68)
    print("Roots:", "; ".join(str(root) for root in roots))
    print("Candidates:", len(files))
    print("Accepted unique:", len(unique_items))
    print("Exact duplicates:", duplicate_count)
    print("Rejected:", rejected)
    print("Accepted bytes:", total_bytes)
    print("Accepted GiB:", round(total_bytes / (1024 ** 3), 4))
    print("Splits:", dict(split_counts))
    print("Manifest:", MANIFEST)
    print("Rejects:", REJECTS)
    print("Summary:", SUMMARY)
    print("=" * 68)


if __name__ == "__main__":
    main()