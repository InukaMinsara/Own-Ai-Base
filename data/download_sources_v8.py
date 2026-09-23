from __future__ import annotations

import argparse
import json
import os
import shutil
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "data" / "sources_v8.json"

REQUESTED_ROOT = Path(
    os.getenv(
        "OWN_AI_V8_DATA_ROOT",
        r"E:\My Drive [Inuka Minsara]\OwnAI_Dataset",
    )
).resolve()

FALLBACK_ROOT = Path(
    os.getenv(
        "OWN_AI_V8_DOWNLOAD_FALLBACK",
        str(ROOT / "data" / "cache" / "phase2_sources"),
    )
).resolve()


def load_catalog():
    return json.loads(
        CATALOG.read_text(
            encoding="utf-8"
        )
    )


def download(url: str, destination: Path):
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    part = destination.with_suffix(
        destination.suffix + ".part"
    )

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "OwnAI-Phase2-DataFetcher/1.0"
        },
    )

    existing = part.stat().st_size if part.exists() else 0

    if existing:
        request.add_header(
            "Range",
            f"bytes={existing}-",
        )

    try:
        with urllib.request.urlopen(
            request,
            timeout=60,
        ) as response:
            mode = "ab" if existing else "wb"

            if existing and response.status != 206:
                mode = "wb"
                existing = 0

            with part.open(mode) as handle:
                shutil.copyfileobj(
                    response,
                    handle,
                    length=1024 * 1024,
                )
    except Exception:
        print(
            "Download interrupted. Partial file kept:",
            part,
        )
        raise

    part.replace(destination)


def choose_root(requested: Path, target_dir: Path) -> Path:
    # Air Live Drive may expose the mount while denying writes inside a
    # particular subdirectory. Probe the exact destination before writing
    # metadata or starting a potentially large download.
    try:
        target_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        probe = target_dir / ".ownai_write_probe"
        probe.write_text(
            "ok",
            encoding="utf-8",
        )
        probe.unlink()
        return requested
    except (OSError, PermissionError):
        FALLBACK_ROOT.mkdir(
            parents=True,
            exist_ok=True,
        )
        print(
            "WARNING: Destination is not writable:",
            target_dir,
        )
        print(
            "Using local staging root:",
            FALLBACK_ROOT,
        )
        return FALLBACK_ROOT


def main():
    parser = argparse.ArgumentParser(
        description="Download approved Own AI Phase 2 sources."
    )
    parser.add_argument(
        "sources",
        nargs="+",
        choices=sorted(
            load_catalog()["approved"]
        ),
    )
    args = parser.parse_args()

    catalog = load_catalog()["approved"]

    for source_id in args.sources:
        item = catalog[source_id]

        requested_dir = (
            REQUESTED_ROOT
            / item["target_dir"]
        )
        download_root = choose_root(
            REQUESTED_ROOT,
            requested_dir,
        )

        target = (
            download_root
            / item["target_dir"]
            / Path(item["url"]).name
        )

        license_file = (
            target.parent
            / "SOURCE_METADATA.json"
        )

        license_file.write_text(
            json.dumps(
                {
                    "source_id": source_id,
                    **item,
                    "requested_root": str(REQUESTED_ROOT),
                    "download_root": str(download_root),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print("=" * 68)
        print("Downloading:", item["name"])
        print("License:", item["license"])
        print("Destination:", target)
        if download_root != REQUESTED_ROOT:
            print("Original cloud path:", REQUESTED_ROOT / item["target_dir"] / Path(item["url"]).name)
        print("=" * 68)

        download(
            item["url"],
            target,
        )

        print("Completed:", target)


if __name__ == "__main__":
    main()
