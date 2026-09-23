from __future__ import annotations

import argparse
import bz2
import hashlib
import html
import json
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MAX_CHARS = int(
    os.getenv("OWN_AI_V8_WIKI_CHUNK_CHARS", "2000000")
)


def strip_wikitext(text):
    text = html.unescape(text)
    text = re.sub(
        r"<ref[^>]*>.*?</ref>",
        " ",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(
        r"<[^>]+>",
        " ",
        text,
    )
    text = re.sub(
        r"\{\{[^{}]*\}\}",
        " ",
        text,
    )
    text = re.sub(
        r"\[\[([^\]|]+\|)?([^\]]+)\]\]",
        r"\2",
        text,
    )
    text = re.sub(
        r"\[\[([^\]]+)\]\]",
        r"\1",
        text,
    )
    text = re.sub(
        r"\[']{2,}",
        "",
        text,
    )
    text = re.sub(
        r"^[#*;:]+\s*",
        "",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"\s+",
        " ",
        text,
    )
    return text.strip()


def local_name(tag):
    return tag.rsplit("}", 1)[-1]


def write_chunk(handle, title, text):
    body = strip_wikitext(text)
    if len(body) < 256:
        return 0

    record = (
        f"Title: {title}\n"
        f"{body}\n\n"
    )
    handle.write(record)
    return len(record)


def extract(source, output_dir, max_chars):
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    part_index = 0
    chars = 0
    article_count = 0
    output = None
    output_path = None

    def open_part():
        nonlocal output, output_path, part_index, chars
        if output:
            output.close()

        output_path = (
            output_dir
            / f"part-{part_index:06d}.txt"
        )
        output = output_path.open(
            "w",
            encoding="utf-8",
        )
        chars = 0
        part_index += 1

    open_part()

    namespace = {
        "mw": "http://www.mediawiki.org/xml/export-0.10/"
    }

    with bz2.open(
        source,
        "rb",
    ) as compressed:
        context = ET.iterparse(
            compressed,
            events=("end",),
        )

        for _event, elem in context:
            if local_name(elem.tag) != "page":
                continue

            title = ""
            text_value = ""
            redirect = False
            namespace_id = "0"

            for child in list(elem):
                name = local_name(child.tag)
                if name == "title":
                    title = child.text or ""
                elif name == "ns":
                    namespace_id = child.text or "0"
                elif name == "redirect":
                    redirect = True
                elif name == "revision":
                    for grandchild in list(child):
                        if local_name(grandchild.tag) == "text":
                            text_value = grandchild.text or ""

            if (
                namespace_id != "0"
                or redirect
                or not text_value
            ):
                elem.clear()
                continue

            clean = strip_wikitext(
                text_value
            )
            if len(clean) < 256:
                elem.clear()
                continue

            record = (
                f"Title: {title.strip()}\n"
                f"{clean}\n\n"
            )

            if (
                chars
                and chars + len(record) > max_chars
            ):
                open_part()

            output.write(record)
            chars += len(record)
            article_count += 1

            if article_count % 1000 == 0:
                output.flush()
                print(
                    f"Articles: {article_count} | "
                    f"Current part chars: {chars}"
                )

            elem.clear()

    if output:
        output.close()

    metadata = {
        "source_file": str(source),
        "source_sha256": hashlib.sha256(
            source.read_bytes()
        ).hexdigest(),
        "articles": article_count,
        "parts": part_index,
        "max_chars": max_chars,
        "license": "CC BY-SA",
        "license_url": "https://meta.wikimedia.org/wiki/Terms_of_Use",
        "extractor": "v8-wikimedia-stream-1",
    }

    (output_dir / "EXTRACTION_METADATA.json").write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 68)
    print("WIKIMEDIA EXTRACTION FINISHED")
    print("Articles:", article_count)
    print("Parts:", part_index)
    print("Output:", output_dir)
    print("=" * 68)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "source",
        type=Path,
    )
    parser.add_argument(
        "output_dir",
        type=Path,
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=DEFAULT_MAX_CHARS,
    )
    args = parser.parse_args()

    extract(
        args.source,
        args.output_dir,
        args.max_chars,
    )


if __name__ == "__main__":
    main()
