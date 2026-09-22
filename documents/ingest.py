from pathlib import Path
import re


def extract_text(path):
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in {".txt", ".md", ".csv", ".json", ".py", ".js", ".html", ".css"}:
        return path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError(
                "PDF support requires pypdf. Install: pip install pypdf"
            ) from exc

        reader = PdfReader(str(path))
        return "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )

    if suffix == ".docx":
        try:
            from docx import Document
        except ImportError as exc:
            raise RuntimeError(
                "DOCX support requires python-docx. Install: pip install python-docx"
            ) from exc

        doc = Document(str(path))
        return "\n".join(
            paragraph.text
            for paragraph in doc.paragraphs
            if paragraph.text.strip()
        )

    raise ValueError(
        f"Unsupported document type: {suffix}"
    )


def clean_text(text):
    text = text.replace("\x00", "")
    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )
    return text.strip()


def ingest(path, output_dir):
    path = Path(path)
    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    text = clean_text(
        extract_text(path)
    )

    if not text:
        raise ValueError("The document contains no extractable text.")

    safe_name = re.sub(
        r"[^a-zA-Z0-9._-]",
        "_",
        path.stem,
    )

    output = output_dir / f"{safe_name}.txt"
    output.write_text(
        text,
        encoding="utf-8",
    )

    return {
        "name": path.name,
        "text_file": str(output),
        "characters": len(text),
    }
