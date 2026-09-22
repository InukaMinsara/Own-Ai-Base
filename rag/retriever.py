from pathlib import Path
from collections import Counter
import math
import re


class LocalRetriever:
    """
    Lightweight dependency-free TF-IDF-style retriever.

    It indexes local .txt/.md files under data/. Retrieved text is runtime
    context; it is not written into model weights.
    """

    WORD_RE = re.compile(
        r"[a-zA-Z0-9_]+",
        re.UNICODE,
    )

    def __init__(self, root):
        self.root = Path(root)
        self.documents = []
        self.df = Counter()
        self.idf = {}
        self.ready = False

    def _words(self, text):
        return [
            word.lower()
            for word in self.WORD_RE.findall(text)
        ]

    def _chunk_text(
        self,
        text,
        chunk_size=1200,
        overlap=180,
    ):
        text = text.strip()

        if not text:
            return []

        chunks = []
        start = 0

        while start < len(text):
            end = min(
                len(text),
                start + chunk_size,
            )

            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            if end >= len(text):
                break

            start = max(
                0,
                end - overlap,
            )

        return chunks

    def build(self):
        self.documents.clear()
        self.df.clear()
        self.ready = False

        source_dirs = [
            self.root / "data" / "raw",
            self.root / "data" / "knowledge",
            self.root / "data" / "uploads",
        ]

        candidates = []

        for source_dir in source_dirs:
            if not source_dir.exists():
                continue

            candidates.extend(
                source_dir.rglob("*.txt")
            )
            candidates.extend(
                source_dir.rglob("*.md")
            )

        seen = set()

        for path in candidates:

            try:
                text = path.read_text(
                    encoding="utf-8"
                )
            except (
                OSError,
                UnicodeDecodeError,
            ):
                continue

            for chunk in self._chunk_text(text):
                key = chunk.strip()

                if not key or key in seen:
                    continue

                seen.add(key)

                words = self._words(chunk)

                if not words:
                    continue

                counts = Counter(words)

                self.documents.append(
                    {
                        "source": str(
                            path.relative_to(
                                self.root
                            )
                        ),
                        "text": chunk,
                        "counts": counts,
                    }
                )

                for word in counts:
                    self.df[word] += 1

        n_docs = max(
            1,
            len(self.documents),
        )

        self.idf = {
            word: math.log(
                (1 + n_docs)
                / (1 + freq)
            ) + 1.0
            for word, freq in self.df.items()
        }

        self.ready = True

        return len(self.documents)

    def search(
        self,
        query,
        top_k=3,
        min_score=0.05,
    ):
        if not self.ready:
            self.build()

        q_words = self._words(query)

        if not q_words:
            return []

        q_counts = Counter(q_words)
        scored = []

        for doc in self.documents:
            score = 0.0

            for word, q_count in q_counts.items():
                tf = doc["counts"].get(
                    word,
                    0,
                )

                if tf == 0:
                    continue

                score += (
                    (
                        1.0
                        + math.log(tf)
                    )
                    * (
                        1.0
                        + math.log(q_count)
                    )
                    * self.idf.get(
                        word,
                        1.0,
                    )
                )

            if score >= min_score:
                scored.append(
                    (score, doc)
                )

        scored.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        return [
            {
                "score": round(
                    score,
                    4,
                ),
                "source": doc["source"],
                "text": doc["text"],
            }
            for score, doc in scored[:top_k]
        ]

    def context(
        self,
        query,
        top_k=3,
        max_chars=2200,
    ):
        results = self.search(
            query,
            top_k=top_k,
        )

        parts = []
        used = 0

        for item in results:
            block = (
                f"[Source: {item['source']}]\n"
                f"{item['text']}\n"
            )

            if used + len(block) > max_chars:
                remaining = max_chars - used

                if remaining <= 80:
                    break

                block = block[:remaining]

            parts.append(block)
            used += len(block)

        return "\n".join(parts), results
