from pathlib import Path
import re

from tools.math_solver import (
    extract_expression,
    looks_like_math,
    solve_expression,
)
from tools.web_search import search_web


class ToolRouter:
    """Deterministic tools for the tiny local model.

    The router handles capabilities that should be exact instead of sampled
    from a 6.5M-parameter language model.
    """

    def __init__(self, root):
        self.root = Path(root)
        self.tools = {
            "calculator": self._calculator,
            "web_search": self._web_search,
            "read_file": self._read_file,
            "list_files": self._list_files,
        }

    def route(self, text):
        if looks_like_math(text):
            try:
                expression = extract_expression(text)
                value = solve_expression(expression)
                return {
                    "tool": "calculator",
                    "answer": f"Result: {value}",
                    "data": {
                        "expression": expression,
                        "value": value,
                    },
                }
            except (SyntaxError, ValueError, ZeroDivisionError):
                pass

        lowered = text.lower()

        search_triggers = (
            "search the web",
            "search online",
            "look this up",
            "latest news",
            "current information",
            "what happened today",
            "google this",
        )

        if any(trigger in lowered for trigger in search_triggers):
            query = re.sub(
                r"^(search the web|search online|look this up)\s*",
                "",
                text,
                flags=re.I,
            ).strip() or text

            try:
                results = search_web(query)
                return {
                    "tool": "web_search",
                    "answer": self._format_search_results(results),
                    "data": {"results": results},
                }
            except Exception as exc:
                return {
                    "tool": "web_search",
                    "answer": (
                        "Web search could not be completed locally: "
                        f"{exc}"
                    ),
                    "data": {"results": []},
                }

        return None

    def _calculator(self, _text):
        return None

    def _web_search(self, _text):
        return None

    def _read_file(self, text):
        name = text.strip()
        base = (self.root / "data").resolve()

        candidate = (base / name).resolve()

        if base not in candidate.parents and candidate != base:
            raise ValueError("File path outside data folder.")

        if not candidate.exists() or not candidate.is_file():
            raise ValueError("File not found.")

        content = candidate.read_text(
            encoding="utf-8",
            errors="ignore",
        )

        return content[:12000]

    def _list_files(self, _text):
        base = self.root / "data"

        return sorted(
            str(path.relative_to(base))
            for path in base.rglob("*")
            if path.is_file()
        )[:500]

    @staticmethod
    def _format_search_results(results):
        if not results:
            return "No web results were found."

        lines = ["Web results:"]
        for i, item in enumerate(results, 1):
            lines.append(
                f"{i}. {item['title']}\n   {item['url']}"
            )

        return "\n".join(lines)
