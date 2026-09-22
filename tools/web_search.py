from html import unescape
from html.parser import HTMLParser
import re
from urllib.parse import quote_plus
from urllib.request import Request, urlopen


class ResultParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.results = []
        self._current = None
        self._capture = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)

        if tag == "a":
            href = attrs.get("href", "")
            cls = attrs.get("class", "")

            if "result__a" in cls:
                self._current = {
                    "title": "",
                    "url": href,
                    "snippet": "",
                }
                self._capture = True

    def handle_endtag(self, tag):
        if tag == "a" and self._capture:
            self._capture = False
            if self._current and self._current["title"]:
                self.results.append(self._current)
                self._current = None

    def handle_data(self, data):
        if self._capture and self._current:
            self._current["title"] += data


def search_web(query, max_results=5):
    query = query.strip()

    if not query:
        return []

    url = (
        "https://html.duckduckgo.com/html/"
        f"?q={quote_plus(query)}"
    )

    request = Request(
        url,
        headers={
            "User-Agent": "OwnAI/7.0 local-search",
        },
    )

    with urlopen(request, timeout=8) as response:
        html = response.read().decode(
            "utf-8",
            errors="ignore",
        )

    parser = ResultParser()
    parser.feed(html)

    results = []

    for item in parser.results[:max_results]:
        href = unescape(item["url"])
        if not href.startswith("http"):
            continue

        title = re.sub(
            r"\s+",
            " ",
            unescape(item["title"]),
        ).strip()

        results.append({
            "title": title,
            "url": href,
        })

    return results
