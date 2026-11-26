"""Make the pydoc-markdown API pages under docs/docs/api MDX-safe.

Run by ``npm run generate-api`` after pydoc-markdown. Outside fenced code blocks and inline code spans every
``{`` and ``}`` is escaped (MDX would read them as JavaScript expressions), and so is ``<`` (docstrings
contain no HTML). Inline code spans written with double
backticks (reST style) become single-backtick spans when their content has no backtick, and HTML entities inside code
spans are decoded (code spans are shown literally).

Usage: python3 docs/scripts/postprocess_api.py docs/docs/api
"""

import html
import pathlib
import re
import sys

_FENCE = re.compile(r"^\s*(```|~~~)")
_TICKS = re.compile(r"`+")


def _process_line(line: str) -> str:
    """Escape braces outside inline code and normalise the code spans of one line."""
    out = []
    pos = 0
    while True:
        m = _TICKS.search(line, pos)
        if m is None:
            break
        # the closing run must have exactly the same number of backticks
        close = re.compile(r"(?<!`)" + re.escape(m.group(0)) + r"(?!`)")
        c = close.search(line, m.end())
        if c is None:
            break
        out.append(_escape(line[pos : m.start()]))
        content = html.unescape(line[m.end() : c.start()])
        if "`" in content:
            out.append(m.group(0) + content + m.group(0))
        else:
            out.append("`" + content.strip() + "`" if content.strip() else m.group(0) + content + m.group(0))
        pos = c.end()
    out.append(_escape(line[pos:]))
    return "".join(out)


_LT = re.compile(r"<")


def _escape(text: str) -> str:
    """Escape MDX expression braces and ``<`` (docstrings contain no HTML) in plain text."""
    return _LT.sub("&lt;", text.replace("{", "\\{").replace("}", "\\}"))


def process(path: pathlib.Path) -> None:
    """Rewrite one Markdown file in place (front matter and fenced blocks unchanged)."""
    lines = path.read_text().split("\n")
    result = []
    in_front = lines[:1] == ["---"]
    in_fence = False
    for i, line in enumerate(lines):
        if in_front:
            result.append(line)
            if i > 0 and line == "---":
                in_front = False
            continue
        if _FENCE.match(line):
            in_fence = not in_fence
            result.append(line)
            continue
        result.append(line if in_fence else _process_line(line))
    path.write_text("\n".join(result))


def main(root: str) -> None:
    """Process every ``.md`` file below ``root``."""
    for path in sorted(pathlib.Path(root).rglob("*.md")):
        process(path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "docs/docs/api")
