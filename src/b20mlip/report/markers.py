"""Marker grammar shared by the README writer (``build``) and the honesty auditor (``audit``).

Grammar (binding for every README produced or audited by this project)::

    number  ::= "<!-- num:" KEY " -->" VALUE "<!-- /num -->"
    KEY     ::= [A-Za-z0-9][A-Za-z0-9_.-]*        # a dotted key of reports/numbers.json
    VALUE   ::= text without "<": exactly ONE numeral (the formatted value) or a numeral-free
                placeholder such as "pending" or "n/a (scale)"
    block   ::= "<!-- gen:start:" NAME " -->" BODY "<!-- gen:end -->"
    NAME    ::= [a-z0-9_-]+                       # blocks never nest; BODY may be inline

A *numeral* is a digit sequence with optional sign, thousands separators, decimals and exponent
(``-3``, ``1,000``, ``12.5``, ``1e-3``). Gate A1 requires every numeral outside gen blocks to sit
inside a number marker, except for the narrow allowlist below (``ALLOWLIST_SPANS``):

* fenced code blocks (three backticks or ``~~~``) and HTML comments;
* URLs, markdown link targets and badges;
* version strings: ``v0.1``, ``Python 3.11``, three-part ``0.3.16``, pins ``name==1.2``;
* ISO dates ``2026-09-18``;
* identifier tokens that START WITH A LETTER and glue their digits with letters, ``-``, ``.``,
  ``+`` or ``′`` only (``T0``, ``B0′``, ``MACE-MPA-0``, ``mp-871``, ``CC-BY-4.0``, ``sha256``);
  a token containing ``=``, whitespace or starting with a digit is never exempt (``8-atom``,
  ``1,000-structure``, ``MAE=12.3``, ``1e-3`` are all numerals).

Inline code spans are NOT exempt. Spell small protocol numbers out in prose ("three seeds")
or generate them inside a gen block.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
NUM_RE = re.compile(
    r"<!--\s*num:(?P<key>[A-Za-z0-9][A-Za-z0-9_.-]*)\s*-->(?P<value>[^<]*)<!--\s*/num\s*-->"
)
GEN_START_RE = re.compile(r"<!--\s*gen:start:(?P<name>[a-z0-9_-]+)\s*-->")
GEN_END_RE = re.compile(r"<!--\s*gen:end\s*-->")
GEN_ANY_RE = re.compile(r"<!--\s*gen:(?:start:[a-z0-9_-]+|end)\s*-->")
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
FENCED_CODE_RE = re.compile(r"^[ \t]*(```|~~~)[^\n]*\n.*?^[ \t]*\1[ \t]*$", re.S | re.M)
HEADING2_RE = re.compile(r"^##(?!#)[ \t]+(?P<title>.+?)[ \t]*#*[ \t]*$", re.M)
TABLE_LINE_RE = re.compile(r"^[ \t]*\|.*\|?[ \t]*$")
TABLE_SEP_RE = re.compile(r"^[ \t]*\|?(?:[ \t]*:?-{3,}:?[ \t]*\|)+[ \t]*:?-*:?[ \t]*\|?[ \t]*$")

NUMERAL_RE = re.compile(r"(?<![\w.])[+\-−]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+\-]?\d+)?")
# Narrow allowlist (see the module docstring). Each pattern is masked BEFORE numerals are sought.
ALLOWLIST_SPANS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("badge", re.compile(r"!\[[^\]]*\]\([^)]*\)")),
    ("url", re.compile(r"https?://[^\s<>)\]]+")),
    ("link-target", re.compile(r"\]\([^)\s]*\)")),
    ("pin", re.compile(r"(?<![\w.])[\w.\-]+==\d[\w.]*")),
    ("python-version", re.compile(r"\bPython \d+\.\d+(?:\.\d+)?\b")),
    ("version", re.compile(r"(?<![\w.])v\d+(?:\.\d+)+\b")),
    ("semver", re.compile(r"(?<![\w.])\d+\.\d+\.\d+(?![\w.])")),
    ("iso-date", re.compile(r"(?<![\w.])\d{4}-\d{2}-\d{2}(?![\w.])")),
)
IDENT_TOKEN_RE = re.compile(r"^[^\W\d_][\w\-–.+′]*$")
_TOKEN_STRIP = "([{\"'*_~`<>)]},.;:!?"


class MarkerError(ValueError):
    """Malformed marker syntax (unbalanced or nested gen blocks)."""


@dataclass(frozen=True)
class NumMarker:
    key: str
    text: str  # the VALUE between the two comments
    start: int  # offset of "<!-- num:"
    end: int  # offset just past "<!-- /num -->"


@dataclass(frozen=True)
class GenBlock:
    name: str
    start: int
    end: int
    body_start: int
    body_end: int


@dataclass(frozen=True)
class Section:
    title: str  # raw heading text ("" for the preamble before the first level-2 heading)
    start: int
    end: int

    @property
    def normalized(self) -> str:
        return self.title.strip().lstrip("#*_ ").lower().strip()


@dataclass
class TableRow:
    line: int  # 1-based line number in the README
    start: int
    end: int
    cells: list[str] = field(default_factory=list)


@dataclass
class Table:
    header: list[str]
    rows: list[TableRow]


# --- formatting and parsing ---------------------------------------------------------------------


def format_value(value: float, fmt: str | None = None) -> str:
    """Canonical text of a published number: integers plain, else four significant digits."""
    if fmt:
        return fmt.format(value)
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"cannot format a non-finite number: {v!r}")
    if v.is_integer() and abs(v) < 1e15:
        return str(int(v))
    s = f"{v:.4g}"
    if "e" in s or "E" in s:
        s = format(Decimal(s), "f")
    return s


def parse_numeral(text: str) -> Decimal | None:
    """``Decimal`` of one numeral (``1,000``, ``−3.5``, ``1e-3``, ``12 %``) or ``None``."""
    s = text.strip().replace("−", "-").replace(",", "").replace(" ", "")
    s = s.rstrip("%").strip()
    if not s:
        return None
    try:
        return Decimal(s)
    except InvalidOperation:
        return None


def value_matches(displayed: str, value: float) -> bool:
    """Does the displayed numeral equal ``value`` up to its rounding (or 4 significant digits)?"""
    d = parse_numeral(displayed)
    if d is None or not d.is_finite():
        return False
    exponent = d.as_tuple().exponent
    quantum = Decimal(10) ** int(exponent) if isinstance(exponent, int) else Decimal(1)
    tol = max(float(quantum) / 2.0, abs(float(value)) * 5e-4)
    return abs(float(d) - float(value)) <= tol + 1e-12


def num_marker(key: str, text: str) -> str:
    if not KEY_RE.match(key):
        raise ValueError(f"invalid number key {key!r}")
    if "<" in text:
        raise ValueError(f"marker text may not contain '<': {text!r}")
    return f"<!-- num:{key} -->{text}<!-- /num -->"


def gen_block(name: str, body: str) -> str:
    if not GEN_START_RE.fullmatch(f"<!-- gen:start:{name} -->"):
        raise ValueError(f"invalid gen block name {name!r}")
    return f"<!-- gen:start:{name} -->{body}<!-- gen:end -->"


# --- scanning ------------------------------------------------------------------------------------


def find_num_markers(text: str) -> list[NumMarker]:
    return [
        NumMarker(key=m.group("key"), text=m.group("value"), start=m.start(), end=m.end())
        for m in NUM_RE.finditer(text)
    ]


def find_gen_blocks(text: str) -> list[GenBlock]:
    """All gen blocks in document order; raises :class:`MarkerError` on nesting or imbalance."""
    blocks: list[GenBlock] = []
    open_block: tuple[str, int, int] | None = None
    for m in GEN_ANY_RE.finditer(text):
        start = GEN_START_RE.fullmatch(m.group(0))
        if start is not None:
            if open_block is not None:
                raise MarkerError(
                    f"gen block {start.group('name')!r} opened inside {open_block[0]!r} "
                    f"at line {line_of(text, m.start())}"
                )
            open_block = (start.group("name"), m.start(), m.end())
            continue
        if open_block is None:
            raise MarkerError(f"gen:end without gen:start at line {line_of(text, m.start())}")
        name, b_start, b_body = open_block
        blocks.append(
            GenBlock(name=name, start=b_start, end=m.end(), body_start=b_body, body_end=m.start())
        )
        open_block = None
    if open_block is not None:
        raise MarkerError(
            f"gen block {open_block[0]!r} at line {line_of(text, open_block[1])} is never closed"
        )
    return blocks


def fenced_code_spans(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in FENCED_CODE_RE.finditer(text)]


def html_comment_spans(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in HTML_COMMENT_RE.finditer(text)]


def allowlist_spans(text: str) -> list[tuple[int, int]]:
    """Spans exempt from the numeral rule: ``ALLOWLIST_SPANS`` patterns plus identifier tokens."""
    spans = [(m.start(), m.end()) for _, pat in ALLOWLIST_SPANS for m in pat.finditer(text)]
    for m in re.finditer(r"\S+", text):
        token = m.group(0)
        lead = len(token) - len(token.lstrip(_TOKEN_STRIP))
        core = token.strip(_TOKEN_STRIP)
        if core and any(ch.isdigit() for ch in core) and IDENT_TOKEN_RE.match(core):
            spans.append((m.start() + lead, m.start() + lead + len(core)))
    return spans


def mask_spans(text: str, spans: list[tuple[int, int]]) -> str:
    """Blank the spans with spaces, keeping every newline so offsets and line numbers survive."""
    chars = list(text)
    for start, end in spans:
        for i in range(max(start, 0), min(end, len(chars))):
            if chars[i] != "\n":
                chars[i] = " "
    return "".join(chars)


def find_numerals(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(0), m.start(), m.end()) for m in NUMERAL_RE.finditer(text)]


def line_of(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def line_text(text: str, pos: int) -> str:
    start = text.rfind("\n", 0, pos) + 1
    end = text.find("\n", pos)
    return text[start : end if end != -1 else len(text)]


def sections(text: str) -> list[Section]:
    """Split by level-2 headings; the preamble (title, one-liner, status) has the title ""."""
    out: list[Section] = []
    prev_start = 0
    prev_title = ""
    for m in HEADING2_RE.finditer(text):
        out.append(Section(title=prev_title, start=prev_start, end=m.start()))
        prev_start, prev_title = m.start(), m.group("title")
    out.append(Section(title=prev_title, start=prev_start, end=len(text)))
    return out


def markdown_tables(text: str, start: int = 0, end: int | None = None) -> list[Table]:
    """Pipe tables between ``start`` and ``end`` (absolute offsets); cells keep their raw text."""
    end = len(text) if end is None else end
    tables: list[Table] = []
    current: list[tuple[int, int, str]] = []  # (line_start, line_end, line)

    def flush() -> None:
        if len(current) >= 2 and TABLE_SEP_RE.match(current[1][2]):
            header = _split_cells(current[0][2])
            rows = [
                TableRow(line=line_of(text, ls), start=ls, end=le, cells=_split_cells(line))
                for ls, le, line in current[2:]
            ]
            tables.append(Table(header=header, rows=rows))
        current.clear()

    pos = start
    while pos < end:
        nl = text.find("\n", pos, end)
        line_end = end if nl == -1 else nl
        line = text[pos:line_end]
        if TABLE_LINE_RE.match(line):
            current.append((pos, line_end, line))
        else:
            flush()
        pos = line_end + 1
    flush()
    return tables


def _split_cells(line: str) -> list[str]:
    body = line.strip()
    if body.startswith("|"):
        body = body[1:]
    if body.endswith("|"):
        body = body[:-1]
    return [c.strip() for c in body.split("|")]


def cell_index(line: str, offset_in_line: int) -> int:
    """Column of a marker at ``offset_in_line`` (``|`` count before it, minus the leading one)."""
    before = line[:offset_in_line]
    count = before.count("|")
    return count - 1 if before.lstrip().startswith("|") else count


__all__ = [
    "ALLOWLIST_SPANS",
    "GEN_END_RE",
    "GEN_START_RE",
    "GenBlock",
    "IDENT_TOKEN_RE",
    "KEY_RE",
    "MarkerError",
    "NUMERAL_RE",
    "NUM_RE",
    "NumMarker",
    "Section",
    "Table",
    "TableRow",
    "allowlist_spans",
    "cell_index",
    "fenced_code_spans",
    "find_gen_blocks",
    "find_num_markers",
    "find_numerals",
    "format_value",
    "gen_block",
    "html_comment_spans",
    "line_of",
    "line_text",
    "markdown_tables",
    "mask_spans",
    "num_marker",
    "parse_numeral",
    "sections",
    "value_matches",
]
