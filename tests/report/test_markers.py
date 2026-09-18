"""Marker grammar, numeral scanner, allowlist, sections and table parsing."""

from __future__ import annotations

from decimal import Decimal

import pytest

from b20mlip.report.markers import (
    MarkerError,
    allowlist_spans,
    cell_index,
    fenced_code_spans,
    find_gen_blocks,
    find_num_markers,
    find_numerals,
    format_value,
    gen_block,
    line_of,
    line_text,
    markdown_tables,
    mask_spans,
    num_marker,
    parse_numeral,
    sections,
    value_matches,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (3.0, "3"),
        (-2.0, "-2"),
        (12.345, "12.35"),
        (0.000012, "0.000012"),
        (123456.7, "123500"),
        (4.558, "4.558"),
        (1.0, "1"),
    ],
)
def test_format_value(value: float, expected: str) -> None:
    assert format_value(value) == expected


def test_format_value_custom_and_nonfinite() -> None:
    assert format_value(0.12345, "{:.2f}") == "0.12"
    with pytest.raises(ValueError, match="non-finite"):
        format_value(float("nan"))


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1,000", Decimal("1000")),
        ("−3.5", Decimal("-3.5")),
        ("1e-3", Decimal("0.001")),
        ("12 %", Decimal("12")),
        ("pending", None),
        ("", None),
    ],
)
def test_parse_numeral(text: str, expected: Decimal | None) -> None:
    assert parse_numeral(text) == expected


@pytest.mark.parametrize(
    ("displayed", "value", "ok"),
    [
        ("12.3", 12.34, True),
        ("12.3", 12.4, False),
        ("1,234", 1234.0, True),
        ("123500", 123456.7, True),
        ("0.012", 0.0123, True),
        ("100", 101.0, False),
        ("pending", 1.0, False),
    ],
)
def test_value_matches(displayed: str, value: float, ok: bool) -> None:
    assert value_matches(displayed, value) is ok


def test_marker_and_block_builders() -> None:
    assert num_marker("a.b", "1.5") == "<!-- num:a.b -->1.5<!-- /num -->"
    assert gen_block("t1", "x") == "<!-- gen:start:t1 -->x<!-- gen:end -->"
    with pytest.raises(ValueError, match="invalid number key"):
        num_marker("bad key", "1")
    with pytest.raises(ValueError, match="'<'"):
        num_marker("k", "<b>1</b>")
    with pytest.raises(ValueError, match="invalid gen block name"):
        gen_block("Bad Name", "x")


def test_find_markers_and_blocks() -> None:
    text = (
        "a <!-- num:k.1 -->1.5<!-- /num --> b\n"
        "<!-- gen:start:one -->inline<!-- gen:end -->\n"
        "<!-- gen:start:two -->\nmulti\n<!-- gen:end -->\n"
    )
    markers = find_num_markers(text)
    assert [(m.key, m.text) for m in markers] == [("k.1", "1.5")]
    blocks = find_gen_blocks(text)
    assert [b.name for b in blocks] == ["one", "two"]
    assert text[blocks[0].body_start : blocks[0].body_end] == "inline"
    assert text[blocks[1].body_start : blocks[1].body_end] == "\nmulti\n"
    assert find_gen_blocks("no blocks") == []


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("<!-- gen:start:a --><!-- gen:start:b --><!-- gen:end -->", "opened inside"),
        ("<!-- gen:end -->", "without gen:start"),
        ("<!-- gen:start:a -->never closed", "never closed"),
    ],
)
def test_malformed_blocks(text: str, match: str) -> None:
    with pytest.raises(MarkerError, match=match):
        find_gen_blocks(text)


def test_allowlist_and_numerals() -> None:
    text = "MACE-MPA-0 mp-871 T0–T4b 8-atom v0.1 Python 3.11 CC-BY-4.0 2026-09-18 1e-3 x"
    masked = mask_spans(text, allowlist_spans(text))
    assert [n for n, _, _ in find_numerals(masked)] == ["8", "1e-3"]
    prose = (
        "see https://x.org/v2/1 and [link](docs/3.md) ![badge](b/95.svg) mace-torch==0.3.16 0.3.16"
    )
    masked = mask_spans(prose, allowlist_spans(prose))
    assert find_numerals(masked) == []
    assert [n for n, _, _ in find_numerals("MAE=12.3, -3 and 1,000 and 12.5e3")] == [
        "12.3",
        "-3",
        "1,000",
        "12.5e3",
    ]
    assert find_numerals("sha256 E0s_qe.json ΔF1 P2₁3 cm⁻¹") == []


def test_mask_keeps_newlines_and_line_helpers() -> None:
    text = "ab\ncd\nef"
    masked = mask_spans(text, [(1, 5)])
    assert masked == "a \n  \nef" and masked.count("\n") == 2
    assert line_of(text, 4) == 2 and line_text(text, 4) == "cd" and line_text(text, 7) == "ef"
    assert fenced_code_spans("x\n```bash\n1 2\n```\ny") == [(2, 17)]


def test_sections() -> None:
    text = "# T\npre\n## Results\nr\n### sub\n## Plan {#x} \np\n"
    secs = sections(text)
    assert [s.title for s in secs] == ["", "Results", "Plan {#x}"]
    assert secs[0].normalized == "" and secs[2].normalized.startswith("plan")
    assert text[secs[1].start : secs[1].end] == "## Results\nr\n### sub\n"


def test_markdown_tables_and_cell_index() -> None:
    text = (
        "intro\n| A | B |\n| --- | --- |\n| r1 | <!-- num:k -->1<!-- /num --> |\n| r2 | x |\n"
        "after\n"
    )
    tables = markdown_tables(text)
    assert len(tables) == 1 and tables[0].header == ["A", "B"]
    assert [r.cells for r in tables[0].rows] == [
        ["r1", "<!-- num:k -->1<!-- /num -->"],
        ["r2", "x"],
    ]
    row = tables[0].rows[0]
    line = text[row.start : row.end]
    assert row.line == 4 and cell_index(line, line.index("<!--")) == 1
    assert cell_index("a | b | c", 5) == 1
    assert markdown_tables("| only header |\n| x |\n") == []  # no separator row -> not a table
