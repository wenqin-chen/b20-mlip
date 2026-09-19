"""Comment-preserving writer for ``configs/cluster/<alias>.yaml``.

The overlay is a single ``cluster:`` mapping whose keys mirror ``ClusterConfig``. Bootstrap
rewrites only the values: the header comment, per-key trailing comments and any other
top-level section survive; unknown (undiscovered or ambiguous) values are written as ``null``
with the candidates in the trailing comment, never invented. A provenance line
(``# last discovered: ...``) is replaced on every write.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from b20mlip.config import Settings, read_yaml

CLUSTER_KEYS: tuple[str, ...] = (
    "alias",
    "control_path",
    "account",
    "partition_cpu",
    "partition_gpu",
    "qos",
    "scratch",
    "modules",
    "qe_cmd",
    "lammps_cmd",
    "micromamba_env",
)
PROVENANCE_PREFIX = "# last discovered:"
DEFAULT_HEADER = (
    "# Cluster overlay: `b20mlip --config configs/cluster/{alias}.yaml ...`\n"
    "# Discovered fields are written by `b20mlip cluster bootstrap` (SPEC.md section 8); "
    "null = unknown.\n"
)
_KEY_LINE = re.compile(r"^(?P<indent>[ \t]+)(?P<key>[A-Za-z_]\w*):(?P<rest>.*)$")
_KEY_INDENT = 2  # indentation of the cluster block's own keys
_VALUE_COLUMN = 26


def yaml_scalar(value: Any) -> str:
    """Flow-style YAML for one value (``null``, ``[a, b]``, quoted strings when needed)."""
    text = yaml.safe_dump(value, default_flow_style=True, width=1_000_000, allow_unicode=True)
    lines = [ln for ln in text.strip().splitlines() if ln.strip() != "..."]
    return " ".join(ln.strip() for ln in lines) or "null"


def _split_comment(rest: str) -> tuple[str, str]:
    """``rest`` of ``key:<rest>`` -> ``(value text, trailing comment without '#')``."""
    m = re.search(r"\s#(.*)$", rest)
    if not m:
        return rest.strip(), ""
    return rest[: m.start()].strip(), m.group(1).strip()


def parse_cluster_yaml(text: str) -> tuple[list[str], dict[str, str], dict[str, str], list[str]]:
    """``(header lines, {key: value text}, {key: comment}, trailing lines after the block)``."""
    header: list[str] = []
    values: dict[str, str] = {}
    comments: dict[str, str] = {}
    trailing: list[str] = []
    state = "header"
    current: str | None = None  # key whose nested block (deeper-indented lines) is being read
    for line in text.splitlines():
        if state == "header":
            if re.match(r"^cluster:\s*(#.*)?$", line):
                state = "block"
            else:
                header.append(line)
            continue
        if state == "block":
            if line.strip() == "":
                continue  # blank lines inside the block are not kept
            if not line[0].isspace():
                state = "trailing"
                trailing.append(line)
                continue
            m = _KEY_LINE.match(line)
            nested = current is not None and (len(line) - len(line.lstrip()) > _KEY_INDENT)
            if nested or (current is not None and line.lstrip().startswith("#")):
                # continuation of a nested mapping/list (e.g. ``resources:``): kept verbatim
                values[current] = values[current] + "\n" + line
                continue
            if line.lstrip().startswith("#"):
                current = None
                continue  # comment-only lines between keys are not kept
            if m and len(m.group("indent")) == _KEY_INDENT:
                value, comment = _split_comment(m.group("rest"))
                values[m.group("key")] = value
                if comment:
                    comments[m.group("key")] = comment
                current = m.group("key") if value == "" else None
            continue
        trailing.append(line)
    return header, values, comments, trailing


def write_cluster_yaml(
    path: str | Path,
    values: dict[str, Any],
    *,
    comments: dict[str, str] | None = None,
    provenance: str | None = None,
    now: datetime | None = None,
) -> Path:
    """Rewrite the ``cluster:`` block of ``path`` with ``values`` (keys in CLUSTER_KEYS order).

    Keys missing from ``values`` keep their current text; ``comments`` override the trailing
    comment of a key (used to list candidates next to a ``null``). The result is validated by
    loading it as a ``Settings`` overlay before it replaces the file.
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8") if p.is_file() else ""
    header, old_values, old_comments, trailing = parse_cluster_yaml(text)
    if not text:
        header = DEFAULT_HEADER.format(alias=values.get("alias", "cluster")).splitlines()
    header = [ln for ln in header if not ln.startswith(PROVENANCE_PREFIX)]
    stamp = (now or datetime.now(UTC)).astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    header.append(f"{PROVENANCE_PREFIX} {stamp}{(' ' + provenance) if provenance else ''}")

    merged_comments = dict(old_comments)
    merged_comments.update({k: v for k, v in (comments or {}).items() if v})
    keys = list(CLUSTER_KEYS) + [k for k in old_values if k not in CLUSTER_KEYS]
    keys += [k for k in values if k not in keys]
    lines = [*header, "cluster:"]
    for key in keys:
        if key in values:
            value_text = yaml_scalar(values[key])
        elif key in old_values:
            value_text = old_values[key]
        else:
            continue
        if "\n" in value_text:  # a nested block kept verbatim (its own comments included)
            comment = merged_comments.get(key, "")
            head = f"  {key}:" + (f"  # {comment}" if comment else "")
            lines.append(head + value_text)
            continue
        line = f"  {key}: {value_text}"
        comment = merged_comments.get(key, "")
        if comment:
            line = f"{line.ljust(_VALUE_COLUMN)} # {comment}"
        lines.append(line)
    if trailing:
        lines.append("")
        lines.extend(trailing)
    out = "\n".join(lines).rstrip("\n") + "\n"

    tmp = p.with_suffix(p.suffix + ".tmp")
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(out, encoding="utf-8")
    try:
        Settings.model_validate(read_yaml(tmp))  # raises on a broken overlay
    except Exception:
        tmp.unlink(missing_ok=True)  # the existing file is left untouched
        raise
    tmp.replace(p)
    return p


__all__ = [
    "CLUSTER_KEYS",
    "PROVENANCE_PREFIX",
    "parse_cluster_yaml",
    "write_cluster_yaml",
    "yaml_scalar",
]
