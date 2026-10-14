"""Structured filters over the cards table -> allowed row indices / oracle ids."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Literal

import numpy as np
import polars as pl

COLOR_ORDER = "WUBRG"

PIP_COLORS = "WUBRGC"

PLAYABLE = ("legal", "restricted")

ColorsMode = Literal["includes", "subset", "exact"]
COLORS_MODES = ("includes", "subset", "exact")

# Single source of truth for the filter keyword arguments of ``filter_expr`` /
# ``allowed_indices``; the MCP server reuses this for tool schemas and docstrings.
FILTER_PARAMS: dict[str, dict[str, Any]] = {
    "color_identity": {
        "type": "string",
        "description": "Commander colour identity: card identity must be within these colours, "
        "e.g. 'UB'; 'C' = colourless only.",
    },
    "colors": {
        "type": "string",
        "description": "Card colours (from its mana cost / colour indicator), e.g. 'RW'; 'C' = colourless. "
        "Interpreted by colors_mode.",
    },
    "colors_mode": {
        "type": "string",
        "enum": list(COLORS_MODES),
        "default": "includes",
        "description": "How `colors` matches: 'includes' = card has all given colours (may have more); "
        "'subset' = card colours are within the given ones (colourless cards pass), e.g. Boros deck = "
        "colors='RW', colors_mode='subset'; 'exact' = card colours equal the given ones.",
    },
    "format": {
        "type": "string",
        "description": "Format the card must be legal (or restricted) in, e.g. 'commander', 'modern', 'pauper'.",
    },
    "types": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Every entry must match a supertype, card type or subtype, e.g. ['legendary', 'creature', 'elf'].",
    },
    "cmc_min": {"type": "number", "description": "Minimum mana value."},
    "cmc_max": {"type": "number", "description": "Maximum mana value."},
    "max_pips": {
        "type": "object",
        "additionalProperties": {"type": "integer"},
        "description": "Upper bound on mana symbols per colour (W,U,B,R,G,C) in the cost of every face, "
        "e.g. {'U': 1} = no double-blue. Hybrid and Phyrexian symbols count for each of their colours.",
    },
    "min_pips": {
        "type": "object",
        "additionalProperties": {"type": "integer"},
        "description": "Lower bound on mana symbols per colour in the cost of some face, e.g. {'R': 2} "
        "(devotion-style counting, hybrid counts for both colours).",
    },
    "has_x": {
        "type": "boolean",
        "description": "true = only cards with {X} (or Y/Z) in a cost; false = exclude them.",
    },
    "has_hybrid": {
        "type": "boolean",
        "description": "true = only cards with hybrid symbols ({G/W}, {2/W}); false = exclude them.",
    },
    "has_phyrexian": {
        "type": "boolean",
        "description": "true = only cards with Phyrexian symbols ({U/P}); false = exclude them.",
    },
    "include_funny": {
        "type": "boolean",
        "default": False,
        "description": "Include Un-set / playtest cards that are not tournament legal.",
    },
}

# Keys that never restrict on their own (see ``has_filters``).
_MODIFIERS = frozenset({"include_funny", "colors_mode"})


def parse_colors(spec: str | Iterable[str] | None) -> set[str] | None:
    """'UB' / 'ub' / ['U','B'] -> {'U','B'}; 'C' or '' -> set() (colorless)."""
    if spec is None:
        return None
    letters = spec if isinstance(spec, str) else "".join(spec)
    letters = letters.upper().replace(",", "").replace(" ", "")
    out = {c for c in letters if c in COLOR_ORDER}
    bad = {c for c in letters if c not in COLOR_ORDER and c != "C"}
    if bad:
        raise ValueError(f"unknown color letters: {sorted(bad)}")
    return out


def parse_pips(spec: Mapping[str, int] | None) -> dict[str, int]:
    """{'u': 1} -> {'U': 1}; validates colour keys (W,U,B,R,G,C) and non-negative ints."""
    out: dict[str, int] = {}
    for k, v in (spec or {}).items():
        key = str(k).strip().upper()
        if key not in PIP_COLORS or len(key) != 1:
            raise ValueError(f"unknown pip colour {k!r}; expected one of {list(PIP_COLORS)}")
        n = int(v)
        if n < 0:
            raise ValueError(f"pip count for {key} must be >= 0, got {v!r}")
        out[key] = n
    return out


def filter_expr(
    df: pl.DataFrame,
    *,
    color_identity: str | Iterable[str] | None = None,
    colors: str | Iterable[str] | None = None,
    colors_mode: ColorsMode = "includes",
    format: str | None = None,
    types: Iterable[str] | None = None,
    cmc_min: float | None = None,
    cmc_max: float | None = None,
    max_pips: Mapping[str, int] | None = None,
    min_pips: Mapping[str, int] | None = None,
    has_x: bool | None = None,
    has_hybrid: bool | None = None,
    has_phyrexian: bool | None = None,
    include_funny: bool = False,
) -> pl.Expr:
    """Build a boolean polars expression for the given filters.

    - ``color_identity``: card identity must be a *subset* of the given colors
      (commander deck-building semantics). ``"C"`` means colorless only.
    - ``colors`` + ``colors_mode``: ``"includes"`` (default) = card colors include all
      given colors; ``"subset"`` = card colors are within the given ones (colorless
      passes); ``"exact"`` = equal. ``"C"`` means colorless.
    - ``format``: card must be ``legal`` or ``restricted`` in that format.
    - ``types``: every entry must match a supertype, card type or subtype (case-insensitive).
    - ``max_pips`` / ``min_pips``: per-colour bounds on mana symbols ({'U': 1}); the card's
      count is the max over its faces, so ``max_pips`` bounds every face and ``min_pips``
      needs one face. Hybrid / Phyrexian symbols count for each colour they contain.
    - ``has_x`` / ``has_hybrid`` / ``has_phyrexian``: ``True`` requires such a symbol in
      some face's cost, ``False`` excludes those cards, ``None`` ignores it.

    See ``FILTER_PARAMS`` for a machine-readable description.
    """
    expr = pl.lit(True)

    ci = parse_colors(color_identity)
    if ci is not None:
        disallowed = [c for c in COLOR_ORDER if c not in ci]
        if disallowed:
            expr &= ~pl.col("color_identity").list.eval(pl.element().is_in(disallowed)).list.any()

    if colors_mode not in COLORS_MODES:
        raise ValueError(f"colors_mode must be one of {COLORS_MODES}, got {colors_mode!r}")
    col = parse_colors(colors)
    if col is not None:
        if colors_mode in ("subset", "exact"):
            others = [c for c in COLOR_ORDER if c not in col]
            if others:
                expr &= ~pl.col("colors").list.eval(pl.element().is_in(others)).list.any()
        if colors_mode in ("includes", "exact"):
            for c in col:
                expr &= pl.col("colors").list.contains(c)
            if not col:
                expr &= pl.col("colors").list.len() == 0

    if format:
        fmt = format.lower()
        fields = {f.name for f in df.schema["legalities"].fields}
        if fmt not in fields:
            raise ValueError(f"unknown format {format!r}; known: {sorted(fields)}")
        expr &= pl.col("legalities").struct.field(fmt).is_in(PLAYABLE)

    if types:
        all_types = pl.concat_list(
            [c for c in ("supertypes", "types", "subtypes") if c in df.columns]
        ).list.eval(pl.element().str.to_lowercase())
        for t in types:
            expr &= all_types.list.contains(t.strip().lower())

    if cmc_min is not None:
        expr &= pl.col("cmc") >= cmc_min
    if cmc_max is not None:
        expr &= pl.col("cmc") <= cmc_max

    for c, n in parse_pips(max_pips).items():
        expr &= pl.col(f"pips_{c}") <= n
    for c, n in parse_pips(min_pips).items():
        expr &= pl.col(f"pips_{c}") >= n
    for flag, want in (("has_x", has_x), ("has_hybrid", has_hybrid), ("has_phyrexian", has_phyrexian)):
        if want is not None:
            expr &= pl.col(flag) if want else ~pl.col(flag)

    if not include_funny:
        expr &= ~pl.col("is_funny")

    return expr


def allowed_indices(df: pl.DataFrame, **filters) -> np.ndarray:
    """Row indices of ``df`` passing the filters (sorted int64 array)."""
    idx = df.with_row_index("_row").filter(filter_expr(df, **filters))["_row"]
    return idx.to_numpy().astype(np.int64)


def allowed_oracle_ids(df: pl.DataFrame, **filters) -> set[str]:
    return set(df.filter(filter_expr(df, **filters))["oracle_id"].to_list())


def has_filters(**filters) -> bool:
    """True if any filter is set to a value that actually restricts the card set."""
    return any(v is not None and v != {} and k not in _MODIFIERS for k, v in filters.items())
