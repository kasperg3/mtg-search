"""Structured card filters -> allowed row indices. Used by the server's tools and by the evaluations' labels."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import numpy as np
import polars as pl

COLOR_ORDER = "WUBRG"
PLAYABLE = ("legal", "restricted")
COLORS_MODES = ("includes", "subset", "exact")


def parse_colors(spec: str | Iterable[str] | None) -> set[str] | None:
    """'UB' / 'ub' / ['U','B'] -> {'U','B'}; 'C' or '' -> set() (colorless)."""
    if spec is None:
        return None
    letters = spec if isinstance(spec, str) else "".join(spec)
    letters = letters.upper().replace(",", "").replace(" ", "")
    bad = {c for c in letters if c not in COLOR_ORDER and c != "C"}
    if bad:
        raise ValueError(f"unknown colour letters {sorted(bad)}: use W, U, B, R, G (or C for colourless)")
    return {c for c in letters if c in COLOR_ORDER}


def filter_expr(
    df: pl.DataFrame,
    *,
    color_identity: str | Iterable[str] | None = None,
    colors: str | Iterable[str] | None = None,
    colors_mode: Literal["includes", "subset", "exact"] = "includes",
    format: str | None = None,
    types: Iterable[str] | None = None,
    cmc_min: float | None = None,
    cmc_max: float | None = None,
    include_funny: bool = False,
) -> pl.Expr:
    """Boolean polars expression for the filters.

    - ``color_identity``: card identity must be within the given colours (Commander rules); "C" = colourless only.
    - ``colors`` + ``colors_mode`` (evaluation labels only): "includes" = card has all given colours,
      "subset" = card colours within them, "exact" = equal.
    - ``format``: card must be legal or restricted in that format.
    - ``types``: every entry must match a supertype, card type or subtype (case-insensitive).
    - ``cmc_min`` / ``cmc_max``: mana value bounds.
    - Un-sets and playtest cards are excluded unless ``include_funny``.
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
            raise ValueError(f"unknown format {format!r}; known formats: {', '.join(sorted(fields))}")
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
    if not include_funny:
        expr &= ~pl.col("is_funny")
    return expr


def allowed_indices(df: pl.DataFrame, **filters) -> np.ndarray:
    """Row indices of ``df`` passing the filters (sorted int64 array)."""
    idx = df.with_row_index("_row").filter(filter_expr(df, **filters))["_row"]
    return idx.to_numpy().astype(np.int64)
