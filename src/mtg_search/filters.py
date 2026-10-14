"""Structured filters over the cards table -> allowed row indices / oracle ids."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import polars as pl

COLOR_ORDER = "WUBRG"

PLAYABLE = ("legal", "restricted")


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


def filter_expr(
    df: pl.DataFrame,
    *,
    color_identity: str | Iterable[str] | None = None,
    colors: str | Iterable[str] | None = None,
    format: str | None = None,
    types: Iterable[str] | None = None,
    cmc_min: float | None = None,
    cmc_max: float | None = None,
    include_funny: bool = False,
) -> pl.Expr:
    """Build a boolean polars expression for the given filters.

    - ``color_identity``: card identity must be a *subset* of the given colors
      (commander deck-building semantics). ``"C"`` means colorless only.
    - ``colors``: card colors must *include all* given colors; ``"C"`` means colorless.
    - ``format``: card must be ``legal`` or ``restricted`` in that format.
    - ``types``: every entry must match a supertype, card type or subtype (case-insensitive).
    """
    expr = pl.lit(True)

    ci = parse_colors(color_identity)
    if ci is not None:
        disallowed = [c for c in COLOR_ORDER if c not in ci]
        if disallowed:
            expr &= ~pl.col("color_identity").list.eval(pl.element().is_in(disallowed)).list.any()

    col = parse_colors(colors)
    if col is not None:
        if col:
            for c in col:
                expr &= pl.col("colors").list.contains(c)
        else:
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
    """True if any filter other than the default include_funny=False is active."""
    return any(v is not None and k != "include_funny" for k, v in filters.items())
