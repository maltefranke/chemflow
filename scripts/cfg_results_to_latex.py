"""Compile conditional-generation eval results from a JSON file into two LaTeX tables.

Usage:
    python scripts/cfg_results_to_latex.py <results.json> [--out-prefix PREFIX] [--use-all]

Produces two tables:
  * Property targeting metrics (validity, MAE, within-tolerance rates, ...).
  * Model quality metrics (energy, novelty, uniqueness, semla stability, KL divergences).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def _fmt(x: Any, digits: int = 3) -> str:
    if x is None:
        return "--"
    if isinstance(x, (int,)) and not isinstance(x, bool):
        return f"{x:d}"
    try:
        return f"{float(x):.{digits}f}"
    except (TypeError, ValueError):
        return str(x)


# Each property: target_keys (conditioning), name_aliases (used for stat lookup),
# target_header, target_digits, mean_header, second_header, second_kind.
_PROPERTIES: dict[str, dict] = {
    "logp": {
        "target_keys": ["target_logp"],
        "aliases": ["logp"],
        "target_header": r"$\log P^{*}$",
        "target_digits": 1,
        "mean_header": r"$\log P$ mean",
        "second_header": r"$\log P$ MAE",
        "second_kind": "mae",
    },
    "qed": {
        "target_keys": ["target_qed"],
        "aliases": ["qed"],
        "target_header": r"QED$^{*}$",
        "target_digits": 2,
        "mean_header": "QED mean",
        "second_header": "QED MAE",
        "second_kind": "mae",
    },
    "natoms": {
        "target_keys": ["target_natoms", "target_n_atoms"],
        "aliases": ["natoms", "n_atoms"],
        "target_header": r"$N^{*}$",
        "target_digits": 0,
        "mean_header": r"$N$ mean",
        "second_header": "Exact match",
        "second_kind": "exact_match_rate",
    },
}


def _stat(s: dict, prop: str, kind: str, allow_bare: bool = False) -> Any:
    """Look up a stat across the property's name aliases and `<prop>_<kind>` /
    `<kind>_<prop>` schemas. `allow_bare` enables single-property `<kind>` keys."""
    candidates: list[str] = []
    for alias in _PROPERTIES[prop]["aliases"]:
        candidates += [f"{alias}_{kind}", f"{kind}_{alias}"]
    if allow_bare:
        candidates.append(kind)
    for k in candidates:
        if k in s:
            return s[k]
    return None


def _entry_target(entry: dict, prop: str) -> Any:
    for k in _PROPERTIES[prop]["target_keys"]:
        if k in entry:
            return entry[k]
    return None


def _properties_present(entries: list[dict], stats_key: str) -> list[str]:
    props: list[str] = []
    for prop, spec in _PROPERTIES.items():
        if any(_entry_target(e, prop) is not None for e in entries):
            props.append(prop)
            continue
        if any(
            _stat(e.get(stats_key, {}), prop, spec["second_kind"]) is not None
            for e in entries
        ):
            props.append(prop)
    return props


def _conditioning_columns(entry: dict) -> list[tuple[str, str]]:
    """Return the conditioning columns present in this entry, as (header, value)."""
    cols: list[tuple[str, str]] = []
    for prop, spec in _PROPERTIES.items():
        v = _entry_target(entry, prop)
        if v is not None:
            cols.append((spec["target_header"], _fmt(v, spec["target_digits"])))
    return cols


def build_property_table(entries: list[dict], use_all: bool = False) -> str:
    stats_key = "all_stats" if use_all else "valid_stats"

    cond_headers = [h for h, _ in _conditioning_columns(entries[0])]
    props = _properties_present(entries, stats_key)

    headers: list[str] = list(cond_headers)
    val_key, val_header = _validity_metric(entries)
    headers.append(val_header)
    for prop in props:
        spec = _PROPERTIES[prop]
        headers += [spec["mean_header"], spec["second_header"]]

    allow_bare = len(props) == 1
    rows: list[list[str]] = []
    for e in entries:
        s = e.get(stats_key, {})
        row = [v for _, v in _conditioning_columns(e)]
        row.append(_fmt(e.get(val_key)))
        for prop in props:
            spec = _PROPERTIES[prop]
            row += [
                rf"{_fmt(_stat(s, prop, 'mean', allow_bare))} $\pm$ {_fmt(_stat(s, prop, 'std', allow_bare))}",
                _fmt(_stat(s, prop, spec["second_kind"], allow_bare)),
            ]
        rows.append(row)

    caption = (
        "Property targeting evaluation"
        + (" (over all generated samples)." if use_all else " (over valid samples).")
    )
    return _render_table(headers, rows, caption=caption, label="tab:cfg_property")


def _validity_metric(entries: list[dict]) -> tuple[str, str]:
    if any("metric/semla-validity" in e for e in entries):
        return "metric/semla-validity", "Semla val."
    return "metric/validity", "Validity"


def build_model_table(entries: list[dict]) -> str:
    val_key, val_header = _validity_metric(entries)
    columns: list[tuple[str, str]] = [
        (val_key, val_header),
        ("metric/uniqueness", "Uniq."),
        ("metric/novelty", "Novelty"),
        ("metric/semla-atom-stability", "Atom stab."),
        ("metric/semla-molecule-stability", "Mol. stab."),
    ]
    # Drop columns that aren't present in any row.
    columns = [(k, h) for k, h in columns if any(k in e for e in entries)]

    cond_headers = [h for h, _ in _conditioning_columns(entries[0])]
    headers = cond_headers + [h for _, h in columns]

    rows = []
    for e in entries:
        row = [v for _, v in _conditioning_columns(e)]
        for k, _ in columns:
            v = e.get(k)
            digits = 2 if "energy" in k or "strain" in k else 3
            row.append(_fmt(v, digits))
        rows.append(row)

    return _render_table(
        headers,
        rows,
        caption="Generative-model quality metrics.",
        label="tab:cfg_model",
    )


def _render_table(
    headers: list[str],
    rows: Iterable[list[str]],
    caption: str,
    label: str,
) -> str:
    n = len(headers)
    col_spec = "l" * n
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(row) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_json", type=Path, help="Path to the CFG results JSON.")
    ap.add_argument(
        "--out-prefix",
        type=Path,
        default=None,
        help="If given, write tables to <prefix>_property.tex and <prefix>_model.tex.",
    )
    ap.add_argument(
        "--use-all",
        action="store_true",
        help="Use all_stats instead of valid_stats for the property table.",
    )
    args = ap.parse_args()

    with args.results_json.open() as f:
        entries = json.load(f)
    if not isinstance(entries, list) or not entries:
        raise SystemExit(f"Expected a non-empty list in {args.results_json}.")

    property_tex = build_property_table(entries, use_all=args.use_all)
    model_tex = build_model_table(entries)

    if args.out_prefix is None:
        print("% ===== Property targeting table =====")
        print(property_tex)
        print()
        print("% ===== Model metrics table =====")
        print(model_tex)
    else:
        prop_path = args.out_prefix.with_name(args.out_prefix.name + "_property.tex")
        model_path = args.out_prefix.with_name(args.out_prefix.name + "_model.tex")
        prop_path.parent.mkdir(parents=True, exist_ok=True)
        prop_path.write_text(property_tex + "\n")
        model_path.write_text(model_tex + "\n")
        print(f"Wrote {prop_path}")
        print(f"Wrote {model_path}")


if __name__ == "__main__":
    main()
