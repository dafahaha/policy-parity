#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""policy-parity — offline action-drift checker for quantized policy exports.

Compare an exported reference action log (FP32) against a quantized/onnx/INT8
action log, aligned row by row, WITHOUT importing torch/onnx/numpy. Both files
are CSVs the user already dumped; the tool reports per-dim MAE, max absolute
action deviation, cosine similarity and the fraction of steps that breach a
tolerance, then returns PASS / WATCH / FAIL.

Pure standard library. No network. Fail-closed on shape mismatch, missing
columns, or non-numeric cells.

Exit codes:
    0  PASS (within tolerance and cosine floor)
    2  WATCH (mild drift; inspect before deployment)
    3  FAIL (breached tolerance / cosine floor / shape mismatch)
    1  usage / IO error
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

DEFAULT_TOL = 0.05          # absolute max |delta| per step (action units)
DEFAULT_COS_MIN = 0.99      # flattened-vector cosine floor

# Columns treated as a row index and excluded when --index-column is omitted.
INDEX_NAMES = {"step", "index", "timestep", "t", "time", "frame", "episode"}


class ShapeMismatchError(ValueError):
    """Rows/labels/dimensions differ between files; treated as FAIL (exit 3),
    not a usage error."""


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_action_matrix(
    path: Path, index_column: str | None
) -> tuple[list[str], list[list[float]], list[str]]:
    """Return (row_labels, matrix, action_column_names). With an index column,
    labels = that column; otherwise labels are positional indices as strings."""
    with path.open("r", newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"{path.name}: empty CSV")
        fields = list(reader.fieldnames)
        # Duplicate header names are silently merged by DictReader (it keeps the
        # last value), which would create a phantom double-weighted dimension.
        if len(fields) != len(set(fields)):
            dup = sorted(f for f, c in Counter(fields).items() if c > 1)
            raise ValueError(f"{path.name}: duplicate header column(s): {dup}")
        if index_column:
            if index_column not in fields:
                raise ValueError(f"{path.name}: index column {index_column!r} not in {fields}")
            value_cols = [f for f in fields if f != index_column]
        else:
            non_index = [f for f in fields if f.lower() not in INDEX_NAMES]
            if non_index:
                value_cols = non_index
            else:
                # Only index-like columns (e.g. a bare `step` file): there is no
                # action data; do not treat row numbers as actions.
                raise ValueError(f"{path.name}: no action columns (only index-like headers)")
        if not value_cols:
            raise ValueError(f"{path.name}: no action columns")
        labels: list[str] = []
        matrix: list[list[float]] = []
        for i, row in enumerate(reader, start=2):  # row 1 is header
            if not any(str(c).strip() for c in row.values()):
                continue
            # Extra fields beyond the header (a row wider than the header) are
            # ragged input; reject instead of silently dropping them.
            extra = row.get(None)
            if extra is not None and any(str(c).strip() for c in extra):
                raise ShapeMismatchError(
                    f"{path.name} row {i}: {len(extra)} extra field(s) beyond the header (ragged row)"
                )
            if index_column:
                label_raw = row[index_column]
                label_text = "" if label_raw is None else str(label_raw).strip()
                if label_text == "":
                    raise ShapeMismatchError(
                        f"{path.name} row {i}: blank index label in {index_column!r}"
                    )
                labels.append(label_text)
            else:
                labels.append(str(i - 2))
            vec: list[float] = []
            for col in value_cols:
                raw = row.get(col)
                text = "" if raw is None else str(raw).strip()
                if text == "":
                    raise ShapeMismatchError(f"{path.name} row {i}: missing/blank cell in {col!r} (ragged row)")
                try:
                    v = float(text)
                except ValueError as exc:
                    raise ValueError(f"{path.name} row {i}: non-numeric {text!r} in {col!r}") from exc
                if not math.isfinite(v):
                    raise ValueError(
                        f"{path.name} row {i}: non-finite value {text!r} (NaN/Inf) in {col!r}; "
                        "clean the log before comparing"
                    )
                vec.append(v)
            matrix.append(vec)
    if not matrix:
        raise ValueError(f"{path.name}: no data rows")
    dims = {len(r) for r in matrix}
    if len(dims) != 1:
        raise ShapeMismatchError(f"{path.name}: ragged rows (dimension count varies: {sorted(dims)})")
    return labels, matrix, value_cols


def align(ref_labels: list[str], ref: list[list[float]],
          q_labels: list[str], q: list[list[float]],
          labeled: bool = False,
          ref_cols: list[str] | None = None,
          q_cols: list[str] | None = None) -> list[tuple[list[float], list[float]]]:
    """Align two matrices by index label when a real index column was given
    (labeled=True); otherwise positional. Fail closed on length/label mismatch
    and on action-column name-set mismatch (same count but reordered/renamed
    streams would otherwise be compared positionally and could hide drift)."""
    # Ordered comparison: catches both renamed columns (set differs) and a
    # permutation (same set, different order) — a reorder would otherwise be
    # paired positionally and silently compare the wrong streams.
    if ref_cols is not None and q_cols is not None and ref_cols != q_cols:
        missing = sorted(set(ref_cols) - set(q_cols))[:5]
        extra = sorted(set(q_cols) - set(ref_cols))[:5]
        reordered = (set(ref_cols) == set(q_cols))
        if reordered:
            raise ShapeMismatchError(
                f"action column order differs; reference: {ref_cols}; quantized: {q_cols}"
            )
        raise ShapeMismatchError(
            f"action columns differ; missing in quantized: {missing}; extra: {extra}"
        )
    if labeled:
        ref_counts = Counter(ref_labels)
        ref_dupes = sorted(l for l, c in ref_counts.items() if c > 1)[:5]
        if ref_dupes:
            raise ShapeMismatchError(f"reference CSV has duplicate index label(s): {ref_dupes}")
        q_counts = Counter(q_labels)
        q_dupes = sorted(l for l, c in q_counts.items() if c > 1)[:5]
        if q_dupes:
            raise ShapeMismatchError(f"quantized CSV has duplicate index label(s): {q_dupes}")
        rset, qset = set(ref_labels), set(q_labels)
        if rset != qset:
            missing = sorted(rset - qset)[:5]
            extra = sorted(qset - rset)[:5]
            raise ShapeMismatchError(
                f"index labels differ; missing in quantized: {missing}; extra: {extra}"
            )
    if len(ref) != len(q):
        raise ShapeMismatchError(f"row count differs: reference {len(ref)} vs quantized {len(q)}")
    if len(ref[0]) != len(q[0]):
        raise ShapeMismatchError(
            f"action dimension differs: reference {len(ref[0])} vs quantized {len(q[0])}"
        )
    # Label-aligned map when a real index column was given, else positional zip
    if labeled:
        q_map = dict(zip(q_labels, q))
        return [(r, q_map[l]) for r, l in zip(ref, ref_labels)]
    return list(zip(ref, q))


# --------------------------------------------------------------------------- #
# Metrics (pure)
# --------------------------------------------------------------------------- #

def per_dim_mae(pairs: list[tuple[list[float], list[float]]]) -> list[float]:
    ndim = len(pairs[0][0])
    out: list[float] = []
    for d in range(ndim):
        out.append(sum(abs(a[d] - b[d]) for a, b in pairs) / len(pairs))
    return out


def max_abs_deviation(pairs: list[tuple[list[float], list[float]]]) -> tuple[float, str]:
    worst = 0.0
    where = "(all deltas zero)"
    for step, (a, b) in enumerate(pairs):
        for d, (x, y) in enumerate(zip(a, b)):
            delta = abs(x - y)
            if delta > worst:
                worst = delta
                where = f"step {step}, dim {d}"
    return worst, where


def flattened_norms(pairs: list[tuple[list[float], list[float]]]) -> tuple[float, float]:
    """Squared norms (na, nb) of the two flattened vectors."""
    na = nb = 0.0
    for a, b in pairs:
        for x, y in zip(a, b):
            na += x * x
            nb += y * y
    return na, nb


def flattened_cosine(pairs: list[tuple[list[float], list[float]]]) -> float | None:
    """Cosine over the flattened vectors; None when either norm is zero
    (cosine is undefined, not zero, in that case)."""
    dot = 0.0
    na, nb = flattened_norms(pairs)
    for a, b in pairs:
        for x, y in zip(a, b):
            dot += x * y
    if na == 0 or nb == 0:
        return None
    return dot / (math.sqrt(na) * math.sqrt(nb))


def step_violation_fraction(pairs: list[tuple[list[float], list[float]]], tol: float) -> float:
    bad = sum(1 for a, b in pairs if max(abs(x - y) for x, y in zip(a, b)) > tol)
    return bad / len(pairs)


def worst_steps(pairs: list[tuple[list[float], list[float]]], n: int = 5) -> list[dict]:
    """The n worst (step, dim, abs-delta) entries, descending."""
    entries: list[dict] = []
    for step, (a, b) in enumerate(pairs):
        for d, (x, y) in enumerate(zip(a, b)):
            entries.append({"step": step, "dim": d, "delta": abs(x - y)})
    entries.sort(key=lambda e: e["delta"], reverse=True)
    return entries[:n]


def parity_verdict(
    pairs: list[tuple[list[float], list[float]]],
    tol: float,
    cos_min: float,
) -> dict:
    """Three-level verdict.

    FAIL  : max|delta| > tol, or cosine < floor, or any step breaches tol.
    WATCH : no hard breach but max|delta| > tol*0.5 or cosine < (cos_min+1)/2
            (mild drift; inspect worst steps before deploying).
    PASS  : otherwise.
    """
    dim_mae = per_dim_mae(pairs)
    max_dev, where = max_abs_deviation(pairs)
    cos = flattened_cosine(pairs)
    na, nb = flattened_norms(pairs)
    viol = step_violation_fraction(pairs, tol)
    reasons: list[str] = []
    notes: list[str] = []
    if max_dev > tol:
        reasons.append(f"max |delta| {max_dev:.6f} exceeds tolerance {tol} ({where})")
    if cos is not None and cos < cos_min:
        reasons.append(f"cosine {cos:.6f} below floor {cos_min}")
    if cos is None:
        # Cosine is undefined on a zero vector; it must NOT silently pass.
        # Both vectors zero -> WATCH (actions agree at zero, direction unverified);
        # exactly one vector zero -> FAIL (asymmetric stream, parity unverifiable).
        if na == 0 and nb == 0:
            notes.append(
                "cosine undefined (both flattened vectors are zero); "
                "direction alignment could not be verified"
            )
        else:
            reasons.append(
                "cosine undefined: exactly one flattened vector is all zero; "
                "an all-zero reference or quantized stream makes parity "
                "unverifiable — check the dump before deploying"
            )

    if reasons:
        level = "FAIL"
    elif notes or max_dev > tol * 0.5 or (cos is not None and cos < (cos_min + 1) / 2):
        level = "WATCH"
    else:
        level = "PASS"
    return {
        "steps": len(pairs),
        "dims": len(pairs[0][0]),
        "per_dim_mae": dim_mae,
        "max_abs_deviation": max_dev,
        "max_at": where,
        "cosine": cos,
        "violation_fraction": viol,
        "tolerance": tol,
        "cosine_floor": cos_min,
        "worst_steps": worst_steps(pairs, 5),
        "verdict": level,
        "reasons": reasons,
        "notes": notes,
    }


# --------------------------------------------------------------------------- #
# Reporting + CLI
# --------------------------------------------------------------------------- #

def format_report(res: dict, ref_name: str, q_name: str) -> str:
    lines = ["# policy-parity report", ""]
    lines.append(f"- reference: {ref_name}")
    lines.append(f"- quantized: {q_name}")
    lines.append(f"- steps: {res['steps']}, action dims: {res['dims']}")
    lines.append(f"- per-dim MAE: {', '.join(f'{m:.6f}' for m in res['per_dim_mae'])}")
    lines.append(f"- max |delta|: {res['max_abs_deviation']:.6f} at {res['max_at']}")
    cos_txt = "cosine undefined (zero vector; not treated as 0)" if res["cosine"] is None else f"{res['cosine']:.6f}"
    lines.append(f"- flattened cosine: {cos_txt} (floor {res['cosine_floor']})")
    lines.append(f"- steps breaching tol {res['tolerance']}: {res['violation_fraction'] * 100:.1f}%")
    lines.append("")
    lines.append("### worst steps (by |delta|)")
    for e in res["worst_steps"][:5]:
        lines.append(f"- step {e['step']}, dim {e['dim']}: |delta| {e['delta']:.6f}")
    lines.append("")
    lines.append(f"## verdict: **{res['verdict']}**")
    for r in res.get("reasons", []):
        lines.append(f"- FAIL: {r}")
    for n in res.get("notes", []):
        lines.append(f"- NOTE: {n}")
    if res["verdict"] == "PASS":
        lines.append("- within tolerance and cosine floor; still validate on hardware")
    elif res["verdict"] == "WATCH":
        lines.append("- mild drift or an unverified signal in the WATCH band; inspect before deploying")
    return "\n".join(lines) + "\n"


class _Parser(argparse.ArgumentParser):
    """ArgumentParser whose usage errors exit 1 (argparse default is 2, which
    collides with the WATCH exit code). --help still exits 0."""

    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(1, f"{self.prog}: error: {message}\n")


def main(argv: list[str] | None = None) -> int:
    ap = _Parser(description="Offline FP32-vs-quantized action parity checker (stdlib).")
    ap.add_argument("reference", help="reference (FP32) action CSV")
    ap.add_argument("quantized", help="quantized/exported action CSV")
    ap.add_argument(
        "--index-column", default=None,
        help="shared row index column (e.g. step). Without it, columns named "
        "step/index/timestep/t/time/frame/episode are auto-excluded and every "
        "other numeric column is treated as an action; if your index has a "
        "different name (tick/frame_idx/...), pass this flag explicitly",
    )
    ap.add_argument("--tol", type=float, default=DEFAULT_TOL, help="max absolute |delta| (default 0.05)")
    ap.add_argument("--cos-min", type=float, default=DEFAULT_COS_MIN, help="cosine floor (default 0.99)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--output", default=None)
    args = ap.parse_args(argv)

    # An explicit empty string is not the same as "no index column" (None);
    # reject it rather than silently switching to auto-detection. Strip a
    # surrounding-whitespace value so the lookup uses the normalized header.
    if args.index_column is not None:
        args.index_column = args.index_column.strip()
        if args.index_column == "":
            print("policy-parity: --index-column must name a header, got an empty string", file=sys.stderr)
            return 1

    if not (math.isfinite(args.tol) and args.tol > 0):
        print("policy-parity: --tol must be a finite number > 0 (action units)", file=sys.stderr)
        return 1
    if not (math.isfinite(args.cos_min) and 0 < args.cos_min <= 1):
        print("policy-parity: --cos-min must be a finite number in (0, 1]", file=sys.stderr)
        return 1

    try:
        r_labels, r_mat, r_cols = load_action_matrix(Path(args.reference), args.index_column)
        q_labels, q_mat, q_cols = load_action_matrix(Path(args.quantized), args.index_column)
        pairs = align(r_labels, r_mat, q_labels, q_mat, labeled=bool(args.index_column),
                      ref_cols=r_cols, q_cols=q_cols)
        res = parity_verdict(pairs, args.tol, args.cos_min)
        rendered = json.dumps(res, indent=2, ensure_ascii=False) if args.json else \
            format_report(res, args.reference, args.quantized)
        if args.output:
            Path(args.output).write_text(rendered + "\n", encoding="utf-8")
        else:
            print(rendered)
    except ShapeMismatchError as exc:
        print(f"policy-parity: FAIL (shape mismatch): {exc}", file=sys.stderr)
        return 3
    except (ValueError, OSError) as exc:
        print(f"policy-parity: error: {exc}", file=sys.stderr)
        return 1

    return {"PASS": 0, "WATCH": 2, "FAIL": 3}[res["verdict"]]


if __name__ == "__main__":
    raise SystemExit(main())
