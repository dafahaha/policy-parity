# -*- coding: utf-8 -*-
"""policy-parity tests (pytest, stdlib only, no network, no torch/onnx)."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SCRIPT = HERE.parent / "skills" / "policy-parity" / "scripts" / "policy_parity.py"
sys.path.insert(0, str(SCRIPT.parent))

import policy_parity as pp  # noqa: E402


def _write(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #

def test_load_matrix_basic(tmp_path: Path):
    p = tmp_path / "ref.csv"
    _write(p, ["step", "a0", "a1"], [[0, 1.0, 2.0], [1, 3.0, 4.0]])
    labels, mat, cols = pp.load_action_matrix(p, None)
    assert labels == ["0", "1"]
    assert mat == [[1.0, 2.0], [3.0, 4.0]]
    assert cols == ["a0", "a1"]


def test_load_matrix_with_index_column(tmp_path: Path):
    p = tmp_path / "ref.csv"
    _write(p, ["step", "a0"], [[7, 1.0], [9, 2.0]])
    labels, mat, cols = pp.load_action_matrix(p, "step")
    assert labels == ["7", "9"]
    assert mat == [[1.0], [2.0]]
    assert cols == ["a0"]


def test_load_bom_and_blank_rows(tmp_path: Path):
    p = tmp_path / "ref.csv"
    p.write_bytes("﻿step,a0\r\n0,1\r\n\r\n1,2\r\n".encode("utf-8"))
    _, mat, _ = pp.load_action_matrix(p, "step")
    assert mat == [[1.0], [2.0]]


def test_non_numeric_cell_raises(tmp_path: Path):
    p = tmp_path / "ref.csv"
    _write(p, ["step", "a0"], [[0, "NaN-ish"]])
    with pytest.raises(ValueError, match="non-numeric"):
        pp.load_action_matrix(p, "step")


def test_ragged_rows_raise(tmp_path: Path):
    p = tmp_path / "ref.csv"
    _write(p, ["a0", "a1"], [[1.0], [2.0, 3.0]])
    with pytest.raises(ValueError, match="ragged"):
        pp.load_action_matrix(p, None)


def test_missing_index_column_raises(tmp_path: Path):
    p = tmp_path / "ref.csv"
    _write(p, ["a0"], [[1.0]])
    with pytest.raises(ValueError, match="index column"):
        pp.load_action_matrix(p, "step")


# --------------------------------------------------------------------------- #
# alignment
# --------------------------------------------------------------------------- #

def test_align_positional():
    pairs = pp.align([], [[1.0]], [], [[1.0]])
    assert pairs == [([1.0], [1.0])]


def test_align_label_matching_orders(tmp_path: Path):
    pairs = pp.align(["1", "0"], [[1.0], [0.0]], ["0", "1"], [[9.0], [8.0]], labeled=True)
    assert pairs == [([1.0], [8.0]), ([0.0], [9.0])]


def test_align_label_mismatch_raises():
    with pytest.raises(ValueError, match="index labels differ"):
        pp.align(["a", "b"], [[1.0]], ["a", "c"], [[1.0]], labeled=True)


def test_align_column_order_permutation_raises():
    # Same set of action columns, different order: must not be paired positionally.
    with pytest.raises(pp.ShapeMismatchError, match="column order differs"):
        pp.align(
            [], [[1.0, 2.0]], [], [[3.0, 4.0]],
            ref_cols=["x", "y"], q_cols=["y", "x"],
        )


def test_align_length_mismatch_raises():
    with pytest.raises(ValueError, match="row count differs"):
        pp.align([], [[1.0], [2.0]], [], [[1.0]])


def test_align_positional_length_mismatch_says_row_count():
    # P3-4: without an index column the labels are positional ("0","1",..).
    # A length mismatch must report "row count differs", not "index labels differ".
    with pytest.raises(ValueError, match="row count differs"):
        pp.align(["0", "1", "2"], [[1.0], [1.0], [1.0]], ["0", "1"], [[1.0], [1.0]])
    # and the positional labels must not be aligned as a real index
    pairs = pp.align(["0", "1"], [[1.0], [2.0]], ["0", "1"], [[9.0], [8.0]])
    assert pairs == [([1.0], [9.0]), ([2.0], [8.0])]  # positional order, not reordered


def test_align_dim_mismatch_raises():
    with pytest.raises(ValueError, match="dimension differs"):
        pp.align([], [[1.0, 2.0]], [], [[1.0]])


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #

def test_per_dim_mae_known():
    pairs = [([0.0, 0.0], [1.0, 2.0]), ([0.0, 0.0], [3.0, 4.0])]
    assert pp.per_dim_mae(pairs) == pytest.approx([2.0, 3.0])


def test_max_abs_deviation_location():
    pairs = [([0.0], [0.1]), ([0.0], [0.4])]
    worst, where = pp.max_abs_deviation(pairs)
    assert worst == pytest.approx(0.4)
    assert "step 1" in where and "dim 0" in where


def test_cosine_identical_is_one():
    pairs = [([1.0, 0.0], [1.0, 0.0]), ([0.0, 1.0], [0.0, 1.0])]
    assert pp.flattened_cosine(pairs) == pytest.approx(1.0)


def test_cosine_zero_vector_is_undefined():
    assert pp.flattened_cosine([([0.0], [1.0])]) is None


def test_cosine_orthogonal():
    assert pp.flattened_cosine([([1.0, 0.0], [0.0, 1.0])]) == pytest.approx(0.0, abs=1e-9)


def test_violation_fraction():
    pairs = [([0.0], [0.02]), ([0.0], [0.2])]
    assert pp.step_violation_fraction(pairs, 0.05) == pytest.approx(0.5)


def test_worst_steps_sorted_descending():
    pairs = [([0.0], [0.02]), ([0.0], [0.2]), ([0.0], [0.1])]
    top = pp.worst_steps(pairs, 2)
    assert top[0]["step"] == 1 and top[0]["delta"] == pytest.approx(0.2)
    assert top[1]["delta"] == pytest.approx(0.1)


# --------------------------------------------------------------------------- #
# verdict
# --------------------------------------------------------------------------- #

def test_verdict_pass_identical():
    pairs = [([0.1, -0.2], [0.1, -0.2]), ([0.3, 0.0], [0.3, 0.0])]
    res = pp.parity_verdict(pairs, 0.05, 0.99)
    assert res["verdict"] == "PASS"
    assert res["reasons"] == []


def test_verdict_fail_on_tolerance():
    pairs = [([0.0], [0.3])]
    res = pp.parity_verdict(pairs, 0.05, 0.99)
    assert res["verdict"] == "FAIL"
    assert any("tolerance" in r for r in res["reasons"])


def test_verdict_fail_on_cosine():
    pairs = [([1.0, 0.0], [0.0, 1.0])]
    res = pp.parity_verdict(pairs, 10.0, 0.99)
    assert res["verdict"] == "FAIL"
    assert any("cosine" in r for r in res["reasons"])


def test_verdict_watch_mild_drift():
    # 0.029 deviation: below tol 0.05, above half 0.025 -> WATCH (both vectors
    # nonzero so cosine is defined; a zero vector would FAIL as asymmetric)
    pairs = [([0.001], [0.03])]
    res = pp.parity_verdict(pairs, 0.05, 0.99)
    assert res["verdict"] == "WATCH"


def test_verdict_watch_cosine_midband():
    # Cosine WATCH band: cosine sits between the floor (0.99) and the midpoint
    # (0.995) while max|delta| stays small (<= tol/2) -> WATCH, not PASS/FAIL.
    pairs = [([1.0, 1.0], [1.0, 0.8])]
    res = pp.parity_verdict(pairs, 10.0, 0.99)
    assert res["cosine"] is not None
    assert 0.99 < res["cosine"] < (0.99 + 1) / 2
    assert res["max_abs_deviation"] <= 10.0 / 2
    assert res["reasons"] == []
    assert res["verdict"] == "WATCH"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _two_files(tmp: Path, ref: list[list], q: list[list]) -> tuple[Path, Path]:
    r = tmp / "ref.csv"
    qf = tmp / "q.csv"
    _write(r, ["step", "a0"], ref)
    _write(qf, ["step", "a0"], q)
    return r, qf


def test_cli_pass_exit_zero(tmp_path: Path, capsys):
    r, q = _two_files(tmp_path, [[0, 0.1], [1, -0.2]], [[0, 0.1], [1, -0.2]])
    code = pp.main([str(r), str(q)])
    assert code == 0
    assert "PASS" in capsys.readouterr().out


def test_cli_fail_exit_three(tmp_path: Path):
    r, q = _two_files(tmp_path, [[0, 0.0]], [[0, 0.3]])
    assert pp.main([str(r), str(q)]) == 3


def test_cli_watch_exit_two(tmp_path: Path):
    r, q = _two_files(tmp_path, [[0, 0.001]], [[0, 0.03]])
    assert pp.main([str(r), str(q)]) == 2


def test_cli_shape_mismatch_exit_three(tmp_path: Path):
    r, q = _two_files(tmp_path, [[0, 0.0], [1, 0.0]], [[0, 0.0]])
    assert pp.main([str(r), str(q)]) == 3


def test_cli_missing_file_exit_one(tmp_path: Path):
    assert pp.main([str(tmp_path / "nope.csv"), str(tmp_path / "x.csv")]) == 1


def test_cli_json_output(tmp_path: Path, capsys):
    import json as _json

    r, q = _two_files(tmp_path, [[0, 0.1]], [[0, 0.1]])
    code = pp.main([str(r), str(q), "--json"])
    assert code == 0
    payload = _json.loads(capsys.readouterr().out)
    assert payload["verdict"] == "PASS"
    assert payload["steps"] == 1
    assert "worst_steps" in payload and len(payload["worst_steps"]) == 1


def test_cli_index_column_aligns_out_of_order(tmp_path: Path, capsys):
    r = tmp_path / "ref.csv"
    q = tmp_path / "q.csv"
    _write(r, ["step", "a0"], [[1, 0.1], [0, 0.2]])
    _write(q, ["step", "a0"], [[0, 0.2001], [1, 0.1001]])
    code = pp.main([str(r), str(q), "--index-column", "step", "--tol", "0.005"])
    out = capsys.readouterr().out
    assert code == 0  # labels align out of order; 0.0001 drift within tol
    assert "PASS" in out


# ---- Round-1 review: NaN fail-closed + undefined cosine + validation ----

def test_load_rejects_nan_cell(tmp_path: Path):
    # P2-1: a NaN cell must not slip through to a false PASS.
    p = tmp_path / "ref.csv"
    _write(p, ["step", "a0"], [[0, "nan"], [1, 2.0]])
    with pytest.raises(ValueError, match="non-finite"):
        pp.load_action_matrix(p, "step")


def test_cli_nan_cell_exit_one(tmp_path: Path):
    r, q = _two_files(tmp_path, [[0, "nan"]], [[0, 0.1]])
    assert pp.main([str(r), str(q)]) == 1


def test_undefined_cosine_both_zero_is_watch_not_pass():
    # P2-2: both vectors zero -> cosine undefined; WATCH with a note, never PASS.
    res = pp.parity_verdict([([0.0], [0.0])], 10.0, 0.99)
    assert res["cosine"] is None
    assert res["verdict"] == "WATCH"
    assert any("undefined" in n for n in res["notes"])


def test_undefined_cosine_asymmetric_zero_is_fail():
    # P2-2: exactly one vector zero -> asymmetric, parity unverifiable -> FAIL.
    res = pp.parity_verdict([([0.0], [0.03])], 0.05, 0.99)
    assert res["cosine"] is None
    assert res["verdict"] == "FAIL"
    assert any("cosine undefined" in r for r in res["reasons"])


def test_duplicate_index_label_raises():
    # P3-1: duplicate labels must fail closed instead of last-wins silently.
    with pytest.raises(pp.ShapeMismatchError, match="duplicate index label"):
        pp.align(["a", "a"], [[1.0], [2.0]], ["a", "a"], [[1.0], [2.0]], labeled=True)


def test_arg_tol_must_be_positive(tmp_path: Path):
    r, q = _two_files(tmp_path, [[0, 0.1]], [[0, 0.1]])
    assert pp.main([str(r), str(q), "--tol", "0"]) == 1


def test_arg_cos_min_range(tmp_path: Path):
    r, q = _two_files(tmp_path, [[0, 0.1]], [[0, 0.1]])
    assert pp.main([str(r), str(q), "--cos-min", "1.5"]) == 1


def test_arg_tol_rejects_nan(tmp_path: Path):
    # A non-finite tol must be rejected cleanly (exit 1), not invert the gate.
    r, q = _two_files(tmp_path, [[0, 0.1]], [[0, 0.1]])
    assert pp.main([str(r), str(q), "--tol", "nan"]) == 1


def test_arg_cos_min_rejects_nan(tmp_path: Path):
    r, q = _two_files(tmp_path, [[0, 0.1]], [[0, 0.1]])
    assert pp.main([str(r), str(q), "--cos-min", "nan"]) == 1


def test_column_named_t_is_excluded_without_index_column(tmp_path: Path):
    # P3-3: a torque column named `t` looks like an index; auto-excluded unless
    # the user passes --index-column explicitly.
    p = tmp_path / "ref.csv"
    _write(p, ["step", "t", "a0"], [[0, 0.5, 1.0], [1, 0.6, 3.0]])
    _, mat, _ = pp.load_action_matrix(p, None)
    assert mat == [[1.0], [3.0]]  # both step and t dropped; only a0 remains


def test_column_named_t_kept_with_explicit_index(tmp_path: Path):
    p = tmp_path / "ref.csv"
    _write(p, ["step", "t", "a0"], [[0, 0.5, 1.0], [1, 0.6, 3.0]])
    _, mat, _ = pp.load_action_matrix(p, "step")
    assert mat == [[0.5, 1.0], [0.6, 3.0]]  # explicit index keeps t as an action dim


# ---- R5 review: column-name mismatch, extra-wide row, usage exit 1, index-only ----

def test_column_name_mismatch_raises(tmp_path: Path):
    r = tmp_path / "ref.csv"
    q = tmp_path / "q.csv"
    _write(r, ["step", "a0", "a1"], [[0, 1.0, 2.0]])
    _write(q, ["step", "a0", "b1"], [[0, 1.0, 2.0]])  # same count, renamed column
    with pytest.raises(pp.ShapeMismatchError, match="action columns differ"):
        rl, rm, rc = pp.load_action_matrix(r, None)
        ql, qm, qc = pp.load_action_matrix(q, None)
        pp.align(rl, rm, ql, qm, ref_cols=rc, q_cols=qc)


def test_extra_wide_row_raises(tmp_path: Path):
    p = tmp_path / "ref.csv"
    # A row with more fields than the header.
    p.write_text("step,a0\n0,1,9\n", encoding="utf-8")
    with pytest.raises(pp.ShapeMismatchError, match="extra field"):
        pp.load_action_matrix(p, None)


def test_index_only_file_raises(tmp_path: Path):
    p = tmp_path / "ref.csv"
    _write(p, ["step", "index"], [[0, 1]])
    with pytest.raises(ValueError, match="no action columns"):
        pp.load_action_matrix(p, None)


def test_usage_error_exits_one():
    with pytest.raises(SystemExit) as e:
        pp.main([])
    assert e.value.code == 1
    with pytest.raises(SystemExit) as e:
        pp.main(["only-one-file"])
    assert e.value.code == 1
