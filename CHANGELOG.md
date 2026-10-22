# Changelog — policy-parity

v1.0.0 — 2026-10-22

## v1 (baseline)
- Load two aligned action CSVs; per-dim MAE, max|delta|, flattened cosine,
  violation fraction; PASS/WATCH/FAIL; Markdown + `--json`; CLI exit codes.
- Initial suite green.

## v2 (fail-closed robustness, bugfix round) — issues found on first test run:
- index column (`step`) was treated as an action dimension in positional mode;
  now auto-excluded known index names, `--index-column` explicit otherwise.
- Labels were compared as ordered lists instead of sets; out-of-order matching now works.
- Zero reference norm made cosine 0.0 and forced a false FAIL; cosine is now
  `None` (undefined) and skipped in the gate.
- Row-count/dim/label mismatch crashed as a usage error (exit 1); now a dedicated
  `ShapeMismatchError` -> FAIL (exit 3), fail-closed.
- Non-numeric / missing cells name the file, row and column.

## v3 (worst-step localization) — quality issue: FAIL told you drift existed but
not where to look.
- `worst_steps()` returns the top-5 (step, dim, |delta|); shown in Markdown and
  included in the JSON payload; new tests cover sorting and payload shape.

## v4 (threshold honesty)
- README/SKILL state that `--tol` is action-scale dependent and 0.05 is a starting
  guess, not a universal safety limit; cosine-undefined behavior documented.

## v5 (spec compliance + manual mutation spot check)
- Frontmatter limited to spec fields; manifest follows official rules.
- Manual predicate-mutation spot check: inverted the cosine gate (`<` -> `>`) ->
  cosine-FAIL test red; reverted, full suite green (29). No automated
  mutation-testing framework is configured.

## Round-1 independent review fixes (v1.1.0)
- P2-2: undefined cosine no longer defaults to WATCH only — both zero vectors
  stay WATCH, exactly one zero stream is FAIL (asymmetric, parity unverifiable);
  the report labels the case "cosine undefined" explicitly.
- P3-1: duplicate index labels fail closed (`ShapeMismatchError` -> exit 3)
  instead of last-wins silently.
- P3-2: `--tol`/`--cos-min` reject non-finite and out-of-range values with a
  clean usage error (NaN, <=0, cos-min>1).
- P3-3: README/SKILL warn that an action column named `t` (torque) is
  auto-excluded; pass `--index-column` to keep it.
- Product: softened the competitor sentence (ORT Quantization Debugging / AMD
  Quark / NVIDIA Polygraphy compare tensors on calibration batches and need a
  runtime/toolchain; this tool compares dumped aligned action CSVs); deleted the
  unsupported "continuous control suffers most"; corrected the 0.05 rad field note
  (random-input analysis is an artifact; ~0.05 rad appears on real calibration
  data, RDK X5 forum 33766); Limits state flattened cosine does not replace
  per-dim MAE; added arXiv:2608.03938 (Jetson ACT 19/20, 18/20, 19/20) for the
  "numeric close != task equivalent" point; manual spot-check wording replaces
  "mutation-checked".
- New tests: asymmetric-zero FAIL, both-zero WATCH, NaN tol/cos-min rejection,
  `t` column excluded/kept. Suite: 35 -> 41.

## Round-1 follow-up diagnostic fix (v1.2.0)
- P3: without an index column the row labels are positional ("0","1",..). A
  length mismatch previously hit the label-set comparison first and reported
  "index labels differ; extra: ['2']"; `align` now takes a `labeled` flag (true
  only when a real index column is given), so positional length mismatches report
  the accurate "row count differs: reference N vs quantized M". Positional labels
  are never reordered as a real index.
- Suite: 41 -> 42.
