# policy-parity

**Offline action-drift checker for quantized RL policy exports.** Compare an FP32
reference action CSV against a quantized/ONNX/INT8 action CSV, row by row, and get
per-dim MAE, max absolute deviation, flattened cosine, breach fraction, and a
PASS / WATCH / FAIL verdict — without importing torch, onnx, or numpy.

Pure Python standard library. No network. Windows / macOS / Linux, Python 3.10+.

## Why this exists

After exporting an RL policy to INT8/ONNX for an embedded or robotic target, the
common failure is never measured end-to-end: the quantized network outputs
slightly different actions, and nobody compares the dumped FP32 rollout against
the quantized one line by line. Numeric closeness still does not prove task
equivalence: on an entry-level Jetson, an ACT policy kept task success at all
three precisions (19/20, 18/20, 19/20 for FP32/FP16/INT8,
arXiv:2608.03938) — a PASS here must still be followed by a hardware eval.

Official debuggers already compare outputs, but at a different layer: ONNX Runtime
[Quantization Debugging](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
inspects weights/activations on calibration batches, AMD
[Quark](https://quark.docs.amd.com/v0.11.1/supported_accelerators/ryzenai/tutorial_xint8_quantize.html)
and NVIDIA [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy)
compare tensor outputs, and both usually require loading an inference runtime or a
tied toolchain. policy-parity only compares the two aligned action CSVs you have
already dumped, in pure stdlib, so it can run in CI.

A field note on the 0.05 rad figure (RDK X5 forum,
<https://forum.d-robotics.cc/t/topic/33766>): running the precision check with
random `np.random.rand` inputs produces a false, large divergence — a measurement
artifact, not quantization error. On real calibration data the float/INT8 action
difference was about 0.05 rad, negligible after the 0.25 action_scale. Use real
rollout data; this tool does not run calibration analysis.

## Install

No build, no dependencies.

- Claude Code: `claude --plugin-dir ./policy-parity`, or register the local
  marketplace in `.claude-plugin/marketplace.json`.
- Portable skill (Codex/Cursor/Gemini CLI/Roo Code): copy `skills/policy-parity/`
  into the client's skills directory.

## Usage

```bash
python skills/policy-parity/scripts/policy_parity.py ref_actions.csv quant_actions.csv \
  --index-column step --tol 0.05 --cos-min 0.99 [--output report.md]
```

- Both CSVs share a row index; auto-excluded index names: `step`, `index`,
  `timestep`, `t`, `time`, `frame`, `episode`.
  **Warning:** an action column literally named `t` (e.g. torque) is also
  auto-excluded as an index name — if your action dim is called `t`, pass
  `--index-column` explicitly with the real index column so `t` stays in the
  action matrix.
- `--tol`: max absolute |delta| per step in action units (default 0.05 — set
  from your action scale; rejected if <= 0 or non-finite).
- `--cos-min`: flattened-vector cosine floor (default 0.99; must be in (0, 1]).
- `--json`: structured payload for CI.
- `--output`: write the report to a file instead of stdout.

### Verdict / exit codes

| code | verdict | meaning |
|---|---|---|
| 0 | PASS | within tolerance and cosine floor; still validate on hardware |
| 2 | WATCH | mild drift (max\|delta\| > tol/2), cosine between the floor and the midpoint `(cos_min+1)/2`, or both-zero undefined cosine; inspect worst steps |
| 3 | FAIL | tolerance/cosine breach or shape mismatch (fail-closed) |
| 1 | usage / IO error |

## Related work

- ONNX Runtime [Quantization Debugging](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html):
  weight/activation tensor comparison on calibration batches; needs the ORT
  runtime. AMD Quark cosine/L2/PSNR output checks and NVIDIA Polygraphy
  max/mean-absdiff cross-backend comparison are bound to their toolchains. This
  tool complements them: it works on already-dumped aligned action trajectories,
  pure stdlib, CI-friendly, and does not replace tensor-level debugging.

## Tests

```bash
python -m pytest tests/ -q
```

46 tests, stdlib only; manual predicate-mutation spot checks (inverting the cosine
gate turns the cosine-FAIL test red; recorded in CHANGELOG).

## Limits (honest)

- Numeric parity only — not a control-success test; PASS does not guarantee the
  task still solves on hardware (see arXiv:2608.03938 above).
- No inference, calibration analysis, or ONNX graph inspection.
- Thresholds are domain-specific; the tool does not choose safety limits.
- Flattened cosine measures global direction agreement over all steps×dims; it
  does not replace the per-dim MAE — a cosine of 0.999 can still hide one joint
  drifting, so read the per-dim MAE and worst-steps table.
- Cosine is undefined (reported as such, never treated as 0) when a vector norm
  is zero: both zero -> WATCH; exactly one zero stream -> FAIL (parity
  unverifiable).

## License

MIT, © 2026 Daizhi Liao (dafahaha).
