---
name: policy-parity
description: Check whether a quantized/exported RL policy (ONNX/INT8/FP16) still produces actions close to the FP32 reference, using two already-dumped action CSVs. Use when the user exports a policy to ONNX/TensorRT/INT8 for an embedded or robotic target, wants to validate deployment drift, or asks "did quantization change my actions?". Computes per-dim MAE, max absolute deviation, cosine and breach fraction; returns PASS/WATCH/FAIL without importing torch or onnx.
license: MIT
compatibility: Python 3.10+, stdlib only; offline; Windows/macOS/Linux; needs two aligned action CSVs, not a live model.
---

# policy-parity — offline action-drift check

## When to activate

- The user exported an RL policy to ONNX/INT8/FP16 for Jetson/robot/embedded deployment and wants to know whether actions drifted.
- Two action logs already exist: reference (FP32) and quantized, aligned by step.
- Do **not** use this to run the model; it compares dumped numeric CSVs only.

## Steps

1. Confirm both CSVs share a row index (`step`/`index`/`timestep`/`t`/`time`/`frame`/`episode` auto-excluded; pass `--index-column` explicitly when names differ). **Caution:** an action column named `t` (torque) is auto-excluded as an index name — pass `--index-column` with the real index so `t` stays as an action dim. Check the same action dimension in both files.
2. Run:

```bash
python skills/policy-parity/scripts/policy_parity.py ref_actions.csv quant_actions.csv \
  [--index-column step] [--tol 0.05] [--cos-min 0.99] [--json] [--output report.md]
```

3. Read the verdict:
   - **PASS (exit 0):** max|delta| <= tol and cosine >= floor; still validate on hardware.
   - **WATCH (exit 2):** mild drift (max|delta| > tol/2), cosine falls between the floor and the midpoint (`cos_min`, `(cos_min+1)/2`), or cosine undefined with both vectors zero; inspect the worst steps before deploying.
   - **FAIL (exit 3):** tolerance or cosine breached, cosine undefined on exactly one zero stream, or rows/labels/dims mismatch (fail-closed).
4. Read the `worst steps` table; map the failing step/dim back to the episode before changing calibration data.
5. Pick `--tol` from the action scale (e.g. radians vs normalized [-1,1]); a fixed 0.05 is a starting guess, not a universal threshold.

## Interpretation notes

- Cosine is undefined (reported as such, never treated as 0) when a vector norm is zero: both zero -> WATCH; exactly one zero stream -> FAIL.
- Per-dim MAE shows which joint/channel drifted; flattened cosine is global direction agreement and does not replace it — read both.
- This is a numeric parity gate, not a control-quality test: small action drift can still change task success on hardware (e.g. Jetson ACT 19/20, 18/20, 19/20 across FP32/FP16/INT8, arXiv:2608.03938) — run the real eval after a PASS.

## Boundaries

- No model inference, no calibration analysis, no ONNX graph inspection. Tensor-level output debugging belongs to ORT Quantization Debugging, AMD Quark, or NVIDIA Polygraphy; this tool only compares dumped aligned action CSVs.
- Does not check latency, memory, or numerical overflow inside the quantizer.
- Thresholds are the user's responsibility; the tool reports, it does not choose domain safety limits.
