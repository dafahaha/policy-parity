# policy-parity

**量化策略导出的离线动作漂移检查器。** 逐行对比 FP32 参考动作 CSV 与量化/ONNX/INT8
动作 CSV，给出逐维 MAE、最大绝对偏差、扁平化余弦、越限比例和 PASS / WATCH / FAIL
判定——不 import torch、onnx、numpy。

纯 Python 标准库。不联网。Windows / macOS / Linux，Python 3.10+。

## 为什么存在

把 RL 策略导出为 INT8/ONNX 部署到嵌入式或机器人目标后，常见失败从未被端到端测量：
量化网络输出略有不同的动作，但没人把 dump 下来的 FP32 rollout 与量化结果逐行对比。
数值接近仍不等于任务等价：在入门级 Jetson 上，一个 ACT 策略三种精度都保住了任务成功率
（FP32/FP16/INT8 为 19/20、18/20、19/20，arXiv:2608.03938）——这里 PASS 之后仍必须做硬件评测。

官方调试器已经在另一层面对比输出：ONNX Runtime
[Quantization Debugging](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
在校准批上检查权重/激活张量，AMD
[Quark](https://quark.docs.amd.com/v0.11.1/supported_accelerators/ryzenai/tutorial_xint8_quantize.html)
与 NVIDIA [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy)
对比张量输出，通常都需要加载推理 runtime 或绑定特定工具链。policy-parity 只对比你已经
dump 好的两份对齐动作 CSV，纯标准库，可以进 CI。

关于 0.05 rad 这个数字的现场说明（RDK X5 论坛，
<https://forum.d-robotics.cc/t/topic/33766>）：用随机 `np.random.rand` 输入做精度分析会
得到虚假的大幅偏差——那是测量伪影，不是量化误差。在真实校准数据下，float/INT8 动作差约
0.05 rad，经 0.25 的 action_scale 后可忽略。请使用真实 rollout 数据；本工具不做校准分析。

## 安装

无需编译，无需依赖。

- Claude Code：`claude --plugin-dir ./policy-parity`，或在
  `.claude-plugin/marketplace.json` 注册本地市场。
- 便携 skill（Codex/Cursor/Gemini CLI/Roo Code）：把 `skills/policy-parity/`
  复制到客户端的 skills 目录。

## 用法

```bash
python skills/policy-parity/scripts/policy_parity.py ref_actions.csv quant_actions.csv \
  --index-column step --tol 0.05 --cos-min 0.99 [--output report.md]
```

- 两个 CSV 共享行索引；自动排除的索引列名：`step`、`index`、`timestep`、`t`、
  `time`、`frame`、`episode`。
  **注意：** 恰好名为 `t` 的动作列（如扭矩 torque）也会被当作索引排除——如果你的动作
  维叫 `t`，请显式传 `--index-column` 指定真实索引列，让 `t` 留在动作矩阵里。
- `--tol`：每步最大绝对 |delta|（动作单位，默认 0.05——按你的动作尺度设定；
  <=0 或非有限值会被干净拒绝）。
- `--cos-min`：扁平化向量余弦下限（默认 0.99；必须在 (0, 1]）。
- `--json`：供 CI 使用的结构化载荷。
- `--output`：把报告写入文件，而非打印到 stdout。

### 判定 / 退出码

| 码 | 判定 | 含义 |
|---|---|---|
| 0 | PASS | 在容差与余弦下限内；仍需硬件验证 |
| 2 | WATCH | 轻微漂移（max\|delta\| > tol/2），或余弦落在下限与中点 `(cos_min+1)/2` 之间，或两向量全零导致余弦未定义；检查最差步骤 |
| 3 | FAIL | 越限/余弦低于下限，或形状不匹配（fail-closed） |
| 1 | 用法 / IO 错误 |

## 相关工作

- ONNX Runtime [Quantization Debugging](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)：
  校准批上的权重/激活张量对比，需要 ORT runtime。AMD Quark 的余弦/L2/PSNR 输出检查与
  NVIDIA Polygraphy 的跨后端 max/mean-absdiff 对比都绑定各自工具链。本工具与之互补：
  处理已 dump 的对齐动作轨迹，纯标准库、适合 CI，不替代张量级调试。

## 测试

```bash
python -m pytest tests/ -q
```

46 个测试，仅标准库；人工谓词变异抽查（翻转余弦闸门会使余弦 FAIL 测试变红；记录在
CHANGELOG）。

## 限制（诚实披露）

- 仅数值一致——不是控制成功测试；PASS 不保证任务在硬件上仍可解（见上方
  arXiv:2608.03938）。
- 不做推理、校准分析或 ONNX 图检查。
- 阈值依赖领域，工具不替你选安全限值。
- 扁平化余弦度量所有步×维的全局方向一致性，不替代逐维 MAE——余弦 0.999 也可能藏着
  某个关节漂移，务必同时看逐维 MAE 和最差步骤表。
- 向量范数为零时余弦未定义（如实标注，绝不当作 0）：两向量全零 → WATCH；
  恰有一个全零流 → FAIL（一致性无法验证）。

## 许可证

MIT，© 2026 Daizhi Liao (dafahaha)。
