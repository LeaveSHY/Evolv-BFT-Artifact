# Experiments Assets

该目录用于承载 Octopus 论文级实验资产的最小真实入口。

当前提供的是**最小可运行实验入口 + 占位绘图入口**，而不是论文级一键复现实验包。

当前包含：

- `run_data_plane_eval.sh`：可运行的最小 Data Plane smoke verification 入口，默认执行 `go test ./octopus/consensus/hotstuff`，并输出 machine-readable summary
- `run_control_plane_eval.py`：可运行的最小 Control Plane evaluation harness，支持按 `scenario × policy × seed` 批量运行，并输出 JSON/CSV 结果文件
- `plot_results.py`：可运行的最小结果消费入口，当前读取 control-plane eval 产物并生成 JSON/Markdown summary，而非稳定论文绘图流水线
- `../deploy/run_local_cluster.sh`：可运行的最小 local multi-node runtime smoke 入口，当前可输出 machine-readable local-cluster smoke summary，用于记录 usage、dependency failure 与真实运行后的结构化 verification 结果

`run_data_plane_eval.sh` 当前边界：

- 必须用 `bash` 运行；
- 默认输出：
  - `data_plane_smoke_summary.json`
  - `go_test_stdout.txt`
  - `go_test_stderr.txt`
- 默认输出目录：`experiments/results/data_plane_eval/`
- 可通过环境变量 `OCTOPUS_DATA_PLANE_EVAL_OUTPUT_DIR` 覆盖输出目录；
- 当前只提供最小 data-plane smoke verification，不构成 paper-grade benchmark、attack study 或 artifact package 证据。

`../deploy/run_local_cluster.sh` 当前边界：

- 必须用 `bash` 运行；
- 默认输出目录：`experiments/results/local_cluster_smoke/`
- 可通过环境变量 `OCTOPUS_LOCAL_CLUSTER_OUTPUT_DIR` 覆盖输出目录；
- 默认输出：
  - `local_cluster_smoke_summary.json`
  - `local_cluster_stdout.txt`
  - `local_cluster_stderr.txt`
- summary 当前会区分：
  - `usage`
  - `failed`
  - `partial`
  - `passed`
- summary 当前包含结构化 `verification`，用于记录：
  - `nodes_requested`
  - `nodes_checked`
  - `reachable_nodes`
  - `committed_nodes`
  - `truthful_outcome`
  - `node_results[]`
- 该入口当前是 minimal multi-node runtime evidence，不构成 paper-grade benchmark、throughput/latency validation 或 artifact package 证据。

`run_control_plane_eval.py` 当前边界：

- 使用仓库内置 deterministic scenario / policy / simulator 进行最小控制面评估；
- 主要用于生成可审阅的 reward / throughput / latency / backlog / reconfiguration 等结果产物；
- 默认输出：
  - `control_plane_eval.json`
  - `control_plane_eval_runs.csv`
  - `control_plane_eval_aggregates.csv`
  - `control_plane_eval_steps.csv`
- `plot_results.py` 当前默认输出：
  - `control_plane_plot_summary.json`
  - `control_plane_plot_summary.md`
- 当前仓库里需要显式区分两类 evidence/schema regime：
  - `octopus-adaptive-v1`：用于 runtime-owned adaptive trace / provenance 边界，来源是 `src/octopus/adaptive`；当前 control-plane eval metadata 中的 `trace_provenance.schema_version` 使用这一 regime，用来说明控制面观测/动作/trace 语义来自运行时自适应路径，而不是根目录 `marl/` 自身成为 runtime truth。
  - `octopus-evidence-v1`：用于 artifact-level `evidence_manifest`，适合 smoke / summary 类产物声明 producer、truth level、claim boundary、evidence kinds 与 excludes；`run_data_plane_eval.sh` 和 `../deploy/run_local_cluster.sh` 当前使用这一 regime。
- 对 control-plane / plot 产物要保持最小真实表述：它们是 reviewable control-plane simulation evidence，并携带 runtime trace provenance；这不等价于 native MARL authority、authoritative runtime proof、或 paper-grade closure。
- 该脚本当前不等价于“真实 runtime + external research service 已完成 paper-grade 闭环验证”；
- 该脚本当前也不构成最终论文图表、benchmark、attack study 或 artifact package 已完成的证据。

最小运行示例：

```bash
python3 experiments/run_control_plane_eval.py \
  --output-dir experiments/results/control_plane_eval \
  --scenarios steady_state membership_disturbance \
  --policies fixed adaptive \
  --seeds 7 13 42
```

约束：

- 所有结果必须由仓库内脚本生成，不能先写图表再倒推脚本；
- 所有参数、输出路径、输入数据格式都需要继续保持显式；
- 当前实验资产仍需继续区分“可运行实验入口”“已验证结果”“论文级可复现资产”；
- 若未来接入真实 runtime ↔ `marl/` service 流水线，文档必须继续明确它是更强闭环验证，而不是自动把当前仓库提升为 paper-grade claim。
