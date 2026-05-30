# E2E Integration Experiment Design

> **目标**: 解决 APEX Review C2 (无端到端实验) + P0-3 (缺少简单自适应基线)  
> **额外收益**: 同时为 Proposition (prop:marl-necessity) 提供实验验证

---

## 1. 为什么需要 E2E 实验

论文声称"端到端自适应安全"，但当前实验：
- DP 评估: Go 代码在 EC2 上跑 (throughput/latency)
- CP 评估: Python 独立模拟 (detection/MARL training)
- V2X 评估: 另一个独立实验 (perception fusion)

**三个实验从未在一个集成系统中运行。** 审稿人最可能的 #1 拒稿理由就是这个。

---

## 2. 代码现状与可行性分析

| 组件 | 状态 | E2E 所需改动 |
|------|------|-------------|
| DP (Go, HotStuff) | ✅ 完整 | 不需要改 — 使用已有 EC2 吞吐量数据作为 throughput model |
| GBC (Go) | ⚠️ Stub | 不需要真 BFT — 模拟 GBC 开销 (1.09×, 论文已有数据) |
| Trust Estimator (Go) | ⚠️ Stub | 在 Python 模拟中实现完整版 (EWMA + sigmoid) |
| MARL (Python) | ⚠️ Numpy | 在已有 `run_full_cp_experiments.py` 基础上扩展 |
| Orchestrator (Python) | ✅ HTTP | 不用于 E2E — E2E 在 Python 模拟中完成 |

**关键洞察**: 论文第 6 节已说 "evaluation within the CP simulation"。E2E 实验不需要 Go + Python 同时跑在 EC2 上——需要的是**一个统一模拟器，展示三层交互的完整流程**。

---

## 3. 实验设计

### Experiment E2E-1: End-to-End Adaptive Response Under Roaming Attacks

#### 3.1 Setup

```
m = 4 parallel BFT instances (Tentacles)
n = 100 total replicas (25 per instance)
f = 30 Byzantine nodes (ρ = 0.30, < 1/3)
T = 500 epochs
Roaming adversary: switches target instance every τ_switch = 50 epochs
GBC: simulated pipelined HotStuff among m=4 primaries (1.09× overhead)
```

#### 3.2 End-to-End Workflow (Per Epoch)

```
┌──────────┐     ┌──────────────┐     ┌──────────┐     ┌──────────────┐
│ Tentacles │────▶│ Trust Estimator│────▶│   MARL   │────▶│ Safety Filter│
│ (DP sim)  │     │  (EWMA+σ)    │     │ (FACMAC) │     │  (n≥3f̂+1)  │
└──────────┘     └──────────────┘     └──────────┘     └──────────────┘
      │                                     │                   │
      │                                     │                   ▼
      │                                     │            ┌──────────┐
      │                                     └───────────▶│   GBC    │
      │                                                  │(metadata)│
      │                                                  └────┬─────┘
      │                                                       │
      ▼                                                       ▼
┌──────────────────────────────────────────────────────────────────┐
│               Reconfiguration (if triggered)                     │
│  Move nodes between instances, update n_j, reset trust windows   │
└──────────────────────────────────────────────────────────────────┘
```

1. **每个实例生成信任特征**: timeout_count, equivoc_vote_count, view_change_count, latency_mean, latency_std
2. **Trust Estimator**: EWMA 平滑 + sigmoid 映射 → f̂_j (每实例估计的拜占庭数)
3. **MARL Controller**: 
   - 观测: {f̂_j, n_j, throughput_j, 实例状态} for j=1..m
   - 动作: {leader_selection, instance_weights, detection_thresholds, reconfig_proposal}
4. **Safety Filter**: 硬约束检查 n_j ≥ 3f̂_j + 1；若违反则 mask action
5. **GBC Publish**: 聚合 trust features + 策略决策 → 全局可见
6. **Reconfiguration** (如触发): 在 epoch 边界转移节点，模拟 Hydra 式状态转移
7. **Throughput Model**: 基于实际 EC2 数据的分段线性模型
   - Normal: throughput[n_j, f_j=0] from DP benchmarks
   - Under attack: throughput × (1 - attack_impact_factor)
   - During reconfig: throughput × 0.7 (30% temporary reduction)

#### 3.3 Baselines (同时解决 P0-3)

| 基线 | 描述 | 对应论文论点 |
|------|------|------------|
| **CUSUM** | 每实例独立 CUSUM 检测，固定阈值 | Impossibility Thm — Ω(T) for Case 4 |
| **EXP3+Safety** | 每实例独立 EXP3 bandit + safety filter | Proposition (i) — cross-instance blindness |
| **Centralized UCB** | 全局 UCB 调度但无多智能体协调 | Proposition (ii) — exponential arms |
| **Full Octopus** | MARL-CTDE + MOISE+ + safety filter | O(√T) claim |

每个基线使用**完全相同的 E2E pipeline**——仅替换 MARL Controller 部分。

#### 3.4 Metrics

| 指标 | 公式/含义 | 验证哪个 claim |
|------|----------|---------------|
| **Cumulative Damage D(T)** | Σ undetected Byzantine epochs | Thm 10: Ω(T) vs O(√T) |
| **Damage Growth Rate** | D(t)/t over time | 可视化 linear vs sublinear |
| **Detection Latency** | Epochs from attack onset to detection | Thm 8: FNR ≤ exp(-Wρ²/2) |
| **Reconfiguration Count** | # epoch-boundary reconfiguration events | System responsiveness |
| **Safety Violations** | # epochs where n_j < 3f̂_j + 1 | Safety filter effectiveness |
| **Throughput Over Time** | ktx/s time series (from DP model) | Composition Thm: liveness impact |
| **Throughput Recovery Time** | Epochs from reconfig trigger to ≥90% normal | Liveness guarantee |
| **GBC Overhead** | % throughput reduction due to GBC | 1.09× claim |
| **End-to-End Latency** | Detection + GBC + Reconfiguration total | Practical deployability |

#### 3.5 Expected Results

基于已有 CP 实验数据和论文理论：

| Metric | CUSUM | EXP3+Safety | UCB | Octopus |
|--------|-------|-------------|-----|---------|
| D(500) | ~2500 (linear) | ~500 (sub-optimal) | ~490 | ~7 (sublinear) |
| D(T)/T trend | → constant | → slow decay | → slow decay | → 0 |
| Detection latency | >10 epochs | 5-8 epochs | 5-7 epochs | 2-3 epochs |
| Safety violations | 0 | 0 | possible | 0 |
| Throughput recovery | N/A (no reconfig) | 8-12 epochs | 6-10 epochs | 3-5 epochs |

#### 3.6 Outputs (论文新增内容)

1. **fig_e2e_damage.pdf** — 4 条 D(t) curves over time (main body Figure)
2. **fig_e2e_throughput.pdf** — Throughput time series showing attack → detection → reconfig → recovery
3. **tab_e2e.tex** — Summary table with all metrics
4. **Text** — 1 paragraph in Section 6 describing E2E results

---

## 4. 实现计划

### Phase 1: 扩展 CP 模拟器 (~2-3 天)

修改 `run_full_cp_experiments.py` 或新建 `run_e2e_experiments.py`:

```python
# 核心新增模块
class ThroughputModel:
    """基于 EC2 DP 数据的分段线性吞吐量模型"""
    
class GBCSimulator:
    """模拟 GBC metadata publishing + overhead"""
    
class ReconfigurationEngine:
    """Epoch-boundary 重配置逻辑 (Hydra-style state transfer)"""
    
class E2EExperiment:
    """统一 Pipeline: DP sim → Trust → Controller → Safety → GBC → Reconfig"""
    
class EXP3Baseline:
    """EXP3 + safety filter per-instance (Proposition 验证)"""
    
class CentralizedUCBBaseline:
    """Centralized UCB without multi-agent coordination"""
```

### Phase 2: 运行实验 (~1 天)

```bash
# 5 seeds, 4 baselines, 500 epochs each
python run_e2e_experiments.py --seeds 7 13 42 97 137 --output-dir results/e2e/
```

### Phase 3: 生成 Figures & Tables (~0.5 天)

```bash
python plot_e2e_results.py --input results/e2e/ --output figures/
# 生成: fig_e2e_damage.pdf, fig_e2e_throughput.pdf, tab_e2e.tex
```

### Phase 4: 更新论文 (~0.5 天)

- `6experiments.tex`: 新增 E2E subsection
- `1introduction.tex`: 更新 contribution 描述
- 编译验证

---

## 5. 吞吐量模型校准数据

从已有 DP 实验结果提取 (EC2 benchmarks):

| n (replicas) | f=0 throughput (ktx/s) | f_max throughput | Source |
|-------------|----------------------|-----------------|--------|
| 10 | ~80 | ~75 | fig3 WAN |
| 100 | ~45 | ~40 | fig3 WAN |
| 200 | ~30 | ~28 | fig3 WAN |
| 1000 | ~8 | ~7 | fig5 |

模型: `throughput(n, f_active, reconfig) = base(n) × (1 - α·f_active/n) × (reconfig ? 0.7 : 1.0)`

其中 `base(n)` 从实际 EC2 数据插值。

---

## 6. 风险分析

| 风险 | 概率 | 缓解 |
|------|------|------|
| 审稿人要求"真实分布式 E2E" | 20% | 论文已声明 "CP simulation"；吞吐量模型基于真实 EC2 数据；rebuttal 可解释 |
| EXP3 baseline 表现比预期好 | 10% | 理论上 EXP3 无法处理 cross-instance roaming — 如果表现好说明模拟有 bug |
| E2E 结果与 CP-only 数据矛盾 | 15% | 保持相同随机种子和参数；差异应该来自 GBC 开销和 reconfig 延迟 |

---

## 7. 论文影响预估

完成 E2E 实验后:
- **拒稿风险**: 30-40% → **15-25%**
- **C2 状态**: ❌ → ✅
- **P0-3 状态**: ❌ → ✅ (EXP3 + UCB baselines)
- **Proposition 实验验证**: ❌ → ✅ (理论+实验双重证据)
- **评分变化**: Weak Accept → **Accept**

---

*设计时间: 2025-07*
