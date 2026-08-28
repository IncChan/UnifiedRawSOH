# Paper-Backup：Terminal Raw CC/CV SOH Estimation

## 1. 一句话故事

仅使用当前 cycle 中协议定义的充电末端 CC/CV 原始电压、电流、相对时间与温度动态，在不构造健康指标、不输入循环次数、也不使用任何 lifetime information 或 lifetime auxiliary supervision 的前提下，学习准确的电池 SOH estimator。

论文核心不是 Mamba 本身，也不是寻找最优充电区间，而是：

> 当完整充电轨迹经常无法获得或没有被连续保存时，协议定义的 terminal CC/CV 数据能否支持接近 full CC+CV reference 的 SOH 估计；针对 CC 与 CV 不同控制机理设计的 phase-aware raw representation 是否能缩小有限观测带来的性能差距？

## 2. 与 Paper-v1 / Paper-v2 的关系

Paper-Backup 是与 Paper-v1、Paper-v2 并列的备用论文路线，不替换、删除或扩展它们。

Paper-v1/Paper-v2 的主线是：

```text
Raw → Unified → Reusable
```

Paper-Backup 主动退出以下问题：

- cross-domain transfer；
- unified five-domain model；
- LODO；
- zero-shot / few-shot；
- MoE；
- domain generalization；
- shared/private representation；
- universal or transferable battery representation。

Paper-Backup 只研究每个 battery family 内的 terminal-raw SOH estimation，以及同一 family 内 strategy-specific 与 dataset-pooled deployment 的取舍。

## 3. 应用边界与任务定义

现实中电池不一定从统一初始 SOC 开始充电，完整 CC/CV 历史可能无法获得或没有被连续保存。但当一次充电进入上截止电压附近并完成 CV 阶段时，充电末端数据相对稳定且更容易形成统一观测。

Paper-Backup 使用由测试协议预先定义、而不是由模型搜索得到的 terminal window：

- XJTU terminal CC：约 4.0–4.2 V；
- 其他 battery families terminal CC：约 3.45–3.6 V，最终精确边界以 canonical preprocessing contract 为准；
- terminal CV：充电电流约从 0.25C 衰减至 0.05C；
- 输入包含当前 cycle 的 raw voltage/current、relative time 和 temperature；
- 只有实际进入规定 terminal CC/CV 区间并满足数据完整性要求的 charging cycle 才属于方法适用范围。

论文不声称 terminal 是完整充电轨迹中理论最优的位置，也不研究自动窗口选择、early/middle/random position comparison 或最小可用窗口搜索。

## 4. 严格无泄漏边界

Proposed raw model 在训练、验证和推理中都不得接收：

- cycle index 或 absolute cycle count；
- normalized cycle coordinate；
- observed lifetime、EOL 或 future cycles；
- lifetime/degradation auxiliary target；
- predicted lifetime coordinate；
- battery/cell ID；
- strategy/condition ID；
- dataset/domain ID；
- `Q_ref`、nominal capacity 或真实 SOH 作为输入；
- IC、DVA 或 handcrafted health indicators。

Paper-Backup proposed model 使用 SOH-only supervision。已有 Paper-v1 中与 cycle/lifetime auxiliary 有关的实现不得被继承为 Paper-Backup 默认行为。

Battery/strategy/dataset metadata 只允许用于：

- split construction 和 leakage audit；
- train sampling；
- strategy-specific job selection；
- metric aggregation 和结果分析。

## 5. 数据与模型单位

使用五个 battery families：

- XJTU；
- MIT；
- LISHEN；
- CATL；
- EVE。

它们在本文中的作用是验证同一套 methodology 在多个异构 battery families 上能否重复成立。默认训练方式是：

```text
XJTU   → XJTU model
MIT    → MIT model
LISHEN → LISHEN model
CATL   → CATL model
EVE    → EVE model
```

因此本文可以声称：

> the same methodology is independently validated within five heterogeneous battery families

但不能声称：

> one model generalizes to unseen battery families

或：

> the same weights work across five battery families

## 6. 方法故事

CC 和 CV 虽然属于连续的 CC/CV charging process，但对应不同控制机制和主要动态量：

- CC：current approximately constant，健康信息主要体现在 terminal voltage evolution；
- CV：voltage approximately constant，健康信息主要体现在 current relaxation；
- temperature：提供额外 thermal response；
- CC 末端状态会影响随后的 CV relaxation。

因此 proposed model 使用：

```text
Terminal CC raw dynamics
        ↓
   CC Mamba encoder
        ↓
   CC→CV state bridge
        ↓
Terminal CV raw dynamics → CV Mamba encoder
        ↓
 CC/CV feature fusion + temperature
        ↓
       SOH
```

Mamba 的角色是自动从 raw sequence 中提取 degradation-related representation。方法贡献应表述为：

> phase-aware selective state-space representation learning for terminal raw CC/CV dynamics

而不是：

> the first Mamba model for battery SOH estimation

## 7. 论文证据链

### E1 — Main terminal-raw SOH estimation

问题：在相同 terminal CC/CV 输入和严格 battery-level test split 下，phase-aware raw model 是否优于 representative feature-based 和 raw-sequence baselines？

五个 battery families 分别比较：

- PINN4SOH-like HI-MLP；
- Raw CNN / Ruan-like transient CC/CV CNN；
- LSTM；
- Transformer；
- joint-sequence Vanilla Mamba；
- Phase-specific Mamba，Ours。

E1 证明 terminal raw SOH estimation 的准确性，以及 proposed phase-aware representation 相对通用 sequence model 的价值。

### E2 — Terminal observation versus full charging

问题：限制为 terminal CC/CV 后，通用模型损失多少性能；phase-aware model 能否缩小这一差距并接近 full charging reference？

重点使用 XJTU、LISHEN、CATL：

| ID | Input | Model | Purpose |
|---|---|---|---|
| E2-A | Full CC+CV | Vanilla Mamba | 连续完整充电轨迹 reference |
| E2-B | Terminal CC+CV | Vanilla Mamba | 隔离 terminal observation 带来的变化 |
| E2-C | Terminal CC only | Single-stream Mamba | 检查 CC 单阶段信息 |
| E2-D | Terminal CV only | Single-stream Mamba | 检查 CV 单阶段信息 |
| E2-E | Terminal CC+CV | Phase-specific Mamba, Ours | 最终 terminal-only proposed system |

三组主要解释是：

1. E2-A vs E2-B：相同 Vanilla Mamba 下，限制输入范围造成的性能变化；
2. E2-B vs E2-E：相同 terminal input 下，phase-aware modeling 带来的增益；
3. E2-A vs E2-E：最终 terminal-only system 与 conventional full-trajectory reference 的系统级差距。

Full CC+CV 是连续序列，因此使用 joint Vanilla Mamba，不额外构造 full phase-aware model。相应结论必须写成 terminal Ours 接近 full-trajectory Vanilla Mamba reference，不能扩大为 terminal input 对所有可能 full-trajectory methods 都非劣。

E2 只描述 terminal window 的实际持续时间、原始点数和相对 full trajectory 的保留比例，不进行位置搜索或大规模 observation-budget sweep。

### E3 — Strategy-specific versus dataset-pooled

问题：同一个 battery family 面对不同 operating strategies 时，是否需要部署 condition-specific estimators，还是一个不接收 strategy ID 的 family-level pooled estimator 已经足够？

重点使用：

- LISHEN：C-rate × DOD conditions；
- XJTU：2C / 3C / R2.5 / R3 / RW / Satellite。

比较：

```text
Strategy-specific:
each strategy's development batteries → one estimator

Dataset-pooled:
all development batteries in one family → one estimator
```

Pooled model 不输入 strategy ID。Strategy metadata 只用于 balanced sampling 和 per-strategy evaluation。评价必须同时报告每个 strategy、strategy-macro average、worst-strategy result，以及部署模型数量和存储成本。

E3 不是 transfer learning，也不预设 pooled 一定更好。pooled 更好、接近、略差或明显更差都有可解释的部署含义。

## 8. 统一实验协议

- test set 使用严格 physical battery-level holdout；
- train/validation 可以在 development batteries 内按 cycle 划分；
- test batteries 不参与 early stopping、hyperparameter selection 或 normalization fitting；
- 所有方法共享相同 split、label、window 和 checkpoint-selection metric；
- raw methods 使用相同可用通道；
- 每个 family 独立训练；
- 使用多个固定随机种子；
- 主结果以 per-battery metric 和 battery-macro aggregation 为准；
- pooled-cycle metric 只能作为补充，不能取代 battery-level result；
- normalization 使用固定物理范围或仅由 development data 得到的统计量；
- Paper-Backup 所有 proposed runs 使用 SOH-only loss。

验证协议应明确写成：

> Validation cycles may originate from the same development batteries used for training, whereas all reported test performance is evaluated on strictly unseen physical batteries.

## 9. 统计报告

至少报告：

- per-test-battery RMSE 和 MAE；
- battery-macro RMSE 和 MAE；
- 多随机种子的 mean ± standard deviation；
- battery-level confidence interval 或 hierarchical bootstrap；
- E2 中 `Terminal Ours - Full Vanilla` 的 per-battery performance difference；
- E3 中 pooled 相对 strategy-specific 的 per-strategy difference。

若使用 practical non-inferiority tolerance，必须在查看正式 test results 前定义，并明确其是系统级 `Terminal Ours` 相对 `Full Vanilla` 的 tolerance。

## 10. 当前不做的内容

当前开发和首轮正式实验不包括：

- E4 robustness；
- noise、missing points、downsampling 或 phase-boundary perturbation；
- early/middle/random position comparison；
- automatic window selection；
- dense observation-budget sweep；
- full phase-aware Mamba；
- cross-domain / LODO / adaptation；
- lifetime auxiliary；
- 大规模 SOTA leaderboard。

Robustness 仅作为审稿后储备，不在当前摘要、贡献或结论中宣称。

## 11. 建议论文 claim

保守主 claim：

> Without handcrafted health indicators or lifetime information, phase-aware selective state-space learning accurately estimates battery SOH from protocol-defined terminal CC/CV raw dynamics.

如果 E2 结果稳定且统计证据支持，可使用：

> Across five independently evaluated battery families, the proposed terminal-only estimator achieves accuracy comparable to a full-trajectory Vanilla Mamba reference while requiring only terminal charging observations.

建议标题方向：

> **Phase-Aware Mamba for Battery State-of-Health Estimation from Terminal CC–CV Charging Dynamics**

