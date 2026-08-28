# Paper-Backup E1–E3 开发 Prompt

> 用途：将本文档完整交给下一位项目开发助理。
>
> 目标：基于当前 `UnifiedRawSOH` 的真实实现，新建隔离的 Paper-Backup namespace，并完成 E1 main estimation、E2 terminal-versus-full、E3 strategy-specific-versus-pooled 的代码、配置、测试、README 和 smoke 骨架。
>
> 本阶段不开发 E4 robustness，不启动正式多 seed 长时间 GPU 训练，不修改 Paper-v1/Paper-v2 的科学协议和已有结果。

## 1. 任务目标

你在仓库根目录工作：

```text
/data1/chenyanxi/lb_project/code/UnifiedRawSOH
```

请直接完成可运行开发，不要只输出计划或伪代码。开发完成后，Paper-Backup 应形成以下独立链路：

```text
E1: five-family terminal-raw benchmark
    HI-MLP / Raw CNN / LSTM / Transformer / Vanilla Mamba / Ours

E2: full-versus-terminal controlled comparison
    Full Vanilla / Terminal Vanilla / CC-only / CV-only / Terminal Ours

E3: within-family deployment comparison
    strategy-specific estimators / one dataset-pooled estimator
```

Human-facing paper label 使用 `Paper-Backup`，Python/config/path namespace 统一使用 `paper_backup`。

## 2. 开始前必须完整阅读

至少阅读：

```text
docs/paper_backup/PAPER_STORY.md
PAPER_STORY.md
EXPERIMENT_PLAN.md
VERSION_INFO.md
README.md

configs/paper_v1/README.md
configs/paper_v1/common/xjtu_raw_c5b_base.json
configs/paper_v1/e1_raw_soh_learning/benchmark/README.md
configs/paper_v1/e1_raw_soh_learning/benchmark/raw_mamba_*.json
configs/paper_v1/e1_raw_soh_learning/benchmark/pinn4soh_onlyf_*.json
configs/paper_v1/e1_raw_soh_learning/ablation/README.md
configs/paper_v1/e1_raw_soh_learning/ablation/raw_ours_no_cycle_aux_*.json

scripts/paper_v1/e1_raw_soh_learning/README.md
scripts/paper_v1/e1_raw_soh_learning/benchmark/run_raw_mamba_benchmark.sh
scripts/paper_v1/e1_raw_soh_learning/benchmark/run_onlyf_benchmark.sh
scripts/paper_v1/e1_raw_soh_learning/ablation/run_raw_ours_no_cycle_aux.sh
scripts/run_seed_batch.sh
scripts/resolve_python_bin.sh
scripts/setup/check_e1_dataset_ready.py

models/c5b_model.py
models/raw_soh_model.py
models/baselines/pinn4soh_no_leak_onlyf.py
trainers/c5b_trainer.py
trainers/pinn4soh_baseline_trainer.py
datasets/base.py
datasets/loaders.py
datasets/baseline_loaders.py
datasets/xjtu.py
datasets/mit.py
datasets/smarthealth.py
datasets/splits.py
datasets/soh_labels.py
evaluation/metrics.py

tests/test_c5b_contract.py
tests/test_paper_v1_interfaces.py
tests/test_e1_no_cycle_aux_config.py
```

同时先运行 `git status --short`。已有修改属于用户，不得回滚、覆盖或顺手清理。若必须修改共享文件，先检查已有 diff，只做带明确 Paper-Backup 开关的最小增量。

## 3. 开发前只读审计

先完成并记录以下审计，再设计实现：

1. 五个 family 的 canonical terminal raw product、Only-F product、split JSON 和 SOH label 是否可用；
2. XJTU、LISHEN、CATL 是否真实保留了可构造 full continuous CC+CV 的 point-level source；
3. full source 是否包含可审计的 physical battery ID、cycle identity、voltage、current、time、temperature 和 label linkage；
4. 当前 terminal preprocessing 的精确物理窗口、边界包含规则和 resampling 语义；
5. 每个 strategy 的 development/test physical battery 数量；
6. 当前主训练入口能否扩展多个 raw baseline，还是需要 Paper-Backup 独立入口。

审计结果写入：

```text
docs/paper_backup/DATA_AND_IMPLEMENTATION_AUDIT.md
```

如果 full CC+CV source 不存在、不完整或无法与 SOH label 和 battery-level split 可靠配对：

- 不得用 terminal product 冒充 full data；
- 不得从 cycle 顺序或文件名猜测身份；
- 将相应 E2 family 标记为 `blocked_by_data`；
- 继续开发不依赖该数据的 model/config/test 骨架和其他可运行 family。

## 4. 硬性隔离要求

### 4.1 保护已有论文版本

不得删除、移动或改写：

```text
configs/paper_v1/
configs/paper_v2/
scripts/paper_v1/
scripts/paper_v2/
docs/paper_v2/
outputs/Paper-v1/
outputs/Paper-v2/
results/Paper-v1/
results/Paper-v2/
```

不得改变现有 `PaperRawSOHModel` 默认结构、state-dict key/shape、V1/V2 config 解析结果、loader 默认语义和历史输出路径。

### 4.2 新代码位置

优先使用：

```text
configs/paper_backup/
scripts/paper_backup/
models/paper_backup/
datasets/paper_backup/
trainers/paper_backup/
evaluation/paper_backup/
tests/paper_backup/
docs/paper_backup/
```

允许复用共享数据、preprocessing、split、SOH label、metrics 和现有 Ours/Only-F 实现，但 Paper-v1/Paper-v2 不得反向 import Paper-Backup。

共享文件只有在无法通过 wrapper/adapter 实现时才允许修改，并必须满足：

1. 新行为由明确的 `paper_backup` config/entry point 启用；
2. 原默认行为不变；
3. 添加 V1/V2 regression test；
4. 在 `docs/paper_backup/DATA_AND_IMPLEMENTATION_AUDIT.md` 记录原因和影响。

### 4.3 独立入口和输出

新增独立入口，例如：

```text
scripts/paper_backup/train.py
scripts/paper_backup/summarize.py
```

Paper-Backup launcher 不继续向根目录 `main.py` 添加难以隔离的分支。运行只能写入：

```text
outputs/Paper-Backup/<experiment>/<model>/<data>/runtime_<time>/seed_<seed>/
```

入口必须校验 `output.paper_version == "Paper-Backup"`。任何 Paper-Backup config 指向 V1/V2 输出目录时立即失败。

## 5. 全局科学 contract

### 5.1 推理输入

Paper-Backup proposed/raw models 只允许接收当前 cycle 的：

- raw voltage/current；
- relative time；
- temperature/DeltaT/T0，保持 Ours 已验证语义。

禁止作为 model input 或 auxiliary target：

- cycle index / absolute cycle count；
- normalized cycle/lifetime coordinate；
- observed lifetime、EOL、future cycles；
- lifetime/degradation auxiliary；
- predicted cycle/lifetime coordinate；
- strategy/condition ID；
- dataset/domain ID；
- physical battery/cell ID；
- `Q_ref`、nominal capacity 或 true SOH；
- IC/DVA/handcrafted HI，除独立 HI-MLP baseline 外。

Ours 必须使用 SOH-only loss，并显式校验：

```text
model.use_cycle_prediction == false
model.use_predicted_cycle_for_soh == false
train.lambda_cycle == 0
```

### 5.2 Split

- test 使用严格 physical battery-level holdout；
- train/validation 可以在 development batteries 内按 cycle 划分；
- test batteries 不参与 early stopping、hyperparameter selection 或 normalization fitting；
- 所有对比方法使用同一 split provenance；
- validation 与 train 来自相同 development batteries 是允许且预期的，不要擅自改为独立 validation-cell protocol。

### 5.3 指标

训练时沿用各 family 已确认的 validation selection metric。正式结果至少保存：

- cycle-level predictions，带 physical battery/strategy provenance；
- per-battery RMSE/MAE；
- battery-macro RMSE/MAE；
- per-strategy metrics；
- seed-level metrics；
- batch mean/std summary。

Pooled-cycle metric 只能作为补充。

## 6. 建议目录骨架

允许根据现有 API 小幅调整文件名，但职责必须清晰：

```text
configs/paper_backup/
  README.md
  common/
    terminal_raw_base.json
    soh_only_contract.json
  e1_main_estimation/
    ours/
    vanilla_mamba/
    raw_cnn/
    lstm/
    transformer/
    hi_mlp/
  e2_charging_information/
    full_vanilla/
    terminal_vanilla/
    terminal_cc_only/
    terminal_cv_only/
    terminal_ours/
  e3_strategy_pooling/
    xjtu/
    lishen/

models/paper_backup/
  __init__.py
  sequence_baselines.py
  model_factory.py
  README.md

datasets/paper_backup/
  __init__.py
  sequence_views.py
  full_cccv.py
  strategy_pooling.py
  README.md

trainers/paper_backup/
  __init__.py
  config_contract.py
  trainer.py

evaluation/paper_backup/
  __init__.py
  aggregation.py
  comparisons.py

scripts/paper_backup/
  train.py
  run_e1.sh
  run_e2.sh
  run_e3.sh
  summarize.py
  README.md

tests/paper_backup/
  __init__.py
  test_config_contract.py
  test_model_contract.py
  test_split_leakage.py
  test_sequence_views.py
  test_e2_pairing.py
  test_e3_pooling.py
  test_v1_v2_regression.py
  smoke_test.py
```

不要为了匹配此建议而移动现有共享文件。

## 7. P0：Paper-Backup contract 与共享骨架

在 E1–E3 前先完成最小 P0，但 P0 不是新的论文实验。

### 7.1 Config contract

入口至少校验：

- `output.paper_version == "Paper-Backup"`；
- experiment ID 只能是 `e1_main_estimation`、`e2_charging_information`、`e3_strategy_pooling`；
- raw/proposed model 没有 cycle/lifetime input 或 auxiliary；
- split file 存在且 battery IDs 在 train/validation/test 之间满足声明协议；
- test battery 与 development battery 无重叠；
- output path 属于 `Paper-Backup`；
- model type/variant 不允许静默 fallback；
- full input config 不允许读取 terminal-only product；
- strategy-pooled config 不允许将 strategy ID 注入 model input。

### 7.2 Model factory

Paper-Backup factory 需要显式构建：

- existing phase-specific `PaperRawSOHModel` SOH-only wrapper，Ours；
- joint Vanilla Mamba；
- single-stream Mamba；
- Raw CNN；
- LSTM；
- Transformer；
- existing Only-F/HI-MLP reference。

不存在的 baseline 必须实际实现并测试，不能仅用配置名把 Ours 包装成不同模型。

### 7.3 Unified batch contract

为 raw sequence models 建立明确 batch schema，至少包含：

```text
input tensors
valid lengths or masks where required
SOH target
physical battery ID (evaluation only)
strategy/condition (sampling/evaluation only)
cycle identity (prediction provenance only)
input view ID: full / terminal_joint / terminal_cc / terminal_cv / terminal_phase
```

metadata 不得进入模型 forward，除非是被允许的 raw time/temperature tensor。

## 8. E1：Main terminal-raw benchmark

### 8.1 Matrix

对 XJTU、MIT、LISHEN、CATL、EVE 分别开发：

| Model ID | Input | Implementation requirement |
|---|---|---|
| `HI-MLP` | handcrafted terminal statistics | 复用现有 no-leak Only-F baseline |
| `RawCNN` | terminal raw CC/CV | Ruan-like representative 1D CNN，不伪称原论文完全复现 |
| `LSTM` | terminal joint raw sequence | 标准 many-to-one sequence baseline |
| `Transformer` | terminal joint raw sequence | 带 mask/position/time 语义的标准 encoder baseline |
| `VanillaMamba` | terminal joint raw sequence | 一个连续 joint-sequence Mamba，不使用 Ours 双 branch/bridge |
| `PhaseMamba` | terminal phase-separated raw sequence | 复用 Ours，SOH-only、无 lifetime auxiliary |

### 8.2 公平协议

- 所有 raw baselines 读取同一 canonical terminal cycles；
- 使用相同 raw channels、labels、splits 和 physical normalization；
- joint models 将 terminal CC 与 terminal CV 按真实时间顺序拼接；
- joint model 不使用显式 strategy/domain/cell ID；
- baseline hidden size/depth 记录在 config，不为每个 test set 单独调参；
- checkpoint selection 和 epoch/patience budget 保持一致；
- Ours 不继承任何 cycle/lifetime head 或 loss。

### 8.3 E1 验收

- 六类 model 都有真实、可区分的 forward path；
- 五个 family 的 config matrix 完整；
- launcher 支持 domain/model/seed dry-run；
- CPU reference smoke 覆盖 shape、loss、backward、checkpoint save/load；
- official Mamba CUDA path 至少完成最小 smoke，未获用户许可不得启动正式训练；
- summary 能输出 family × model 表和 per-battery result。

## 9. E2：Terminal observation versus full charging

### 9.1 Matrix

重点为 XJTU、LISHEN、CATL：

| Model ID | Input view | Model |
|---|---|---|
| `Full-VanillaMamba` | continuous full CC+CV | joint Vanilla Mamba |
| `Terminal-VanillaMamba` | terminal CC+CV | 同一 Vanilla Mamba family |
| `Terminal-CC-Mamba` | terminal CC only | single-stream Mamba |
| `Terminal-CV-Mamba` | terminal CV only | single-stream Mamba |
| `Terminal-PhaseMamba` | terminal CC+CV | Ours |

Full CC+CV 是连续序列，直接使用 Vanilla Mamba；当前阶段不开发 full phase-aware Mamba。

### 9.2 Data views

同一 accepted physical cycle 尽量生成以下 paired views：

```text
full_cccv
terminal_joint
terminal_cc
terminal_cv
terminal_phase
```

必须保留相同：

- physical battery ID；
- cycle identity；
- SOH label；
- train/validation/test assignment。

`full_cccv` 必须来自真实完整 point-level charging record。Terminal view 使用已确认的 canonical physical window，不重新搜索范围。

### 9.3 Full/terminal matching

E2 的主比较应使用 full 和 terminal 都存在的 matched physical cycles。若完整轨迹缺失导致 cycle 被排除：

- 输出 exclusion reason；
- 记录每个 battery/strategy 的 matched coverage；
- 所有 E2 model 使用同一 matched cohort；
- 不允许不同 model 各自静默丢样本。

Full sequence 可使用 padding/mask、length-aware batching 或可审计的 resampling，但不得压缩到明显破坏 reference 的程度。相关选择只能用 development data 确定。

### 9.4 E2 输出

除通用指标外，生成：

- terminal actual duration distribution；
- terminal/full raw point ratio；
- terminal/full time-span ratio；
- `Full Vanilla → Terminal Vanilla` per-battery difference；
- `Terminal Vanilla → Terminal Ours` per-battery difference；
- `Full Vanilla → Terminal Ours` per-battery system difference；
- CC-only、CV-only、Ours 的互补性表。

如实现 confidence interval，resampling unit 必须是 physical battery，不能把同一 battery 的 cycles 当成独立样本。

### 9.5 E2 验收

- full source provenance 审计完成；
- full/terminal paired-cycle invariant 有自动测试；
- 五种 view/model config 对三个重点 family 完整，数据缺失者明确标记 blocker；
- summary 能独立计算三组主要比较；
- 不包含 window-position search 或 full phase-aware model。

## 10. E3：Strategy-specific versus dataset-pooled

### 10.1 范围

只开发：

- XJTU：2C / 3C / R2.5 / R3 / RW / Satellite；
- LISHEN：canonical C-rate × DOD conditions。

使用 E1 的 terminal `PhaseMamba` Ours 和同一 SOH-only contract。

### 10.2 Strategy-specific

每个 strategy：

- 只使用该 strategy 的 development batteries 训练；
- 使用该 strategy 固定的 held-out test battery/batteries 测试；
- validation cycles 仍可来自同一 strategy 的 development batteries；
- 输出独立 model/checkpoint/result ID。

### 10.3 Dataset-pooled

每个 family 训练一个 pooled estimator：

- development set 是该 family 所有 strategies 的 development batteries 并集；
- test set 是各 strategy 原有 held-out test batteries 的并集；
- model forward 不接收 strategy ID；
- strategy metadata 只用于 train balancing 和 evaluation；
- 默认使用 strategy → battery → cycle 的分层平衡采样，避免长寿命电池或大 strategy 支配训练；
- 同时保存 sampling audit。

不要把 XJTU 和 LISHEN 合并训练；这不是 cross-domain experiment。

### 10.4 E3 输出

生成：

- 每个 strategy 的 specific vs pooled RMSE/MAE；
- strategy-macro average；
- worst-strategy metric；
- per-battery result；
- pooled-specific paired difference；
- deployed model count；
- total checkpoint storage；
- inference-time model-selection requirement。

### 10.5 E3 验收

- strategy metadata 从 canonical source 显式得到，不从 cycle order 猜测；
- specific/pooled 使用相同 test batteries；
- pooled model input 不包含 strategy ID；
- sampler audit 能证明 strategy/battery sampling 语义；
- launcher 支持 XJTU/LISHEN、specific/pooled、seed dry-run；
- summary 生成 per-strategy 和 macro deployment comparison。

## 11. 测试要求

所有测试优先使用小型 synthetic tensors/metadata，不依赖正式训练。至少覆盖：

### Config / isolation

- Paper-Backup output namespace；
- experiment/model/view ID 合法性；
- lifetime/cycle auxiliary 被拒绝；
- full config 读取 terminal-only source 被拒绝；
- V1/V2 config 解析和 state-dict contract 不变。

### Models

- 六类 E1 model 参数和 forward path 可区分；
- output shape、finite loss、backward；
- checkpoint round-trip；
- Vanilla Mamba 没有 CC/CV dual branch 或 bridge；
- Ours 的 cycle/lifetime head 关闭。

### Data / leakage

- physical test batteries 与 development batteries 无重叠；
- train/validation 同属 development cohort 被允许；
- full/terminal views label、battery、cycle、split identity 一致；
- metadata 未进入 model features；
- E3 pooled test cohort 等于各 strategy test cohort 并集。

### Evaluation

- battery-macro 不被长寿命 battery 的 cycle 数量加权；
- strategy-macro 计算正确；
- paired comparison 只使用共同 batteries/cycles；
- missing/blocked family 不被静默计为零或跳过。

## 12. Launcher 与运行安全

Shell launcher 应：

- 使用 `scripts/resolve_python_bin.sh`；
- 默认支持 `DRY_RUN=1`；
- 显式选择 experiment/model/family/seed；
- 启动前运行 config 和 dataset readiness check；
- 一个 seed 一个独立目录；
- 不覆盖已有 runtime；
- 正式 CUDA/Mamba job 缺少 backend 或 dataset 时 fail fast；
- 不自动下载依赖；
- 不启动用户未要求的长时间训练。

## 13. README 与状态记录

开发完成后至少更新：

```text
configs/paper_backup/README.md
scripts/paper_backup/README.md
models/paper_backup/README.md
datasets/paper_backup/README.md
docs/paper_backup/DATA_AND_IMPLEMENTATION_AUDIT.md
docs/paper_backup/DEVELOPMENT_STATUS.md
```

`DEVELOPMENT_STATUS.md` 使用表格记录：

```text
component | status | config | command | output | tests | last verified | limitation
```

状态必须区分：

- `planned`；
- `implemented`；
- `unit-tested`；
- `smoke-tested`；
- `runnable`；
- `blocked_by_data`；
- `formally_trained`。

没有正式结果时不得写出 paper performance claim。

## 14. 当前明确不开发

- E4 robustness；
- noise/missing/downsampling/boundary perturbation；
- early/middle/random segment comparison；
- automatic window selection；
- dense observation-budget sweep；
- full phase-aware Mamba；
- cross-domain/unified/LODO/transfer/MoE；
- lifetime auxiliary；
- 正式大规模 GPU experiment；
- 论文结果表中的虚构数字。

若实现过程中发现这些内容有帮助，只记录为 future/reviewer-requested work，不要擅自扩展当前 scope。

## 15. 最终交付与验收顺序

按以下顺序工作并逐阶段验证：

1. 完成只读 data/implementation audit；
2. 建立 Paper-Backup config contract、entry point 和 regression protection；
3. 实现 E1 baseline/model/config/launcher/evaluation；
4. smoke E1；
5. 实现 E2 full/terminal paired views 和 matrix；
6. smoke E2，数据缺失项明确记录 blocker；
7. 实现 E3 specific/pooled composition、sampler 和 comparison；
8. smoke E3；
9. 运行所有 Paper-Backup tests 和相关 V1/V2 regression tests；
10. 更新 README、audit、status，最后检查 `git diff --check` 和 `git status --short`。

最终回复用户时只报告：

- 实际新增/修改的文件；
- E1/E2/E3 各自达到的真实状态；
- 执行过的测试和 smoke；
- 数据或 backend blocker；
- 未启动正式训练；
- 没有改动 Paper-v1/Paper-v2 的既有科学协议。

