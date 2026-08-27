# Paper V2 P0–P2 开发 Prompt

> 用途：将本文档完整交给下一位开发助理。
>
> 目标：在不破坏 Paper V1 的前提下，完成 Paper V2 的 P0–P2 代码与协议骨架：V2 隔离、Residual MoE/Dense Adapter、分层采样、hierarchical episodic DG，以及五折 zero-cell LODO 可运行链路。
>
> 本阶段不实现 P3 one-trajectory router adaptation，不实现 P4 企业实验，不启动正式长时间 GPU 训练。

## 1. 你的任务

你在仓库根目录工作：

```text
/data1/chenyanxi/lb_project/code/UnifiedRawSOH
```

请直接完成 P0、P1、P2 的实际开发、测试、文档更新和 smoke 验证，而不是只给计划或伪代码。在开发中保持以下科学主线：

```text
BOL health-aligned RawMamba Base
→ parameter-matched Dense Adapter / Residual MoE
→ hierarchical episodic domain generalization
→ strict five-fold zero-cell LODO
```

要求每一个阶段都能独立验证，不允许把 MoE、DG 和后续 adaptation 一次性糊在同一条无法消融的路径中。

## 2. 必须先读的现有文件

开始修改前，完整阅读与任务直接相关的文件，至少包括：

```text
docs/paper_v2/P0_P2_DEVELOPMENT_PROMPT.md
configs/paper_v2/README.md
configs/paper_v2/common/bol_soh_base.json
configs/paper_v2/e2_full_domain/raw_mamba_domain_balanced.json
configs/paper_v2/e3_lodo_zero_cell/base.json
configs/paper_v2/e3_lodo_zero_cell/lodo_*.json
scripts/paper_v2/README.md
scripts/paper_v2/run_bol_soh_retraining.sh
models/c5b_model.py
models/raw_soh_model.py
trainers/c5b_trainer.py
datasets/loaders.py
datasets/soh_labels.py
evaluation/paper_v2_metrics.py
tests/test_paper_v2_bol.py
tests/paper_v2_smoke_test.py
```

同时检查当前 `git status --short`。仓库可能已经有用户未提交的修改；这些改动属于用户，不得回滚、覆盖、重写或顺手整理。如果必须修改与用户脏改动重叠的文件，先识别已有 diff，仅做最小、可区分的增量修改；无法安全合并时停止并向用户说明。

## 3. 已有基础：不要重新实现

当前仓库已经有：

- Paper V2 BOL 标签：`bol_peak_mean_top5_first100_v1`；
- V2 E1 single-domain Base/FeatureMLP 配置；
- V2 E2 full-domain RawMamba Base 配置；
- V2 E3 五折 zero-cell RawMamba/FeatureMLP Base 配置；
- `outputs/Paper-v2/` 输出命名空间；
- hierarchical metric 生成与汇总基础；
- no-cycle-aux RawMamba Base；
- 已有 domain/battery 平衡采样和 LODO loader 基础。

应复用这些经验证接口，但不得为了“更干净”而重构 V1 路径。

## 4. 硬性隔离约束

### 4.1 V1 保护

不得删除、移动或改写以下路径中的现有文件，除非经用户明确同意：

```text
configs/paper_v1/
scripts/paper_v1/
evaluation/paper_v1/
outputs/Paper-v1/
results/Paper-v1/
```

不得改变 V1 `PaperRawSOHModel` 默认结构、state-dict key/shape、V1 config 解析结果、loader 默认语义或输出路径。

### 4.2 V2 新实现位置

原则上将新实现放在：

```text
models/paper_v2/
datasets/paper_v2/
trainers/paper_v2/
evaluation/paper_v2/
configs/paper_v2/
scripts/paper_v2/
tests/paper_v2/
docs/paper_v2/
```

允许 import 现有 V1/共享接口，但不要反向让 V1 代码 import V2 实现。

如果确实必须修改 `datasets/loaders.py` 等共享文件，必须同时满足：

1. 新行为由明确的 Paper V2 config 开关启用；
2. V1 config 的默认路径不变；
3. 有针对 V1 不变性的回归测试；
4. 在对应 V2 README 中记录该共享改动。

### 4.3 独立入口

新增 V2 Python 入口，例如：

```text
scripts/paper_v2/train.py
```

新 P1/P2 启动脚本只能调用 V2 入口。不继续往根目录 `main.py` 中增加 V2 分支。现有 BOL baseline 启动器先保留，不做破坏性迁移。

### 4.4 输出隔离

新运行只能写入：

```text
outputs/Paper-v2/<experiment>/<model>/<data>/runtime_<time>/seed_<seed>/
```

启动前校验 `output.paper_version == "Paper-v2"`。配置为 V2 却指向 Paper-v1 输出时必须立即失败。

## 5. 科学与数据协议

### 5.1 推理输入

模型推理只允许使用当前 cycle 的：

- CC/CV voltage-current raw signal；
- relative time；
- temperature/DeltaT/T0，与已有 Base 保持一致。

router 只能接收 `z_base`，不得接收：

- dataset/domain ID；
- strategy/condition ID；
- battery/cell ID；
- cycle index、normalized lifetime 或 EOL；
- `Q_ref`、nominal capacity 或真实 SOH。

domain/strategy/cell 信息只可用于采样、episode 构造、split 校验、评价聚合和路由分析。

### 5.2 标签

使用已实现的 Paper V2 BOL SOH 标签规则，不在 P0–P2 内再改标签定义。`Q_ref` 只是标签 provenance，不得进入输入或 normalization。

### 5.3 层次

数据层次固定为：

```text
dataset/domain → strategy group → physical cell → cycle
```

`strategy group` 默认来自样本的稳定 `condition` 元数据：

- XJTU：2C / 3C / R2.5 / R3 / RW / satellite；
- MIT：source-date group；
- SmartHealth：C-rate/DOD condition。

如果某个 adapter 的 `condition` 不足以表示 strategy，不得用 cycle 顺序或文件名暗中推测；应增加显式、可审计的 metadata mapping，并记录在 README/config 中。

### 5.4 真实 LODO 边界

每个五折 LODO fold 中，真实 target dataset 不得参与：

- train/validation；
- early stopping/checkpoint selection；
- expert 数量、Top-k、loss weight 选择；
- normalization/statistics fitting；
- episode 构造；
- router/expert load-balancing 统计拟合。

target 在 P2 zero-cell 中只贡献固定 test cells 的最终评价。

## 6. 建议新增结构

可根据现有 API 小幅调整文件名，但责任边界应保持清晰：

```text
models/paper_v2/
  __init__.py
  residual_moe.py
  dense_adapter.py
  raw_mamba_moe.py
  README.md

datasets/paper_v2/
  __init__.py
  hierarchy.py
  hierarchical_sampler.py
  episodic_sampler.py
  README.md

trainers/paper_v2/
  __init__.py
  config_contract.py
  common.py
  seen_domain.py
  mldg.py
  README.md

evaluation/paper_v2/
  __init__.py
  routing.py
  README.md

scripts/paper_v2/
  train.py
  run_e2_seen_domain.sh
  run_e3_zero_cell.sh
  README.md

tests/paper_v2/
  __init__.py
  test_model_contract.py
  test_hierarchical_sampler.py
  test_episodic_dg.py
  test_lodo_leakage.py
  test_v1_regression.py
```

不要移动已有 `evaluation/paper_v2_metrics.py`、`tests/test_paper_v2_bol.py` 或现有 configs/scripts；新代码可以导入它们。

## 7. P0：隔离骨架与回归保护

### 7.1 目标

- 建立 V2 独立 Python package、训练入口和 README 索引；
- 为 V2 config 建立严格 contract validation；
- 固化 V1 不变性测试；
- 记录现有 V2 BOL baseline 的真实实现状态。

### 7.2 V2 config contract

V2 入口至少校验：

- `output.paper_version == "Paper-v2"`；
- `data.label_mode == "bol_peak_relative"`；
- `data.bol_reference_rule == "bol_peak_mean_top5_first100_v1"`；
- `model.use_cycle_prediction == false`；
- `model.use_predicted_cycle_for_soh == false`；
- `train.lambda_cycle == 0`；
- P2 LODO 中 source/target 非空、无重叠；
- target test-only 协议字段齐全；
- 输出路径属于 `Paper-v2`；
- model/trainer variant 不允许静默 fallback 到 Base。

无效配置必须 fail fast，不得自动修补或忽略关键字段。

### 7.3 V1 回归

添加不依赖正式数据/GPU 的回归测试，至少覆盖：

- V1 config 仍解析到 V1 model/trainer；
- V1 `PaperRawSOHModel` 的 state-dict keys/shapes 不变；
- V1 默认 forward 在固定 seed/eval 模式下不因 V2 import 变化；
- V1 配置不会写入 Paper-v2，V2 配置不会写入 Paper-v1。

### 7.4 P0 验收

- V2 入口可以加载并验证 existing Base config；
- `--help` 可用；
- dry-run 只打印 resolved jobs，不创建 output/checkpoint；
- 原有单元测试全部通过；
- README 标明 existing/scaffolded/smoke-tested/runnable 状态，不得夸大。

## 8. P1：Residual MoE、Dense Adapter 与 E2

### 8.1 Base 复用

复用现有 `PaperRawSOHModel.encode()` 生成 `z_base`。V2 不修改 Base 的 CC/CV branch、bridge、pooling 或 T0 语义。

V2 wrapper 应显式暴露：

```text
encode_base(...) -> z_base
compose(z_base) -> z_out + routing_aux
predict_from_composed_feature(z_out, t0) -> soh_pred
forward_with_aux(...) -> structured dict
```

`forward_with_aux` 至少返回：

```text
soh_pred
z_base
z_out
balance_loss
router_logits
router_probabilities
topk_indices
topk_weights
expert_load
expert_importance
```

对 Base/Dense 对照，返回字段尽可能保持统一，不适用的 routing 字段可为 `None`。

### 8.2 Residual MoE

实现：

\[
z_{out}=z_{base}+\sum_{k\in TopK}\alpha_k(z_{base})E_k(z_{base})
\]

默认结构：

```text
num_experts = 8
top_k = 2
expert_bottleneck_dim = 16
router_input = z_base only
expert = bottleneck residual MLP
```

约束：

- `1 <= top_k <= num_experts`；
- top-k 后权重在选中 experts 上重新归一化，每个样本权重和为 1；
- expert 的最后线性层零初始化，或使用等价的严格零输出初始化；
- 零初始化时，`z_out == z_base` 在数值容差内成立；
- router/expert 不依赖 domain 索引；
- 不得预先把 expert 绑定到数据集或 strategy；
- 提供有梯度的 load-balancing loss，并在 README 说明精确公式；
- 记录 soft importance、hard load、routing entropy 和 top-k 使用率；
- 任何一个 batch 中不能因未选 expert 或空路由导致 NaN。

balance loss 的具体定义可选用经典 importance/load 乘积形式，但必须：

1. 对 router 可导；
2. 对 batch size=1 也有定义；
3. 在代码、README 和 test 中保持一致。

### 8.3 Dense Adapter 对照

在同一 `z_base` 上插入 residual dense bottleneck adapter，作为参数量对照：

\[
z_{out}=z_{base}+A(z_{base})
\]

不允许使用 domain/strategy 输入。应在配置或构建器中显式记录 Dense 隐层维度和参数量。

以“总新增可训练参数”匹配 MoE，目标误差不超过 5%；若受整数 hidden dimension 限制无法达到，选最近值并在结果中报告精确差异，不得隐藏。

### 8.4 损失

P1 MoE-ERM：

\[
L=L_{SOH}+\lambda_{balance}L_{balance}
\]

其中 `L_SOH` 与 Base 使用相同定义。P1 不加 DG、domain adversarial、MMD/CORAL、expert decorrelation 或其他额外损失。

### 8.5 E2 配置

建立独立可比配置：

```text
Unified Base
Unified parameter-matched Dense Adapter
Unified Residual MoE-ERM
```

保持相同：

- 五个 domain；
- BOL label rule；
- no-cycle auxiliary；
- raw encoder 配置；
- train/validation/test split；
- optimizer/scheduler/early-stopping 原则；
- hierarchical metric aggregation。

E2 的 early stopping 使用 source validation domain-macro RMSE。

### 8.6 P1 测试与验收

至少覆盖：

- expert 零初始化的 Base/MoE 输出等价；
- top-k 个数、权重归一化、tensor shape；
- balance loss 有限、可 backward；
- router 和选中 experts 获得有限梯度；
- router API 不接受 domain/strategy/cycle metadata；
- Dense/MoE 新增参数量比较；
- Base/Dense/MoE 的 CPU `torch_reference` forward/backward smoke；
- checkpoint 保存后严格重载；
- V1 checkpoint/state-dict contract 不变。

P1 只需完成 wiring smoke，不要伪造 E2 正式结果。

## 9. P2：分层采样、Episodic DG 与 Zero-cell LODO

### 9.1 分层平衡采样

实现真正的四层采样，不只是给 cycle 设置一个静态 weight：

```text
均匀选 dataset/domain
→ 均匀选 strategy group
→ 均匀选 physical cell
→ 从该 cell 中均匀选 cycle
```

要求：

- 只用于 train，val/test 始终 deterministic sequential；
- 采样长度由 config 显式定义，默认可与原 train dataset 长度相同；
- 支持 seed 与 `set_epoch(epoch)`，同 seed/epoch 完全可复现；
- 不依赖 DataLoader worker 调度顺序；
- 对空 strategy/cell 显式报错；
- 输出采样 hierarchy audit：domain、strategy、cell 的 inventory 和实际抽样计数；
- 不将 target dataset 加入 source sampler。

用小型 synthetic index dataset 测试近似均衡性，不要依赖本地正式数据才能验证。

### 9.2 Episode 类型

每个真实 LODO fold 只在四个 source domains 内构造 episode。

#### Dataset-level pseudo-LODO

```text
从四个 source datasets 选一个 pseudo-target
其余三个为 meta-train
```

pseudo-target 必须整个 dataset 留出。

#### Strategy-level pseudo-domain

```text
从所有 source (dataset, strategy) environments 中
选一个完整 environment 为 pseudo-target
其余 environments 为 meta-train
```

同一 physical cell 的 cycles 不得出现在 episode 两侧。如果数据语义使一个 cell 跨多 strategy，应以 cell-disjoint 为更高优先级，并在 audit 中记录因此扩大的留出范围。

默认 episode 比例：

```text
dataset-level = 0.5
strategy-level = 0.5
```

比例可配置，但只能通过 source-internal validation 选择，不得用真实 target 结果选择。

### 9.3 First-order MLDG

实现一步 first-order MLDG：

\[
w'=w-\alpha\nabla_wL_{meta-train}(w)
\]

\[
L=L_{ERM}(w)+\beta L_{pseudo-target}(w')+
\lambda_{balance}L_{balance}
\]

关键要求：

- 一阶近似必须在代码和 README 中明确，不得写成 full second-order MAML；
- inner step 不得污染 outer optimizer state；
- fast weights 必须从当前 model 一步更新获得；
- pseudo-target gradient 按参数名正确回传/组合到原模型；
- 不为实现 MLDG 引入未经允许的新依赖；
- 对 requires-grad=false、None gradient、unused parameter 和零初始 expert 有明确处理；
- source validation 仍独立用于 early stopping，不被当作 episodic train 数据；
- 记录 ERM、meta-train、pseudo-target、balance 各自的 loss，不只记录总 loss；
- inner learning rate `alpha`、outer weight `beta`、balance weight 全部来自 config 并写入 resolved config/checkpoint。

可以采用不依赖二阶图的参数名映射/手动一阶梯度组合，但必须用小模型数值测试验证与所声称的 first-order 更新一致。

### 9.4 P2 对照组

五折 zero-cell 至少支持：

```text
Unified Base-ERM
Unified Dense Adapter-ERM
Unified MoE-ERM
Unified MoE-DG
```

Base/Dense/MoE-ERM 和 MoE-DG 必须共享相同真实 LODO split、BOL label、Base encoder 参数和评价代码。

不允许将 Base 的普通 ERM 训练偷换成 DG 训练；每个 model/trainer variant 必须在 config 和输出 manifest 中可识别。

### 9.5 LODO 输出 provenance

每个 run 至少保存：

```text
resolved_config.json
run_manifest.json
split_info.json
history.json
test_metrics.json
test_metrics_by_domain.json
hierarchical metric CSVs
best.pt
routing_summary.json           # MoE variants
sampling_audit.json
episode_audit.json             # DG variant
```

`split_info.json`/audit 必须能证明：

- 真实 source/target domain 列表；
- train/val/test 中的 cell inventory；
- target 未参与 train/val；
- normalization 未使用 target-fitted statistics；
- episode 只在 source train 内构造；
- pseudo-target 类型和留出 environment；
- sampler 的 domain/strategy/cell 计数。

### 9.6 P2 测试与验收

至少覆盖：

- 分层采样 reproducibility；
- 小型非平衡 synthetic dataset 上的 domain/strategy/cell 近似均匀性；
- dataset-level episode 整体留出；
- strategy-level episode 整体留出；
- episode 两侧 cell-disjoint；
- 真实 target 不在 source sampler/episode/validation 中；
- first-order inner update 的快参数确实变化；
- outer update 的梯度有限且包含 pseudo-target 贡献；
- `beta=0` 时退化为对应 ERM 路径；
- `lambda_balance=0` 时不添加 balance loss；
- Base/Dense/MoE-ERM/MoE-DG 的 CPU bounded smoke；
- 五个 LODO configs 全部通过静态 leakage/config validation。

CPU smoke 应限制在极少样本、1 epoch，使用 `torch_reference`；它只验证 wiring，不产生论文结论。

## 10. Config 设计要求

尽量沿用现有 `base_config` 继承，避免五折配置复制整个物理 normalization。建议新增：

```text
configs/paper_v2/common/moe_base.json
configs/paper_v2/common/dg_base.json

configs/paper_v2/e2_full_domain/base/
configs/paper_v2/e2_full_domain/dense_adapter/
configs/paper_v2/e2_full_domain/moe_erm/

configs/paper_v2/e3_lodo_zero_cell/base_erm/
configs/paper_v2/e3_lodo_zero_cell/dense_adapter_erm/
configs/paper_v2/e3_lodo_zero_cell/moe_erm/
configs/paper_v2/e3_lodo_zero_cell/moe_dg/
```

为了不破坏现有 V2 baseline，不要立即移动已有 `e2_full_domain/raw_mamba_domain_balanced.json` 和 `e3_lodo_zero_cell/lodo_*.json`。在 README 中将其标记为 existing Base compatibility configs，新配置使用新路径。

模型配置建议显式包含：

```json
{
  "variant": "residual_moe",
  "num_experts": 8,
  "top_k": 2,
  "expert_bottleneck_dim": 16,
  "expert_init": "zero_output",
  "router_input": "z_base"
}
```

DG 配置建议显式包含：

```json
{
  "trainer": "first_order_mldg",
  "inner_steps": 1,
  "inner_learning_rate": 0.001,
  "beta": 1.0,
  "dataset_episode_probability": 0.5,
  "strategy_episode_probability": 0.5
}
```

上述数值是开发默认值，不是已验证最优值。不得使用真实 held-out target 选择它们。

## 11. 启动器要求

P1/P2 shell launcher 应：

- 默认 `DRY_RUN=1`；
- 显式接受 `SEEDS`、`GPU_IDS`、`JOBS_PER_GPU`、`TARGET_DOMAINS`、`RESUME`；
- dry-run 不创建 output；
- 启动前进行 config/data/checkpoint readiness validation；
- 禁止静默 fallback 到 legacy dataset、rated-relative label 或 Paper-v1 config；
- 每个 child process 的 seed/output 独立；
- 记录完整 resolved config；
- 本任务中只运行 dry-run 和 bounded smoke，不自动运行正式 jobs。

## 12. README 同步规则

每次新增或改变模型、trainer、config 或 launcher，必须在同一次开发中更新对应 README。不允许最后一次性补文档。

状态词汇固定为：

```text
planned
scaffolded
smoke-tested
runnable
formal-results-complete
deprecated
```

每个组件至少记录：

| 字段 | 内容 |
|---|---|
| Component | 模型、trainer 或脚本名称 |
| Status | 上述固定状态 |
| Config | 对应配置路径 |
| Command | 最小可执行命令 |
| Output | 输出路径 |
| Tests | 已通过的测试 |
| Last verified | 最近验证日期 |
| Limitations | 未完成项和限制 |

文档职责：

- `docs/paper_v2/README.md`：P0–P4 总状态和实现索引；
- `models/paper_v2/README.md`：模型公式、参数量、forward/checkpoint contract；
- `datasets/paper_v2/README.md`：hierarchy、strategy mapping、sampler/episode 语义；
- `trainers/paper_v2/README.md`：ERM/MLDG 公式、伪代码和限制；
- `evaluation/paper_v2/README.md`：metrics/routing audit；
- `configs/paper_v2/README.md`：可运行 config 矩阵；
- `scripts/paper_v2/README.md`：dry-run/smoke/formal 命令与脚本状态。

只有通过对应 smoke 和 contract tests 后才能标记 `runnable`。P0–P2 开发完成不等于 `formal-results-complete`。

## 13. 测试执行顺序

开发中使用由小到大的验证顺序：

1. 单文件语法/import 测试；
2. 新增 V2 unit tests；
3. 现有 Paper V2 BOL tests；
4. 全量 `tests/test_*.py` 回归；
5. V2 CPU bounded smoke；
6. launcher dry-run job matrix；
7. 最终重跑全量回归。

建议命令形式（根据环境的 Python 路径调整）：

```bash
python -m unittest discover -s tests/paper_v2 -p 'test_*.py'
python -m unittest tests.test_paper_v2_bol
python -m unittest discover -s tests -p 'test_*.py'
python tests/paper_v2_smoke_test.py
bash scripts/paper_v2/run_e2_seen_domain.sh   # DRY_RUN=1
bash scripts/paper_v2/run_e3_zero_cell.sh     # DRY_RUN=1
```

如果环境缺少 CUDA/mamba-ssm，正式 backend 测试可报告未执行，但 CPU `torch_reference` 结构 smoke 不能略过。不得为了让测试变绿而将 formal backend 静默改为 reference backend。

## 14. 实现质量要求

- 优先小而明确的 API，不复制整个 V1 trainer，但也不为强行抽象而重构 V1；
- 所有随机选择由显式 seed/generator 控制；
- config 字段必须校验，不要对关键科学参数静默使用猜测值；
- checkpoint 保存 resolved config、split provenance 和 trainer/model variant；
- 报错信息应指出具体 domain/strategy/cell/config 字段；
- 不引入 cycle index 作为模型输入；
- 不生成假数据、假结果或占位 checkpoint；
- 不安装新依赖，除非必要且已得到用户同意；
- 不执行 git commit/push，除非用户明确要求；
- 不删除用户输出、数据、split 或历史 checkpoint。

## 15. 明确不在本任务范围内

不得在本次 P0–P2 开发中擅自实现：

- one-cell/one-trajectory support-query adaptation；
- router-only inner/outer meta-adaptation；
- head-only/full-FT/scratch one-cell 主表；
- enterprise E4 adapter 或伪企业数据；
- domain adversarial、CORAL/MMD、IRM、GroupDRO 等额外方法；
- expert 与电化学退化机理的因果绑定；
- 正式多 seed GPU 训练；
- 基于 held-out target 的调参；
- 论文数值结论或“已证明可迁移”的声称。

## 16. 最终交付物

完成时应交付：

1. P0–P2 新增/修改代码；
2. Base/Dense/MoE-ERM/MoE-DG 配置；
3. E2/E3 dry-run launcher；
4. V2 model/sampler/episode/MLDG/leakage/V1-regression tests；
5. CPU bounded smoke；
6. 各 V2 README 和总状态文档；
7. 一份简明最终报告，包含：

```text
- 实际完成的组件
- 未完成/被阻塞的项目
- 关键设计决策
- 新增与修改文件列表
- 执行的测试命令与结果
- smoke/dry-run 结果
- V1 不变性证据
- 已知风险
- 进入正式 pilot 前还需要的事项
```

最终报告必须区分：

- “实现已完成”；
- “结构 smoke 已通过”；
- “正式训练/结果未执行”。

## 17. 最终验收 Checklist

只有以下项全部成立，才能称 P0–P2 开发完成：

- [ ] 现有用户改动未被回滚或覆盖。
- [ ] V1 config/script/model/checkpoint/output 协议未被破坏。
- [ ] P1/P2 代码位于独立 V2 namespace。
- [ ] V2 独立入口和 dry-run launcher 可用。
- [ ] Residual MoE 零初始等价性测试通过。
- [ ] Router 只输入 `z_base`。
- [ ] Top-k、balance loss、routing audit 完成并有测试。
- [ ] Dense Adapter 参数量匹配并报告差异。
- [ ] 四层 hierarchical sampler 完成、可复现、有 audit。
- [ ] Dataset-level 和 strategy-level episodes 均可运行。
- [ ] Episode 两侧 physical-cell disjoint。
- [ ] First-order MLDG 的 inner/outer 更新通过数值测试。
- [ ] 五个真实 LODO fold 的 target leakage 静态校验全部通过。
- [ ] Base/Dense/MoE-ERM/MoE-DG 均有独立 config 和唯一 output ID。
- [ ] 全量现有单元测试通过。
- [ ] CPU bounded smoke 通过。
- [ ] 所有新增/改变组件均同步更新 README。
- [ ] 未实现 P3/P4，未启动正式长训练，未伪造论文结果。

