# Paper-Backup 数据与实现审计

审计日期：2026-08-27

本文件记录 Paper-Backup 开发前的只读审计。数据文件本身仍由仓库的
`.gitignore` 忽略；这里只记录路径、字段和可复核的数量，不写入结果数字。

## 1. Canonical terminal 产品

| family | canonical RAW | canonical Only-F | canonical identity | development / test |
|---|---|---|---|---:|
| XJTU | `datasets/XJTU_raw` | `datasets/XJTU_features` | `<condition>_battery-<n>`，cycle | 43 / 12 physical batteries |
| MIT | `datasets/MIT_raw` | `datasets/MIT_features` | `mit_p###`，one-based global physical cycle | 100 / 24 physical cells |
| LISHEN | `datasets/SmartHealth_raw/smarthealth_lishen40` | matching family directory | `logical_sequence_id`，canonical chronological cycle | 18 / 9 logical batteries |
| CATL | `datasets/SmartHealth_raw/smarthealth_catl280` | matching family directory | `logical_sequence_id`，canonical chronological cycle | 6 / 3 logical batteries |
| EVE | `datasets/SmartHealth_raw/smarthealth_eve280` | matching family directory | `logical_sequence_id`，canonical chronological cycle | 6 / 3 logical batteries |

实际 inventory：XJTU canonical RAW 为 55 个 battery CSV（2C=8、3C=15、
R2.5=8、R3=8、RW=8、satellite=8）；MIT canonical physical RAW 为 124
个 CSV；SmartHealth canonical RAW 为 LISHEN/CATL/EVE 分别 27/9/9 个
logical-sequence CSV。所有 canonical 文件均包含 point-level 电压、电流、
relative time、温度、CC/CV segment 和 cycle label linkage。

这些产品已经经过 Paper-v1 的 terminal contract，但它们不保存完整充电
轨迹：XJTU 只写出 CC `4.0--4.195 V` 与 CV `0.5--0.1 A` 的末端窗口；
MIT 只写出 phase-aware CC `3.45--3.60 V` 与 nominal-C-rate CV
`0.25C--0.05C`；SmartHealth 只写出推断 phase 后的 CC `3.45--3.58 V`
与 CV `0.25C--0.05C`。窗口边界按 canonical preprocessor 的包含规则
和 tolerance 执行，随后现有 terminal adapter 对 CC 电压/CV 电流做固定
长度插值。SmartHealth 的 CV selection tolerance 是 `±0.002C`，温度不做
插值或填补。

## 2. Full source 审计

| family | 本机 source | 是否含 full point-level charge | 当前是否已有配对 canonical full product | Paper-Backup 状态 |
|---|---|---|---|---|
| XJTU | `/data1/chenyanxi/lb_project/datasets/XJTU battery dataset` | 是，56 个 MATLAB v5 `.mat`，每 cycle 保留 charging stage 的完整 V/I/T/time | 否 | 可由新 adapter 读取；无 source 配置时 blocked |
| LISHEN | `/data1/chenyanxi/lb_project/datasets/SmartHealth/LISHEN` | 是，GB18030 source 的 `恒流恒压充电` point-level event | 否 | 可由 source-linked adapter 读取；无 source 配置时 blocked |
| CATL | `/data1/chenyanxi/lb_project/datasets/SmartHealth/CATL` | 是，同上 | 否 | 可由 source-linked adapter 读取；无 source 配置时 blocked |
| MIT | `/data1/chenyanxi/lb_project/datasets/A123 Dataset` | 是，3 个 batch HDF5/MAT，保留 cycles 的 V/I/T/time/Qc/Qd | 否；MIT 不在本阶段 E2 重点范围 | 审计保留，暂不进入 E2 matrix |

XJTU 原始 `.mat` 由 `preprocess/XJTUBatteryClass.py` 的 `get_one_cycle`
和 charging stage 提供 full source；MIT HDF5 的 `cycles['I'/'V'/'T'/'t']`
和容量数组也可审计。SmartHealth 原始 CSV 具备 `循环号`、`工步类型`、
`绝对时间`、电流、电压、充电/放电容量和（大部分）温度字段；canonical
RAW 额外保留了 source file、source cycle、source absolute interval、
source row 和 calibration label linkage，可用于按明确 provenance 配对。

但是，当前仓库没有把这些 full source 物化为可直接训练的
`Paper-Backup/full_cccv` 产品，也没有把 machine-specific source root 写入
tracked config。因此 E2 full job 不会把 `datasets/*_raw` 当作 full 数据，
而是要求显式 `full_data_root`/full-source adapter；缺少时返回
`blocked_by_data`。任何 full/terminal 配对必须以
`(physical battery ID, canonical cycle identity)` 为 key，并逐项校验
label、strategy 和 split provenance 一致。

当用户在 E2 terminal config 中提供 `matched_full_data_root` 时，terminal
loader 会复用同一个 full source 做配对，并只保留匹配的 physical cycles；
这样 Full/Terminal 模型可以共享完全相同的 cohort。当前 tracked terminal
configs 将该字段留为 `null`，所以 terminal-only smoke 可以独立运行，但
在 full source 配置完成前不宣称 E2 system-level paired result。

## 3. Strategy 与 split

- XJTU strategy 直接来自 canonical `condition`：`2C`, `3C`, `R2.5`,
  `R3`, `RW`, `satellite`。`paper_v1_mixed_split.json` 每个 strategy
  固定 battery-4 和 battery-8 为 test；3C 的其余 battery 数量因 canonical
  inventory 为 15 而不同，但 test 规则仍由 JSON 所有。
- LISHEN strategy 直接来自 canonical `condition`，即 3 个 C-rate 与 3
  个 DOD 的 9 个组合。每个 condition 的 3 个 logical sequences 中 2 个
  development、1 个 test，由
  `smarthealth_lishen40_cell_split.json` 的显式 JSON assignment 指定。
- CATL/EVE 同样是 0.5C × 20/60/100%DOD 的 condition split；它们只作为
  E1 family，E2 重点配置会记录 full-source blocker。
- validation 是 development battery 内的 mixed-cycle split；train/val
  physical battery overlap 是 JSON 声明的预期协议。test battery 不参与
  early stopping、normalization fitting 或 hyperparameter selection。

## 4. 现有实现边界与 Paper-Backup 处理

1. 现有 `datasets.loaders` 能构造 terminal phase sample，但它面向
   Paper-v1 C5B，且 `PaperRawSOHModel` 默认带 cycle prediction head 和
   cycle auxiliary。Paper-Backup 不直接复用这个默认训练入口。
2. Paper-Backup 新 namespace 提供 SOH-only Ours wrapper、真实 Transformer、
   joint Vanilla/single-stream sequence baseline、HI-MLP adapter 和独立
   trainer。Ours 在 config 和 runtime 同时校验
   `use_cycle_prediction=false`、`use_predicted_cycle_for_soh=false`、
   `lambda_cycle=0`。
3. E1 按本阶段范围只配置 HI-MLP、Transformer、Ours；Raw CNN、LSTM 和
   E1 Vanilla Mamba 不列入本阶段 E1 runnable matrix。Vanilla/single-stream
   Mamba 仍为 E2 所需的独立模型实现。
4. E2 full view 只接受真实 full point-level source；terminal view 使用
   canonical terminal window，不重新搜索 window position。没有 source 或
   pairing 不完整的 family 不会静默删样本或计为零，而会标为
   `blocked_by_data`。
5. E3 的 strategy metadata 只用于 split、分层 sampler 和 aggregation，
   不进入任何 model forward；pooled estimator 的 test cohort 必须等于
   各 strategy test cohort 的并集。

## 5. 共享文件影响

本阶段不修改 `configs/paper_v1/`、`configs/paper_v2/`、对应 scripts、
历史 outputs/results 或共享 loader 的默认语义。所有新增训练、full view、
strategy pooling、evaluation 和 output layout 均放在 `paper_backup` namespace；
共享数据 adapter 只以只读方式被调用。
