# C题第一、二问代码框架

本目录只实现第一、二问。程序以附件中的 `EgoDemo.zip`、`EgoPro.zip`、`EgoStandard.zip` 和官方环境划分表为数据来源，不合成训练样本，也不把人的手部状态解释成商品位置。

## 文件分工

- `real_data_pipeline.py`：解读三种原始数据结构；按 `episode_uuid` 对齐官方划分；将21个三维关键点压缩为坐标逐维中位数；对齐坏帧标记和语义阶段；保留时间重复、标签覆盖等审计信息。
- `models.py`：实现共用的六维常速度卡尔曼模型，并在代码注释中说明状态定义、过程噪声和观测残差尺度。
- `run_analysis.py`：总入口。分为数据审计、第一问质量分类、第一问状态预测、第二问人工遮挡回放、自然低质量区间统计和图片输出。
- `requirements.txt`：运行所需的 Python 包。

## 关键函数

- `frame_features`：只由当前和过去的手部坐标构造时间间隔、速度和加速度特征；坐标完整性单独作为硬性质量门槛，避免作为模型输入泄漏到标签。
- `fit_quality_model`：仅在训练集拟合逻辑回归，仅用验证集选择坏帧阈值，测试集只做最终评价；同时输出多数类基线、坏帧召回率、精确率、平衡准确率和 Brier 分数。
- `estimate_motion_parameters`：仅由训练环境估计状态模型尺度，并按环境重抽样给出经验区间。
- `one_step_state_replay`：比较末值保持、常速度外推、卡尔曼预测；每条记录单独评价，再按数据源和环境汇总。
- `predict_hidden_window`：第二问的因果回放。窗口内真实坐标只供评估使用；只有在请求返回时，才将原始记录里真实存在的坐标交给模型更新。
- `audit_quality_runs`：统计原始低质量标记连续段及下一次合格观测的恢复时间；不对这些低质量坐标计算“真值误差”。
- `save_figures`：从结果汇总表生成英文 PNG 图，避免图中的中文字体影响论文排版。

## PyCharm 运行

在 PyCharm 打开本目录的上一级项目，选择安装了 `requirements.txt` 依赖的解释器，并将下面命令配置为运行参数：

```text
--data-root "<题目数据目录>/Restocking_Problem_and_Data_EN"
--output outputs
```

也可以在 PyCharm Terminal 执行：

```text
python run_analysis.py --data-root "<题目数据目录>/Restocking_Problem_and_Data_EN" --output outputs
```

## 主要结果文件

- `outputs/record_audit.csv`：每条记录、左右手的帧数、重复时间戳、坐标完整率、质量标签覆盖率和语义阶段覆盖率。
- `outputs/q1_quality_metrics.csv`：验证集阈值选择结果与测试集质量分类结果。
- `outputs/q1_state_metrics.csv`、`q1_state_by_stage.csv`：状态预测误差的环境和阶段汇总。
- `outputs/q2_summary.csv`、`q2_by_stage.csv`：不同遮挡时长下的误差、区间覆盖、区间半径、请求成本和等待时间。
- `outputs/low_quality_events.csv`、`low_quality_summary.csv`：自然记录中低质量标记的持续与恢复情况。
- `outputs/figures/`：状态误差、遮挡误差、请求成本与区间覆盖图。
- `outputs/results.json`：关键模型参数、实验条件、样本计数和输出文件名。

## 固定实验条件与解释边界

第二问预先固定遮挡时长为 0.25、0.5、1、1.5、2 秒，单次返回延迟为 1 秒，每个窗口最多请求 3 次，每次请求成本为 1 个实验单位。自适应复核阈值和名义90%区间尺度仅用验证集校准。末值、常速度、固定间隔复核和自适应复核在同一遮挡窗口及预算下比较。

语义阶段沿用附件自然语言分段，不将其强行改名为统一的“取货/放置”等动作标签。参考轨迹来自合格的留出记录，不是独立物理真值。人工遮挡结果不代表机器人现场性能；位置坐标的误差单位沿用源记录坐标单位。

数维杯论文格式规范要求参赛论文用英文，程序源码须作为附录或支撑材料提供；这些要求约束论文与最终提交材料，不改变 Python 代码的运行规则。
