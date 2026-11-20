# C题第一、二问：手部状态与遮挡回放

## 做法

这版按建模竞赛里容易解释、能复现的思路做：先把每只手的21个三维关键点取坐标中位数，作为该帧手部代表点；再用帧间速度和加速度变化、时间间隔、坐标是否完整，加上题目提供的坏帧标记，训练逻辑回归判断观测是否合格。模型不把手的位置当成商品位置。

状态预测用常速度卡尔曼滤波。对照组是“保持最后一次位置”和“按前两帧速度外推”。重新观测策略在位置不确定性超过验证集校准阈值时发出请求，按附件中的1秒返回延迟处理，最多请求3次。复核只使用该真实记录里确实存在的合格样本。

## 数据边界

- 训练数据只取题目附带的 EgoDemo、EgoPro、EgoStandard 三个压缩包。
- 训练、验证、测试严格使用 `record_split.csv` 的环境划分；不会把同一环境拆到不同集合。
- EgoDemo坏帧标签来自 `bad_frame_ratio.json`；EgoPro/EgoStandard来自 `annotation_bad_frame_pose_hand.csv`。
- 问题二只在测试集真实连续片段上人工遮蔽观测，以被遮蔽的原始样本作评估参考，不合成轨迹，也不补造缺失样本。
- 坐标/字段或官方数据划分对不上时程序直接报错，不猜测映射。

## 运行

在 PyCharm 选择包含 pandas、numpy、scikit-learn、pyarrow 的 Python 环境，运行 `run_analysis.py` 并传入数据目录：

```text
--data-root <解压目录>/Restocking_Problem_and_Data_EN
--output outputs
```

结果写入输出目录：`record_audit.csv`、`q1_observation_audit.csv`、`q1_quality_metrics.csv`、`q2_test_replay.csv`、`q2_summary.csv` 和 `results.json`。测试集指标只在所有规则固定后计算。回放实验不等同于真实机器人现场性能。
