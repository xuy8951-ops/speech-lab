# Week 2（S1）日志

## 完成
- 迁入 tiny_gru（659,713 参数，双向 GRU 谱掩蔽），同时实现训练 forward 与 Enhancer 波形接口。
- 通用 Trainer：Adam+余弦退火、梯度裁剪、best/last 断点、history.json，无具体模型/数据名。
- 跑通 `python train/run.py configs/exp/001_gru_overfit.yaml`（600 轮，4 样本纯过拟合）。

## 过拟合验证结论（训练链路无 bug）
- 恒等基线 L1 幅度损失 1.093 → 训练后 0.190（**下降 83%**）；
- 4 个训练样本 SNR 平均提升 **+8.8 dB**（如 white: 2.62→12.49 dB）。

## 关键认知（为什么 loss 不趋零）
掩蔽类模型 mask≤1，而噪声相位抵消处 |clean|>|noisy|，理论上无法完美重建，
存在残差下限。因此过拟合判据改为"相对恒等基线的损失降幅 + 训练集 SNR 提升"，
而不是教条地追求 loss→0。

## 泛化差距（教材级现象）
同一模型在冻结的未见测试集上平均 SNR 提升仅 **+1.99 dB**（训练集 +8.8 dB）。
→ 训练集表现 ≠ 泛化能力；S2 必须上真实数据 + 正规验证集划分。

## 下周（S2）
- 接入 VoiceBank-DEMAND；冻结真实测试 manifest；加入验证集早停与 W&B 日志。
