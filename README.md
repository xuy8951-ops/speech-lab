# speech-lab

面向中文真实场景的语音增强研究仓库：从传统信号处理到预训练大模型迁移。

> 当前进度：**S1 已完成**（S0 骨架 + tiny_gru 训练链路，过拟合验证通过）。
> 路线图：S2 VoiceBank-DEMAND 真实数据 → S3 中英文落差实验（主创新）→ M4 阵列 → M6 水声迁移。

## 快速开始

```bash
pip install -r requirements.txt
python eval/run.py configs/exp/000_baseline.yaml
# 输出 runs/000_baseline/scores.jsonl（逐样本明细）与 summary.csv（分组汇总）
python train/run.py configs/exp/001_gru_overfit.yaml   # S1 训练冒烟（CPU约10分钟）
python -m pytest tests/ -q
```

## S1 训练链路验证（tiny_gru，659,713 参数）

- 恒等基线 L1 损失 1.093 → 0.190（-83%）；训练样本 SNR 平均 **+8.8 dB**；
- 未见测试集平均仅 +1.99 dB → 泛化差距直观可见，真实数据与验证集划分是 S2 的重点；
- 注：掩蔽≤1 的模型存在理论残差（相位抵消处无法完全重建），故不以 loss→0 为判据。

## S0 基线结果（合成冒烟测试集，谱门控）

| 噪声类型 | 输入 SNR | 输出 SNR | 提升 |
|---|---|---|---|
| white | 0 dB | 8.27 dB | **+8.27** |
| white | 5 dB | 12.60 dB | +7.60 |
| pink | 0 dB | 1.51 dB | +1.51 |
| babble | 0 dB | 0.00 dB | +0.00（算法假设失效，见下） |

**已知的诚实结论**：谱门控假设噪声频谱平坦，因此对白噪有效、对粉噪有限、对 babble（类语音噪声）基本无效。
这正是后续引入深度模型的动机之一，不是 bug。

## 架构要点

- `speechlab/models/base.py`：所有算法实现统一 `Enhancer.enhance((B,T))->(B,T)` 接口；
- `speechlab/engine/registry.py`：YAML 配置按名字实例化算法，新增算法不改引擎；
- `speechlab/dsp/stft.py`：全局唯一 STFT 入口，可逆性有单元测试；
- `speechlab/eval/runner.py`：逐样本 `scores.jsonl` + 按语言×噪声×SNR 的 `summary.csv`，所有数字可溯源；
- 测试集固定 seed 冻结（crc32 跨进程稳定），指标永远可比。

## 目录

```
configs/          # 实验 YAML（一次实验=一个文件）
speechlab/        # 包：data / dsp / baselines / models / engine / eval
eval/run.py       # 评测入口
train/run.py      # 训练入口
tests/            # STFT 可逆性、混合 SNR 精度
runs/             # 实验输出（gitignore）
```

## 出处

- 谱门控实现迁移自本人原型 speech-denoiser-mini（合成数据实测验证）；
- 软掩蔽思路另见本人对 timsainb/noisereduce 的 fork 优化（soft-mask-stationary 分支）。
