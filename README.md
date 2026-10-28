# speech-lab

面向中文真实场景的语音增强研究仓库：从传统信号处理到预训练大模型迁移。

> 当前进度：**S2 真实数据 pilot 已完成**（VoiceBank-DEMAND、冻结测试 manifest、验证早停、824 条正式评测）。
> 路线图：全量 VoiceBank 训练 → S3 中英文落差实验（主创新）→ M4 阵列 → M6 水声迁移。

## 快速开始

```bash
pip install -r requirements.txt
python eval/run.py configs/exp/000_baseline.yaml
# 输出 runs/000_baseline/scores.jsonl（逐样本明细）与 summary.csv（分组汇总+overall）
python train/run.py configs/exp/001_gru_overfit.yaml   # S1 训练冒烟（CPU约10分钟）
python -m pytest tests/ -q
```

## S2 VoiceBank-DEMAND v1（真实数据 CPU pilot）

```bash
# 16 kHz 镜像；全量时把 --only test 改为 --only all
python scripts/download_voicebank_16k.py --only test
python scripts/prepare_voicebank.py --only test
# 本次小数据复现：p226 验证候选 + p228 训练候选
python scripts/fetch_voicebank_rowgroups.py --shard 0 --row-groups 0,8,9
python scripts/prepare_voicebank.py --only train --allow-partial
python train/run.py configs/exp/002_gru_voicebank_v1.yaml
python eval/run.py configs/exp/002_gru_voicebank_v1.yaml
```

最佳验证权重为 epoch 2，epoch 5 触发早停。824 条冻结测试集总体结果：SNR **+1.197 dB**、SI-SDR **+1.296 dB**、STOI **-0.036**。
注意：本次仅使用单训练说话人 128 条与单验证说话人 64 条，是全链路 pilot，不与全量结果或 SOTA 直接比较；PESQ 因 Windows 缺少 MSVC 记 NaN。详见 `docs/log/week3.md`。

## S2 多说话人对照 v2

```bash
# 训练改为 p227+p228，验证/测试继续复用 v1 清单
python scripts/prepare_voicebank.py --only train --allow-partial \
  --manifest-dir data/manifests/voicebank_demand_multispk --val-speakers p226
python train/run.py configs/exp/003_gru_voicebank_multispk.yaml
python eval/run.py configs/exp/003_gru_voicebank_multispk.yaml
```

训练条数仍为 128（p227=70、p228=58），模型、seed、验证集和 824 条冻结测试集均不变。最佳为 epoch 11，val L1 从 v1 的 0.11481 降至 0.06100。
v2 总体结果：SNR **+5.522 dB**、SI-SDR **+5.306 dB**、STOI **-0.032**；17.5 dB 输入的 SNR 变化由 v1 的 **-4.184 dB** 改善为 **+0.185 dB**。
这说明本设置下增加说话人多样性有效，但 STOI 尚未改善，且仍是小数据对照而非全量训练或 SOTA。逐样本与分组结果见 `runs/003_gru_voicebank_multispk/`。

### v3 对数幅度复合损失（负结果）

```bash
python train/run.py configs/exp/004_gru_voicebank_logloss.yaml
python eval/run.py configs/exp/004_gru_voicebank_logloss.yaml
```

只增加对数幅度 L1 来保护弱语音，其他条件与 v2 相同。结果 SNR 增益降至 **+5.030 dB**、SI-SDR 增益降至 **+4.763 dB**、STOI 降至 **-0.038**；824 条中 560 条 STOI 比 v2 更低。
因此 v3 假设未获支持，没有取代 v2。负结果与复现命令已保留在 `docs/log/week3.md`。

### 相位诊断与 v4 理想比例掩蔽（IRM）

Oracle 诊断显示：即使沿用带噪相位，理想掩蔽仍可让 STOI 提高约 0.066，因此当前主要瓶颈是掩蔽学习，而非相位本身。v4 仅将训练目标改为直接 IRM L1：

```bash
python scripts/analyze_phase_oracle.py  # 使用 clean 的诊断上限，不是模型成绩
python train/run.py configs/exp/005_gru_voicebank_irm.yaml
python eval/run.py configs/exp/005_gru_voicebank_irm.yaml
```

v4 最佳为 epoch 9，结果为 SNR **+4.557 dB**、SI-SDR **+4.576 dB**、STOI **-0.014**；634/824 条样本的 STOI 优于 v2，17.5 dB 档 SNR 增益提高到 **+1.036 dB**。
结论是明确的取舍：**v2 保持最高 SNR，v4 提供目前最好的 STOI 保持能力，但并非全面优于 v2**。

### v5 幅度＋IRM 组合损失

```bash
python train/run.py configs/exp/006_gru_voicebank_mag_irm.yaml
python eval/run.py configs/exp/006_gru_voicebank_mag_irm.yaml --split val
python eval/run.py configs/exp/006_gru_voicebank_mag_irm.yaml
```

λ=0.25 预先由验证损失量级确定，并通过验证门槛后才进入冻结测试。v5 的 SNR 增益为 **+5.304 dB**，接近 v2；但 STOI 为 **-0.030**，没有保住 v4 的 **-0.014** 优势。
因此当前仍是 **v2 最高 SNR、v4 最佳 STOI**。验证集也未能预测 v4 的测试 STOI 优势，后续应先扩大多说话人验证集，而不是继续看测试集扫 λ。

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
