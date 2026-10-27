# speech-lab

面向中文真实场景的语音增强研究仓库：从传统信号处理到预训练大模型迁移。

> 当前进度：**S0 垂直切片已完成**（统一接口 + 注册表 + 谱门控基线 + 合成评测集 + 指标汇总）。
> 路线图：S1 深度模型训练 → S2 VoiceBank-DEMAND 真实数据 → S3 中英文落差实验（主创新）→ M4 阵列 → M6 水声迁移。

## 快速开始

```bash
pip install -r requirements.txt
python eval/run.py configs/exp/000_baseline.yaml
# 输出 runs/000_baseline/scores.jsonl（逐样本明细）与 summary.csv（分组汇总）
python -m pytest tests/ -q
```

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
tests/            # STFT 可逆性、混合 SNR 精度
runs/             # 实验输出（gitignore）
```

## 出处

- 谱门控实现迁移自本人原型 speech-denoiser-mini（合成数据实测验证）；
- 软掩蔽思路另见本人对 timsainb/noisereduce 的 fork 优化（soft-mask-stationary 分支）。
