# Week 3（S2）日志

## 完成
- 接入 VoiceBank-DEMAND：官方 48 kHz 下载脚本（断点续传、ZIP CRC、SHA256）与 16 kHz 镜像脚本。
- 清洗约定：单声道、有限值检查、成对长度检查、统一 16 kHz float32；保留官方 noise/SNR 标签。
- 冻结官方测试集：824 条，`test.jsonl` + `test.sha256`，并记录每条 clean/noisy 音频 SHA256；评测前自动校验。
- 数据集注册入口复用 `DATASETS`，训练/评测仍由 YAML 驱动，样本结构保持 `clean/noisy/meta`。
- Trainer 支持可变长度 batch（padding 不参与 L1）、验证早停、逐轮 JSONL、环境/配置快照、恢复最佳权重。
- 新增 STOI；PESQ 因本机无 MSVC 无法构建，按既有规范记 NaN，不阻塞评测。

## 本次 v1 的数据边界（重要）
官方与镜像整包下载均出现严重限速，因此本次不是全量训练，而是真实数据 CPU pilot：
- 训练候选：p228 的 200 条，YAML 固定抽取 128 条；
- 验证候选：p226 的 100 条，YAML 固定抽取 64 条；
- 测试：官方 p232/p257 共 824 条，未参与训练或早停；
- 训练/验证各只有一个说话人，结果用于验证真实数据全链路，不能与全量 VoiceBank 结果或 SOTA 直接比较。

## 训练与早停
配置：`configs/exp/002_gru_voicebank_v1.yaml`，tiny_gru 659,713 参数，batch=8，lr=0.001。

- epoch 1：train L1=0.10369，val L1=0.12473；
- epoch 2：train L1=0.08581，val L1=0.11481（最佳）；
- epoch 3–5 验证损失回升，patience=3 触发早停；
- 评测使用 epoch 2 的 `best.pt`，不是最后一轮。

## 首个正式指标（824 条冻结测试集）
| 指标 | noisy | enhanced | delta |
|---|---:|---:|---:|
| SNR (dB) | 8.446 | 9.643 | **+1.197** |
| SI-SDR (dB) | 8.446 | 9.742 | **+1.296** |
| STOI | 0.921 | 0.885 | **-0.036** |
| PESQ | NaN | NaN | NaN（可选依赖未安装） |

分组观察：2.5 dB 输入上 SNR +5.796 dB，但 17.5 dB 输入上 SNR -4.184 dB；模型在低 SNR 有增益，在高 SNR 明显破坏语音。
同时，STOI 在所有噪声组均下降，说明波形级 SNR/SI-SDR 改善不等于可懂度改善，后续需增加多说话人数据并检查损失与掩蔽目标。

## 多说话人对照 v2

目的：在不扩大模型、不增加训练条数、不换考题的前提下，检查训练说话人多样性是否改善泛化。

- 训练候选：p227+p228 共 647 条；固定 seed 实际抽取 128 条，其中 p227=70、p228=58。
- 验证：与 v1 完全相同的 p226 64 条；测试：与 v1 完全相同的 p232/p257 824 条。
- 模型仍为 659,713 参数 tiny_gru；seed=2026、batch=8、lr=0.001、最多 12 轮、早停 patience=3 均不变。
- v2 最佳为 epoch 11，val L1=0.06100；epoch 12 跑完，未触发早停。v1 最佳 val L1=0.11481。
- 本次加入完整断点续训（模型、优化器、调度器、早停状态与随机数状态），保障长任务恢复；它不改变评测指标的计算方式。

| 指标 | noisy | v1 enhanced / delta | v2 enhanced / delta |
|---|---:|---:|---:|
| SNR (dB) | 8.446 | 9.643 / +1.197 | **13.968 / +5.522** |
| SI-SDR (dB) | 8.446 | 9.742 / +1.296 | **13.752 / +5.306** |
| STOI | 0.921 | 0.885 / -0.036 | **0.889 / -0.032** |

| 输入 SNR 档 | v1 delta SNR | v2 delta SNR |
|---|---:|---:|
| 2.5 dB | +5.796 dB | **+9.626 dB** |
| 7.5 dB | +3.292 dB | **+7.630 dB** |
| 12.5 dB | -0.182 dB | **+4.579 dB** |
| 17.5 dB | -4.184 dB | **+0.185 dB** |

结论：在本小数据设置下，增加一个训练说话人后，未见说话人的波形级指标全面提升，高 SNR 平均退化基本消除。但 17.5 dB 档中 cafe/living/psquare 仍分别下降 0.395/0.408/0.523 dB，不能称为所有条件均已解决。
STOI 仍总体下降 0.032，且五类噪声均为负，说明 SNR/SI-SDR 的大幅改善尚未转化为可懂度改善。后续应检查掩蔽目标与损失，而不是只继续追求 SNR。
这仍是两个训练说话人、128 条训练样本的受控对照，不是全量 VoiceBank 训练，也不与 SOTA 直接比较。

复现 v2（复用已下载的 p226/p227/p228 与冻结测试集）：

```bash
python scripts/prepare_voicebank.py --only train --allow-partial \
  --manifest-dir data/manifests/voicebank_demand_multispk --val-speakers p226
python train/run.py configs/exp/003_gru_voicebank_multispk.yaml
python eval/run.py configs/exp/003_gru_voicebank_multispk.yaml
```

## 对数幅度复合损失 v3（未改善）

假设：线性幅度 L1 由高能量频谱点主导，可能忽略弱辅音和字尾；增加对数幅度误差，或许能保留弱语音并改善 STOI。

保持 v2 的训练/验证/测试清单、128 条训练样本、模型、seed 和训练超参数不变，只将训练目标改为：

`L = magnitude_L1 + 1.0 × log1p_magnitude_L1`

- 最佳为 epoch 11；复合 val_loss=0.09577。该总损失定义与 v2 不同，不能直接横向比较。
- 同口径 val magnitude L1=0.06005，略低于 v2 的 0.06100，但测试指标没有改善。

| 指标 | v2 | v3 |
|---|---:|---:|
| delta SNR | **+5.522 dB** | +5.030 dB |
| delta SI-SDR | **+5.306 dB** | +4.763 dB |
| delta STOI | **-0.032** | -0.038 |
| 17.5 dB 档 delta SNR | **+0.185 dB** | -0.478 dB |

v3 在四个 SNR 档和五类噪声上均逊于 v2。逐样本配对比较中，824 条有 542 条 SNR 更低、560 条 STOI 更低。
结论：本设置下，增加等权对数幅度项没有解决可懂度下降，反而损害去噪与高 SNR 保持能力；v2 仍是当前最佳。可能原因是保护弱成分时也保留了弱噪声，但这仅是待验证解释，不能由本次指标直接证明。
这是一次有复现记录的负结果，不应隐藏或继续宣称“弱语音保护有效”。

复现 v3：

```bash
python train/run.py configs/exp/004_gru_voicebank_logloss.yaml
python eval/run.py configs/exp/004_gru_voicebank_logloss.yaml
```

## 相位 oracle 误差分析

脚本：`scripts/analyze_phase_oracle.py`。该分析故意使用测试 clean 构造上限，只用于定位误差，不是可部署模型成绩，也不参与训练或模型选择。

| 诊断构造（均沿用带噪相位） | delta SNR | delta SI-SDR | delta STOI |
|---|---:|---:|---:|
| 干净幅度谱 | +11.683 dB | +11.633 dB | +0.0678 |
| 理想比例掩蔽（IRM） | +11.298 dB | +11.255 dB | +0.0660 |

结论：即使不估计干净相位，理想幅度/掩蔽也能大幅提高 STOI，因此当前主要瓶颈是学习掩蔽的质量，而不是带噪相位本身。v3 的间接对数幅度监督没有解决问题，v4 改为直接学习 IRM。

## 理想比例掩蔽监督 v4

保持 v2/v3 的数据、模型、seed 和训练超参数不变，只将目标改为 `clip(clean_mag / noisy_mag, 0, 1)` 的掩蔽 L1。

- 最佳为 epoch 9，val IRM L1=0.23614；epoch 12 触发早停。该损失口径与幅度 L1 不可直接比较。
- 824 条冻结测试集结果如下。

| 指标 | v2 幅度监督 | v4 IRM 监督 |
|---|---:|---:|
| delta SNR | **+5.522 dB** | +4.557 dB |
| delta SI-SDR | **+5.306 dB** | +4.576 dB |
| delta STOI | -0.032 | **-0.014** |
| 17.5 dB 档 delta SNR | +0.185 dB | **+1.036 dB** |

逐样本配对比较：v4 有 634/824 条 STOI 优于 v2，但有 583/824 条 SNR 低于 v2。四个 SNR 档和五类噪声的 STOI 均比 v2 更好。
结论：直接 IRM 监督显著缓解可懂度下降，并改善高 SNR 保持能力，但总体去噪增益降低。当前应保留两个明确取舍：**v2 是最高 SNR 版本，v4 是最佳 STOI 保持版本**，不能宣称 v4 全面优于 v2。

复现诊断与 v4：

```bash
python scripts/analyze_phase_oracle.py
python train/run.py configs/exp/005_gru_voicebank_irm.yaml
python eval/run.py configs/exp/005_gru_voicebank_irm.yaml
```

## 幅度＋IRM 组合损失 v5

目标：结合 v2 的幅度监督与 v4 的 IRM 监督，寻找 SNR 与 STOI 的折中。损失为 `magnitude_L1 + 0.25 × IRM_L1`；λ=0.25 预先根据验证损失量级确定（v2 约 0.061、v4 约 0.236），不是根据冻结测试成绩调参。

评测入口新增 `--split val`，验证结果写入各实验的 `validation/` 子目录，不覆盖正式测试结果。预先固定入选规则：v5 只有在验证集上 SNR 高于 v4 且 STOI 不差于 v4，才进入冻结测试。

| 验证集（p226 64 条） | v2 | v4 | v5 |
|---|---:|---:|---:|
| delta SNR | 4.752 dB | 2.595 dB | 4.645 dB |
| delta STOI | -0.021 | -0.028 | -0.021 |

v5 通过验证门槛。但需要注意：v4 在正式测试中的 STOI 优势并未在这个单说话人小验证集上出现，说明验证集与测试集存在分布差异。

v5 最佳为 epoch 11，随后在同一 824 条冻结测试集评测：

| 指标 | v2 | v4 | v5 |
|---|---:|---:|---:|
| delta SNR | **+5.522 dB** | +4.557 dB | +5.304 dB |
| delta SI-SDR | **+5.306 dB** | +4.576 dB | +5.143 dB |
| delta STOI | -0.032 | **-0.014** | -0.030 |
| 17.5 dB 档 delta SNR | +0.185 dB | **+1.036 dB** | -0.032 dB |

v5 相对 v2 有 511/824 条 STOI 更好，但平均仅提高 0.00231；SNR 平均低 0.218 dB。它没有保住 v4 的可懂度优势，也没有实现预期的双重优势。
当前结论不变：**v2 保持最高 SNR，v4 保持最佳 STOI**。v5 是接近 v2 的折中版本，不作为新突破宣传。
后续不应继续根据冻结测试集扫 λ；应先扩大为多说话人验证集，提高模型选择的可靠性。

复现 v5：

```bash
python train/run.py configs/exp/006_gru_voicebank_mag_irm.yaml
python eval/run.py configs/exp/006_gru_voicebank_mag_irm.yaml --split val
python eval/run.py configs/exp/006_gru_voicebank_mag_irm.yaml
```

## 复现
```bash
# 本次真实数据 pilot（只取训练 row groups，不触碰 test 训练）
python scripts/fetch_voicebank_rowgroups.py --shard 0 --row-groups 0,8,9
python scripts/prepare_voicebank.py --only train --allow-partial

# 测试集（已下载时自动跳过）
python scripts/download_voicebank_16k.py --only test
python scripts/prepare_voicebank.py --only test

python train/run.py configs/exp/002_gru_voicebank_v1.yaml
python eval/run.py configs/exp/002_gru_voicebank_v1.yaml
python -m pytest tests/ -q
```

全量数据时运行 `python scripts/download_voicebank_16k.py --only all`，再运行不带 `--allow-partial` 的 `prepare_voicebank.py`，并删除 YAML 的 `max_train/max_val` 限制。

## 踩坑记录
1. 仓库交接估计数据约 1 GB；官方 28spk 48 kHz 四包实际约 5.3 GB，16 kHz 镜像约 2.28 GB。
2. 官方页面新 bitstream 链接需转 `server/api/core/bitstreams/<id>/content`，否则可能得到 HTML 而非 ZIP。
3. Windows 上 `pesq` 需要 MSVC 编译；pystoi 可直接安装，PESQ 保持可选降级。
4. 真实音频长度不一，S1 的 `np.stack` 不适用；改为 batch 内 padding + 有效帧掩码，未改模型接口。
