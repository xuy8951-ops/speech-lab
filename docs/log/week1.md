# Week 1（S0）日志

## 完成
- 仓库骨架：统一 Enhancer 接口、注册表、YAML 配置、EvalRunner、两个单元测试。
- 迁入原 speech-denoiser-mini 的谱门控实现（Adapter 模式包装为 Enhancer）。
- 跑通 `python eval/run.py configs/exp/000_baseline.yaml`，产出 9 样本明细与汇总。

## 踩坑记录
1. 指标未注册：registry 需要 import 触发装饰器，已在 runner 中显式 import metrics 模块。
2. SI-SDR 异常：合成 babble 与干净语音相关性过高（f0/包络不随 seed 变化），
   已改为参数随 seed 变化；同时将测试集 seed 从 hash() 换成 crc32（跨进程稳定）。
3. babble 上谱门控提升为 0：非 bug，是"噪声频谱平坦"假设失效，写入 README 作为引入深度模型的动机。

## 下周（S1）
- 迁入 tiny_gru，写 Trainer，10 分钟数据过拟合验证训练链路。
