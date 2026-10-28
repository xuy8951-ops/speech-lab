"""配置驱动的数据集入口；各适配器统一返回 clean/noisy/meta 样本。"""
from speechlab.data.synthesize import frozen_test_set, make_train_cases
from speechlab.data.voicebank import load_voicebank_cases
from speechlab.engine.registry import DATASETS


@DATASETS.register("synth_overfit")
def _synth_overfit(cfg, split):
    if split != "train":
        raise ValueError("synth_overfit 仅提供 train；S1 保持训练集即验证集")
    return make_train_cases(
        n=cfg["n_train"],
        duration=cfg.get("duration", 2.0),
        seed=cfg.get("seed", 1234),
    )


@DATASETS.register("frozen_synth")
def _frozen_synth(cfg, split):
    if split != "test":
        raise ValueError("frozen_synth 仅提供 test")
    return frozen_test_set(n_per_case=cfg.get("n_per_case", 2))


@DATASETS.register("voicebank_demand")
def _voicebank_demand(cfg, split):
    return load_voicebank_cases(cfg, split)


def load_cases(cfg, split):
    """按 YAML 的 data.name 取数据，入口脚本无需判断具体数据集。"""
    return DATASETS.get(cfg["name"])(cfg, split)
