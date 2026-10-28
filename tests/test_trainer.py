import json

import torch

from speechlab.data.synthesize import make_train_cases
from speechlab.engine.trainer import Trainer


class ConstantMask(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logit = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, mag):
        return torch.sigmoid(self.logit).expand_as(mag)


def test_variable_length_and_early_stop(tmp_path):
    cases = make_train_cases(n=2, duration=0.10, seed=7)
    cases.append(make_train_cases(n=1, duration=0.15, seed=8)[0])
    cfg = {
        "train": {
            "epochs": 20,
            "batch": 2,
            "lr": 0.0,
            "log_every": 1,
            "early_stop": {"metric": "val_loss", "patience": 1},
        }
    }
    model = ConstantMask()
    history = Trainer(model, cfg, tmp_path).fit(cases, cases)
    assert len(history) == 2
    assert (tmp_path / "best.pt").exists()
    assert (tmp_path / "last.pt").exists()
    assert len((tmp_path / "history.jsonl").read_text().splitlines()) == 2
    summary = json.loads((tmp_path / "training_summary.json").read_text())
    assert summary["stopped_early"] is True
    assert summary["best_epoch"] == 1


def test_resume_training_state(tmp_path):
    cases = make_train_cases(n=2, duration=0.10, seed=11)
    cfg = {"train": {"epochs": 1, "batch": 2, "lr": 0.001, "log_every": 1}}
    Trainer(ConstantMask(), cfg, tmp_path).fit(cases, cases)
    assert (tmp_path / "trainer_state.pt").exists()

    cfg["train"]["epochs"] = 2
    cfg["train"]["resume"] = True
    history = Trainer(ConstantMask(), cfg, tmp_path).fit(cases, cases)
    assert [row["epoch"] for row in history] == [1, 2]
    assert len((tmp_path / "history.jsonl").read_text().splitlines()) == 2


def test_magnitude_log_loss_and_default_compatibility(tmp_path):
    enhanced = torch.tensor([[[0.25, 1.0]]])
    clean = torch.tensor([[[0.0, 0.5]]])
    valid = torch.ones((1, 1, 1))
    expected_mag_l1 = (torch.tensor(0.25) + torch.tensor(0.5)) / 2
    expected_log_l1 = (torch.log1p(torch.tensor(0.25)) +
                       (torch.log1p(torch.tensor(1.0)) - torch.log1p(torch.tensor(0.5)))) / 2

    default = Trainer(ConstantMask(), {
        "train": {"epochs": 1, "batch": 1, "lr": 0.0}
    }, tmp_path / "default")
    noisy = torch.ones_like(enhanced)
    terms = default._loss_terms(enhanced, noisy, clean, valid)
    assert torch.isclose(terms["loss"], expected_mag_l1)
    assert "log_l1" not in terms

    composite = Trainer(ConstantMask(), {
        "train": {"epochs": 1, "batch": 1, "lr": 0.0,
                  "loss": {"name": "mag_log_l1", "log_weight": 0.5}}
    }, tmp_path / "composite")
    terms = composite._loss_terms(enhanced, noisy, clean, valid)
    assert torch.isclose(terms["mag_l1"], expected_mag_l1)
    assert torch.isclose(terms["log_l1"], expected_log_l1)
    assert torch.isclose(terms["loss"], expected_mag_l1 + 0.5 * expected_log_l1)


def test_ideal_ratio_mask_loss(tmp_path):
    trainer = Trainer(ConstantMask(), {
        "train": {"epochs": 1, "batch": 1, "lr": 0.0, "loss": {"name": "irm_l1"}}
    }, tmp_path)
    mask = torch.tensor([[[0.5, 0.25]]])
    noisy = torch.tensor([[[0.5, 2.0]]])
    clean = torch.tensor([[[0.0, 0.5]]])
    valid = torch.ones((1, 1, 1))
    terms = trainer._loss_terms(mask, noisy, clean, valid)
    assert torch.isclose(terms["irm_l1"], torch.tensor(0.25))
    assert torch.isclose(terms["loss"], terms["irm_l1"])
    assert torch.isclose(terms["mag_l1"], torch.tensor(0.125))

    combined = Trainer(ConstantMask(), {
        "train": {"epochs": 1, "batch": 1, "lr": 0.0,
                  "loss": {"name": "mag_irm_l1", "irm_weight": 0.25}}
    }, tmp_path / "combined")
    terms = combined._loss_terms(mask, noisy, clean, valid)
    assert torch.isclose(terms["mag_l1"], torch.tensor(0.125))
    assert torch.isclose(terms["irm_l1"], torch.tensor(0.25))
    assert torch.isclose(terms["loss"], torch.tensor(0.1875))
