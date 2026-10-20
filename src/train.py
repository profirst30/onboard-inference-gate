"""Train a small CNN on EuroSAT and export it to ONNX: the "partner model" fed to the bench.

    python -m src.train [--epochs 6]
"""
import argparse
import json

import numpy as np
import torch
from torch import nn

from .data import CLASSES, ROOT, load, to_input

MODELS = ROOT / "models"


def make_model() -> nn.Module:
    def block(c_in, c_out):
        return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(), nn.MaxPool2d(2))
    return nn.Sequential(block(3, 16), block(16, 32), block(32, 64), block(64, 96),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(96, len(CLASSES)))


def accuracy(model: nn.Module, x: np.ndarray, y: np.ndarray) -> float:
    model.eval()
    with torch.no_grad():
        pred = torch.cat([model(torch.from_numpy(to_input(x[i:i + 512]))).argmax(1) for i in range(0, len(x), 512)])
    return float((pred.numpy() == y).mean())


def main(epochs: int) -> None:
    torch.manual_seed(0)
    d = load()
    model = make_model()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=4e-3, total_steps=epochs * (len(d["y_train"]) // 128 + 1))
    loss_fn = nn.CrossEntropyLoss()
    rng = np.random.default_rng(0)
    for epoch in range(epochs):
        model.train()
        for i in np.array_split(rng.permutation(len(d["y_train"])), len(d["y_train"]) // 128 + 1):
            x = d["x_train"][i]
            if rng.random() < 0.5:
                x = x[:, :, ::-1]  # horizontal flip: an overhead view has no left or right
            loss = loss_fn(model(torch.from_numpy(to_input(x))), torch.from_numpy(d["y_train"][i]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        print(f"epoch {epoch + 1}/{epochs}: validation accuracy {accuracy(model, d['x_val'], d['y_val']):.3f}")

    MODELS.mkdir(exist_ok=True)
    model.eval()
    torch.onnx.export(model, torch.zeros(1, 3, 64, 64), MODELS / "eurosat_fp32.onnx", input_names=["image"],
                      output_names=["logits"], dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=17, dynamo=False)
    acc = accuracy(model, d["x_test"], d["y_test"])
    (MODELS / "train_summary.json").write_text(json.dumps({"epochs": epochs, "test_accuracy_pytorch": acc}, indent=2))
    print(f"test accuracy (PyTorch): {acc:.3f} -> {MODELS / 'eurosat_fp32.onnx'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=6)
    main(p.parse_args().epochs)
