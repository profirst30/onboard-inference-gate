"""EuroSAT RGB (Sentinel-2, 10 classes, 64x64), loaded once and cached as .npz."""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "eurosat.npz"
CLASSES = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
           "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]


def load() -> dict[str, np.ndarray]:
    """Return x_train, y_train, x_val, y_val, x_test, y_test (uint8 NHWC images)."""
    if not CACHE.exists():
        from torchvision.datasets import EuroSAT  # only needed on the first run
        ds = EuroSAT(str(ROOT / "data"), download=True)
        x = np.stack([np.asarray(img.convert("RGB")) for img, _ in ds])
        y = np.array([label for _, label in ds], dtype=np.int64)
        idx = np.random.default_rng(0).permutation(len(y))  # fixed, reproducible 80/10/10 split
        n_val, n_test = len(y) // 10, len(y) // 10
        parts = {"test": idx[:n_test], "val": idx[n_test:n_test + n_val], "train": idx[n_test + n_val:]}
        np.savez_compressed(CACHE, **{f"{k}_{s}": a[i] for s, i in parts.items() for k, a in (("x", x), ("y", y))})
    with np.load(CACHE) as f:
        return dict(f)


def to_input(x_uint8: np.ndarray) -> np.ndarray:
    """uint8 NHWC -> float32 NCHW in [0, 1], the model's input format."""
    return (x_uint8.astype(np.float32) / 255.0).transpose(0, 3, 1, 2).copy()
