"""Static INT8 quantization (QDQ format) of the FP32 model, calibrated on training images.

    python -m src.quantize [--calib 512] [--methods minmax entropy percentile] [--exclude NODE ...] [--reduce-range]

One INT8 model per calibration method: the method decides how each activation range is chosen.
--exclude keeps nodes in FP32; --reduce-range stores weights on 7 bits (see README, "What the numbers say").
"""
import argparse
from pathlib import Path

import numpy as np
from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process

from .data import load, to_input
from .train import MODELS


class Calibration(CalibrationDataReader):
    """Feeds calibration images in batches of 32; ORT records the range of every activation."""

    def __init__(self, x: np.ndarray):
        self.batches = iter([to_input(x[i:i + 32]) for i in range(0, len(x), 32)])

    def get_next(self):
        b = next(self.batches, None)
        return None if b is None else {"image": b}


METHODS = {"minmax": CalibrationMethod.MinMax, "entropy": CalibrationMethod.Entropy,
           "percentile": CalibrationMethod.Percentile}


def preprocess() -> Path:
    pre = MODELS / "eurosat_fp32_pre.onnx"
    quant_pre_process(str(MODELS / "eurosat_fp32.onnx"), str(pre))  # shape inference + fusions, recommended before quantizing
    return pre


def quantize(pre: Path, out: Path, x: np.ndarray, method: str = "minmax", exclude: list[str] = (),
             reduce_range: bool = False) -> None:
    # reduce_range stores weights on 7 bits. On x86 CPUs without VNNI, U8S8 kernels add pairs of u8*s8
    # products into int16 and can saturate; 7-bit weights keep those sums in range.
    quantize_static(str(pre), str(out), Calibration(x), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,  # U8S8: the fast path on x86 CPUs
                    calibrate_method=METHODS[method], nodes_to_exclude=list(exclude), reduce_range=reduce_range)


def main(n_calib: int, methods: list[str], exclude: list[str], reduce_range: bool) -> None:
    x = load()["x_train"][:n_calib]  # training images: the test split stays untouched for evaluation
    pre = preprocess()
    for name in methods:
        out = MODELS / f"eurosat_int8_{name}{'_mixed' if exclude else ''}{'_rr' if reduce_range else ''}.onnx"
        quantize(pre, out, x, name, exclude, reduce_range)
        print(f"-> {out}")
    pre.unlink()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--calib", type=int, default=512)
    p.add_argument("--methods", nargs="+", choices=list(METHODS), default=["minmax"])
    p.add_argument("--exclude", nargs="+", default=[], help="node names kept in FP32 (see src.sensitivity)")
    p.add_argument("--reduce-range", action="store_true", help="7-bit weights, against U8S8 saturation on CPUs without VNNI")
    a = p.parse_args()
    main(a.calib, a.methods, a.exclude, a.reduce_range)
