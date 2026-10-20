"""Which layer costs the accuracy? Quantize every Conv/Gemm except one, and measure each time.

    python -m src.sensitivity
"""
import json
import shutil
import tempfile
from pathlib import Path

import onnx

from .bench import DOCS, RESULTS, accuracy
from .data import load
from .quantize import preprocess, quantize


def main() -> None:
    x = load()["x_train"][:512]
    pre = preprocess()
    nodes = [n.name for n in onnx.load(str(pre)).graph.node if n.op_type in ("Conv", "Gemm")]
    providers = ["CPUExecutionProvider"]
    res = {"fp32": accuracy(Path(__file__).resolve().parent.parent / "models" / "eurosat_fp32.onnx", providers)}
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "m.onnx"
        for kept in [None, *nodes]:
            quantize(pre, out, x, exclude=[kept] if kept else [])
            res[kept or "all int8"] = accuracy(out, providers)
            print(f"{kept or 'all int8':30} {100 * res[kept or 'all int8']:.2f} %")
    pre.unlink()
    path = RESULTS.with_name("sensitivity.json")
    path.write_text(json.dumps(res, indent=2))
    shutil.copyfile(path, DOCS.with_name("sensitivity.json"))


if __name__ == "__main__":
    main()
