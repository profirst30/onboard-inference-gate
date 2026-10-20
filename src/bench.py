"""Bench: latency, memory, size, accuracy and node placement per execution provider.

    python -m src.bench [--threads 1 4] [--providers CPUExecutionProvider]

Each latency and memory measurement runs in a fresh process, so one model cannot skew the next.
On another target (a Jetson with TensorRT, for instance), change --providers.
"""
import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

from .data import ROOT, load, to_input
from .train import MODELS

RESULTS = ROOT / "results" / "bench.json"
DOCS = ROOT / "docs" / "results.json"


def peak_mb() -> float:
    """Peak memory of this process since it started, in MB."""
    if sys.platform == "win32":
        import psutil
        return psutil.Process().memory_info().peak_wset / 2**20
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**10  # ru_maxrss is in KB on Linux


def session(path: Path, threads: int, providers: list[str], profile_dir: str | None = None) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if profile_dir:
        so.enable_profiling = True
        so.profile_file_prefix = str(Path(profile_dir) / "profile")
    return ort.InferenceSession(str(path), so, providers=providers)


def measure(path: Path, threads: int, providers: list[str], runs: int = 300, warmup: int = 30) -> dict:
    """Single-image latency (batch 1) and memory added by the session. Called in a fresh process."""
    # Random input: a CNN's latency does not depend on the image content, and loading the dataset here
    # would raise the memory peak before the measurement and hide what the session itself allocates.
    x = np.random.default_rng(0).random((1, 3, 64, 64), dtype=np.float32)
    before = peak_mb()
    s = session(path, threads, providers)
    for _ in range(warmup):
        s.run(None, {"image": x})
    t = []
    for _ in range(runs):
        t0 = time.perf_counter_ns()
        s.run(None, {"image": x})
        t.append((time.perf_counter_ns() - t0) / 1e6)
    t = np.array(t)
    return {"latency_ms": {"p50": float(np.percentile(t, 50)), "p95": float(np.percentile(t, 95)),
                           "p99": float(np.percentile(t, 99)), "mean": float(t.mean())},
            "session_mb": round(peak_mb() - before, 2), "runs": runs}


def placement(path: Path, providers: list[str]) -> dict[str, dict[str, int]]:
    """Nodes run by each provider, per operator type, from ORT profiling (optimized graph)."""
    with tempfile.TemporaryDirectory() as tmp:
        s = session(path, 1, providers, profile_dir=tmp)
        s.run(None, {"image": to_input(load()["x_test"][:1])})
        events = json.loads(Path(s.end_profiling()).read_text())
    c = Counter((e["args"]["op_name"], e["args"]["provider"]) for e in events
                if e.get("cat") == "Node" and e.get("name", "").endswith("_kernel_time"))
    out: dict[str, dict[str, int]] = {}
    for (op, prov), n in sorted(c.items()):
        out.setdefault(op, {})[prov] = n
    return out


def accuracy(path: Path, providers: list[str]) -> float:
    d = load()
    s = session(path, 0, providers)  # 0 = one thread per core: evaluation is not a latency measurement
    pred = np.concatenate([s.run(None, {"image": to_input(d["x_test"][i:i + 256])})[0].argmax(1)
                           for i in range(0, len(d["y_test"]), 256)])
    return float((pred == d["y_test"]).mean())


def describe(path: Path) -> dict:
    m = onnx.load(str(path))
    return {"size_bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "opset": max(o.version for o in m.opset_import if o.domain in ("", "ai.onnx")),
            "op_types_graph": dict(Counter(n.op_type for n in m.graph.node))}


def cpu_name() -> str:
    if sys.platform == "win32":
        r = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                           capture_output=True, text=True)
        return r.stdout.strip() or platform.processor()
    return platform.processor() or platform.machine()


def main(threads: list[int], providers: list[str]) -> None:
    models = sorted(MODELS.glob("eurosat_*.onnx"))
    res = {"machine": {"cpu": cpu_name(), "os": platform.platform(), "onnxruntime": ort.__version__,
                       "providers": providers, "test_images": int(len(load()["y_test"]))},
           "models": {}, "runs": []}
    for m in models:
        print(f"{m.name}: accuracy, placement...")
        res["models"][m.stem] = {**describe(m), "accuracy": accuracy(m, providers), "placement": placement(m, providers)}
        for t in threads:
            print(f"  latency, {t} thread(s)...")
            out = subprocess.run([sys.executable, "-m", "src.bench", "--measure", str(m), str(t), *providers],
                                 capture_output=True, text=True, check=True, cwd=ROOT)
            res["runs"].append({"model": m.stem, "threads": t, **json.loads(out.stdout.strip().splitlines()[-1])})
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(res, indent=2))
    shutil.copyfile(RESULTS, DOCS)
    print(f"-> {RESULTS}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--measure"]:
        print(json.dumps(measure(Path(sys.argv[2]), int(sys.argv[3]), sys.argv[4:])))
    else:
        p = argparse.ArgumentParser()
        p.add_argument("--threads", type=int, nargs="+", default=[1, 4])
        p.add_argument("--providers", nargs="+", default=["CPUExecutionProvider"])
        a = p.parse_args()
        main(a.threads, a.providers)
