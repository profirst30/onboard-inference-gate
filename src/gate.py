"""Validation gate: a model only goes to a target if it fits that target's budgets,
and the target refuses a file that does not match its manifest.

    python -m src.gate check                                   # budgets, from results/bench.json
    python -m src.gate package models/eurosat_int8_minmax_mixed.onnx --target cpu-1-thread
    python -m src.gate verify  models/eurosat_int8_minmax_mixed.onnx --target cpu-1-thread
"""
import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import onnx
import yaml

from .data import ROOT

BUDGETS = ROOT / "budgets.yaml"
BENCH = ROOT / "results" / "bench.json"
GATE = ROOT / "results" / "gate.json"


def violations(model: str, bench: dict, target: dict, reference: str) -> list[str]:
    """Budgets that `model` exceeds on `target`; empty means accepted."""
    info = bench["models"][model]
    run = next((r for r in bench["runs"] if r["model"] == model and r["threads"] == target["threads"]), None)
    if run is None:
        return [f"no measurement at {target['threads']} thread(s)"]
    drop = 100 * (bench["models"][reference]["accuracy"] - info["accuracy"])
    checks = [
        ("latency p99", run["latency_ms"]["p99"], target["max_latency_p99_ms"], "ms"),
        ("size", info["size_bytes"] / 1024, target["max_model_kb"], "KB"),
        ("session memory", run["session_mb"], target["max_session_mb"], "MB"),
        ("accuracy drop", drop, target["max_accuracy_drop_pts"], "pts"),
        ("opset", info["opset"], target["max_opset"], ""),
    ]
    return [f"{name} {v:.2f} {u} > {limit} {u}".rstrip() for name, v, limit, u in checks if v > limit]


def check(bench: dict, budgets: dict) -> dict:
    return {model: {t: violations(model, bench, spec, budgets["reference"]) for t, spec in budgets["targets"].items()}
            for model in bench["models"]}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest(path: Path, target: str) -> dict:
    m = onnx.load(str(path))
    inp = m.graph.input[0]
    return {"model": path.name, "sha256": sha256(path), "target": target,
            "opset": max(o.version for o in m.opset_import if o.domain in ("", "ai.onnx")),
            "input": {"name": inp.name,
                      "shape": [d.dim_param or d.dim_value for d in inp.type.tensor_type.shape.dim]},
            "quantized": any(n.op_type == "QuantizeLinear" for n in m.graph.node)}


def verify(path: Path, man: dict, target: str, budgets: dict) -> list[str]:
    """Checks the target runs before loading a model; empty means it may be loaded."""
    if not path.exists():
        return [f"missing file: {path}"]
    errors = []
    if sha256(path) != man["sha256"]:
        errors.append("SHA-256 differs from the manifest: modified or wrong file")
    if man["target"] != target:
        errors.append(f"built for target {man['target']}, not {target}")
    spec = budgets["targets"].get(target)
    if spec is None:
        errors.append(f"unknown target: {target}")
    elif man["opset"] > spec["max_opset"]:
        errors.append(f"opset {man['opset']} > {spec['max_opset']} supported by the target")
    if man["input"]["shape"] != budgets["input_shape"]:
        errors.append(f"input {man['input']['shape']} instead of {budgets['input_shape']}")
    return errors


def results_table(bench: dict, res: dict) -> str:
    """Markdown summary for the README, built from the measurements."""
    m = bench["machine"]
    lines = [f"Measured on {m['cpu']}, {m['os']}, ONNX Runtime {m['onnxruntime']}.", "",
             "| model | size (KB) | accuracy | threads | p50 (ms) | p99 (ms) | session (MB) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in bench["runs"]:
        info = bench["models"][r["model"]]
        lines.append(f"| {r['model']} | {info['size_bytes'] / 1024:.1f} | {100 * info['accuracy']:.1f} % | {r['threads']} "
                     f"| {r['latency_ms']['p50']:.3f} | {r['latency_ms']['p99']:.3f} | {r['session_mb']:.1f} |")
    lines += ["", "| model | " + " | ".join(next(iter(res.values()))) + " |", "|---|" + "---|" * len(next(iter(res.values())))]
    for model, by_target in res.items():
        lines.append(f"| {model} | " + " | ".join("accepted" if not v else "rejected: " + "; ".join(v) for v in by_target.values()) + " |")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["check", "package", "verify"])
    p.add_argument("model", nargs="?", type=Path)
    p.add_argument("--target")
    a = p.parse_args()
    budgets = yaml.safe_load(BUDGETS.read_text(encoding="utf-8"))

    if a.cmd == "check":
        bench = json.loads(BENCH.read_text())
        res = check(bench, budgets)
        GATE.write_text(json.dumps(res, indent=2))
        shutil.copyfile(GATE, ROOT / "docs" / "gate.json")
        readme = ROOT / "README.md"
        text = re.sub(r"<!-- results -->.*<!-- /results -->", lambda _: f"<!-- results -->\n{results_table(bench, res)}\n<!-- /results -->",
                      readme.read_text(encoding="utf-8"), flags=re.S)
        readme.write_text(text, encoding="utf-8")
        for model, by_target in res.items():
            for t, v in by_target.items():
                print(f"{model:15} {t:15} {'ACCEPTED' if not v else 'REJECTED: ' + '; '.join(v)}")
        return 0

    if not a.model or not a.target:
        p.error("package and verify need a model and --target")
    man_path = a.model.with_suffix(".manifest.json")
    if a.cmd == "package":
        man_path.write_text(json.dumps(manifest(a.model, a.target), indent=2))
        print(f"-> {man_path}")
        return 0
    errors = verify(a.model, json.loads(man_path.read_text()), a.target, budgets)
    print("OK: the model may be loaded" if not errors else "REJECTED:\n  " + "\n  ".join(errors))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
