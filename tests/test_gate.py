import onnx
from onnx import TensorProto, helper

from src.gate import check, manifest, verify

BUDGETS = {
    "reference": "fp32",
    "input_shape": ["batch", 3, 64, 64],
    "targets": {"small": {"threads": 1, "max_latency_p99_ms": 2.0, "max_model_kb": 128,
                          "max_session_mb": 16, "max_accuracy_drop_pts": 1.0, "max_opset": 17}},
}


def bench(int8_acc=0.95, int8_p99=1.0):
    run = lambda m, p99: {"model": m, "threads": 1, "latency_ms": {"p99": p99}, "session_mb": 5.0}
    return {"models": {"fp32": {"accuracy": 0.955, "size_bytes": 300 * 1024, "opset": 17},
                       "int8": {"accuracy": int8_acc, "size_bytes": 90 * 1024, "opset": 17}},
            "runs": [run("fp32", 1.5), run("int8", int8_p99)]}


def test_budgets():
    res = check(bench(), BUDGETS)
    assert res["int8"]["small"] == []
    assert res["fp32"]["small"] == ["size 300.00 KB > 128 KB"]
    res = check(bench(int8_acc=0.90, int8_p99=3.0), BUDGETS)
    assert [v.split()[0] for v in res["int8"]["small"]] == ["latency", "accuracy"]


def tiny_model(path, opset=17):
    x = helper.make_tensor_value_info("image", TensorProto.FLOAT, ["batch", 3, 64, 64])
    y = helper.make_tensor_value_info("out", TensorProto.FLOAT, ["batch", 3, 64, 64])
    g = helper.make_graph([helper.make_node("Relu", ["image"], ["out"])], "g", [x], [y])
    onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid("", opset)]), str(path))


def test_verify_accepts_then_rejects(tmp_path):
    p = tmp_path / "m.onnx"
    tiny_model(p)
    man = manifest(p, "small")
    assert verify(p, man, "small", BUDGETS) == []
    assert "not autre" in verify(p, man, "autre", BUDGETS)[0]
    p.write_bytes(p.read_bytes() + b"x")  # file altered after the manifest
    assert "SHA-256" in verify(p, man, "small", BUDGETS)[0]


def test_verify_rejects_opset(tmp_path):
    p = tmp_path / "m.onnx"
    tiny_model(p, opset=19)
    assert verify(p, manifest(p, "small"), "small", BUDGETS) == ["opset 19 > 17 supported by the target"]
