# Onboard inference gate

**A partner model only goes to orbit if it fits the target's budget, and the target refuses a file that was not built for it.**

Personal project, not affiliated with or endorsed by Loft Orbital. Demo page: https://profirst30.github.io/onboard-inference-gate/

## The problem

As I understand it from the [ML Runtime Engineer posting](https://jobs.lever.co/loftorbital/7bf12750-6b1f-4bac-a1ab-6709d13c60fa) and the [On-Orbit AI page](https://loftorbital.com/on-orbit-ai/): partners hand over ONNX models, and each one has to become an engine for a specific onboard accelerator, inside power and thermal budgets. Before upload, the ground side must show that a model fits. On board, the runtime must reject an engine compiled for another target, or a corrupted file.

This repository is a small, CPU-only version of that path, built in a day.

## What it does

1. **Partner model** (`src/train.py`): a 4-block CNN trained on [EuroSAT](https://github.com/phelber/EuroSAT) RGB (Sentinel-2, 10 land-cover classes, 64×64), exported to ONNX (opset 17).
2. **INT8** (`src/quantize.py`): static post-training quantization with ONNX Runtime, QDQ format, per-channel weights, calibrated on 512 training images. MinMax, Entropy and Percentile calibration are available; on this model none of them closed the accuracy gap.
3. **Sensitivity** (`src/sensitivity.py`): quantize every Conv/Gemm but one, and measure accuracy each time. It shows which layer costs the accuracy; keeping only that layer in FP32 gives the `mixed` model.
4. **Bench** (`src/bench.py`): for each model, size, SHA-256, accuracy on 2,700 held-out images, and which execution provider runs each node of the optimized graph (from ORT profiling). For each thread count, p50/p95/p99 latency over 300 single-image runs and the memory added by the session, each in a fresh process.
5. **Gate** (`src/gate.py check`): budgets per target in `budgets.yaml` (p99 latency, size, session memory, accuracy drop against FP32, max opset). A model is accepted only if it passes every check.
6. **Manifest** (`src/gate.py package` / `verify`): SHA-256, target, opset and input shape. `verify` exits non-zero on any mismatch, before the model is loaded.

## Results

<!-- results -->
Measured on AMD Ryzen 7 5825U with Radeon Graphics, Windows-11-10.0.26200-SP0, ONNX Runtime 1.31.0.

| model | size (KB) | accuracy | threads | p50 (ms) | p99 (ms) | session (MB) |
|---|---:|---:|---:|---:|---:|---:|
| eurosat_fp32 | 314.6 | 92.9 % | 1 | 0.319 | 0.443 | 8.0 |
| eurosat_fp32 | 314.6 | 92.9 % | 4 | 0.131 | 0.178 | 9.0 |
| eurosat_int8_minmax | 92.4 | 90.9 % | 1 | 0.372 | 0.611 | 7.8 |
| eurosat_int8_minmax | 92.4 | 90.9 % | 4 | 0.152 | 0.182 | 9.2 |
| eurosat_int8_minmax_mixed | 92.3 | 92.9 % | 1 | 0.441 | 0.668 | 8.0 |
| eurosat_int8_minmax_mixed | 92.3 | 92.9 % | 4 | 0.353 | 0.515 | 9.1 |

| model | cpu-1-thread | cpu-4-threads |
|---|---|---|
| eurosat_fp32 | rejected: size 314.61 KB > 128 KB | accepted |
| eurosat_int8_minmax | rejected: accuracy drop 2.00 pts > 1.0 pts | rejected: accuracy drop 2.00 pts > 1.0 pts |
| eurosat_int8_minmax_mixed | accepted | accepted |
<!-- /results -->

The table above is written by `python -m src.gate check` from `results/bench.json`; no number is typed by hand. The budgets in `budgets.yaml` are examples chosen to illustrate the check, not real spacecraft budgets.

## What the numbers say

- Full INT8 cuts the file to under a third of the FP32 size but fails the accuracy budget, whatever the calibration method.
- The sensitivity run points at a single layer, the first convolution (3 input channels, raw pixels). Keeping it in FP32 brings accuracy back to the FP32 level at almost no size cost.
- On this x86 CPU, INT8 is **not** faster than FP32 for a model this small: the quantize/dequantize steps and layout changes cost more than the integer arithmetic saves. INT8 here buys size, not speed. On an accelerator with native INT8 (Hailo, TensorRT INT8) the trade-off has to be measured again, which is exactly why the gate runs per target.

## Limits

- Measured on a laptop CPU only. I have no Jetson, Hailo or AMD hardware, so TensorRT and the other providers are untested. `src/bench.py --providers` accepts them, but those numbers do not exist yet.
- No power or thermal measurement, which is the real constraint on board.
- Latency on a laptop is noisy (power management, background load): p99 moves between runs.
- Session memory is the rise of the process's peak working set (peak RSS on Linux): an approximation.
- The model is deliberately small; real partner models are much larger and stress memory far more.

## Reproduce

```sh
python -m venv .venv && .venv/Scripts/activate        # source .venv/bin/activate on Linux
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
python -m src.train          # downloads EuroSAT (~90 MB) on the first run
python -m src.quantize
python -m src.sensitivity
python -m src.quantize --exclude /0/0.0/Conv   # Git Bash on Windows: prefix with MSYS_NO_PATHCONV=1
python -m src.bench          # --threads 1 4, --providers CPUExecutionProvider
python -m src.gate check
python -m src.gate package models/eurosat_int8_minmax_mixed.onnx --target cpu-1-thread
python -m src.gate verify  models/eurosat_int8_minmax_mixed.onnx --target cpu-1-thread
python -m pytest tests
```

The demo page (`docs/index.html`) reads `docs/results.json`, `docs/gate.json` and `docs/sensitivity.json`: serve the folder (`python -m http.server -d docs`) to view it locally.

Data: EuroSAT (Helber, Bischke, Dengel and Borth, IEEE JSTARS, 2019), MIT licence; contains modified Copernicus Sentinel data.

Nobel Djessou · [LinkedIn](https://www.linkedin.com/in/nobel-djessou/) · [GitHub](https://github.com/profirst30)
