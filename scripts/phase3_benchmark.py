#!/usr/bin/env python
"""Phase 3 CPU benchmark: DistilBERT vs MiniLM, FP32 vs INT8."""
from __future__ import annotations

import json
import platform
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.src.config import PROCESSED_DIR, REPORTS_DIR  # noqa: E402

MODELS = {
    "distilbert-squad-fp32": "distilbert/distilbert-base-uncased-distilled-squad",
    "minilm-squad2-fp32": "deepset/minilm-uncased-squad2",
}
VAL_DIR = PROCESSED_DIR / "val_eval_features"
SAMPLE_WINDOWS = 128
BATCH_SIZES = [1, 2, 4, 8, 16]
THREADS = 8


def monitor_memory(stop, output):
    process = psutil.Process()
    peak = 0.0
    while not stop.is_set():
        peak = max(peak, process.memory_info().rss / 1e9)
        time.sleep(0.05)
    output.append(peak)


def load_windows(limit):
    import pyarrow.parquet as pq

    shard = sorted(VAL_DIR.glob("shard_*.parquet"))[0]
    return pq.read_table(shard, columns=["input_ids"]).column(
        "input_ids"
    ).to_pylist()[:limit]


def make_batch(windows, pad_id):
    size = max(map(len, windows))
    inputs = torch.tensor([
        w + [pad_id] * (size - len(w)) for w in windows
    ])
    masks = torch.tensor([
        [1] * len(w) + [0] * (size - len(w)) for w in windows
    ])
    return inputs, masks


def forward_seconds(model, windows, batch_size, pad_id):
    total = 0.0

    for i in range(0, len(windows), batch_size):
        inputs, masks = make_batch(
            windows[i:i + batch_size], pad_id
        )

        start = time.perf_counter()
        with torch.inference_mode():
            model(input_ids=inputs, attention_mask=masks)
        total += time.perf_counter() - start

    return total


def best_spans(model, windows, pad_id):
    """Return best start/end indexes and confidence for each window."""
    results = []

    for i in range(0, len(windows), 16):
        inputs, masks = make_batch(windows[i:i + 16], pad_id)

        with torch.inference_mode():
            output = model(input_ids=inputs, attention_mask=masks)

        starts = torch.softmax(output.start_logits, dim=-1)
        ends = torch.softmax(output.end_logits, dim=-1)

        for k in range(len(inputs)):
            n = int(masks[k].sum())
            best = (0, 0, -1.0)

            for start in range(1, n):
                for end in range(start, min(start + 20, n)):
                    score = float(starts[k, start] * ends[k, end])
                    if score > best[2]:
                        best = (start, end, score)

            results.append((best[0], best[1], round(best[2], 5)))

    return results


def benchmark(name, model, pad_id, windows):
    forward_seconds(model, windows[:8], 8, pad_id)
    results = []

    for batch in BATCH_SIZES:
        seconds = forward_seconds(model, windows, batch, pad_id)
        results.append({
            "batch_size": batch,
            "wall_seconds": round(seconds, 2),
            "windows_per_second": round(len(windows) / seconds, 2),
            "ms_per_window": round(1000 * seconds / len(windows), 1),
        })
        print(f"  [{name}] batch={batch}: {len(windows) / seconds:.2f} win/s")

    return results, max(results, key=lambda x: x["windows_per_second"])


def main() -> int:
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer

    torch.set_num_threads(THREADS)
    windows = load_windows(SAMPLE_WINDOWS)

    print(f"Sample: {len(windows)} windows, threads={torch.get_num_threads()}")

    stop = threading.Event()
    peak = []
    threading.Thread(
        target=monitor_memory,
        args=(stop, peak),
        daemon=True,
    ).start()

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "device": "cpu",
        "cpu": {
            "processor": platform.processor(),
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "torch_threads": THREADS,
            "total_ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
        },
        "torch_version": torch.__version__,
        "sample_windows": len(windows),
        "variants": {},
        "int8_output_diff": {},
    }

    try:
        for name, model_id in MODELS.items():
            print(f"Benchmarking {name} ({model_id})")

            tokenizer = AutoTokenizer.from_pretrained(
                model_id, use_fast=True
            )
            model = AutoModelForQuestionAnswering.from_pretrained(model_id)
            model.eval()

            results, best = benchmark(
                name, model, tokenizer.pad_token_id, windows
            )

            report["variants"][name] = {
                "model_id": model_id,
                "tokenizer_max_len": tokenizer.model_max_length,
                "results": results,
                "best": best,
            }

            if name == "minilm-squad2-fp32":
                fp32 = best_spans(model, windows, tokenizer.pad_token_id)

                print("Quantizing MiniLM to INT8...")
                quantized = torch.ao.quantization.quantize_dynamic(
                    model,
                    {torch.nn.Linear},
                    dtype=torch.qint8,
                )
                quantized.eval()

                q_results, q_best = benchmark(
                    "minilm-squad2-int8",
                    quantized,
                    tokenizer.pad_token_id,
                    windows,
                )

                report["variants"]["minilm-squad2-int8"] = {
                    "model_id": model_id + " (dynamic INT8)",
                    "tokenizer_max_len": tokenizer.model_max_length,
                    "results": q_results,
                    "best": q_best,
                }

                int8 = best_spans(
                    quantized, windows, tokenizer.pad_token_id
                )

                same = sum(
                    a[:2] == b[:2] for a, b in zip(fp32, int8)
                )
                deltas = [
                    abs(a[2] - b[2])
                    for a, b in zip(fp32, int8)
                ]

                report["int8_output_diff"] = {
                    "windows_compared": len(fp32),
                    "identical_best_spans": same,
                    "identical_rate": round(same / len(fp32), 4),
                    "mean_abs_confidence_delta": round(
                        sum(deltas) / len(deltas), 6
                    ),
                    "max_abs_confidence_delta": round(
                        max(deltas), 6
                    ),
                }

                print(f"  INT8 identical spans: {same}/{len(fp32)}")
                del quantized

            del model

    finally:
        stop.set()
        time.sleep(0.1)

    report["peak_rss_gb_during_benchmark"] = (
        round(peak[0], 2) if peak else None
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    json_path = REPORTS_DIR / "phase3_benchmark.json"
    json_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    variants = report["variants"]
    md = [
        "# ContractIQ — Phase 3 CPU Model Benchmark",
        "",
        f"_Generated: {report['generated_at']}_ · "
        f"{report['cpu']['physical_cores']} cores · "
        f"torch threads {THREADS} · "
        f"{len(windows)} validation windows",
        "",
        "| Variant | Best batch | Windows/s | ms/window |",
        "|---|---:|---:|---:|",
    ]

    for name, data in variants.items():
        best = data["best"]
        md.append(
            f"| {name} | {best['batch_size']} | "
            f"{best['windows_per_second']} | "
            f"{best['ms_per_window']} |"
        )

    diff = report["int8_output_diff"]
    if diff:
        md += [
            "",
            f"INT8 output agreement: "
            f"{diff['identical_best_spans']}/{diff['windows_compared']} "
            f"identical best spans "
            f"({diff['identical_rate'] * 100:.1f}%), "
            f"mean |Δconfidence| = "
            f"{diff['mean_abs_confidence_delta']}",
        ]

    md += [
        "",
        "## Per-batch throughput (windows/s)",
        "",
        "| Batch | " + " | ".join(variants) + " |",
        "|---|" + "---:|" * len(variants),
    ]

    for batch in BATCH_SIZES:
        row = [str(batch)]
        for data in variants.values():
            result = next(
                x for x in data["results"]
                if x["batch_size"] == batch
            )
            row.append(str(result["windows_per_second"]))
        md.append("| " + " | ".join(row) + " |")

    md += [
        "",
        f"Peak RSS during benchmark: "
        f"{report['peak_rss_gb_during_benchmark']} GB",
        "",
    ]

    md_path = REPORTS_DIR / "phase3_benchmark.md"
    md_path.write_text("\n".join(md), encoding="utf-8")

    print(f"\nWrote {json_path} and {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
