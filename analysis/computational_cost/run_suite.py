"""Serial MS cost benchmark; never runs models concurrently."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def summarize(output):
    rows = []
    for file in output.glob("*/events.jsonl"):
        for line in file.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            if event["event"] == "end":
                rows.append(event)
    (output / "measurements.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    lines = ["# MS computational-cost measurements", "",
             "Single seed measurements, not uncertainty estimates. Inference is measured on the final trained model, with fixed inputs already prepared. RAM is sampled process working set; GPU memory is PyTorch allocated memory, not whole-device usage.", "",
             "| Method | Task | Train s | N2V training s | RF feature extraction s | Evaluation s | Inference ms | Peak RAM MiB | Peak GPU allocated MiB |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        stages = row["stages"]
        def seconds(key):
            return stages.get(key, {}).get("total_seconds", 0)
        lines.append(f"| {row['method']} | {row['task']} | {seconds('train')+seconds('rf_fit'):.3f} | {seconds('run_node2vec'):.3f} | {seconds('extract_features'):.3f} | {seconds('ttest'):.3f} | {1000*row['inference_mean_seconds']:.3f} | {row['peak_process_ram_mib']:.1f} | {row['peak_gpu_allocated_mib']:.1f} |")
    (output / "summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-root", required=True)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--task", default="all")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.glob("*/events.jsonl")):
        p.error("Use a fresh output directory to avoid mixing benchmark runs.")
    methods = ["RF", "TriNet", "N2V-MLP", "TriMoGCL", "HANAMI"]
    completed = []
    for method in methods:
        status = dict(state="running", method=method, completed=completed, updated=time.ctime())
        (args.output/"status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        command = [sys.executable, "-u", str(Path(__file__).with_name("benchmark_ms.py")),
                   "--baseline-root", args.baseline_root, "--method", method,
                   "--output", str(args.output), "--epochs", str(args.epochs),
                   "--seed", str(args.seed), "--task", args.task]
        with (args.output/f"{method}.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        summarize(args.output)
        if result.returncode:
            status.update(state="failed", exit_code=result.returncode)
            (args.output/"status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
            raise SystemExit(result.returncode)
        completed.append(method)
    (args.output/"status.json").write_text(json.dumps(dict(state="complete", completed=completed,
                                                         updated=time.ctime()), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
