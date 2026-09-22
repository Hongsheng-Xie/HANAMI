"""Run all five native MS implementations serially across ten matched seeds.

Resume skips only jobs with successful exit markers and seven completed tasks.
Failures and timing flags are retained, never silently replaced or excluded.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

METHODS = ["RF", "TriNet", "N2V-MLP", "TriMoGCL", "HANAMI"]
SEEDS = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90]
DATASET = "ms"


def save_text(path, content):
    """Retry transient Windows file locks without changing any measurements."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    for attempt in range(10):
        try:
            temporary.write_text(content, encoding="utf-8")
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(min(0.25 * 2**attempt, 2))


def save(path, value):
    save_text(path, json.dumps(value, indent=2))


def read_events(file):
    if not file.exists():
        return []
    events = []
    for line in file.read_text(encoding="utf-8").splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            # The worker can be in the middle of appending its latest event.
            continue
    return events


def summarize(output):
    tasks, jobs, flags = [], [], []
    for seed in SEEDS:
        for method in METHODS:
            directory = output / f"seed_{seed}" / method
            events = read_events(directory / "events.jsonl")
            ends = [row for row in events if row["event"] == "end"]
            tasks.extend(ends)
            flags.extend(dict(seed=seed, method=method, task=row["task"], flags=row["timing_flags"])
                         for row in ends if row.get("timing_flags"))
            if len(ends) != 7 or len({row["task"] for row in ends}) != 7:
                continue
            if not (directory / "success.json").exists():
                continue
            training = sum(row["stages"].get(stage, {}).get("total_seconds", 0)
                           for row in ends for stage in ("train", "rf_fit", "run_node2vec"))
            jobs.append(dict(seed=seed, method=method, training_minutes=training/60,
                             inference_ms=sum(row["inference_mean_seconds"] for row in ends)*1000,
                             peak_ram_gib=max(row["peak_process_ram_mib"] for row in ends)/1024,
                             peak_gpu_gib=max(row["peak_gpu_allocated_mib"] for row in ends)/1024,
                             test_motifs=sum(row["test_motifs"] for row in ends),
                             timing_flags=sum(len(row.get("timing_flags", [])) for row in ends)))
    save(output/"completed_tasks.json", tasks)
    save(output/"seed_level_costs.json", jobs)
    save(output/"timing_flags.json", flags)
    total_jobs = len(SEEDS) * len(METHODS)
    lines = [f"# {DATASET.upper()} computational-cost benchmark ({len(SEEDS)} seed(s))", "",
             f"Completed {len(tasks)}/{total_jobs * 7} tasks and {len(jobs)}/{total_jobs} complete model-seed jobs.", "",
             "Training and inference are summed over seven tasks within each seed. Memory is the peak across those tasks. Mean and sample SD below are calculated across completed seeds, not across tasks. Partial results are provisional; timing flags require review and are not automatically excluded.", "",
             "| Model | Seeds complete | Training min, mean ± SD | Inference ms, mean ± SD | Peak RAM GiB, mean ± SD | Peak GPU GiB, mean ± SD | Flagged calls |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        group = [row for row in jobs if row["method"] == method]
        def fmt(key):
            if not group:
                return "pending"
            values = [row[key] for row in group]
            sd = f"{statistics.stdev(values):.2f}" if len(values)>1 else "not available"
            return f"{statistics.mean(values):.2f} ± {sd}"
        lines.append(f"| {method} | {len(group)}/{len(SEEDS)} | {fmt('training_minutes')} | {fmt('inference_ms')} | {fmt('peak_ram_gib')} | {fmt('peak_gpu_gib') if method!='RF' else 'CPU fitting/inference'} | {sum(row['timing_flags'] for row in group)} |")
    lines.extend(["", "Node2Vec training is included. Offline input-feature generation, data preparation, and evaluation are excluded from training time. These measurements retain each implementation's features and graph processing. They are not an identical-feature or identical-graph architecture ablation."])
    save_text(output/"summary.md", "\n".join(lines)+"\n")
    if len(SEEDS) == 1:
        table = [f"# {DATASET.upper()} computational cost, seed {SEEDS[0]}", "",
                 "Total training time across seven tasks; maximum allocated GPU memory across tasks.", "",
                 "| Model | Total training time | Peak GPU memory |",
                 "|---|---:|---:|"]
        for method in METHODS:
            row = next((job for job in jobs if job["method"] == method), None)
            if row is None:
                table.append(f"| {method} | Pending | Pending |")
            else:
                gpu = "CPU only" if method == "RF" else f"{row['peak_gpu_gib']:.2f} GiB"
                table.append(f"| {method} | {row['training_minutes']:.2f} min | {gpu} |")
        if flags:
            table.extend(["", "Timing flags are present. Raw times are retained pending review."])
        save_text(output/"cost_table.md", "\n".join(table)+"\n")
    return len(tasks), len(jobs), len(flags)


def main():
    global SEEDS, METHODS, DATASET
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--dataset", choices=["ms", "drkg"], default="ms")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--method-order", nargs="+", choices=METHODS)
    parser.add_argument("--epochs", type=int, default=150)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must be distinct.")
    if args.epochs < 1:
        parser.error("Epochs must be positive.")
    if len(set(args.methods)) != len(args.methods):
        parser.error("Methods must be distinct.")
    if args.method_order is None:
        args.method_order = args.methods
    if len(args.method_order) != len(args.methods) or set(args.method_order) != set(args.methods):
        parser.error("Method order must contain every selected method exactly once.")
    SEEDS = args.seeds
    METHODS = args.method_order
    DATASET = args.dataset
    total_jobs = len(SEEDS) * len(METHODS)
    total_tasks = total_jobs * 7
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    worker = Path(__file__).with_name("benchmark_ms.py")
    configuration = dict(seeds=SEEDS, methods=METHODS, epochs=args.epochs, task="all", inference_repeats=20,
                         baseline_root=str(args.baseline_root.resolve()), python=sys.executable,
                         worker_sha256=hashlib.sha256(worker.read_bytes()).hexdigest())
    configuration["dataset"] = DATASET
    manifest = output/"configuration.json"
    if manifest.exists():
        if not args.resume:
            parser.error("Output already contains a run. Use --resume or a fresh directory.")
        if json.loads(manifest.read_text()) != configuration:
            parser.error("Configuration changed. Use a fresh output directory.")
    else:
        save(manifest, configuration)
    completed = []
    # Fingerprints must stay fixed across seeds; split hashes must match across methods.
    fingerprints, split_hashes = {}, {}
    start = time.perf_counter()
    for seed in SEEDS:
        for method in METHODS:
            directory = output/f"seed_{seed}"/method
            marker = directory/"success.json"
            if marker.exists():
                completed.append(dict(seed=seed, method=method))
            else:
                if (directory/"events.jsonl").exists():
                    raise RuntimeError(f"Incomplete attempt retained at {directory}; review before retrying.")
                directory.mkdir(parents=True, exist_ok=True)
                command = [sys.executable, "-u", str(worker), "--baseline-root", str(args.baseline_root),
                           "--method", method, "--dataset", DATASET, "--seed", str(seed), "--epochs", str(args.epochs), "--task", "all",
                           "--inference-repeats", "20", "--output", str(output/f"seed_{seed}")]
                with (directory/"console.log").open("w", encoding="utf-8") as log:
                    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                    while True:
                        code = process.poll()
                        task_count, job_count, flag_count = summarize(output)
                        events = read_events(directory/"events.jsonl")
                        last = events[-1] if events else {}
                        progress = next((row for row in reversed(events) if row["event"]=="progress" and row["task"]==last.get("task")), {})
                        save(output/"status.json", dict(state="running" if code is None else "checking",
                             seed=seed, method=method, current_task=last.get("task"),
                             completed_epochs=progress.get("epochs"), worker_pid=process.pid,
                             completed_tasks=task_count, total_tasks=total_tasks, completed_jobs=job_count,
                             total_jobs=total_jobs, flagged_tasks=flag_count, updated=datetime.now().isoformat(),
                             elapsed_seconds=time.perf_counter()-start))
                        if code is not None:
                            break
                        time.sleep(10)
                if code != 0:
                    raise RuntimeError(f"{method} seed {seed} exited {code}; see {directory/'console.log'}")
                ends = [row for row in events if row["event"]=="end"]
                if len(ends)!=7 or len({row["task"] for row in ends})!=7:
                    raise RuntimeError("Expected exactly seven completed tasks")
            provenance = json.loads((directory/"provenance.json").read_text())
            fingerprint = {key:provenance[key] for key in ("feature_files", "source_files", f"{DATASET}_arrays")}
            if method in fingerprints and fingerprints[method] != fingerprint:
                raise RuntimeError(f"Source or input files changed for {method}")
            fingerprints[method] = fingerprint
            events = read_events(directory/"events.jsonl")
            data = next(row for row in events if row["event"]=="data")
            if seed in split_hashes and split_hashes[seed] != data["split_sha256"]:
                raise RuntimeError(f"Sample splits differ across models for seed {seed}")
            split_hashes[seed] = data["split_sha256"]
            if not marker.exists():
                save(marker, dict(seed=seed, method=method, tasks=7, exit_code=0, verified=datetime.now().isoformat()))
                completed.append(dict(seed=seed, method=method))
            summarize(output)
    task_count, job_count, flag_count = summarize(output)
    save(output/"status.json", dict(state="complete_pending_timing_review" if flag_count else "complete",
         completed_tasks=task_count, total_tasks=total_tasks, completed_jobs=job_count, total_jobs=total_jobs,
         flagged_tasks=flag_count, updated=datetime.now().isoformat(), elapsed_seconds=time.perf_counter()-start))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Never leave a stale 'running' status after a coordinator failure.
        if "--output" in sys.argv:
            output = Path(sys.argv[sys.argv.index("--output")+1])
            if output.exists():
                save(output/"status.json", dict(state="failed", error=str(exc), updated=datetime.now().isoformat()))
        raise
