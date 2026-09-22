"""Rebuild cost summaries from archived completed-task records, without training.

No timing outliers are removed. Each run remains separate; historical reruns are
not substituted unless an existing audit explicitly specifies the replacement.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics

METHODS = ["RF", "TriNet", "N2V-MLP", "TriMoGCL", "HANAMI"]
METRICS = ["training_minutes", "inference_ms", "peak_ram_gib", "peak_gpu_gib"]
REPO = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_tasks(tasks):
    jobs = []
    groups = defaultdict(list)
    for row in tasks:
        groups[row["method"], row["seed"]].append(row)
    for (method, seed), rows in sorted(groups.items()):
        if len(rows) != 7 or len({row["task"] for row in rows}) != 7:
            continue
        jobs.append(dict(method=method, seed=seed, tasks=7,
            training_minutes=sum(row["stages"].get(stage, {}).get("total_seconds", 0)
                for row in rows for stage in ("train", "rf_fit", "run_node2vec")) / 60,
            inference_ms=sum(row["inference_mean_seconds"] for row in rows) * 1000,
            peak_ram_gib=max(row["peak_process_ram_mib"] for row in rows) / 1024,
            peak_gpu_gib=max(row["peak_gpu_allocated_mib"] for row in rows) / 1024,
            timing_flags=sum(len(row.get("timing_flags", [])) for row in rows)))
    return jobs


def verify_inventory(directory):
    for item in read(directory / "archive_inventory.json")["files"]:
        actual = hashlib.sha256((directory / item["path"]).read_bytes()).hexdigest()
        if actual != item["published_sha256"]:
            raise ValueError(f"Archived record changed: {directory.name}/{item['path']}")


def read_tasks(directory):
    # Prefer actual worker events. For original single-seed runs, the archived
    # root status proves suite completion; later suites have success markers.
    complete = (read(directory / "status.json").get("state") == "complete"
                if (directory / "status.json").exists() else False)
    rows = []
    for event_file in sorted(directory.rglob("events.jsonl")):
        if not (event_file.with_name("success.json").exists() or complete):
            continue
        events = [json.loads(line) for line in event_file.read_text().splitlines() if line.strip()]
        rows.extend(row for row in events if row.get("event") == "end")
    return rows


def build(data_root, output):
    output.mkdir(parents=True, exist_ok=True)
    manifests = sorted(data_root.glob("*/archive_inventory.json"))
    summaries = {}
    lines = ["# Reconstructed computational-cost records", "",
        "Each archive is reported separately. Timing flags and incomplete runs are retained; no new exclusions or substitutions are made.", "",
        "Training includes Node2Vec fitting but excludes offline input-feature generation, preparation, and evaluation. Within each seed, times are summed across seven tasks and memory is the maximum across tasks. Means below average complete model-seed jobs. RF fitting/inference is CPU-only.", ""]
    for manifest in manifests:
        directory = manifest.parent
        verify_inventory(directory)
        if (directory / "verified_measurements.json").exists():
            tasks = read(directory / "verified_measurements.json")
            audit = read(directory / "timing_audit.json")
            # Validate the recorded historical replacement, not a newly chosen one.
            new = audit["new_task"]
            assert next(row for row in tasks if row["method"] == new["method"] and row["task"] == new["task"]) == new
        else:
            tasks = read_tasks(directory)
        jobs = summarize_tasks(tasks)
        summary = []
        lines.extend([f"## {directory.name}", "",
            "| Method | Complete seeds | Training min | Inference ms | Peak RAM GiB | Peak GPU GiB | Flagged calls |",
            "|---|---:|---:|---:|---:|---:|---:|"])
        for method in METHODS:
            group = [row for row in jobs if row["method"] == method]
            if not group:
                continue
            record = dict(method=method, seeds=[row["seed"] for row in group], n_seeds=len(group),
                          timing_flags=sum(row["timing_flags"] for row in group))
            for metric in METRICS:
                values = [row[metric] for row in group]
                record[metric] = statistics.mean(values)
                record[metric + "_sample_sd"] = statistics.stdev(values) if len(values) > 1 else None
            summary.append(record)
            gpu = "CPU only" if method == "RF" else f"{record['peak_gpu_gib']:.2f}"
            lines.append(f"| {method} | {len(group)} | {record['training_minutes']:.2f} | {record['inference_ms']:.2f} | {record['peak_ram_gib']:.2f} | {gpu} | {record['timing_flags']} |")
        lines.append("")
        summaries[directory.name] = dict(jobs=jobs, summary=summary)
    (output / "recorded_summaries.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    (output / "recorded_summaries.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    # DRKG Supplementary Table 2 has an exact, complete source run.
    drkg = summaries["drkg_seed1_full_20260912_190355"]["summary"]
    expected = {"RF": 570.38, "TriNet": 0.82, "N2V-MLP": 9.40, "TriMoGCL": 85.66, "HANAMI": 265.93}
    assert len(drkg) == 5
    for row in drkg:
        assert row["n_seeds"] == 1 and row["seeds"] == [1]
        assert round(row["training_minutes"], 2) == expected[row["method"]]
    table = ["# Supplementary Table 2 source values (DRKG)", "",
             "Measured seed: 1. Complete seven-task run per method; not a ten-seed average.", "",
             "| Method | Training (min) | Inference (ms) | Peak RAM (GiB) | Peak GPU (GiB) |",
             "|---|---:|---:|---:|---:|"]
    for row in drkg:
        gpu = "CPU only" if row["method"] == "RF" else f"{row['peak_gpu_gib']:.2f}"
        table.append(f"| {row['method']} | {row['training_minutes']:.2f} | {row['inference_ms']:.2f} | {row['peak_ram_gib']:.2f} | {gpu} |")
    (output / "supplementary_table2_drkg.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    print(f"Verified {len(manifests)} archives; rebuilt summaries and DRKG Supplementary Table 2.")
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO / "data/computational_cost")
    parser.add_argument("--output", type=Path, default=REPO / "results/computational_cost")
    args = parser.parse_args()
    build(args.data_root, args.output)
