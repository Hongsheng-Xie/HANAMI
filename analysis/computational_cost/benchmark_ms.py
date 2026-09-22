"""Instrument existing MS/DRKG entry points without rewriting model or loss code."""
import argparse
import ast
import functools
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import threading
import time
import traceback

import numpy as np
import psutil
import torch
import sklearn
import torch_geometric

REPO = Path(__file__).resolve().parents[2]
ENTRIES = {"HANAMI": "main.py", "TriMoGCL": "main_tri_binary.py",
           "TriNet": "main_tri_binary_TriNet.py", "N2V-MLP": "main_tri_binary_N2V_MLP.py",
           "RF": "main_tri_binary_RF.py"}


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Recorder:
    def __init__(self, options, namespace):
        self.opt, self.ns = options, namespace
        self.path = Path(options.output) / options.method
        self.path.mkdir(parents=True, exist_ok=True)
        self.key = "preparation"
        self.samples = {}
        self.stop = threading.Event()
        self.rss_peak = psutil.Process().memory_info().rss
        self.monitor = threading.Thread(target=self.sample_ram, daemon=True)
        self.monitor.start()

    def sample_ram(self):
        proc = psutil.Process()
        while not self.stop.wait(0.05):
            self.rss_peak = max(self.rss_peak, proc.memory_info().rss)

    def event(self, kind, **values):
        row = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), method=self.opt.method,
                   seed=self.opt.seed, task=str(self.key), event=kind, **values)
        with (self.path / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        if kind in ("begin", "end", "progress", "error"):
            print(json.dumps(row), flush=True)

    def wrap(self, name, function):
        @functools.wraps(function)
        def measured(*args, **kwargs):
            sync()
            start = time.perf_counter()
            result = function(*args, **kwargs)
            sync()
            seconds = time.perf_counter() - start
            self.samples.setdefault(name, []).append(seconds)
            self.event(name, seconds=seconds)
            if name == "train" and len(self.samples[name]) % 5 == 0:
                self.event("progress", epochs=len(self.samples[name]),
                           training_seconds=sum(self.samples[name]))
            return result
        return measured

    def install(self):
        ns = self.ns
        if self.opt.method == "RF":
            # sklearn and NumPy operate on CPU; avoid implicit CUDA->NumPy conversion.
            ns["device"] = torch.device("cpu")
        ns["args"].res_dir = str(self.path / "original_metrics") + os.sep
        original_prepare = self.wrap("prepare_data", ns["prepare_data"])

        def prepare(args):
            data, positives, negatives = original_prepare(args)
            # Loaded features are fixed inputs, not a retained serialization-time graph.
            data.x = data.x.detach()
            if self.opt.task != "all":
                keys = list(positives[0])
                chosen = keys[0] if self.opt.task == "first" else next(
                    key for key in keys if str(key) == self.opt.task)
                positives = [{chosen: split[chosen]} for split in positives]
                negatives = [{chosen: split[chosen]} for split in negatives]
            split_hash = hashlib.sha256()
            for sets in (positives, negatives):
                for split in sets:
                    for key, values in split.items():
                        split_hash.update(str(key).encode())
                        split_hash.update(values.cpu().numpy().tobytes())
            self.event("data", feature_shape=list(data.x.shape),
                       dataset=self.opt.dataset, nodes=int(data.num_nodes),
                       prepared_graph_edges=int(data.train_graph.shape[1]),
                       split_sha256=split_hash.hexdigest(), tasks=list(map(str, positives[0])))
            return data, positives, negatives

        ns["prepare_data"] = prepare
        for name in ("train", "ttest", "run_node2vec", "extract_features"):
            if name in ns:
                ns[name] = self.wrap(name, ns[name])
        if self.opt.method == "RF":
            base = ns["RandomForestClassifier"]
            recorder = self

            class TimedForest(base):
                def fit(self, *args, **kwargs):
                    return recorder.wrap("rf_fit", super().fit)(*args, **kwargs)

            ns["RandomForestClassifier"] = TimedForest

    def begin(self, key):
        self.key = key
        self.samples = {}
        sync()
        self.start = time.perf_counter()
        self.rss_peak = psutil.Process().memory_info().rss
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.event("begin")

    def inference(self):
        ns = self.ns
        if self.opt.method == "RF":
            return ns["rf"].predict_proba(ns["X_test"])
        model = ns["model"]
        model.eval()
        x = ns["x_combined"] if self.opt.method == "N2V-MLP" else ns["data"].x
        edges = [ns["poslist"][2][self.key], ns["neglist"][2][self.key]]
        with torch.no_grad():
            h = None if self.opt.method == "TriNet" else model(x, ns["edges_in_graph"])
            outputs = []
            for edge in edges:
                if self.opt.method == "TriNet":
                    out = model(x, edge)
                else:
                    feat = model.pred(h, edge)
                    if self.opt.method in ("HANAMI", "TriMoGCL"):
                        feat = torch.cat((feat, model.pooling2(h, edge)), dim=1)
                    out = model.classifier(feat)
                outputs.append(torch.softmax(out, dim=1)[:, 1])
            return torch.cat(outputs).cpu().numpy()

    def end(self):
        sync()
        total = time.perf_counter() - self.start
        memory = dict(peak_process_ram_mib=self.rss_peak / 2**20,
                      peak_gpu_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                      peak_gpu_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)
        self.inference()  # Untimed warm-up, no parameter updates.
        durations = []
        for _ in range(self.opt.inference_repeats):
            sync()
            start = time.perf_counter()
            result = self.inference()
            sync()
            durations.append(time.perf_counter() - start)
        ns = self.ns
        graph = ns["edges_in_graph"]
        # Flag potential pauses/contention without deleting or replacing observations.
        train_times = self.samples.get("train", [])
        timing_flags = []
        if train_times:
            usual = statistics.median(train_times)
            threshold = max(30.0, 10 * usual)
            timing_flags = [dict(epoch=index + 1, seconds=value, threshold_seconds=threshold)
                            for index, value in enumerate(train_times) if value > threshold]
        self.event("end", task_wall_seconds=total,
                   timing_flags=timing_flags,
                   stages={name: dict(calls=len(times), total_seconds=sum(times),
                                     mean_seconds=statistics.mean(times))
                           for name, times in self.samples.items()},
                   inference_mean_seconds=statistics.mean(durations),
                   inference_sd_seconds=statistics.stdev(durations) if len(durations)>1 else 0,
                   inference_repeats=len(durations), test_motifs=len(result),
                   train_motifs=len(ns["poslist"][0][self.key])+len(ns["neglist"][0][self.key]),
                   input_graph_edges=int(graph.shape[1]), feature_width=int(ns["data"].x.shape[1]),
                   parameter_count=None if self.opt.method=="RF" else sum(p.numel() for p in ns["model"].parameters()),
                   best_validation_auroc=float(ns["Best_Val_from_maf1"]),
                   test_metrics=[float(v) for v in ns["Best_metrics"]], **memory)


class Instrument(ast.NodeTransformer):
    def __init__(self, options):
        self.opt = options
        self.seed_nodes = self.task_nodes = 0

    def visit_Assign(self, node):
        if any(isinstance(t, ast.Name) and t.id == "seeds" for t in node.targets):
            self.seed_nodes += 1
            node.value = ast.List(elts=[ast.Constant(self.opt.seed)], ctx=ast.Load())
            return [ast.Expr(ast.Call(ast.Attribute(ast.Name("_cost", ast.Load()), "install", ast.Load()), [], [])), node]
        return self.generic_visit(node)

    def visit_For(self, node):
        self.generic_visit(node)
        if isinstance(node.target, ast.Tuple) and any(isinstance(t, ast.Name) and t.id == "key" for t in node.target.elts):
            self.task_nodes += 1
            node.body.insert(0, ast.parse("_cost.begin(key)").body[0])
            node.body.append(ast.parse("_cost.end()").body[0])
        return node


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--method", required=True, choices=ENTRIES)
    parser.add_argument("--dataset", choices=["ms", "drkg"], default="ms")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--task", default="all", help="all, first, or an exact task key")
    parser.add_argument("--inference-repeats", type=int, default=20)
    parser.add_argument("--output", required=True)
    opt = parser.parse_args()
    root = REPO if opt.method == "HANAMI" else opt.baseline_root.resolve()
    source = root / ENTRIES[opt.method]
    data_root = root / "data"
    # Existing modules import siblings by name; use a fresh process per method.
    sys.path.insert(0, str(root))
    sys.argv = [str(source), "--data-name", opt.dataset, "--input_dir", str(data_root)+os.sep,
                "--seed", str(opt.seed), "--epoch-num", str(1 if opt.method=="RF" else opt.epochs)]
    ns = {"__name__": "__main__", "__file__": str(source)}
    recorder = Recorder(opt, ns)
    ns["_cost"] = recorder
    provenance = dict(method=opt.method, source=str(source), source_sha256=digest(source),
                      training_device="cpu" if opt.method == "RF" else "cuda",
                      options={k:str(v) if isinstance(v,Path) else v for k,v in vars(opt).items()},
                      hardware=dict(cpu=platform.processor(), cpu_logical=psutil.cpu_count(),
                                    ram_gib=psutil.virtual_memory().total/2**30,
                                    gpu=torch.cuda.get_device_name(0)),
                      software=dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
                                    sklearn=sklearn.__version__, pyg=torch_geometric.__version__,
                                    torch_cpu_threads=torch.get_num_threads()),
                      feature_files={}, source_files={})
    for name in ("dise", "drug", "gene"):
        feature = data_root / opt.dataset / f"{name}_{'All' if opt.method=='HANAMI' else 'feat'}.pth"
        provenance["feature_files"][str(feature)] = digest(feature)
    provenance[f"{opt.dataset}_arrays"] = {file.name: digest(file) for file in (data_root/opt.dataset).glob("*.npy")}
    for file in root.glob("*.py"):
        provenance["source_files"][file.name] = digest(file)
    (recorder.path / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    tree = ast.parse(source.read_text(encoding="utf-8-sig"), filename=str(source))
    modifier = Instrument(opt)
    tree = modifier.visit(tree)
    assert modifier.seed_nodes == 1 and modifier.task_nodes == 1, "Unsupported entry-point structure"
    try:
        exec(compile(ast.fix_missing_locations(tree), str(source), "exec"), ns)
    except Exception:
        recorder.event("error", traceback=traceback.format_exc())
        raise
    finally:
        recorder.stop.set()
        recorder.monitor.join()


if __name__ == "__main__":
    main()
