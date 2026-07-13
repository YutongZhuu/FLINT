#!/usr/bin/env python3
import csv
import glob
import os
import re
import statistics
import sys
from collections import defaultdict

paths = sys.argv[1:] or glob.glob("build/profile/run-*.txt")
if not paths:
    raise SystemExit("usage: summarize_onnxmlir_profile.py <profile-output.txt>...")

groups = {
    "onnx.Conv": ("onnx.Conv",),
    "onnx.Add": ("onnx.Add",),
    "onnx.Sigmoid + onnx.Mul": ("onnx.Sigmoid", "onnx.Mul"),
    "onnx.MaxPool": ("onnx.MaxPool",),
}

per_run = []
for path in paths:
    totals = defaultdict(float)
    counts = defaultdict(int)
    total_instrumented = 0.0
    with open(path, newline="") as f:
        for row in csv.reader(f, skipinitialspace=True):
            if not row or row[0] != "==PERF-REPORT==":
                continue
            # Format:
            # ==PERF-REPORT==, op, node-name, before|after, delta-sec, cumulative-sec
            if len(row) < 6 or row[3].strip() != "after":
                continue
            op = row[1].strip()
            elapsed = float(row[4])
            totals[op] += elapsed
            counts[op] += 1
            total_instrumented += elapsed
    time_path = re.sub(r"/run-(\d+)\.txt$", r"/time-\1.txt", path)
    process_time = {}
    if os.path.exists(time_path):
        with open(time_path) as tf:
            for line in tf:
                parts = line.strip().split()
                if len(parts) == 2 and parts[0] in {"real", "user", "sys"}:
                    process_time[parts[0]] = float(parts[1])
    per_run.append((path, totals, counts, total_instrumented, process_time))

print("Per-run totals, seconds:")
for path, totals, counts, total, process_time in per_run:
    print(f"\n{path}")
    for label, ops in groups.items():
        sec = sum(totals[op] for op in ops)
        n = sum(counts[op] for op in ops)
        pct = (100.0 * sec / total) if total else 0.0
        print(f"  {label:24s} {sec:9.6f}s  {pct:6.2f}%  count={n}")
    print(f"  {'instrumented total':24s} {total:9.6f}s")
    if process_time:
        cpu = process_time.get("user", 0.0) + process_time.get("sys", 0.0)
        real = process_time.get("real", 0.0)
        coverage = (100.0 * total / real) if real else 0.0
        print(f"  {'process wall time':24s} {real:9.6f}s")
        print(f"  {'process CPU time':24s} {cpu:9.6f}s  user={process_time.get('user', 0.0):.6f}s sys={process_time.get('sys', 0.0):.6f}s")
        print(f"  {'instr/wall coverage':24s} {coverage:9.2f}%")

print("\nAverages across runs:")
for label, ops in groups.items():
    values = [sum(totals[op] for op in ops) for _, totals, _, _, _ in per_run]
    count_values = [sum(counts[op] for op in ops) for _, _, counts, _, _ in per_run]
    avg = statistics.mean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    pct_values = [
        (100.0 * value / total) if total else 0.0
        for value, (_, _, _, total, _) in zip(values, per_run)
    ]
    print(
        f"  {label:24s} {avg:9.6f}s ± {stdev:8.6f}s"
        f"  avg_pct={statistics.mean(pct_values):6.2f}%"
        f"  count={round(statistics.mean(count_values))}"
    )

total_values = [total for _, _, _, total, _ in per_run]
print(
    f"  {'instrumented total':24s} {statistics.mean(total_values):9.6f}s"
    f" ± {(statistics.stdev(total_values) if len(total_values) > 1 else 0.0):8.6f}s"
)

real_values = [t["real"] for *_, t in per_run if "real" in t]
user_values = [t["user"] for *_, t in per_run if "user" in t]
sys_values = [t["sys"] for *_, t in per_run if "sys" in t]
if real_values:
    cpu_values = [u + s for u, s in zip(user_values, sys_values)]
    coverage_values = [
        (100.0 * instr / real) if real else 0.0
        for instr, real in zip(total_values, real_values)
    ]
    print(
        f"  {'process wall time':24s} {statistics.mean(real_values):9.6f}s"
        f" ± {(statistics.stdev(real_values) if len(real_values) > 1 else 0.0):8.6f}s"
    )
    print(
        f"  {'process CPU time':24s} {statistics.mean(cpu_values):9.6f}s"
        f" ± {(statistics.stdev(cpu_values) if len(cpu_values) > 1 else 0.0):8.6f}s"
    )
    print(
        f"  {'instr/wall coverage':24s} {statistics.mean(coverage_values):9.2f}%"
    )
