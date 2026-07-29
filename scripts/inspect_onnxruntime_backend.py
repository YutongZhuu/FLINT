#!/usr/bin/env python3
"""Dump ONNX Runtime's optimized graph and aggregate its per-node CPU profile."""

import argparse
import collections
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("input")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("build/ort-inspect"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    optimized = args.output_dir / "optimized.onnx"
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.optimized_model_filepath = str(optimized)
    options.enable_profiling = True
    options.profile_file_prefix = str(args.output_dir / "profile")
    session = ort.InferenceSession(
        args.model, sess_options=options, providers=["CPUExecutionProvider"]
    )

    image = np.fromfile(args.input, dtype=np.float32).reshape(1, 3, 640, 640)
    feeds = {session.get_inputs()[0].name: image}
    outputs = session.run(None, feeds)
    for _ in range(args.runs):
        outputs = session.run(None, feeds)
    profile_path = Path(session.end_profiling())

    prediction = outputs[0]
    if prediction.ndim != 3 or prediction.shape[0] != 1 or prediction.shape[2] < 6:
        raise RuntimeError(
            f"expected YOLO output shaped [1, candidates, fields], got "
            f"{prediction.shape}"
        )
    rows = prediction[0]
    classes = rows[:, 5:].argmax(axis=1)
    class_scores = rows[np.arange(rows.shape[0]), classes + 5]
    scores = rows[:, 4] * class_scores
    best_candidate = int(scores.argmax())
    best_class = int(classes[best_candidate])
    best_box = rows[best_candidate, :4]

    print("onnxruntime:", ort.__version__)
    print("available providers:", ort.get_available_providers())
    print("session providers:", session.get_providers())
    print("optimized graph:", optimized)
    print("output shape:", list(prediction.shape))
    print(
        f"best raw candidate: index={best_candidate} class={best_class} "
        f"score={scores[best_candidate]:.6f} "
        f"box_xywh={best_box.tolist()}"
    )

    events = json.loads(profile_path.read_text())
    totals = collections.Counter()
    providers = collections.Counter()
    names = collections.Counter()
    for event in events:
        event_args = event.get("args", {})
        if event.get("cat") != "Node" or "dur" not in event:
            continue
        op_name = event_args.get("op_name", "unknown")
        provider = event_args.get("provider", "unknown")
        duration = float(event["dur"])
        totals[op_name] += duration
        providers[provider] += duration
        names[(event.get("name", "unknown"), op_name)] += duration

    print("profile:", profile_path)
    print("time by provider (profile total across runs):")
    for provider, duration in providers.most_common():
        print(f"  {provider}: {duration / 1000:.3f} ms")
    print("time by operator:")
    for op_name, duration in totals.most_common():
        print(f"  {op_name}: {duration / 1000:.3f} ms")
    print("top nodes:")
    for (name, op_name), duration in names.most_common(20):
        print(f"  {duration / 1000:.3f} ms  {op_name}  {name}")


if __name__ == "__main__":
    main()
