#!/usr/bin/env python3
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("input", type=Path)
    parser.add_argument("actual", type=Path)
    parser.add_argument("--reference-output", type=Path)
    args = parser.parse_args()

    for label, path in (
        ("model", args.model),
        ("input", args.input),
        ("actual output", args.actual),
    ):
        if not path.is_file():
            parser.error(f"{label} file does not exist: {path}")

    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(
        str(args.model), providers=["CPUExecutionProvider"]
    )
    model_input = session.get_inputs()[0]
    shape = tuple(int(dim) for dim in model_input.shape)
    input_data = np.fromfile(args.input, dtype=np.float32).reshape(shape)
    reference = session.run(None, {model_input.name: input_data})[0]
    actual = np.fromfile(args.actual, dtype=np.float32)

    if args.reference_output:
        reference.tofile(args.reference_output)
    reference = reference.reshape(-1)
    if actual.size != reference.size:
        raise SystemExit(
            f"output size mismatch: actual={actual.size}, reference={reference.size}"
        )

    difference = np.abs(actual - reference)
    worst = int(np.argmax(difference))
    print(f"output_shape={session.get_outputs()[0].shape}")
    print(f"elements={actual.size}")
    print(f"all_finite={np.isfinite(actual).all() and np.isfinite(reference).all()}")
    print(f"max_abs={difference.max():.9g}")
    print(f"mean_abs={difference.mean():.9g}")
    print(f"rmse={np.sqrt(np.mean((actual - reference) ** 2)):.9g}")
    print(f"exact_percent={100 * np.mean(actual == reference):.6f}")
    for tolerance in (1e-4, 1e-3, 1e-2):
        percent = 100 * np.mean(difference <= tolerance)
        print(f"within_{tolerance:g}_percent={percent:.6f}")
    print(
        f"worst_index={worst} actual={actual[worst]:.9g} "
        f"reference={reference[worst]:.9g}"
    )


if __name__ == "__main__":
    main()
