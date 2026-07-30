# Tests

Run the MyAccel QDQ runtime tests:

```bash
./tests/myaccel/test_qdq_runtime.sh
```

This exercises both `CPU=1` host execution and the INT8 accelerator boundary
with host-side bias and requantization. It also verifies that a 6x6 INT8
convolution does not invoke XRT and uses the software fallback.

Run the ordinary C++ numerical models for all HLS kernels:

```bash
./scripts/test_myaccel_hls_kernels.sh
```

Run the mocked KV260 build, firmware, and XRT profiling helper tests (Vitis is
not required):

```bash
./tests/test_kv260_build_helpers.sh
```

Verify that the YOLOv5n INT8 model lowers to 60 fused INT8 convolution calls:

```bash
./tests/myaccel/test_qdq_fusion.sh
```

First copy the fused output produced on the KV260 back to this repository:

```bash
scp ubuntu@100.112.84.110:/home/ubuntu/dev/int8-fused-test/build/output-fused-v2.bin \
  build/output-fused-v2.bin
```

Then compare it with ONNX Runtime using the same model and input:

```bash
python3 tests/compare_onnx_output.py \
  build/yolov5n-int8.onnx \
  build/bus-input-aarch64.bin \
  build/output-fused-v2.bin
```
