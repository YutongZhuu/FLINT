MODEL_URL ?= https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.onnx
AARCH64_SYSROOT ?= build/aarch64-linux-jammy-sysroot
ONNX_MLIR_BIN ?= third_party/onnx-mlir/build-host-exact/Release/bin/onnx-mlir
MODEL := models/yolov5n.onnx
FP32_MODEL := build/yolov5n-fp32.onnx
LOCAL_MYACCEL_OUTPUT := build/yolov5n-myaccel.so
LOCAL_MYACCEL_DRIVER := build/yolo-myaccel-driver
BUS_INPUT_AARCH64 := build/bus-input-aarch64.bin

.PHONY: all model local-myaccel run-local-myaccel verify-local-myaccel sysroot-jammy cross-yolo-accelerator-aarch64 package-aarch64 run-aarch64-docker clean distclean

all: local-myaccel package-aarch64

model: $(MODEL)

$(MODEL):
	@mkdir -p models
	curl --fail --location --retry 3 --output $@ $(MODEL_URL)

$(FP32_MODEL): $(MODEL) scripts/fp16_to_fp32.py
	@mkdir -p build
	python3 -m venv .venv
	.venv/bin/pip install --quiet -r requirements.txt
	.venv/bin/python scripts/fp16_to_fp32.py $(MODEL) $@

$(BUS_INPUT_AARCH64): samples/bus.jpg scripts/preprocess_yolo.py requirements.txt
	@mkdir -p build
	python3 -m venv .venv
	.venv/bin/pip install --quiet -r requirements.txt
	.venv/bin/python scripts/preprocess_yolo.py samples/bus.jpg $@

local-myaccel: $(FP32_MODEL)
	ONNX_MLIR_BIN="$(ONNX_MLIR_BIN)" ./scripts/build_yolo_myaccel_local.sh

run-local-myaccel: local-myaccel
	./$(LOCAL_MYACCEL_DRIVER) build/bus-input.bin build/bus-myaccel.bin

verify-local-myaccel: local-myaccel
	test -s $(LOCAL_MYACCEL_OUTPUT)
	test -x $(LOCAL_MYACCEL_DRIVER)
	file $(LOCAL_MYACCEL_OUTPUT)
	file $(LOCAL_MYACCEL_DRIVER)
	rg 'my_conv_f32' build/yolov5n-myaccel.ll

sysroot-jammy:
	AARCH64_SYSROOT="$(AARCH64_SYSROOT)" ./scripts/fetch_ubuntu_aarch64_sysroot.sh

cross-yolo-accelerator-aarch64: $(FP32_MODEL)
	AARCH64_SYSROOT="$(AARCH64_SYSROOT)" \
	ONNX_MLIR_BIN="$(ONNX_MLIR_BIN)" \
	./scripts/cross_compile_aarch64_llvm.sh

package-aarch64: cross-yolo-accelerator-aarch64 $(BUS_INPUT_AARCH64)
	./scripts/package_aarch64_artifacts.sh

run-aarch64-docker: package-aarch64
	./scripts/run_yolo_aarch64_docker.sh

clean:
	rm -rf build models .venv

distclean: clean
	rm -rf third_party/onnx-mlir/build-myaccel \
	       third_party/onnx-mlir/build-host-exact \
	       third_party/onnx-mlir/build-host-cross
