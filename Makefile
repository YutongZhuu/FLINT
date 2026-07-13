ONNX_MLIR_IMAGE ?= ghcr.io/onnxmlir/onnx-mlir:latest
PLATFORM ?= linux/amd64
MODEL_URL ?= https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.onnx
AARCH64_SYSROOT ?= build/aarch64-linux-jammy-sysroot
ONNX_MLIR_BIN ?= third_party/onnx-mlir/build-host-exact/Release/bin/onnx-mlir
MODEL := models/yolov5n.onnx
FP32_MODEL := build/yolov5n-fp32.onnx
OUTPUT := build/yolov5n.so
DRIVER := build/yolo_driver
BUS_INPUT_AARCH64 := build/bus-input-aarch64.bin

.PHONY: all model compile driver run verify verify-accelerator run-yolo-accelerator run-yolo-accelerator-arm64 sysroot-jammy cross-yolo-accelerator-aarch64 package-aarch64 run-aarch64-docker clean distclean

all: compile driver verify

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

compile: $(OUTPUT)

$(OUTPUT): $(FP32_MODEL)
	docker run --rm --platform $(PLATFORM) \
		-v "$(CURDIR):/work" -w /work \
		$(ONNX_MLIR_IMAGE) \
		-O3 --EmitLib -o build/yolov5n $(FP32_MODEL)

driver: $(DRIVER)

$(DRIVER): $(OUTPUT) driver/yolo_driver.cpp
	docker run --rm --platform $(PLATFORM) --entrypoint /bin/bash \
		-v "$(CURDIR):/work" -w /work \
		$(ONNX_MLIR_IMAGE) -lc \
		'g++ -std=c++17 -O2 -Wall -Wextra -I/usr/local/include \
		 driver/yolo_driver.cpp build/yolov5n.so \
		 -Wl,-rpath,"\$$ORIGIN" -o $(DRIVER)'

run: driver
	docker run --rm --platform $(PLATFORM) --entrypoint /work/$(DRIVER) \
		-v "$(CURDIR):/work" -w /work $(ONNX_MLIR_IMAGE)

verify:
	test -s $(OUTPUT)
	test -x $(DRIVER)
	file $(OUTPUT)
	file $(DRIVER)

# End-to-end verification of the same Conv->krnl.call->LLVM call path used by
# MyAccel. The stock compiler flag is used because the release SDK omits headers
# required to rebuild onnx-mlir itself.
verify-accelerator:
	@mkdir -p build
	.venv/bin/python scripts/make_conv_test.py
	docker run --rm --platform $(PLATFORM) \
		-v "$(CURDIR):/work" -w /work $(ONNX_MLIR_IMAGE) \
		--ops-for-call=Conv --preserveMLIR --preserveLLVMIR -O0 \
		-o build/conv-test-accel build/conv-test.onnx
	rg 'call void @Conv|llvm.call @Conv' build/conv-test-accel.ll build/conv-test-accel.llvm.mlir
	docker run --rm --platform $(PLATFORM) --entrypoint /bin/bash \
		-v "$(CURDIR):/work" -w /work $(ONNX_MLIR_IMAGE) -lc \
		'gcc -std=c11 -O2 -Wall -Wextra -I/usr/local/include \
		 driver/conv_test_driver.c \
		 third_party/onnx-mlir/src/Accelerators/MyAccel/Runtime/MyConv.c \
		 build/conv-test-accel.so -lm -Wl,-rpath,"\$$ORIGIN" \
		 -o build/conv-test-driver && build/conv-test-driver'

run-yolo-accelerator: $(FP32_MODEL)
	./scripts/run_yolo_myaccel.sh

run-yolo-accelerator-arm64: $(FP32_MODEL)
	PLATFORM=linux/arm64 \
	MYACCEL_IMAGE=onnx-mlir-myaccel-dev-arm64 \
	MYACCEL_DOCKERFILE=Dockerfile.myaccel.arm64 \
	MYACCEL_MODE=native \
	./scripts/run_yolo_myaccel.sh

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
