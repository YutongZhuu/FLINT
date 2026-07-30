#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "onnx-mlir/Runtime/OMTensor.h"
#include "onnx-mlir/Runtime/OMTensorList.h"

extern "C" OMTensorList *run_main_graph(OMTensorList *inputs);
extern "C" void omInstrumentPrint() __attribute__((weak));

namespace {
using Clock = std::chrono::steady_clock;

constexpr int64_t kChannels = 3;
constexpr int64_t kHeight = 640;
constexpr int64_t kWidth = 640;
constexpr int64_t kElements = kChannels * kHeight * kWidth;

double milliseconds(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

std::vector<float> loadInput(const std::string &path) {
  std::vector<float> data(kElements);
  if (path.empty())
    return data; // Black, already-normalized image.

  std::ifstream input(path, std::ios::binary);
  if (!input)
    throw std::runtime_error("cannot open input file: " + path);
  input.read(reinterpret_cast<char *>(data.data()),
             static_cast<std::streamsize>(data.size() * sizeof(float)));
  if (input.gcount() != static_cast<std::streamsize>(data.size() * sizeof(float)) ||
      input.peek() != std::ifstream::traits_type::eof())
    throw std::runtime_error(
        "input must contain exactly 1x3x640x640 float32 values in NCHW order");
  return data;
}

void printShape(const OMTensor *tensor) {
  const int64_t rank = omTensorGetRank(tensor);
  const int64_t *shape = omTensorGetShape(tensor);
  std::cout << "output shape: [";
  for (int64_t i = 0; i < rank; ++i)
    std::cout << (i ? ", " : "") << shape[i];
  std::cout << "]\n";
}
} // namespace

int main(int argc, char **argv) {
  try {
    const auto programStart = Clock::now();
    if (argc > 3) {
      std::cerr << "usage: " << argv[0]
                << " [input-f32-nchw.bin] [output-f32.bin]\n";
      return 2;
    }

    const auto inputLoadStart = Clock::now();
    std::vector<float> inputData = loadInput(argc >= 2 ? argv[1] : "");
    const auto inputLoadEnd = Clock::now();

    const auto inputSetupStart = Clock::now();
    int64_t inputShape[] = {1, kChannels, kHeight, kWidth};
    OMTensor *input = omTensorCreate(
        inputData.data(), inputShape, 4, ONNX_TYPE_FLOAT);
    if (!input)
      throw std::runtime_error("failed to create input tensor");

    OMTensor *inputArray[] = {input};
    OMTensorList *inputs = omTensorListCreate(inputArray, 1);
    const auto inputSetupEnd = Clock::now();

    const auto inferenceStart = Clock::now();
    OMTensorList *outputs = run_main_graph(inputs);
    const auto inferenceEnd = Clock::now();
    omTensorListDestroy(inputs);
    if (!outputs || omTensorListGetSize(outputs) < 1)
      throw std::runtime_error("model did not return an output");

    // Ultralytics YOLOv5 exports output0 (decoded predictions) first, followed
    // by three auxiliary detection-head tensors.
    OMTensor *output = omTensorListGetOmtByIndex(outputs, 0);
    if (!output)
      throw std::runtime_error("model returned a null primary output");
    printShape(output);
    const float *values = static_cast<const float *>(omTensorGetDataPtr(output));

    const auto outputWriteStart = Clock::now();
    if (argc == 3) {
      std::ofstream dump(argv[2], std::ios::binary);
      if (!dump)
        throw std::runtime_error("cannot open output file");
      dump.write(reinterpret_cast<const char *>(values),
          static_cast<std::streamsize>(omTensorGetNumElems(output) * sizeof(float)));
    }
    const auto outputWriteEnd = Clock::now();

    const auto postprocessStart = Clock::now();
    const int64_t *shape = omTensorGetShape(output);
    if (omTensorGetRank(output) != 3 || shape[0] != 1 || shape[2] < 6)
      throw std::runtime_error("unexpected YOLO output layout");

    const int64_t candidates = shape[1];
    const int64_t fields = shape[2];
    float bestScore = -std::numeric_limits<float>::infinity();
    int64_t bestCandidate = -1;
    int64_t bestClass = -1;
    for (int64_t i = 0; i < candidates; ++i) {
      const float *row = values + i * fields;
      const auto best = std::max_element(row + 5, row + fields);
      const float score = row[4] * *best;
      if (std::isfinite(score) && score > bestScore) {
        bestScore = score;
        bestCandidate = i;
        bestClass = best - (row + 5);
      }
    }

    if (bestCandidate < 0)
      throw std::runtime_error("model output contains no finite candidates");
    const float *box = values + bestCandidate * fields;
    std::cout << "best raw candidate: index=" << bestCandidate
              << " class=" << bestClass << " score=" << bestScore
              << " box_xywh=[" << box[0] << ", " << box[1] << ", "
              << box[2] << ", " << box[3] << "]\n";
    const auto postprocessEnd = Clock::now();
    if (omInstrumentPrint)
      omInstrumentPrint();
    omTensorListDestroy(outputs);
    const auto programEnd = Clock::now();
    std::cerr << "YOLO_PROFILE input_load="
              << milliseconds(inputLoadStart, inputLoadEnd)
              << " input_setup=" << milliseconds(inputSetupStart, inputSetupEnd)
              << " inference=" << milliseconds(inferenceStart, inferenceEnd)
              << " output_write="
              << milliseconds(outputWriteStart, outputWriteEnd)
              << " postprocess="
              << milliseconds(postprocessStart, postprocessEnd)
              << " total=" << milliseconds(programStart, programEnd)
              << " ms\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
