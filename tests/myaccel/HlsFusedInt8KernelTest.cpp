#include "Conv3x3ActDual1x1Int8Kernel.h"
#include "Int8Requantize.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct Case {
  const char *name;
  int n;
  int c;
  int h;
  int w;
  int mid;
  int outputA;
  int outputB;
  int padLeft;
  int padTop;
  int strideH;
  int strideW;
  int xZeroPoint;
  int convWeightZeroPoint;
  int convOutputZeroPoint;
  int activatedZeroPoint;
  int weightAZeroPoint;
  int outputAZeroPoint;
  int weightBZeroPoint;
  int outputBZeroPoint;
  float convMultiplier;
  float multiplierA;
  float multiplierB;
  bool identityLut;
};

uint32_t floatBits(float value) {
  uint32_t bits;
  static_assert(sizeof(bits) == sizeof(value), "binary32 expected");
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

class DeterministicValues {
public:
  explicit DeterministicValues(uint32_t seed) : state(seed) {}

  int next(int low, int high) {
    state = state * 1664525U + 1013904223U;
    return low + (int)(state % (uint32_t)(high - low + 1));
  }

private:
  uint32_t state;
};

template <typename T>
void fillValues(std::vector<T> &values, DeterministicValues &generator,
    int low, int high) {
  for (T &value : values)
    value = (T)generator.next(low, high);
}

int outputSize(int input, int pad, int stride) {
  return (input + 2 * pad - 3) / stride + 1;
}

std::vector<int8_t> makeActivationLut(const Case &testCase) {
  std::vector<int8_t> lut(256);
  for (int i = 0; i < 256; ++i) {
    const int quantized = i - 128;
    if (testCase.identityLut) {
      lut[i] = (int8_t)quantized;
      continue;
    }

    // A deterministic nonlinear mapping that changes quantization. It is not
    // presented as SiLU; it verifies that the LUT and activated zero point are
    // honored independently of the 3x3 output quantization.
    const int centered = quantized - testCase.convOutputZeroPoint;
    const int transformed = centered < 0 ? centered / 4 : centered * 3 / 4;
    const int mapped = std::max(
        -128, std::min(127, testCase.activatedZeroPoint + transformed));
    lut[i] = (int8_t)mapped;
  }
  return lut;
}

void reference(const Case &testCase, int oh, int ow,
    const std::vector<int8_t> &x, const std::vector<int8_t> &convWeight,
    const std::vector<int32_t> &convBias,
    const std::vector<int8_t> &activationLut,
    const std::vector<int8_t> &weightA, const std::vector<int32_t> &biasA,
    const std::vector<int8_t> &weightB, const std::vector<int32_t> &biasB,
    std::vector<int8_t> &outputA, std::vector<int8_t> &outputB) {
  const uint32_t convMultiplierBits = floatBits(testCase.convMultiplier);
  const uint32_t multiplierABits = floatBits(testCase.multiplierA);
  const uint32_t multiplierBBits = floatBits(testCase.multiplierB);
  std::vector<int8_t> activated(
      (size_t)testCase.n * testCase.mid * oh * ow);

  for (int n = 0; n < testCase.n; ++n) {
    for (int mid = 0; mid < testCase.mid; ++mid) {
      for (int outputRow = 0; outputRow < oh; ++outputRow) {
        for (int outputColumn = 0; outputColumn < ow; ++outputColumn) {
          int32_t accumulator = 0;
          for (int c = 0; c < testCase.c; ++c) {
            for (int kh = 0; kh < 3; ++kh) {
              for (int kw = 0; kw < 3; ++kw) {
                const int inputRow = outputRow * testCase.strideH -
                                         testCase.padTop +
                                     kh;
                const int inputColumn = outputColumn * testCase.strideW -
                                            testCase.padLeft +
                                        kw;
                if (inputRow < 0 || inputRow >= testCase.h ||
                    inputColumn < 0 || inputColumn >= testCase.w)
                  continue;
                const size_t xIndex =
                    ((size_t)n * testCase.c + c) * testCase.h * testCase.w +
                    (size_t)inputRow * testCase.w + inputColumn;
                const size_t weightIndex =
                    (((size_t)mid * testCase.c + c) * 3 + kh) * 3 + kw;
                accumulator +=
                    ((int32_t)x[xIndex] - testCase.xZeroPoint) *
                    ((int32_t)convWeight[weightIndex] -
                        testCase.convWeightZeroPoint);
              }
            }
          }
          const int8_t quantized = myaccel_int8::requantize(accumulator,
              convBias[mid], convMultiplierBits,
              testCase.convOutputZeroPoint);
          const size_t index =
              ((size_t)n * testCase.mid + mid) * oh * ow +
              (size_t)outputRow * ow + outputColumn;
          activated[index] = activationLut[(int)quantized + 128];
        }
      }
    }
  }

  for (int n = 0; n < testCase.n; ++n) {
    for (int outputRow = 0; outputRow < oh; ++outputRow) {
      for (int outputColumn = 0; outputColumn < ow; ++outputColumn) {
        for (int output = 0; output < testCase.outputA; ++output) {
          int32_t accumulator = 0;
          for (int mid = 0; mid < testCase.mid; ++mid) {
            const size_t activationIndex =
                ((size_t)n * testCase.mid + mid) * oh * ow +
                (size_t)outputRow * ow + outputColumn;
            accumulator +=
                ((int32_t)activated[activationIndex] -
                    testCase.activatedZeroPoint) *
                ((int32_t)weightA[(size_t)output * testCase.mid + mid] -
                    testCase.weightAZeroPoint);
          }
          const size_t outputIndex =
              ((size_t)n * testCase.outputA + output) * oh * ow +
              (size_t)outputRow * ow + outputColumn;
          outputA[outputIndex] = myaccel_int8::requantize(accumulator,
              biasA[output], multiplierABits, testCase.outputAZeroPoint);
        }

        for (int output = 0; output < testCase.outputB; ++output) {
          int32_t accumulator = 0;
          for (int mid = 0; mid < testCase.mid; ++mid) {
            const size_t activationIndex =
                ((size_t)n * testCase.mid + mid) * oh * ow +
                (size_t)outputRow * ow + outputColumn;
            accumulator +=
                ((int32_t)activated[activationIndex] -
                    testCase.activatedZeroPoint) *
                ((int32_t)weightB[(size_t)output * testCase.mid + mid] -
                    testCase.weightBZeroPoint);
          }
          const size_t outputIndex =
              ((size_t)n * testCase.outputB + output) * oh * ow +
              (size_t)outputRow * ow + outputColumn;
          outputB[outputIndex] = myaccel_int8::requantize(accumulator,
              biasB[output], multiplierBBits, testCase.outputBZeroPoint);
        }
      }
    }
  }
}

bool compare(const std::string &caseName, const char *branch,
    const std::vector<int8_t> &expected,
    const std::vector<int8_t> &actual) {
  if (expected.size() != actual.size()) {
    std::cerr << caseName << " " << branch << " size mismatch\n";
    return false;
  }
  for (size_t i = 0; i < expected.size(); ++i) {
    if (expected[i] != actual[i]) {
      std::cerr << caseName << " " << branch << " mismatch at " << i
                << ": expected " << (int)expected[i] << ", got "
                << (int)actual[i] << "\n";
      return false;
    }
  }
  return true;
}

bool runCase(const Case &testCase, uint32_t seed) {
  const int oh = outputSize(testCase.h, testCase.padTop, testCase.strideH);
  const int ow = outputSize(testCase.w, testCase.padLeft, testCase.strideW);
  DeterministicValues generator(seed);
  std::vector<int8_t> x(
      (size_t)testCase.n * testCase.c * testCase.h * testCase.w);
  std::vector<int8_t> convWeight(
      (size_t)testCase.mid * testCase.c * 3 * 3);
  std::vector<int32_t> convBias(testCase.mid);
  std::vector<int8_t> weightA(
      (size_t)testCase.outputA * testCase.mid);
  std::vector<int32_t> biasA(testCase.outputA);
  std::vector<int8_t> weightB(
      (size_t)testCase.outputB * testCase.mid);
  std::vector<int32_t> biasB(testCase.outputB);
  fillValues(x, generator, -31, 29);
  fillValues(convWeight, generator, -8, 7);
  fillValues(convBias, generator, -400, 400);
  fillValues(weightA, generator, -9, 8);
  fillValues(biasA, generator, -300, 300);
  fillValues(weightB, generator, -7, 10);
  fillValues(biasB, generator, -300, 300);
  const std::vector<int8_t> activationLut = makeActivationLut(testCase);

  std::vector<int8_t> expectedA(
      (size_t)testCase.n * testCase.outputA * oh * ow);
  std::vector<int8_t> expectedB(
      (size_t)testCase.n * testCase.outputB * oh * ow);
  std::vector<int8_t> actualA(expectedA.size(), (int8_t)0x55);
  std::vector<int8_t> actualB(expectedB.size(), (int8_t)0x55);
  reference(testCase, oh, ow, x, convWeight, convBias, activationLut, weightA,
      biasA, weightB, biasB, expectedA, expectedB);

  conv3x3_act_dual1x1_i8_kernel(x.data(), convWeight.data(), convBias.data(),
      activationLut.data(), weightA.data(), biasA.data(), actualA.data(),
      weightB.data(), biasB.data(), actualB.data(), testCase.n, testCase.c,
      testCase.h, testCase.w, testCase.mid, oh, ow, testCase.outputA,
      testCase.outputB, testCase.padLeft, testCase.padTop, testCase.strideH,
      testCase.strideW, testCase.xZeroPoint, testCase.convWeightZeroPoint,
      floatBits(testCase.convMultiplier), testCase.convOutputZeroPoint,
      testCase.activatedZeroPoint, testCase.weightAZeroPoint,
      floatBits(testCase.multiplierA), testCase.outputAZeroPoint,
      testCase.weightBZeroPoint, floatBits(testCase.multiplierB),
      testCase.outputBZeroPoint);

  const bool matches = compare(testCase.name, "branch A", expectedA, actualA) &&
                       compare(testCase.name, "branch B", expectedB, actualB);
  if (matches)
    std::cout << "PASS " << testCase.name << "\n";
  return matches;
}

} // namespace

int main() {
  const Case cases[] = {
      {"identity-lut-stride1-tails", 2, 5, 5, 7, 7, 5, 3, 1, 1, 1, 1,
          -3, 2, -9, -9, 1, 7, -2, -11, 0.037125f, 0.05125f,
          0.02875f, true},
      {"nonlinear-lut-stride2-near-limits", 1, 31, 4, 5, 63, 31, 29, 1,
          1, 2, 2, 4, -3, 6, -17, 2, -5, 3, 9, 0.014625f,
          0.023875f, 0.031375f, false},
  };

  bool ok = true;
  uint32_t seed = 0x4d595df4U;
  for (const Case &testCase : cases) {
    ok = runCase(testCase, seed) && ok;
    seed ^= 0x9e3779b9U;
  }
  return ok ? 0 : 1;
}
