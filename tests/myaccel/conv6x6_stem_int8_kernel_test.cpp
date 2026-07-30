#include "Conv6x6StemInt8Kernel.h"
#include "Int8Requantize.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace {

struct StemCase {
  const char *name;
  int n;
  int c;
  int h;
  int inputW;
  int m;
  int oh;
  int ow;
  int padLeft;
  int padTop;
  int strideH;
  int strideW;
  int xZeroPoint;
  int wZeroPoint;
  uint32_t multiplierBits;
  int outputZeroPoint;
};

std::vector<uint32_t> packInt8(const std::vector<int8_t> &bytes) {
  std::vector<uint32_t> words((bytes.size() + 3) / 4, 0);
  for (size_t i = 0; i < bytes.size(); ++i)
    words[i / 4] |=
        (uint32_t)(uint8_t)bytes[i] << ((i % 4) * 8);
  return words;
}

int8_t unpackInt8(const std::vector<uint32_t> &words, size_t index) {
  return (int8_t)(uint8_t)(
      (words[index / 4] >> ((index % 4) * 8)) & 0xffU);
}

void referenceStem(const StemCase &test, const std::vector<int8_t> &x,
    const std::vector<int8_t> &weight, const std::vector<int32_t> &bias,
    std::vector<int8_t> &output) {
  for (int n = 0; n < test.n; ++n)
    for (int m = 0; m < test.m; ++m)
      for (int oh = 0; oh < test.oh; ++oh)
        for (int ow = 0; ow < test.ow; ++ow) {
          int32_t accumulator = 0;
          for (int c = 0; c < test.c; ++c)
            for (int kh = 0; kh < 6; ++kh)
              for (int kw = 0; kw < 6; ++kw) {
                const int ih = oh * test.strideH + kh - test.padTop;
                const int iw = ow * test.strideW + kw - test.padLeft;
                int32_t input = test.xZeroPoint;
                if (ih >= 0 && ih < test.h && iw >= 0 && iw < test.inputW) {
                  const size_t xIndex =
                      ((size_t)n * test.c + c) * test.h * test.inputW +
                      (size_t)ih * test.inputW + iw;
                  input = x[xIndex];
                }
                const size_t weightIndex =
                    (((size_t)m * test.c + c) * 6 + kh) * 6 + kw;
                accumulator +=
                    (input - test.xZeroPoint) *
                    ((int32_t)weight[weightIndex] - test.wZeroPoint);
              }
          const size_t outputIndex =
              ((size_t)n * test.m + m) * test.oh * test.ow +
              (size_t)oh * test.ow + ow;
          output[outputIndex] = myaccel_int8::requantize(accumulator, bias[m],
              test.multiplierBits, test.outputZeroPoint);
        }
}

bool runCase(const StemCase &test, std::mt19937 &generator) {
  const size_t xCount =
      (size_t)test.n * test.c * test.h * test.inputW;
  const size_t weightCount = (size_t)test.m * test.c * 6 * 6;
  const size_t outputCount =
      (size_t)test.n * test.m * test.oh * test.ow;

  std::uniform_int_distribution<int> valueDistribution(-128, 127);
  std::uniform_int_distribution<int32_t> biasDistribution(-10000, 10000);
  std::vector<int8_t> x(xCount);
  std::vector<int8_t> weight(weightCount);
  std::vector<int32_t> bias(test.m);
  for (int8_t &value : x)
    value = (int8_t)valueDistribution(generator);
  for (int8_t &value : weight)
    value = (int8_t)valueDistribution(generator);
  for (int32_t &value : bias)
    value = biasDistribution(generator);

  std::vector<int8_t> expected(outputCount);
  referenceStem(test, x, weight, bias, expected);

  std::vector<uint32_t> packedX = packInt8(x);
  std::vector<uint32_t> packedWeight = packInt8(weight);
  std::vector<uint32_t> packedOutput(outputCount / 4, 0xdeadbeefU);
  conv6x6_stem_i8_kernel(packedX.data(), packedWeight.data(), bias.data(),
      packedOutput.data(), test.n, test.c, test.h, test.inputW, test.m,
      test.oh, test.ow, test.padLeft, test.padTop, test.strideH,
      test.strideW, test.xZeroPoint, test.wZeroPoint, test.multiplierBits,
      test.outputZeroPoint);

  for (size_t i = 0; i < outputCount; ++i) {
    const int8_t actual = unpackInt8(packedOutput, i);
    if (actual != expected[i]) {
      std::fprintf(stderr,
          "FAIL %-32s index=%zu expected=%d actual=%d\n", test.name, i,
          (int)expected[i], (int)actual);
      return false;
    }
  }

  std::printf("PASS %-32s outputs=%zu\n", test.name, outputCount);
  return true;
}

bool testInvalidAlignmentNoWrite() {
  std::vector<uint32_t> x(256, 0);
  std::vector<uint32_t> weight(256, 0);
  std::vector<int32_t> bias(8, 0);
  std::vector<uint32_t> output(64, 0x5a5a5a5aU);

  conv6x6_stem_i8_kernel(x.data(), weight.data(), bias.data(), output.data(),
      1, 3, 6, 6, 5, 1, 4, 0, 0, 1, 1, 0, 0, 0x3f800000U, 0);
  for (uint32_t value : output)
    if (value != 0x5a5a5a5aU)
      return false;

  conv6x6_stem_i8_kernel(x.data(), weight.data(), bias.data(), output.data(),
      1, 3, 6, 8, 5, 1, 6, 0, 0, 1, 1, 0, 0, 0x3f800000U, 0);
  for (uint32_t value : output)
    if (value != 0x5a5a5a5aU)
      return false;

  std::printf("PASS %-32s\n", "invalid alignment no-write");
  return true;
}

} // namespace

int main() {
  const StemCase cases[] = {
      {"stride1 padding and m tail", 1, 3, 7, 8, 5, 5, 8, 2, 2, 1, 1,
          3, -5, 0x3c800000U, 11},
      {"stride2 three-channel stem", 2, 3, 8, 8, 8, 4, 4, 2, 2, 2, 2,
          -7, 9, 0x3b800000U, -13},
      {"four-channel output tail", 1, 4, 9, 12, 11, 4, 8, 1, 1, 1, 1,
          5, -11, 0x3dcccccdU, 7},
  };

  std::mt19937 generator(0x6a6a493U);
  bool passed = true;
  for (const StemCase &test : cases)
    passed = runCase(test, generator) && passed;
  passed = testInvalidAlignmentNoWrite() && passed;
  return passed ? 0 : 1;
}
