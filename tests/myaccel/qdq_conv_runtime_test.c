#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "onnx-mlir/Runtime/OMTensor.h"

void my_conv_qdq_i8(OMTensor *, const OMTensor *, const OMTensor *,
    const OMTensor *, const OMTensor *, const OMTensor *, const OMTensor *,
    const OMTensor *, const OMTensor *, const OMTensor *, const OMTensor *,
    const OMTensor *, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t);

// MyConv.c also contains the optional FP32/XRT entry point. It is irrelevant
// to this test, so provide a link stub.
int myaccel_xrt_conv2d_f32(const float *x, const float *weight,
    const float *bias, float *y, int n, int c, int h, int w, int m, int kh,
    int kw, int oh, int ow, int dh, int dw, int cPerGroup, int group,
    int padLeft, int padTop, int sh, int sw, int hasBias) {
  (void)x; (void)weight; (void)bias; (void)y; (void)n; (void)c; (void)h;
  (void)w; (void)m; (void)kh; (void)kw; (void)oh; (void)ow; (void)dh;
  (void)dw; (void)cPerGroup; (void)group; (void)padLeft; (void)padTop;
  (void)sh; (void)sw; (void)hasBias;
  return 0;
}

static int i8XrtCalls;

// Stand in for the FPGA kernel: compute centered INT8 dot products, add bias,
// and apply the same float32 requantization contract as the HLS kernel.
int myaccel_xrt_conv2d_i8(const int8_t *x, const int8_t *weight,
    const int32_t *bias, int8_t *output, int nSize, int cSize, int hSize,
    int wSize, int mSize, int khSize, int kwSize, int ohSize, int owSize,
    int dh, int dw, int cPerGroup, int group, int padLeft, int padTop, int sh,
    int sw, int xZeroPoint, int wZeroPoint, uint32_t requantMultiplierBits,
    int outputZeroPoint) {
  ++i8XrtCalls;
  float multiplier = 0.0f;
  memcpy(&multiplier, &requantMultiplierBits, sizeof(multiplier));
  const int mPerGroup = mSize / group;
  for (int n = 0; n < nSize; ++n)
    for (int m = 0; m < mSize; ++m) {
      const int groupIndex = m / mPerGroup;
      for (int oh = 0; oh < ohSize; ++oh)
        for (int ow = 0; ow < owSize; ++ow) {
          int32_t sum = 0;
          for (int cg = 0; cg < cPerGroup; ++cg)
            for (int kh = 0; kh < khSize; ++kh)
              for (int kw = 0; kw < kwSize; ++kw) {
                const int ih = oh * sh + kh * dh - padTop;
                const int iw = ow * sw + kw * dw - padLeft;
                int32_t input = xZeroPoint;
                if (ih >= 0 && ih < hSize && iw >= 0 && iw < wSize) {
                  const int c = groupIndex * cPerGroup + cg;
                  input =
                      x[((n * cSize + c) * hSize + ih) * wSize + iw];
                }
                const int32_t kernel =
                    weight[((m * cPerGroup + cg) * khSize + kh) * kwSize +
                           kw];
                sum +=
                    (input - xZeroPoint) * (kernel - wZeroPoint);
              }
          int32_t quantized = (int32_t)nearbyintf(
                                  (float)((int64_t)sum + bias[m]) * multiplier) +
                              outputZeroPoint;
          if (quantized < -128)
            quantized = -128;
          else if (quantized > 127)
            quantized = 127;
          output[((n * mSize + m) * ohSize + oh) * owSize + ow] =
              (int8_t)quantized;
        }
    }
  return 1;
}

static OMTensor *tensor(void *data, int64_t *shape, int64_t rank,
    OM_DATA_TYPE type) {
  return omTensorCreate(data, shape, rank, type);
}

static int checkOutput(const char *name, const char *route,
    const int8_t *actual, const int8_t *expected, int64_t count) {
  int ok = memcmp(actual, expected, (size_t)count) == 0;
  printf("%s %s (%s)\n", ok ? "PASS" : "FAIL", name, route);
  if (!ok) {
    for (int64_t i = 0; i < count; ++i)
      if (actual[i] != expected[i])
        fprintf(stderr, "  index %lld: expected %d, got %d\n",
            (long long)i, expected[i], actual[i]);
  }
  return ok;
}

static int runCase(const char *name, int8_t *x, int64_t *xShape, int8_t *w,
    int64_t *wShape, int32_t *bias, int8_t *actual, int64_t *yShape,
    const int8_t *expected, int64_t count, float xScale, int8_t xZeroPoint,
    float wScale, int8_t wZeroPoint, float biasScale, int32_t biasZeroPoint,
    float yScale, int8_t yZeroPoint, int64_t pad, int64_t stride,
    int expectXrt) {
  int64_t scalarShape[1] = {1};
  OMTensor *tensors[] = {
      tensor(actual, yShape, 4, ONNX_TYPE_INT8),
      tensor(x, xShape, 4, ONNX_TYPE_INT8),
      tensor(&xScale, scalarShape, 1, ONNX_TYPE_FLOAT),
      tensor(&xZeroPoint, scalarShape, 1, ONNX_TYPE_INT8),
      tensor(w, wShape, 4, ONNX_TYPE_INT8),
      tensor(&wScale, scalarShape, 1, ONNX_TYPE_FLOAT),
      tensor(&wZeroPoint, scalarShape, 1, ONNX_TYPE_INT8),
      tensor(bias, &wShape[0], 1, ONNX_TYPE_INT32),
      tensor(&biasScale, scalarShape, 1, ONNX_TYPE_FLOAT),
      tensor(&biasZeroPoint, scalarShape, 1, ONNX_TYPE_INT32),
      tensor(&yScale, scalarShape, 1, ONNX_TYPE_FLOAT),
      tensor(&yZeroPoint, scalarShape, 1, ONNX_TYPE_INT8),
  };

  const int callsBeforeCpu = i8XrtCalls;
  setenv("CPU", "1", 1);
  my_conv_qdq_i8(tensors[0], tensors[1], tensors[2], tensors[3], tensors[4],
      tensors[5], tensors[6], tensors[7], tensors[8], tensors[9], tensors[10],
      tensors[11], 1, 1, 1, pad, pad, stride, stride);
  int ok = checkOutput(name, "CPU=1 host", actual, expected, count);
  if (i8XrtCalls != callsBeforeCpu) {
    fprintf(stderr, "FAIL %s: CPU=1 invoked the XRT entry point\n", name);
    ok = 0;
  }

  memset(actual, 0, (size_t)count);
  setenv("CPU", "0", 1);
  const int callsBeforeXrt = i8XrtCalls;
  my_conv_qdq_i8(tensors[0], tensors[1], tensors[2], tensors[3], tensors[4],
      tensors[5], tensors[6], tensors[7], tensors[8], tensors[9], tensors[10],
      tensors[11], 1, 1, 1, pad, pad, stride, stride);
  ok &= checkOutput(name,
      expectXrt ? "CPU=0 full INT8 XRT" : "CPU=0 host fallback",
      actual, expected, count);
  const int expectedCalls = callsBeforeXrt + (expectXrt ? 1 : 0);
  if (i8XrtCalls != expectedCalls) {
    fprintf(stderr, "FAIL %s: expected %s XRT invocation\n", name,
        expectXrt ? "exactly one" : "no");
    ok = 0;
  }

  for (int i = 0; i < 12; ++i)
    omTensorDestroy(tensors[i]);
  return ok;
}

int main(void) {
  unsetenv("CPU");
  unsetenv("MYACCEL_FORCE_CPU");
  int ok = 1;
  int64_t xShape[4] = {1, 1, 1, 6};
  int64_t wShape[4] = {1, 1, 1, 1};
  int64_t yShape[4] = {1, 1, 1, 6};
  int8_t x[6] = {1, 3, -1, -3, 127, -128};
  int8_t w[1] = {1};
  int32_t bias[1] = {0};
  int8_t actual[6] = {0};
  const int8_t tiesExpected[6] = {0, 2, 0, -2, 64, -64};
  ok &= runCase("ties-to-even", x, xShape, w, wShape, bias, actual, yShape,
      tiesExpected, 6, 0.5f, 0, 1.0f, 0, 0.5f, 0, 1.0f, 0, 0, 1, 1);

  int8_t shiftedX[6] = {2, 4, 0, -2, 2, 4};
  int8_t shiftedW[1] = {-1};
  int32_t shiftedBias[1] = {2};
  const int8_t shiftedExpected[6] = {7, 7, 5, 5, 7, 7};
  memset(actual, 0, sizeof(actual));
  ok &= runCase("zero-points-and-bias", shiftedX, xShape, shiftedW, wShape,
      shiftedBias, actual, yShape, shiftedExpected, 6, 0.5f, 1, 1.0f, -2,
      0.5f, 0, 1.0f, 5, 0, 1, 1);

  const int8_t generalBiasExpected[6] = {1, 2, 0, -1, 64, -64};
  memset(actual, 0, sizeof(actual));
  ok &= runCase("mismatched-bias-scale-host-fallback", x, xShape, w,
      wShape, shiftedBias, actual, yShape, generalBiasExpected, 6, 0.5f, 0,
      1.0f, 0, 0.25f, 0, 1.0f, 0, 0, 1, 0);

  memset(actual, 0, sizeof(actual));
  ok &= runCase("nonzero-bias-zero-point-host-fallback", x, xShape, w,
      wShape, shiftedBias, actual, yShape, generalBiasExpected, 6, 0.5f, 0,
      1.0f, 0, 0.5f, 1, 1.0f, 0, 0, 1, 0);

  int64_t satXShape[4] = {1, 1, 1, 2};
  int64_t satYShape[4] = {1, 1, 1, 2};
  int8_t satX[2] = {127, -128};
  int8_t satActual[2] = {0};
  const int8_t satExpected[2] = {127, -128};
  ok &= runCase("signed-saturation", satX, satXShape, w, wShape, bias,
      satActual, satYShape, satExpected, 2, 2.0f, 0, 1.0f, 0, 2.0f, 0,
      0.01f, 0, 0, 1, 1);

  int64_t imageShape[4] = {1, 1, 3, 3};
  int64_t kernelShape[4] = {1, 1, 3, 3};
  int8_t image[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  int8_t kernel[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  int8_t imageActual[9] = {0};
  const int8_t imageExpected[9] = {4, 6, 4, 6, 9, 6, 4, 6, 4};
  ok &= runCase("3x3-im2col-padding", image, imageShape, kernel, kernelShape,
      bias, imageActual, imageShape, imageExpected, 9, 1.0f, 0, 1.0f, 0,
      1.0f, 0, 1.0f, 0, 1, 1, 1);

  int64_t sixBySixInputShape[4] = {1, 1, 6, 6};
  int64_t sixBySixWeightShape[4] = {1, 1, 6, 6};
  int64_t sixBySixOutputShape[4] = {1, 1, 1, 1};
  int8_t sixBySixInput[36];
  int8_t sixBySixWeight[36];
  for (int i = 0; i < 36; ++i) {
    sixBySixInput[i] = 1;
    sixBySixWeight[i] = 1;
  }
  int8_t sixBySixActual[1] = {0};
  const int8_t sixBySixExpected[1] = {36};
  ok &= runCase("6x6-host-fallback", sixBySixInput, sixBySixInputShape,
      sixBySixWeight, sixBySixWeightShape, bias, sixBySixActual,
      sixBySixOutputShape, sixBySixExpected, 1, 1.0f, 0, 1.0f, 0, 1.0f, 0,
      1.0f, 0, 0, 1, 0);
  return ok ? 0 : 1;
}
