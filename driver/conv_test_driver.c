#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include "onnx-mlir/Runtime/OMTensor.h"
#include "onnx-mlir/Runtime/OMTensorList.h"

extern OMTensorList *run_main_graph(OMTensorList *);

int main(void) {
  float input[16];
  for (int i = 0; i < 16; ++i)
    input[i] = (float)(i + 1);
  int64_t shape[4] = {1, 1, 4, 4};
  OMTensor *x = omTensorCreate(input, shape, 4, ONNX_TYPE_FLOAT);
  OMTensor *array[1] = {x};
  OMTensorList *inputs = omTensorListCreate(array, 1);
  OMTensorList *outputs = run_main_graph(inputs);
  omTensorListDestroy(inputs);
  OMTensor *y = omTensorListGetOmtByIndex(outputs, 0);
  const float *actual = (const float *)omTensorGetDataPtr(y);
  const float expected[4] = {54.5f, 63.5f, 90.5f, 99.5f};
  int ok = 1;
  for (int i = 0; i < 4; ++i) {
    printf("y[%d] = %.1f (expected %.1f)\n", i, actual[i], expected[i]);
    if (fabsf(actual[i] - expected[i]) > 1e-5f)
      ok = 0;
  }
  omTensorListDestroy(outputs);
  puts(ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
