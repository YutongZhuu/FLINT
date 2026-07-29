#ifndef MYACCEL_CONV1X1_INT8_KERNEL_H
#define MYACCEL_CONV1X1_INT8_KERNEL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MYACCEL_CONV1X1_INT8_MAX_INPUT_CHANNELS 512

// Computes centered INT8 dot products for a 1x1, stride-1, pad-0,
// dilation-1, group-1 convolution. Bias and requantization stay on the host.
void conv1x1_i8_kernel(const int8_t *x, const int8_t *weight,
    int32_t *accumulator, int n_size, int c_size, int h_size,
    int input_w_size, int m_size, int x_zero_point, int w_zero_point);

#ifdef __cplusplus
}
#endif

#endif
