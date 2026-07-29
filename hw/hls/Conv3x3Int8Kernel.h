#ifndef MYACCEL_CONV3X3_INT8_KERNEL_H
#define MYACCEL_CONV3X3_INT8_KERNEL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MYACCEL_CONV3X3_INT8_MAX_INPUT_CHANNELS 128
#define MYACCEL_CONV3X3_INT8_MAX_STRIDE 2

// Computes centered INT8 dot products for a 3x3, dilation-1, group-1
// convolution. Bias and requantization stay on the host.
void conv3x3_i8_kernel(const int8_t *x, const int8_t *weight,
    int32_t *accumulator, int n_size, int c_size, int h_size,
    int input_w_size, int m_size, int oh_size, int ow_size, int pad_left,
    int pad_top, int stride_h, int stride_w, int x_zero_point,
    int w_zero_point);

#ifdef __cplusplus
}
#endif

#endif
