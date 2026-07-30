#ifndef MYACCEL_CONV3X3_INT8_KERNEL_H
#define MYACCEL_CONV3X3_INT8_KERNEL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MYACCEL_CONV3X3_INT8_MAX_INPUT_CHANNELS 128
#define MYACCEL_CONV3X3_INT8_MAX_STRIDE 2

// Computes and requantizes a 3x3, dilation-1, group-1 convolution. x, weight,
// and output expose the unchanged logical INT8 NCHW/OIHW byte layouts as
// little-endian groups of four bytes per AXI word. c_size, input_w_size, and
// ow_size must therefore all be multiples of four. Bias is in accumulator
// units. requant_multiplier_bits contains the raw IEEE-754 binary32 bits of a
// positive, finite, normal multiplier.
void conv3x3_i8_kernel(const uint32_t *x, const uint32_t *weight,
    const int32_t *bias, uint32_t *output, int n_size, int c_size,
    int h_size, int input_w_size, int m_size, int oh_size, int ow_size,
    int pad_left, int pad_top, int stride_h, int stride_w, int x_zero_point,
    int w_zero_point, uint32_t requant_multiplier_bits,
    int output_zero_point);

#ifdef __cplusplus
}
#endif

#endif
