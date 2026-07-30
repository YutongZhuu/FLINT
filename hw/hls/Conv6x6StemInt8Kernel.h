#ifndef MYACCEL_CONV6X6_STEM_INT8_KERNEL_H
#define MYACCEL_CONV6X6_STEM_INT8_KERNEL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MYACCEL_CONV6X6_STEM_INT8_MAX_INPUT_CHANNELS 4
#define MYACCEL_CONV6X6_STEM_INT8_MAX_STRIDE 2

// Computes and requantizes a 6x6, group-1, dilation-1 convolution intended
// for an INT8 model stem. x and output use NCHW layout, weight uses OIHW
// layout, and all three logical INT8 arrays are exposed to HLS as
// little-endian groups of four bytes per uint32_t. input_w_size and ow_size
// must be divisible by four. Bias is in accumulator units. The caller
// supplies padding, strides, and output dimensions. requant_multiplier_bits
// contains the raw IEEE-754 binary32 bits of a positive, finite, normal
// multiplier.
void conv6x6_stem_i8_kernel(const uint32_t *x, const uint32_t *weight,
    const int32_t *bias, uint32_t *output, int n_size, int c_size, int h_size,
    int input_w_size, int m_size, int oh_size, int ow_size, int pad_left,
    int pad_top, int stride_h, int stride_w, int x_zero_point,
    int w_zero_point, uint32_t requant_multiplier_bits,
    int output_zero_point);

#ifdef __cplusplus
}
#endif

#endif
