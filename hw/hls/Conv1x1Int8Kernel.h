#ifndef MYACCEL_CONV1X1_INT8_KERNEL_H
#define MYACCEL_CONV1X1_INT8_KERNEL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MYACCEL_CONV1X1_INT8_MAX_INPUT_CHANNELS 512

// Computes and requantizes a 1x1, stride-1, pad-0, dilation-1, group-1
// convolution. x, weight, and output use the unchanged logical INT8 byte
// layout, exposed to HLS as little-endian groups of four bytes per uint32_t.
// Activation/output words contain four consecutive spatial pixels; weight
// words contain four consecutive input channels. c_size and h_size *
// input_w_size must both be divisible by four. Bias is in accumulator units.
// requant_multiplier_bits contains the raw IEEE-754 binary32 bits of a
// positive, finite, normal multiplier.
void conv1x1_i8_kernel(const uint32_t *x, const uint32_t *weight,
    const int32_t *bias, uint32_t *output, int n_size, int c_size, int h_size,
    int input_w_size, int m_size, int x_zero_point, int w_zero_point,
    uint32_t requant_multiplier_bits, int output_zero_point);

#ifdef __cplusplus
}
#endif

#endif
