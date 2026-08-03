#ifndef MYACCEL_CONV3X3_INT8_KERNEL_H
#define MYACCEL_CONV3X3_INT8_KERNEL_H

#include <stdint.h>

#define MYACCEL_CONV3X3_INT8_MAX_INPUT_CHANNELS 128
#define MYACCEL_CONV3X3_INT8_MAX_STRIDE 2
#define MYACCEL_CONV3X3_INT8_OUTPUT_STRIPE_WIDTH 32
#define MYACCEL_CONV3X3_INT8_INPUT_AXI_BITS 128

#if defined(__cplusplus) && defined(__SYNTHESIS__)
#include <ap_int.h>
typedef ap_uint<MYACCEL_CONV3X3_INT8_INPUT_AXI_BITS>
    myaccel_conv3x3_i8_input_axi_t;
#else
// Native tests and C runtime code retain the logical four-byte packed-word
// view. Vitis HLS sees the 128-bit type above at the kernel boundary.
typedef uint32_t myaccel_conv3x3_i8_input_axi_t;
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Computes and requantizes a 3x3, dilation-1, group-1 convolution. x, weight,
// and output expose the unchanged logical INT8 NCHW/OIHW byte layouts as
// little-endian groups of four bytes per uint32_t. During synthesis, x is
// explicitly exposed as 128-bit AXI beats; native tests preserve the uint32_t
// pointer ABI and assemble the same beats internally. The generated csynth
// report remains the physical-width gate. c_size, input_w_size, and ow_size
// must all be multiples of four. Bias is in accumulator units.
// requant_multiplier_bits contains the raw IEEE-754 binary32 bits of a
// positive, finite, normal multiplier.
void conv3x3_i8_kernel(const myaccel_conv3x3_i8_input_axi_t *x,
    const uint32_t *weight, const int32_t *bias, uint32_t *output,
    int n_size, int c_size, int h_size, int input_w_size, int m_size,
    int oh_size, int ow_size, int pad_left, int pad_top, int stride_h,
    int stride_w, int x_zero_point, int w_zero_point,
    uint32_t requant_multiplier_bits, int output_zero_point);

#ifdef __cplusplus
}
#endif

#endif
