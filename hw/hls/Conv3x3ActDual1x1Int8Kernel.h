#ifndef MYACCEL_CONV3X3_ACT_DUAL1X1_INT8_KERNEL_H
#define MYACCEL_CONV3X3_ACT_DUAL1X1_INT8_KERNEL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MYACCEL_FUSED_INT8_MAX_INPUT_CHANNELS 32
#define MYACCEL_FUSED_INT8_MAX_MID_CHANNELS 64
#define MYACCEL_FUSED_INT8_MAX_BRANCH_OUTPUT_CHANNELS 32
#define MYACCEL_FUSED_INT8_MAX_STRIDE 2

// Computes this NCHW/OIHW block in one kernel invocation:
//
//   3x3 convolution -> 256-entry INT8 activation LUT -> two 1x1 convolutions
//
// The 3x3 result is retained in an on-chip spatial tile and is never exposed
// through an AXI port. activation_lut[(int)q + 128] maps every possible signed
// INT8 3x3 output q to the quantized activation consumed by both branches.
// An identity LUT therefore has activation_lut[i] == (int8_t)(i - 128) and
// activated_zero_point == conv3x3_output_zero_point. A future SiLU LUT may use
// a different activated_zero_point without changing this interface.
//
// All three biases are in their convolution's accumulator units. Each requant
// multiplier argument contains the raw IEEE-754 binary32 bits of a positive,
// finite, normal multiplier. The 3x3 convolution is dilation-1/group-1 with
// caller-supplied symmetric-stride-compatible output dimensions and left/top
// padding. Both downstream 1x1 convolutions are stride-1/pad-0/group-1.
void conv3x3_act_dual1x1_i8_kernel(const int8_t *x,
    const int8_t *conv3x3_weight, const int32_t *conv3x3_bias,
    const int8_t *activation_lut, const int8_t *branch_a_weight,
    const int32_t *branch_a_bias, int8_t *branch_a_output,
    const int8_t *branch_b_weight, const int32_t *branch_b_bias,
    int8_t *branch_b_output, int n_size, int c_size, int h_size,
    int input_w_size, int mid_size, int oh_size, int ow_size,
    int branch_a_size, int branch_b_size, int pad_left, int pad_top,
    int stride_h, int stride_w, int x_zero_point,
    int conv3x3_weight_zero_point, uint32_t conv3x3_multiplier_bits,
    int conv3x3_output_zero_point, int activated_zero_point,
    int branch_a_weight_zero_point, uint32_t branch_a_multiplier_bits,
    int branch_a_output_zero_point, int branch_b_weight_zero_point,
    uint32_t branch_b_multiplier_bits, int branch_b_output_zero_point);

#ifdef __cplusplus
}
#endif

#endif
