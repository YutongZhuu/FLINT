#include "Conv3x3ActDual1x1Int8Kernel.h"
#include "Int8Requantize.h"

#include <stdint.h>

#ifdef __SYNTHESIS__
#include <ap_int.h>
#endif

namespace {

constexpr int kInputParallel = 2;
constexpr int kMidParallel = 4;
constexpr int kBranchInputParallel = 4;
constexpr int kBranchOutputParallel = 4;
constexpr int kPixelTile = 8;
constexpr int kInputTileHeight = 3;
constexpr int kInputTileWidth =
    (kPixelTile - 1) * MYACCEL_FUSED_INT8_MAX_STRIDE + 3;

#ifdef __SYNTHESIS__
using centered_t = ap_int<9>;
#else
using centered_t = int16_t;
#endif

static int32_t dspMultiply(centered_t lhs, centered_t rhs) {
#pragma HLS INLINE
  int32_t product;
#pragma HLS BIND_OP variable = product op = mul impl = dsp
  product = lhs * rhs;
  return product;
}

static int32_t reduce18(const int32_t value[18]) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable = value complete
  int32_t level1[9];
#pragma HLS ARRAY_PARTITION variable = level1 complete

Reduce18Level1Loop:
  for (int i = 0; i < 9; ++i) {
#pragma HLS UNROLL
    level1[i] = value[2 * i] + value[2 * i + 1];
  }

  const int32_t level2_0 = level1[0] + level1[1];
  const int32_t level2_1 = level1[2] + level1[3];
  const int32_t level2_2 = level1[4] + level1[5];
  const int32_t level2_3 = level1[6] + level1[7];
  const int32_t level3_0 = level2_0 + level2_1;
  const int32_t level3_1 = level2_2 + level2_3;
  return (level3_0 + level3_1) + level1[8];
}

static int32_t reduce4(const int32_t value[4]) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable = value complete
  return (value[0] + value[1]) + (value[2] + value[3]);
}

} // namespace

extern "C" void conv3x3_act_dual1x1_i8_kernel(const int8_t *x,
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
    uint32_t branch_b_multiplier_bits, int branch_b_output_zero_point) {
#pragma HLS INTERFACE m_axi port = x offset = slave bundle = gmem0 \
    max_read_burst_length = 64 num_read_outstanding = 16
#pragma HLS INTERFACE m_axi port = conv3x3_weight offset = slave bundle = gmem1 \
    max_read_burst_length = 64 num_read_outstanding = 16
#pragma HLS INTERFACE m_axi port = conv3x3_bias offset = slave bundle = gmem1 \
    max_read_burst_length = 64 num_read_outstanding = 16
#pragma HLS INTERFACE m_axi port = activation_lut offset = slave bundle = gmem1 \
    max_read_burst_length = 64 num_read_outstanding = 16
#pragma HLS INTERFACE m_axi port = branch_a_weight offset = slave bundle = gmem2 \
    max_read_burst_length = 64 num_read_outstanding = 16
#pragma HLS INTERFACE m_axi port = branch_a_bias offset = slave bundle = gmem2 \
    max_read_burst_length = 64 num_read_outstanding = 16
#pragma HLS INTERFACE m_axi port = branch_a_output offset = slave bundle = gmem2 \
    max_write_burst_length = 64 num_write_outstanding = 16
#pragma HLS INTERFACE m_axi port = branch_b_weight offset = slave bundle = gmem3 \
    max_read_burst_length = 64 num_read_outstanding = 16
#pragma HLS INTERFACE m_axi port = branch_b_bias offset = slave bundle = gmem3 \
    max_read_burst_length = 64 num_read_outstanding = 16
#pragma HLS INTERFACE m_axi port = branch_b_output offset = slave bundle = gmem3 \
    max_write_burst_length = 64 num_write_outstanding = 16
#pragma HLS INTERFACE s_axilite port = x bundle = control
#pragma HLS INTERFACE s_axilite port = conv3x3_weight bundle = control
#pragma HLS INTERFACE s_axilite port = conv3x3_bias bundle = control
#pragma HLS INTERFACE s_axilite port = activation_lut bundle = control
#pragma HLS INTERFACE s_axilite port = branch_a_weight bundle = control
#pragma HLS INTERFACE s_axilite port = branch_a_bias bundle = control
#pragma HLS INTERFACE s_axilite port = branch_a_output bundle = control
#pragma HLS INTERFACE s_axilite port = branch_b_weight bundle = control
#pragma HLS INTERFACE s_axilite port = branch_b_bias bundle = control
#pragma HLS INTERFACE s_axilite port = branch_b_output bundle = control
#pragma HLS INTERFACE s_axilite port = n_size bundle = control
#pragma HLS INTERFACE s_axilite port = c_size bundle = control
#pragma HLS INTERFACE s_axilite port = h_size bundle = control
#pragma HLS INTERFACE s_axilite port = input_w_size bundle = control
#pragma HLS INTERFACE s_axilite port = mid_size bundle = control
#pragma HLS INTERFACE s_axilite port = oh_size bundle = control
#pragma HLS INTERFACE s_axilite port = ow_size bundle = control
#pragma HLS INTERFACE s_axilite port = branch_a_size bundle = control
#pragma HLS INTERFACE s_axilite port = branch_b_size bundle = control
#pragma HLS INTERFACE s_axilite port = pad_left bundle = control
#pragma HLS INTERFACE s_axilite port = pad_top bundle = control
#pragma HLS INTERFACE s_axilite port = stride_h bundle = control
#pragma HLS INTERFACE s_axilite port = stride_w bundle = control
#pragma HLS INTERFACE s_axilite port = x_zero_point bundle = control
#pragma HLS INTERFACE s_axilite port = conv3x3_weight_zero_point bundle = control
#pragma HLS INTERFACE s_axilite port = conv3x3_multiplier_bits bundle = control
#pragma HLS INTERFACE s_axilite port = conv3x3_output_zero_point bundle = control
#pragma HLS INTERFACE s_axilite port = activated_zero_point bundle = control
#pragma HLS INTERFACE s_axilite port = branch_a_weight_zero_point bundle = control
#pragma HLS INTERFACE s_axilite port = branch_a_multiplier_bits bundle = control
#pragma HLS INTERFACE s_axilite port = branch_a_output_zero_point bundle = control
#pragma HLS INTERFACE s_axilite port = branch_b_weight_zero_point bundle = control
#pragma HLS INTERFACE s_axilite port = branch_b_multiplier_bits bundle = control
#pragma HLS INTERFACE s_axilite port = branch_b_output_zero_point bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

  if (!x || !conv3x3_weight || !conv3x3_bias || !activation_lut ||
      !branch_a_weight || !branch_a_bias || !branch_a_output ||
      !branch_b_weight || !branch_b_bias || !branch_b_output || n_size <= 0 ||
      c_size <= 0 || c_size > MYACCEL_FUSED_INT8_MAX_INPUT_CHANNELS ||
      h_size <= 0 || input_w_size <= 0 || mid_size <= 0 ||
      mid_size > MYACCEL_FUSED_INT8_MAX_MID_CHANNELS || oh_size <= 0 ||
      ow_size <= 0 || branch_a_size <= 0 ||
      branch_a_size > MYACCEL_FUSED_INT8_MAX_BRANCH_OUTPUT_CHANNELS ||
      branch_b_size <= 0 ||
      branch_b_size > MYACCEL_FUSED_INT8_MAX_BRANCH_OUTPUT_CHANNELS ||
      pad_left < 0 || pad_top < 0 || stride_h <= 0 ||
      stride_h > MYACCEL_FUSED_INT8_MAX_STRIDE || stride_w <= 0 ||
      stride_w > MYACCEL_FUSED_INT8_MAX_STRIDE ||
      !myaccel_int8::isPositiveNormalMultiplier(conv3x3_multiplier_bits) ||
      !myaccel_int8::isPositiveNormalMultiplier(branch_a_multiplier_bits) ||
      !myaccel_int8::isPositiveNormalMultiplier(branch_b_multiplier_bits))
    return;

  centered_t conv3x3_weight_cache[MYACCEL_FUSED_INT8_MAX_MID_CHANNELS]
                                        [MYACCEL_FUSED_INT8_MAX_INPUT_CHANNELS]
                                        [3][3];
  int32_t conv3x3_bias_cache[MYACCEL_FUSED_INT8_MAX_MID_CHANNELS];
  int8_t lut_cache[256];
  centered_t branch_a_weight_cache
      [MYACCEL_FUSED_INT8_MAX_BRANCH_OUTPUT_CHANNELS]
      [MYACCEL_FUSED_INT8_MAX_MID_CHANNELS];
  centered_t branch_b_weight_cache
      [MYACCEL_FUSED_INT8_MAX_BRANCH_OUTPUT_CHANNELS]
      [MYACCEL_FUSED_INT8_MAX_MID_CHANNELS];
  int32_t branch_a_bias_cache
      [MYACCEL_FUSED_INT8_MAX_BRANCH_OUTPUT_CHANNELS];
  int32_t branch_b_bias_cache
      [MYACCEL_FUSED_INT8_MAX_BRANCH_OUTPUT_CHANNELS];
#pragma HLS ARRAY_RESHAPE variable = conv3x3_weight_cache cyclic \
    factor = kMidParallel dim = 1
#pragma HLS ARRAY_RESHAPE variable = conv3x3_weight_cache cyclic \
    factor = kInputParallel dim = 2
#pragma HLS ARRAY_PARTITION variable = conv3x3_weight_cache complete dim = 3
#pragma HLS ARRAY_PARTITION variable = conv3x3_weight_cache complete dim = 4
#pragma HLS BIND_STORAGE variable = conv3x3_weight_cache type = ram_2p impl = uram
#pragma HLS ARRAY_PARTITION variable = conv3x3_bias_cache cyclic \
    factor = kMidParallel
#pragma HLS ARRAY_PARTITION variable = lut_cache complete
#pragma HLS ARRAY_RESHAPE variable = branch_a_weight_cache cyclic \
    factor = kBranchOutputParallel dim = 1
#pragma HLS ARRAY_RESHAPE variable = branch_a_weight_cache cyclic \
    factor = kBranchInputParallel dim = 2
#pragma HLS ARRAY_RESHAPE variable = branch_b_weight_cache cyclic \
    factor = kBranchOutputParallel dim = 1
#pragma HLS ARRAY_RESHAPE variable = branch_b_weight_cache cyclic \
    factor = kBranchInputParallel dim = 2
#pragma HLS BIND_STORAGE variable = branch_a_weight_cache type = ram_2p impl = bram
#pragma HLS BIND_STORAGE variable = branch_b_weight_cache type = ram_2p impl = bram
#pragma HLS ARRAY_PARTITION variable = branch_a_bias_cache cyclic \
    factor = kBranchOutputParallel
#pragma HLS ARRAY_PARTITION variable = branch_b_bias_cache cyclic \
    factor = kBranchOutputParallel

LoadActivationLutLoop:
  for (int i = 0; i < 256; ++i) {
#pragma HLS PIPELINE II = 1
    lut_cache[i] = activation_lut[i];
  }

LoadConv3x3BiasLoop:
  for (int m = 0; m < mid_size; ++m) {
#pragma HLS PIPELINE II = 1
    conv3x3_bias_cache[m] = conv3x3_bias[m];
  }

LoadConv3x3WeightMidLoop:
  for (int m = 0; m < mid_size; ++m) {
  LoadConv3x3WeightChannelLoop:
    for (int c = 0; c < c_size; ++c) {
    LoadConv3x3WeightRowLoop:
      for (int kh = 0; kh < 3; ++kh) {
      LoadConv3x3WeightColumnLoop:
        for (int kw = 0; kw < 3; ++kw) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = conv3x3_weight_cache inter false
          const uint32_t index =
              (((uint32_t)m * c_size + c) * 3 + kh) * 3 + kw;
          conv3x3_weight_cache[m][c][kh][kw] =
              (centered_t)((int32_t)conv3x3_weight[index] -
                           conv3x3_weight_zero_point);
        }
      }
    }
  }

LoadBranchAOutputLoop:
  for (int output = 0; output < branch_a_size; ++output) {
    branch_a_bias_cache[output] = branch_a_bias[output];
  LoadBranchAInputLoop:
    for (int mid = 0; mid < mid_size; ++mid) {
#pragma HLS PIPELINE II = 1
      branch_a_weight_cache[output][mid] =
          (centered_t)((int32_t)branch_a_weight[(uint32_t)output * mid_size +
                                                mid] -
                       branch_a_weight_zero_point);
    }
  }

LoadBranchBOutputLoop:
  for (int output = 0; output < branch_b_size; ++output) {
    branch_b_bias_cache[output] = branch_b_bias[output];
  LoadBranchBInputLoop:
    for (int mid = 0; mid < mid_size; ++mid) {
#pragma HLS PIPELINE II = 1
      branch_b_weight_cache[output][mid] =
          (centered_t)((int32_t)branch_b_weight[(uint32_t)output * mid_size +
                                                mid] -
                       branch_b_weight_zero_point);
    }
  }

BatchLoop:
  for (int n = 0; n < n_size; ++n) {
  OutputRowLoop:
    for (int oh = 0; oh < oh_size; ++oh) {
    OutputColumnTileLoop:
      for (int ow_base = 0; ow_base < ow_size; ow_base += kPixelTile) {
        int32_t mid_accum[MYACCEL_FUSED_INT8_MAX_MID_CHANNELS][kPixelTile];
        int8_t activated_tile[MYACCEL_FUSED_INT8_MAX_MID_CHANNELS]
                              [kPixelTile];
        int32_t branch_a_accum
            [MYACCEL_FUSED_INT8_MAX_BRANCH_OUTPUT_CHANNELS][kPixelTile];
        int32_t branch_b_accum
            [MYACCEL_FUSED_INT8_MAX_BRANCH_OUTPUT_CHANNELS][kPixelTile];
#pragma HLS ARRAY_RESHAPE variable = mid_accum cyclic \
    factor = kMidParallel dim = 1
#pragma HLS BIND_STORAGE variable = mid_accum type = ram_2p impl = bram
#pragma HLS ARRAY_RESHAPE variable = activated_tile cyclic \
    factor = kBranchInputParallel dim = 1
#pragma HLS BIND_STORAGE variable = activated_tile type = ram_2p impl = bram
#pragma HLS ARRAY_PARTITION variable = branch_a_accum cyclic \
    factor = kBranchOutputParallel dim = 1
#pragma HLS ARRAY_PARTITION variable = branch_b_accum cyclic \
    factor = kBranchOutputParallel dim = 1
#pragma HLS BIND_STORAGE variable = branch_a_accum type = ram_2p impl = bram
#pragma HLS BIND_STORAGE variable = branch_b_accum type = ram_2p impl = bram

      InitMidOutputGroupLoop:
        for (int mid_base = 0; mid_base < mid_size;
             mid_base += kMidParallel) {
        InitMidPixelLoop:
          for (int pixel = 0; pixel < kPixelTile; ++pixel) {
#pragma HLS PIPELINE II = 1
          InitMidLaneLoop:
            for (int mid_lane = 0; mid_lane < kMidParallel; ++mid_lane) {
#pragma HLS UNROLL
              mid_accum[mid_base + mid_lane][pixel] = 0;
            }
          }
        }

      Conv3x3InputChannelTileLoop:
        for (int c_base = 0; c_base < c_size;
             c_base += kInputParallel) {
          centered_t input_tile[kInputParallel][kInputTileHeight]
                               [kInputTileWidth];
#pragma HLS ARRAY_PARTITION variable = input_tile complete dim = 1
#pragma HLS ARRAY_PARTITION variable = input_tile complete dim = 2
#pragma HLS ARRAY_PARTITION variable = input_tile complete dim = 3
          const int remaining_pixels = ow_size - ow_base;
          const int valid_pixels =
              remaining_pixels < kPixelTile ? remaining_pixels : kPixelTile;
          const int patch_width = (valid_pixels - 1) * stride_w + 3;
          const int input_row_base = oh * stride_h - pad_top;
          const int input_column_base = ow_base * stride_w - pad_left;

        LoadInputLaneLoop:
          for (int input_lane = 0; input_lane < kInputParallel;
               ++input_lane) {
            const int c = c_base + input_lane;
          LoadInputRowLoop:
            for (int local_ih = 0; local_ih < kInputTileHeight; ++local_ih) {
            LoadInputColumnLoop:
              for (int local_iw = 0; local_iw < kInputTileWidth; ++local_iw) {
#pragma HLS PIPELINE II = 1
                const int ih = input_row_base + local_ih;
                const int iw = input_column_base + local_iw;
                if (c < c_size && local_iw < patch_width && ih >= 0 &&
                    ih < h_size && iw >= 0 && iw < input_w_size) {
                  const uint32_t index =
                      ((uint32_t)n * c_size + c) * h_size * input_w_size +
                      (uint32_t)ih * input_w_size + iw;
                  input_tile[input_lane][local_ih][local_iw] =
                      (centered_t)((int32_t)x[index] - x_zero_point);
                } else {
                  input_tile[input_lane][local_ih][local_iw] = 0;
                }
              }
            }
          }

        Conv3x3MidOutputGroupLoop:
          for (int mid_base = 0; mid_base < mid_size;
               mid_base += kMidParallel) {
          Conv3x3PixelLoop:
            for (int pixel = 0; pixel < kPixelTile; ++pixel) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = mid_accum inter false
              centered_t input_values[kInputParallel * 3 * 3];
#pragma HLS ARRAY_PARTITION variable = input_values complete
            CaptureInputLaneLoop:
              for (int input_lane = 0; input_lane < kInputParallel;
                   ++input_lane) {
#pragma HLS UNROLL
              CaptureInputRowLoop:
                for (int kh = 0; kh < 3; ++kh) {
#pragma HLS UNROLL
                CaptureInputColumnLoop:
                  for (int kw = 0; kw < 3; ++kw) {
#pragma HLS UNROLL
                    const int index = (input_lane * 3 + kh) * 3 + kw;
                    input_values[index] = input_tile[input_lane][kh]
                                                        [pixel * stride_w + kw];
                  }
                }
              }

            Conv3x3MidLaneLoop:
              for (int mid_lane = 0; mid_lane < kMidParallel; ++mid_lane) {
#pragma HLS UNROLL
                const int mid = mid_base + mid_lane;
                int32_t products[kInputParallel * 3 * 3];
#pragma HLS ARRAY_PARTITION variable = products complete
              Conv3x3ProductInputLaneLoop:
                for (int input_lane = 0; input_lane < kInputParallel;
                     ++input_lane) {
#pragma HLS UNROLL
                  const int c = c_base + input_lane;
                Conv3x3ProductRowLoop:
                  for (int kh = 0; kh < 3; ++kh) {
#pragma HLS UNROLL
                  Conv3x3ProductColumnLoop:
                    for (int kw = 0; kw < 3; ++kw) {
#pragma HLS UNROLL
                      const int index = (input_lane * 3 + kh) * 3 + kw;
                      const centered_t weight_value =
                          mid < mid_size && c < c_size
                              ? conv3x3_weight_cache[mid][c][kh][kw]
                              : centered_t(0);
                      products[index] =
                          dspMultiply(input_values[index], weight_value);
                    }
                  }
                }
                mid_accum[mid][pixel] += reduce18(products);
              }
            }
          }
        }

      RequantizeActivationMidLoop:
        for (int mid = 0; mid < mid_size; ++mid) {
        RequantizeActivationPixelLoop:
          for (int pixel = 0; pixel < kPixelTile; ++pixel) {
#pragma HLS PIPELINE II = 1
            // Requantization includes an exact 24x24 binary32-significand
            // multiply. Keep one lane here; replicating it four times saves
            // little beside the much larger convolution loops.
            const int8_t quantized = myaccel_int8::requantize(
                mid_accum[mid][pixel], conv3x3_bias_cache[mid],
                conv3x3_multiplier_bits, conv3x3_output_zero_point);
            activated_tile[mid][pixel] = lut_cache[(int)quantized + 128];
          }
        }

      InitBranchOutputGroupLoop:
        for (int output_base = 0;
             output_base < MYACCEL_FUSED_INT8_MAX_BRANCH_OUTPUT_CHANNELS;
             output_base += kBranchOutputParallel) {
        InitBranchPixelLoop:
          for (int pixel = 0; pixel < kPixelTile; ++pixel) {
#pragma HLS PIPELINE II = 1
          InitBranchOutputLaneLoop:
            for (int output_lane = 0;
                 output_lane < kBranchOutputParallel; ++output_lane) {
#pragma HLS UNROLL
              branch_a_accum[output_base + output_lane][pixel] = 0;
              branch_b_accum[output_base + output_lane][pixel] = 0;
            }
          }
        }

        const int max_branch_size =
            branch_a_size > branch_b_size ? branch_a_size : branch_b_size;
      BranchInputGroupLoop:
        for (int mid_base = 0; mid_base < mid_size;
             mid_base += kBranchInputParallel) {
        BranchOutputGroupLoop:
          for (int output_base = 0; output_base < max_branch_size;
               output_base += kBranchOutputParallel) {
          BranchPixelLoop:
            for (int pixel = 0; pixel < kPixelTile; ++pixel) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = branch_a_accum inter false
#pragma HLS DEPENDENCE variable = branch_b_accum inter false
              centered_t activation_values[kBranchInputParallel];
#pragma HLS ARRAY_PARTITION variable = activation_values complete
            CenterActivationLoop:
              for (int input_lane = 0;
                   input_lane < kBranchInputParallel; ++input_lane) {
#pragma HLS UNROLL
                const int mid = mid_base + input_lane;
                activation_values[input_lane] =
                    mid < mid_size
                        ? (centered_t)((int32_t)activated_tile[mid][pixel] -
                                       activated_zero_point)
                        : centered_t(0);
              }

            BranchOutputLaneLoop:
              for (int output_lane = 0;
                   output_lane < kBranchOutputParallel; ++output_lane) {
#pragma HLS UNROLL
                const int output = output_base + output_lane;
                int32_t products_a[kBranchInputParallel];
                int32_t products_b[kBranchInputParallel];
#pragma HLS ARRAY_PARTITION variable = products_a complete
#pragma HLS ARRAY_PARTITION variable = products_b complete
              BranchProductInputLoop:
                for (int input_lane = 0;
                     input_lane < kBranchInputParallel; ++input_lane) {
#pragma HLS UNROLL
                  const int mid = mid_base + input_lane;
                  const centered_t weight_a =
                      output < branch_a_size && mid < mid_size
                          ? branch_a_weight_cache[output][mid]
                          : centered_t(0);
                  const centered_t weight_b =
                      output < branch_b_size && mid < mid_size
                          ? branch_b_weight_cache[output][mid]
                          : centered_t(0);
                  products_a[input_lane] =
                      dspMultiply(activation_values[input_lane], weight_a);
                  products_b[input_lane] =
                      dspMultiply(activation_values[input_lane], weight_b);
                }
                if (output < branch_a_size)
                  branch_a_accum[output][pixel] += reduce4(products_a);
                if (output < branch_b_size)
                  branch_b_accum[output][pixel] += reduce4(products_b);
              }
            }
          }
        }

      StoreBranchAOutputLoop:
        for (int output = 0; output < branch_a_size; ++output) {
        StoreBranchAPixelLoop:
          for (int pixel = 0; pixel < kPixelTile; ++pixel) {
#pragma HLS PIPELINE II = 1
            const int ow = ow_base + pixel;
            if (ow < ow_size) {
              const uint32_t index =
                  ((uint32_t)n * branch_a_size + output) * oh_size * ow_size +
                  (uint32_t)oh * ow_size + ow;
              branch_a_output[index] = myaccel_int8::requantize(
                  branch_a_accum[output][pixel],
                  branch_a_bias_cache[output], branch_a_multiplier_bits,
                  branch_a_output_zero_point);
            }
          }
        }

      StoreBranchBOutputLoop:
        for (int output = 0; output < branch_b_size; ++output) {
        StoreBranchBPixelLoop:
          for (int pixel = 0; pixel < kPixelTile; ++pixel) {
#pragma HLS PIPELINE II = 1
            const int ow = ow_base + pixel;
            if (ow < ow_size) {
              const uint32_t index =
                  ((uint32_t)n * branch_b_size + output) * oh_size * ow_size +
                  (uint32_t)oh * ow_size + ow;
              branch_b_output[index] = myaccel_int8::requantize(
                  branch_b_accum[output][pixel],
                  branch_b_bias_cache[output], branch_b_multiplier_bits,
                  branch_b_output_zero_point);
            }
          }
        }
      }
    }
  }
}
