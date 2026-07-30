#include "Conv3x3Int8Kernel.h"
#include "Int8Requantize.h"

#include <stdint.h>

#ifdef __SYNTHESIS__
#include <ap_int.h>
#endif

namespace {

constexpr int kInputParallel = 4;
constexpr int kOutputParallel = 8;
constexpr int kOutputBlock = 16;
constexpr int kOutputTileHeight = 2;
constexpr int kOutputTileWidth = 8;
constexpr int kInputTileHeight =
    (kOutputTileHeight - 1) * MYACCEL_CONV3X3_INT8_MAX_STRIDE + 3;
constexpr int kInputTileWidth =
    (kOutputTileWidth - 1) * MYACCEL_CONV3X3_INT8_MAX_STRIDE + 3;
constexpr int kInputPlaneSize = kInputTileHeight * kInputTileWidth;
constexpr int kInputTileValues = kInputParallel * kInputPlaneSize;
constexpr int kOutputTilePixels = kOutputTileHeight * kOutputTileWidth;
constexpr int kComputeIterations =
    (kOutputBlock / kOutputParallel) * kOutputTilePixels;

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

static int32_t reduce36(const int32_t value[36]) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable = value complete
  int32_t level1[18];
  int32_t level2[9];
#pragma HLS ARRAY_PARTITION variable = level1 complete
#pragma HLS ARRAY_PARTITION variable = level2 complete

ReduceLevel1Loop:
  for (int i = 0; i < 18; ++i) {
#pragma HLS UNROLL
    level1[i] = value[2 * i] + value[2 * i + 1];
  }
ReduceLevel2Loop:
  for (int i = 0; i < 9; ++i) {
#pragma HLS UNROLL
    level2[i] = level1[2 * i] + level1[2 * i + 1];
  }

  const int32_t level3_0 = level2[0] + level2[1];
  const int32_t level3_1 = level2[2] + level2[3];
  const int32_t level3_2 = level2[4] + level2[5];
  const int32_t level3_3 = level2[6] + level2[7];
  const int32_t level4_0 = level3_0 + level3_1;
  const int32_t level4_1 = level3_2 + level3_3;
  return (level4_0 + level4_1) + level2[8];
}

} // namespace

extern "C" void conv3x3_i8_kernel(const int8_t *x, const int8_t *weight,
    const int32_t *bias, int8_t *output, int n_size, int c_size, int h_size,
    int input_w_size, int m_size, int oh_size, int ow_size, int pad_left,
    int pad_top, int stride_h, int stride_w, int x_zero_point,
    int w_zero_point, uint32_t requant_multiplier_bits,
    int output_zero_point) {
#pragma HLS INTERFACE m_axi port = x offset = slave bundle = gmem0 \
    max_read_burst_length = 64 num_read_outstanding = 16
#pragma HLS INTERFACE m_axi port = weight offset = slave bundle = gmem1 \
    max_read_burst_length = 64 num_read_outstanding = 16
#pragma HLS INTERFACE m_axi port = bias offset = slave bundle = gmem2 \
    max_read_burst_length = 64 num_read_outstanding = 16
#pragma HLS INTERFACE m_axi port = output offset = slave bundle = gmem3 \
    max_write_burst_length = 64 num_write_outstanding = 16
#pragma HLS INTERFACE s_axilite port = x bundle = control
#pragma HLS INTERFACE s_axilite port = weight bundle = control
#pragma HLS INTERFACE s_axilite port = bias bundle = control
#pragma HLS INTERFACE s_axilite port = output bundle = control
#pragma HLS INTERFACE s_axilite port = n_size bundle = control
#pragma HLS INTERFACE s_axilite port = c_size bundle = control
#pragma HLS INTERFACE s_axilite port = h_size bundle = control
#pragma HLS INTERFACE s_axilite port = input_w_size bundle = control
#pragma HLS INTERFACE s_axilite port = m_size bundle = control
#pragma HLS INTERFACE s_axilite port = oh_size bundle = control
#pragma HLS INTERFACE s_axilite port = ow_size bundle = control
#pragma HLS INTERFACE s_axilite port = pad_left bundle = control
#pragma HLS INTERFACE s_axilite port = pad_top bundle = control
#pragma HLS INTERFACE s_axilite port = stride_h bundle = control
#pragma HLS INTERFACE s_axilite port = stride_w bundle = control
#pragma HLS INTERFACE s_axilite port = x_zero_point bundle = control
#pragma HLS INTERFACE s_axilite port = w_zero_point bundle = control
#pragma HLS INTERFACE s_axilite port = requant_multiplier_bits bundle = control
#pragma HLS INTERFACE s_axilite port = output_zero_point bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

  if (!x || !weight || !bias || !output || n_size <= 0 || c_size <= 0 ||
      c_size > MYACCEL_CONV3X3_INT8_MAX_INPUT_CHANNELS || h_size <= 0 ||
      input_w_size <= 0 || m_size <= 0 || oh_size <= 0 || ow_size <= 0 ||
      stride_h <= 0 || stride_h > MYACCEL_CONV3X3_INT8_MAX_STRIDE ||
      stride_w <= 0 || stride_w > MYACCEL_CONV3X3_INT8_MAX_STRIDE ||
      !myaccel_int8::isPositiveNormalMultiplier(requant_multiplier_bits))
    return;

OutputBlockLoop:
  for (int m_block = 0; m_block < m_size; m_block += kOutputBlock) {
    centered_t weight_cache[kOutputBlock]
                             [MYACCEL_CONV3X3_INT8_MAX_INPUT_CHANNELS][3][3];
    int32_t bias_cache[kOutputBlock];
#pragma HLS ARRAY_RESHAPE variable = weight_cache cyclic \
    factor = kOutputParallel dim = 1
#pragma HLS ARRAY_RESHAPE variable = weight_cache cyclic \
    factor = kInputParallel dim = 2
#pragma HLS ARRAY_PARTITION variable = weight_cache complete dim = 3
#pragma HLS ARRAY_PARTITION variable = weight_cache complete dim = 4
// The nine independently-read tap planes use URAM so their 2,592-bit/cycle
// aggregate bandwidth does not consume distributed LUT RAM.
#pragma HLS BIND_STORAGE variable = weight_cache type = ram_2p impl = uram
#pragma HLS ARRAY_PARTITION variable = bias_cache complete

  LoadBiasLoop:
    for (int local_m = 0; local_m < kOutputBlock; ++local_m) {
#pragma HLS PIPELINE II = 1
      const int m = m_block + local_m;
      bias_cache[local_m] = m < m_size ? bias[m] : 0;
    }

  LoadWeightOutputLoop:
    for (int local_m = 0; local_m < kOutputBlock; ++local_m) {
      const int m = m_block + local_m;
    LoadWeightChannelLoop:
      for (int c = 0; c < c_size; ++c) {
      LoadWeightRowLoop:
        for (int kh = 0; kh < 3; ++kh) {
        LoadWeightColumnLoop:
          for (int kw = 0; kw < 3; ++kw) {
#pragma HLS PIPELINE II = 1
// Every flattened loop iteration writes a distinct OIHW cache element. Vitis
// otherwise infers a false loop-carried read/write dependence on weight_cache
// and schedules this preload at II=2.
#pragma HLS DEPENDENCE variable = weight_cache inter false
            if (m < m_size) {
              const uint32_t weight_index =
                  (((uint32_t)m * c_size + c) * 3 + kh) * 3 + kw;
              weight_cache[local_m][c][kh][kw] =
                  (centered_t)((int32_t)weight[weight_index] - w_zero_point);
            } else {
              weight_cache[local_m][c][kh][kw] = 0;
            }
          }
        }
      }
    }

  BatchLoop:
    for (int n = 0; n < n_size; ++n) {
    OutputTileRowLoop:
      for (int oh_base = 0; oh_base < oh_size;
           oh_base += kOutputTileHeight) {
      OutputTileColumnLoop:
        for (int ow_base = 0; ow_base < ow_size;
             ow_base += kOutputTileWidth) {
          centered_t input_tile[2][kInputParallel][kInputTileHeight]
                               [kInputTileWidth];
          int32_t accum[kOutputBlock][kOutputTileHeight][kOutputTileWidth];
#pragma HLS ARRAY_PARTITION variable = input_tile complete dim = 1
#pragma HLS ARRAY_PARTITION variable = input_tile complete dim = 2
#pragma HLS ARRAY_PARTITION variable = input_tile complete dim = 3
#pragma HLS ARRAY_PARTITION variable = input_tile complete dim = 4
// Pack eight output lanes into one 256-bit word and retain tile positions as
// RAM depth. Partitioning tile positions would waste one BRAM per shallow bank.
#pragma HLS ARRAY_RESHAPE variable = accum cyclic \
    factor = kOutputParallel dim = 1
#pragma HLS BIND_STORAGE variable = accum type = ram_2p impl = bram

        InitAccumOutputGroupLoop:
          for (int output_base = 0; output_base < kOutputBlock;
               output_base += kOutputParallel) {
          InitAccumRowLoop:
            for (int local_oh = 0; local_oh < kOutputTileHeight; ++local_oh) {
            InitAccumColumnLoop:
              for (int local_ow = 0; local_ow < kOutputTileWidth; ++local_ow) {
#pragma HLS PIPELINE II = 1
              InitAccumLaneLoop:
                for (int output_lane = 0;
                     output_lane < kOutputParallel; ++output_lane) {
#pragma HLS UNROLL
                  accum[output_base + output_lane][local_oh][local_ow] = 0;
                }
              }
            }
          }

          const int remaining_oh = oh_size - oh_base;
          const int remaining_ow = ow_size - ow_base;
          const int valid_oh = remaining_oh < kOutputTileHeight
                                   ? remaining_oh
                                   : kOutputTileHeight;
          const int valid_ow = remaining_ow < kOutputTileWidth
                                   ? remaining_ow
                                   : kOutputTileWidth;
          const int patch_height = (valid_oh - 1) * stride_h + 3;
          const int patch_width = (valid_ow - 1) * stride_w + 3;
          const int input_row_base = oh_base * stride_h - pad_top;
          const int input_column_base = ow_base * stride_w - pad_left;
          const int input_channel_tiles =
              (c_size + kInputParallel - 1) / kInputParallel;

// Prime one bank. Each following phase consumes the current bank while the
// AXI port fills the other bank with the next input-channel patch.
        PreloadFirstInputTileLoop:
          for (int load_index = 0; load_index < kInputTileValues;
               ++load_index) {
#pragma HLS PIPELINE II = 1
            const int input_lane = load_index / kInputPlaneSize;
            const int input_position = load_index % kInputPlaneSize;
            const int local_ih = input_position / kInputTileWidth;
            const int local_iw = input_position % kInputTileWidth;
            const int c = input_lane;
            const int ih = input_row_base + local_ih;
            const int iw = input_column_base + local_iw;
            if (c < c_size && local_ih < patch_height &&
                local_iw < patch_width && ih >= 0 && ih < h_size && iw >= 0 &&
                iw < input_w_size) {
              const uint32_t x_index =
                  ((uint32_t)n * c_size + c) * h_size * input_w_size +
                  (uint32_t)ih * input_w_size + iw;
              input_tile[0][input_lane][local_ih][local_iw] =
                  (centered_t)((int32_t)x[x_index] - x_zero_point);
            } else {
              input_tile[0][input_lane][local_ih][local_iw] = 0;
            }
          }

        InputChannelTileLoop:
          for (int channel_tile = 0; channel_tile < input_channel_tiles;
               ++channel_tile) {
            const int c_base = channel_tile * kInputParallel;
            const int current_buffer = channel_tile & 1;
            const int next_buffer = current_buffer ^ 1;
            const bool has_next = channel_tile + 1 < input_channel_tiles;
            const int phase_iterations =
                has_next ? kInputTileValues : kComputeIterations;

          OverlapInputLoadComputeLoop:
            for (int phase = 0; phase < phase_iterations; ++phase) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = input_tile inter false
#pragma HLS DEPENDENCE variable = accum inter false
              if (has_next) {
                const int input_lane = phase / kInputPlaneSize;
                const int input_position = phase % kInputPlaneSize;
                const int local_ih = input_position / kInputTileWidth;
                const int local_iw = input_position % kInputTileWidth;
                const int c = c_base + kInputParallel + input_lane;
                const int ih = input_row_base + local_ih;
                const int iw = input_column_base + local_iw;
                if (c < c_size && local_ih < patch_height &&
                    local_iw < patch_width && ih >= 0 && ih < h_size &&
                    iw >= 0 && iw < input_w_size) {
                  const uint32_t x_index =
                      ((uint32_t)n * c_size + c) * h_size * input_w_size +
                      (uint32_t)ih * input_w_size + iw;
                  input_tile[next_buffer][input_lane][local_ih][local_iw] =
                      (centered_t)((int32_t)x[x_index] - x_zero_point);
                } else {
                  input_tile[next_buffer][input_lane][local_ih][local_iw] = 0;
                }
              }

              if (phase < kComputeIterations) {
                const int output_group = phase / kOutputTilePixels;
                const int output_position = phase % kOutputTilePixels;
                const int output_base = output_group * kOutputParallel;
                const int local_oh = output_position / kOutputTileWidth;
                const int local_ow = output_position % kOutputTileWidth;
              ComputeOutputLaneLoop:
                for (int output_lane = 0;
                     output_lane < kOutputParallel; ++output_lane) {
#pragma HLS UNROLL
                  const int local_m = output_base + output_lane;
                  int32_t products[kInputParallel * 3 * 3];
#pragma HLS ARRAY_PARTITION variable = products complete
                ProductInputLaneLoop:
                  for (int input_lane = 0;
                       input_lane < kInputParallel; ++input_lane) {
#pragma HLS UNROLL
                    const int c = c_base + input_lane;
                  ProductRowLoop:
                    for (int kh = 0; kh < 3; ++kh) {
#pragma HLS UNROLL
                    ProductColumnLoop:
                      for (int kw = 0; kw < 3; ++kw) {
#pragma HLS UNROLL
                        const int product_index =
                            (input_lane * 3 + kh) * 3 + kw;
                        const centered_t weight_value =
                            c < c_size
                                ? weight_cache[local_m][c][kh][kw]
                                : centered_t(0);
                        const int local_ih = local_oh * stride_h + kh;
                        const int local_iw = local_ow * stride_w + kw;
                        products[product_index] = dspMultiply(
                            input_tile[current_buffer][input_lane][local_ih]
                                      [local_iw],
                            weight_value);
                      }
                    }
                  }
                  accum[local_m][local_oh][local_ow] += reduce36(products);
                }
              }
            }
          }

        StoreOutputLoop:
          for (int local_m = 0; local_m < kOutputBlock; ++local_m) {
            const int m = m_block + local_m;
          StoreOutputRowLoop:
            for (int local_oh = 0; local_oh < kOutputTileHeight; ++local_oh) {
              const int oh = oh_base + local_oh;
            StoreOutputColumnLoop:
              for (int local_ow = 0; local_ow < kOutputTileWidth;
                   ++local_ow) {
#pragma HLS PIPELINE II = 1
                const int ow = ow_base + local_ow;
                if (m < m_size && oh < oh_size && ow < ow_size) {
                  const uint32_t output_index =
                      ((uint32_t)n * m_size + m) * oh_size * ow_size +
                      (uint32_t)oh * ow_size + ow;
                  output[output_index] = myaccel_int8::requantize(
                      accum[local_m][local_oh][local_ow],
                      bias_cache[local_m], requant_multiplier_bits,
                      output_zero_point);
                }
              }
            }
          }
        }
      }
    }
  }
}
