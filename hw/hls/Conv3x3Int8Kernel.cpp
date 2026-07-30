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
constexpr int kPackedBytes = 4;
constexpr int kKernelElements = 3 * 3;
constexpr int kWeightWordsPerInputGroup =
    kInputParallel * kKernelElements / kPackedBytes;
constexpr int kInputTileHeight =
    (kOutputTileHeight - 1) * MYACCEL_CONV3X3_INT8_MAX_STRIDE + 3;
constexpr int kInputTileWidth =
    (kOutputTileWidth - 1) * MYACCEL_CONV3X3_INT8_MAX_STRIDE + 3;

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

static int8_t unpackInt8(uint32_t word, int byte_lane) {
#pragma HLS INLINE
  return (int8_t)((word >> (byte_lane * 8)) & 0xffU);
}

static uint32_t packInt8(const int8_t bytes[kPackedBytes]) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable = bytes complete
  return (uint32_t)(uint8_t)bytes[0] |
         ((uint32_t)(uint8_t)bytes[1] << 8) |
         ((uint32_t)(uint8_t)bytes[2] << 16) |
         ((uint32_t)(uint8_t)bytes[3] << 24);
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

extern "C" void conv3x3_i8_kernel(const uint32_t *x,
    const uint32_t *weight, const int32_t *bias, uint32_t *output,
    int n_size, int c_size, int h_size, int input_w_size, int m_size,
    int oh_size, int ow_size, int pad_left, int pad_top, int stride_h,
    int stride_w, int x_zero_point, int w_zero_point,
    uint32_t requant_multiplier_bits, int output_zero_point) {
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
      (c_size & (kPackedBytes - 1)) != 0 ||
      (input_w_size & (kPackedBytes - 1)) != 0 ||
      (ow_size & (kPackedBytes - 1)) != 0 ||
      stride_h <= 0 || stride_h > MYACCEL_CONV3X3_INT8_MAX_STRIDE ||
      stride_w <= 0 || stride_w > MYACCEL_CONV3X3_INT8_MAX_STRIDE ||
      !myaccel_int8::isPositiveNormalMultiplier(requant_multiplier_bits))
    return;

OutputBlockLoop:
  for (int m_block = 0; m_block < m_size; m_block += kOutputBlock) {
    uint32_t weight_cache[kOutputBlock]
                         [MYACCEL_CONV3X3_INT8_MAX_INPUT_CHANNELS /
                             kInputParallel]
                         [kWeightWordsPerInputGroup];
    int32_t bias_cache[kOutputBlock];
#pragma HLS ARRAY_RESHAPE variable = weight_cache cyclic \
    factor = kOutputParallel dim = 1
#pragma HLS ARRAY_PARTITION variable = weight_cache complete dim = 3
// Four input channels contain 36 OIHW bytes, exactly nine aligned words. Keep
// those nine words in independently-read URAM planes. Reshaping eight output
// lanes supplies 2,304 raw weight bits per cycle; the unrolled compute lanes
// center the resulting 72 INT8 operands without a preload byte-scatter mux.
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
    LoadWeightInputGroupLoop:
      for (int input_group = 0; input_group < c_size / kInputParallel;
           ++input_group) {
      LoadWeightWordLoop:
        for (int word_in_group = 0;
             word_in_group < kWeightWordsPerInputGroup; ++word_in_group) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = weight_cache inter false
          const uint32_t output_word_base =
              (uint32_t)m * c_size * kKernelElements / kPackedBytes;
          weight_cache[local_m][input_group][word_in_group] =
              m < m_size
                  ? weight[output_word_base +
                           input_group * kWeightWordsPerInputGroup +
                           word_in_group]
                  : 0;
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
          centered_t
              input_tile[kInputParallel][kInputTileHeight][kInputTileWidth];
          int32_t accum[kOutputBlock][kOutputTileHeight][kOutputTileWidth];
#pragma HLS ARRAY_PARTITION variable = input_tile complete dim = 1
#pragma HLS ARRAY_PARTITION variable = input_tile complete dim = 2
#pragma HLS ARRAY_PARTITION variable = input_tile complete dim = 3
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

        InputChannelTileLoop:
          for (int c_base = 0; c_base < c_size;
               c_base += kInputParallel) {
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

          LoadInputChannelLoop:
            for (int input_lane = 0; input_lane < kInputParallel;
                 ++input_lane) {
              const int c = c_base + input_lane;
            LoadInputRowLoop:
              for (int local_ih = 0; local_ih < kInputTileHeight;
                   ++local_ih) {
              LoadInputWordGroupLoop:
                for (int local_iw_base = 0;
                     local_iw_base < kInputTileWidth;
                     local_iw_base += kPackedBytes) {
#pragma HLS PIPELINE II = 1
                  const int ih = input_row_base + local_ih;
                  const int group_first_iw =
                      input_column_base + local_iw_base;
                  const int remaining_patch_columns =
                      patch_width - local_iw_base;
                  const int active_group_bytes =
                      remaining_patch_columns <= 0
                          ? 0
                          : (remaining_patch_columns < kPackedBytes
                                    ? remaining_patch_columns
                                    : kPackedBytes);
                  const int group_last_iw =
                      group_first_iw + active_group_bytes - 1;
                  const int valid_first_iw =
                      group_first_iw < 0 ? 0 : group_first_iw;
                  const int valid_last_iw = group_last_iw >= input_w_size
                                                ? input_w_size - 1
                                                : group_last_iw;
                  const bool valid_row = c < c_size &&
                                         local_ih < patch_height && ih >= 0 &&
                                         ih < h_size;
                  const bool has_valid_columns =
                      active_group_bytes > 0 &&
                      valid_first_iw <= valid_last_iw &&
                      valid_first_iw < input_w_size && valid_last_iw >= 0;

                  uint32_t first_word = 0;
                  uint32_t second_word = 0;
                  int first_word_column = 0;
                  int second_word_column = 0;
                  if (valid_row && has_valid_columns) {
                    first_word_column = valid_first_iw & ~(kPackedBytes - 1);
                    second_word_column = valid_last_iw & ~(kPackedBytes - 1);
                    const uint32_t row_byte_index =
                        ((uint32_t)n * c_size + c) * h_size * input_w_size +
                        (uint32_t)ih * input_w_size;
                    const uint32_t row_word_index =
                        row_byte_index / kPackedBytes;
                    first_word =
                        x[row_word_index + first_word_column / kPackedBytes];
                    second_word =
                        second_word_column == first_word_column
                            ? first_word
                            : x[row_word_index +
                                second_word_column / kPackedBytes];
                  }

                ScatterInputByteLoop:
                  for (int byte_lane = 0; byte_lane < kPackedBytes;
                       ++byte_lane) {
#pragma HLS UNROLL
                    const int local_iw = local_iw_base + byte_lane;
                    const int iw = group_first_iw + byte_lane;
                    if (local_iw < kInputTileWidth) {
                      if (valid_row && local_iw < patch_width && iw >= 0 &&
                          iw < input_w_size) {
                        const int word_column =
                            iw & ~(kPackedBytes - 1);
                        const uint32_t packed_input =
                            word_column == first_word_column ? first_word
                                                             : second_word;
                        input_tile[input_lane][local_ih][local_iw] =
                            (centered_t)((int32_t)unpackInt8(
                                             packed_input,
                                             iw & (kPackedBytes - 1)) -
                                         x_zero_point);
                      } else {
                        input_tile[input_lane][local_ih][local_iw] = 0;
                      }
                    }
                  }
                }
              }
            }

          OutputSubBlockLoop:
            for (int output_base = 0; output_base < kOutputBlock;
                 output_base += kOutputParallel) {
            ComputeOutputRowLoop:
              for (int local_oh = 0; local_oh < kOutputTileHeight;
                   ++local_oh) {
              ComputeOutputColumnLoop:
                for (int local_ow = 0; local_ow < kOutputTileWidth;
                     ++local_ow) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = accum inter false
                ComputeOutputLaneLoop:
                  for (int output_lane = 0;
                       output_lane < kOutputParallel; ++output_lane) {
#pragma HLS UNROLL
                    const int local_m = output_base + output_lane;
                    uint32_t packed_weights[kWeightWordsPerInputGroup];
                    int32_t products[kInputParallel * 3 * 3];
#pragma HLS ARRAY_PARTITION variable = packed_weights complete
#pragma HLS ARRAY_PARTITION variable = products complete
                  ReadPackedWeightWordLoop:
                    for (int word = 0;
                         word < kWeightWordsPerInputGroup; ++word) {
#pragma HLS UNROLL
                      packed_weights[word] =
                          weight_cache[local_m]
                                      [c_base / kInputParallel][word];
                    }
                  ProductInputLaneLoop:
                    for (int input_lane = 0;
                         input_lane < kInputParallel; ++input_lane) {
#pragma HLS UNROLL
                    ProductRowLoop:
                      for (int kh = 0; kh < 3; ++kh) {
#pragma HLS UNROLL
                      ProductColumnLoop:
                        for (int kw = 0; kw < 3; ++kw) {
#pragma HLS UNROLL
                          const int product_index =
                              (input_lane * 3 + kh) * 3 + kw;
                          const int packed_word = product_index / kPackedBytes;
                          const int packed_byte =
                              product_index & (kPackedBytes - 1);
                          const centered_t weight_value =
                              m_block + local_m < m_size
                                  ? (centered_t)((int32_t)unpackInt8(
                                                     packed_weights[packed_word],
                                                     packed_byte) -
                                                 w_zero_point)
                                  : centered_t(0);
                          const int local_ih = local_oh * stride_h + kh;
                          const int local_iw = local_ow * stride_w + kw;
                          products[product_index] =
                              dspMultiply(input_tile[input_lane][local_ih]
                                                    [local_iw],
                                  weight_value);
                        }
                      }
                    }
                    accum[local_m][local_oh][local_ow] +=
                        reduce36(products);
                  }
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
              int8_t output_bytes[kPackedBytes];
#pragma HLS ARRAY_PARTITION variable = output_bytes complete
            StoreOutputColumnLoop:
              for (int local_ow = 0; local_ow < kOutputTileWidth;
                   ++local_ow) {
#pragma HLS PIPELINE II = 1
                const int ow = ow_base + local_ow;
                if (m < m_size && oh < oh_size && ow < ow_size) {
                  const int byte_lane = local_ow & (kPackedBytes - 1);
                  output_bytes[byte_lane] = myaccel_int8::requantize(
                      accum[local_m][local_oh][local_ow],
                      bias_cache[local_m], requant_multiplier_bits,
                      output_zero_point);
                  // ow_size and the eight-column tile are word aligned. Emit
                  // one packed AXI word after four scalar requantizer cycles.
                  if (byte_lane == kPackedBytes - 1) {
                    const uint32_t output_byte_index =
                        ((uint32_t)n * m_size + m) * oh_size * ow_size +
                        (uint32_t)oh * ow_size + ow -
                        (kPackedBytes - 1);
                    output[output_byte_index / kPackedBytes] =
                        packInt8(output_bytes);
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
