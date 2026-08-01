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
constexpr int kPackedInputRowWords =
    (kInputTileWidth + kPackedBytes - 1) / kPackedBytes;
constexpr int kPackedInputBankWords =
    kInputParallel * kInputTileHeight * kPackedInputRowWords;
constexpr int kPackedInputBufferWords = 2 * kPackedInputBankWords;

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

static int packedInputIndex(
    int bank, int input_lane, int local_ih, int word_offset) {
#pragma HLS INLINE
  return bank * kPackedInputBankWords +
         (input_lane * kInputTileHeight + local_ih) *
             kPackedInputRowWords +
         word_offset;
}

// Load the valid portion of one four-channel patch as contiguous packed words.
// Padding is deliberately excluded from this loop so the x access is an
// unconditional, monotonically-increasing AXI read that HLS can burst.
static void loadPackedInputTile(const uint32_t *x,
    uint32_t packed_input[kPackedInputBufferWords], int bank, int n,
    int c_size, int h_size, int input_w_size, int c_base, int patch_height,
    int patch_width, int input_row_base, int input_column_base) {
#pragma HLS INLINE off
  const int first_local_row =
      input_row_base < 0 ? -input_row_base : 0;
  const int rows_inside_image = h_size - input_row_base;
  const int last_local_row =
      rows_inside_image < patch_height ? rows_inside_image : patch_height;

  const int first_valid_column =
      input_column_base < 0 ? 0 : input_column_base;
  const int patch_last_column = input_column_base + patch_width - 1;
  const int last_valid_column =
      patch_last_column < input_w_size ? patch_last_column : input_w_size - 1;
  const bool has_valid_region = first_local_row < last_local_row &&
                                first_valid_column <= last_valid_column;
  const int first_word_column =
      has_valid_region
          ? first_valid_column & ~(kPackedBytes - 1)
          : 0;
  const int last_word_column =
      has_valid_region
          ? last_valid_column & ~(kPackedBytes - 1)
          : -kPackedBytes;
  const int valid_word_count =
      has_valid_region
          ? (last_word_column - first_word_column) / kPackedBytes + 1
          : 0;

LoadPackedInputLaneLoop:
  for (int input_lane = 0; input_lane < kInputParallel; ++input_lane) {
    const int c = c_base + input_lane;
  LoadPackedInputRowLoop:
    for (int local_ih = first_local_row; local_ih < last_local_row;
         ++local_ih) {
      const int ih = input_row_base + local_ih;
      const uint32_t row_byte_index =
          ((uint32_t)n * c_size + c) * h_size * input_w_size +
          (uint32_t)ih * input_w_size;
      const uint32_t first_source_word =
          row_byte_index / kPackedBytes + first_word_column / kPackedBytes;
    BurstLoadPackedInputWords:
      for (int word_offset = 0; word_offset < valid_word_count;
           ++word_offset) {
#pragma HLS PIPELINE II = 1
        packed_input[packedInputIndex(
            bank, input_lane, local_ih, word_offset)] =
            x[first_source_word + word_offset];
      }
    }
  }
}

// Expand one packed staging-bank into the fully-partitioned register window.
// Keeping this conversion separate from the AXI loop removes conditional DDR
// reads while preserving the 36 scalar activation reads needed by compute.
static void expandPackedInputTile(
    const uint32_t packed_input[kPackedInputBufferWords], int bank,
    centered_t input_tile[kInputParallel][kInputTileHeight][kInputTileWidth],
    int h_size, int input_w_size, int patch_height, int patch_width,
    int input_row_base, int input_column_base, int x_zero_point) {
#pragma HLS INLINE off
ClearInputTileLaneLoop:
  for (int input_lane = 0; input_lane < kInputParallel; ++input_lane) {
  ClearInputTileRowLoop:
    for (int local_ih = 0; local_ih < kInputTileHeight; ++local_ih) {
    ClearInputTileWordLoop:
      for (int local_iw_base = 0; local_iw_base < kInputTileWidth;
           local_iw_base += kPackedBytes) {
#pragma HLS PIPELINE II = 1
      ClearInputTileByteLoop:
        for (int byte_lane = 0; byte_lane < kPackedBytes; ++byte_lane) {
#pragma HLS UNROLL
          const int local_iw = local_iw_base + byte_lane;
          if (local_iw < kInputTileWidth)
            input_tile[input_lane][local_ih][local_iw] = 0;
        }
      }
    }
  }

  const int first_local_row =
      input_row_base < 0 ? -input_row_base : 0;
  const int rows_inside_image = h_size - input_row_base;
  const int last_local_row =
      rows_inside_image < patch_height ? rows_inside_image : patch_height;
  const int first_valid_column =
      input_column_base < 0 ? 0 : input_column_base;
  const int patch_last_column = input_column_base + patch_width - 1;
  const int last_valid_column =
      patch_last_column < input_w_size ? patch_last_column : input_w_size - 1;
  const bool has_valid_region = first_local_row < last_local_row &&
                                first_valid_column <= last_valid_column;
  const int first_word_column =
      has_valid_region
          ? first_valid_column & ~(kPackedBytes - 1)
          : 0;
  const int last_word_column =
      has_valid_region
          ? last_valid_column & ~(kPackedBytes - 1)
          : -kPackedBytes;
  const int valid_word_count =
      has_valid_region
          ? (last_word_column - first_word_column) / kPackedBytes + 1
          : 0;

ExpandInputLaneLoop:
  for (int input_lane = 0; input_lane < kInputParallel; ++input_lane) {
  ExpandInputRowLoop:
    for (int local_ih = first_local_row; local_ih < last_local_row;
         ++local_ih) {
    ExpandInputWordLoop:
      for (int word_offset = 0; word_offset < valid_word_count;
           ++word_offset) {
#pragma HLS PIPELINE II = 1
        const uint32_t packed_value = packed_input[packedInputIndex(
            bank, input_lane, local_ih, word_offset)];
      ExpandInputByteLoop:
        for (int byte_lane = 0; byte_lane < kPackedBytes; ++byte_lane) {
#pragma HLS UNROLL
          const int iw = first_word_column + word_offset * kPackedBytes +
                         byte_lane;
          const int local_iw = iw - input_column_base;
          if (iw >= first_valid_column && iw <= last_valid_column &&
              local_iw >= 0 && local_iw < patch_width &&
              local_iw < kInputTileWidth) {
            input_tile[input_lane][local_ih][local_iw] =
                (centered_t)((int32_t)unpackInt8(packed_value, byte_lane) -
                             x_zero_point);
          }
        }
      }
    }
  }
}

// Port 0 reads the current half while port 1 fills the other half. The DATAFLOW
// boundary is intentional: it asks HLS to overlap the current-bank expansion
// with the next-bank AXI burst without duplicating the physical staging RAM.
static void loadNextAndExpandCurrent(const uint32_t *x,
    uint32_t packed_input[kPackedInputBufferWords], int current_bank,
    int next_bank,
    centered_t input_tile[kInputParallel][kInputTileHeight][kInputTileWidth],
    int n, int c_size, int h_size, int input_w_size, int next_c_base,
    int patch_height, int patch_width, int input_row_base,
    int input_column_base, int x_zero_point) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW
  loadPackedInputTile(x, packed_input, next_bank, n, c_size, h_size,
      input_w_size, next_c_base, patch_height, patch_width, input_row_base,
      input_column_base);
  expandPackedInputTile(packed_input, current_bank, input_tile, h_size,
      input_w_size, patch_height, patch_width, input_row_base,
      input_column_base, x_zero_point);
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
          uint32_t packed_input[kPackedInputBufferWords];
          centered_t
              input_tile[kInputParallel][kInputTileHeight][kInputTileWidth];
          int32_t accum[kOutputBlock][kOutputTileHeight][kOutputTileWidth];
#pragma HLS BIND_STORAGE variable = packed_input type = ram_s2p impl = uram
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
          const int input_channel_tiles = c_size / kInputParallel;

// Prime bank zero. Subsequent iterations read/expand one address range while
// filling the other through the SDP write port.
        PrimePackedInputTile:
          loadPackedInputTile(x, packed_input, 0, n, c_size, h_size,
              input_w_size, 0, patch_height, patch_width, input_row_base,
              input_column_base);

        InputChannelTileLoop:
          for (int channel_tile = 0; channel_tile < input_channel_tiles;
               ++channel_tile) {
            const int c_base = channel_tile * kInputParallel;
            const int current_bank = channel_tile & 1;
            const int next_bank = current_bank ^ 1;
            const bool has_next = channel_tile + 1 < input_channel_tiles;

            if (has_next) {
              loadNextAndExpandCurrent(x, packed_input, current_bank,
                  next_bank, input_tile, n, c_size, h_size, input_w_size,
                  c_base + kInputParallel, patch_height, patch_width,
                  input_row_base, input_column_base, x_zero_point);
            } else {
              expandPackedInputTile(packed_input, current_bank, input_tile,
                  h_size, input_w_size, patch_height, patch_width,
                  input_row_base, input_column_base, x_zero_point);
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
