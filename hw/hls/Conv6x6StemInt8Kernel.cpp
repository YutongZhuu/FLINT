#include "Conv6x6StemInt8Kernel.h"
#include "Int8Requantize.h"

#include <stdint.h>

#ifdef __SYNTHESIS__
#include <ap_int.h>
#endif

namespace {

constexpr int kKernelSize = 6;
constexpr int kKernelElements = kKernelSize * kKernelSize;
constexpr int kInt8ValuesPerWord = 4;
constexpr int kWeightWordsPerInputChannel =
    kKernelElements / kInt8ValuesPerWord;
constexpr int kOutputParallel = 4;
constexpr int kOutputBlock = 8;
constexpr int kWidthTile = 8;
constexpr int kMaximumPatchWidth =
    (kWidthTile - 1) * MYACCEL_CONV6X6_STEM_INT8_MAX_STRIDE + kKernelSize;
constexpr int kPatchWordsPerRow =
    (kInt8ValuesPerWord - 1 + kMaximumPatchWidth +
        kInt8ValuesPerWord - 1) /
    kInt8ValuesPerWord;
constexpr int kPatchLoadIterations = kKernelSize * kPatchWordsPerRow;
constexpr int kComputeIterations =
    (kOutputBlock / kOutputParallel) * kWidthTile;
constexpr int kOverlapIterations =
    kPatchLoadIterations > kComputeIterations ? kPatchLoadIterations
                                              : kComputeIterations;

static_assert(kKernelElements % kInt8ValuesPerWord == 0,
    "each input-channel weight plane must contain complete packed words");
static_assert(kPatchWordsPerRow == 6,
    "the packed patch must cover an unaligned stride-two width tile");

#ifdef __SYNTHESIS__
using centered_t = ap_int<9>;
#else
using centered_t = int16_t;
#endif

static int32_t unpackLittleEndianInt8(uint32_t packed, int byteIndex) {
#pragma HLS INLINE
  const uint32_t value = (packed >> (byteIndex * 8)) & 0xffU;
  return value < 0x80U ? (int32_t)value : (int32_t)value - 256;
}

static uint32_t replicateInt8Byte(int value) {
#pragma HLS INLINE
  const uint32_t byteValue = (uint32_t)value & 0xffU;
  return byteValue * 0x01010101U;
}

static uint32_t appendLittleEndianInt8(
    uint32_t packed, int byteIndex, int8_t value) {
#pragma HLS INLINE
  const uint32_t byteValue = (uint32_t)(uint8_t)value;
  const uint32_t shifted = byteValue << (byteIndex * 8);
  return byteIndex == 0 ? shifted : packed | shifted;
}

static int floorDivideByFour(int value) {
#pragma HLS INLINE
  const int quotient = value / kInt8ValuesPerWord;
  const int remainder = value % kInt8ValuesPerWord;
  return remainder < 0 ? quotient - 1 : quotient;
}

static int32_t dspMultiply(centered_t lhs, centered_t rhs) {
#pragma HLS INLINE
  int32_t product;
#pragma HLS BIND_OP variable = product op = mul impl = dsp
  product = lhs * rhs;
  return product;
}

static int32_t reduce36(const int32_t values[kKernelElements]) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable = values complete
  int32_t level1[18];
  int32_t level2[9];
  int32_t level3[5];
  int32_t level4[3];
#pragma HLS ARRAY_PARTITION variable = level1 complete
#pragma HLS ARRAY_PARTITION variable = level2 complete
#pragma HLS ARRAY_PARTITION variable = level3 complete
#pragma HLS ARRAY_PARTITION variable = level4 complete

ReduceLevel1Loop:
  for (int i = 0; i < 18; ++i) {
#pragma HLS UNROLL
    level1[i] = values[2 * i] + values[2 * i + 1];
  }
ReduceLevel2Loop:
  for (int i = 0; i < 9; ++i) {
#pragma HLS UNROLL
    level2[i] = level1[2 * i] + level1[2 * i + 1];
  }
ReduceLevel3Loop:
  for (int i = 0; i < 4; ++i) {
#pragma HLS UNROLL
    level3[i] = level2[2 * i] + level2[2 * i + 1];
  }
  level3[4] = level2[8];
  level4[0] = level3[0] + level3[1];
  level4[1] = level3[2] + level3[3];
  level4[2] = level3[4];
  return (level4[0] + level4[1]) + level4[2];
}

} // namespace

extern "C" void conv6x6_stem_i8_kernel(const uint32_t *x,
    const uint32_t *weight, const int32_t *bias, uint32_t *output, int n_size,
    int c_size, int h_size, int input_w_size, int m_size, int oh_size,
    int ow_size, int pad_left, int pad_top, int stride_h, int stride_w,
    int x_zero_point, int w_zero_point, uint32_t requant_multiplier_bits,
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
      c_size > MYACCEL_CONV6X6_STEM_INT8_MAX_INPUT_CHANNELS || h_size <= 0 ||
      input_w_size <= 0 || m_size <= 0 || oh_size <= 0 || ow_size <= 0 ||
      pad_left < 0 || pad_top < 0 || stride_h <= 0 ||
      stride_h > MYACCEL_CONV6X6_STEM_INT8_MAX_STRIDE || stride_w <= 0 ||
      stride_w > MYACCEL_CONV6X6_STEM_INT8_MAX_STRIDE ||
      input_w_size % kInt8ValuesPerWord != 0 ||
      ow_size % kInt8ValuesPerWord != 0 ||
      !myaccel_int8::isPositiveNormalMultiplier(requant_multiplier_bits))
    return;

  const int input_words_per_row = input_w_size / kInt8ValuesPerWord;
  const int input_words_per_channel = h_size * input_words_per_row;
  const int output_words_per_row = ow_size / kInt8ValuesPerWord;
  const int weight_words_per_output =
      c_size * kWeightWordsPerInputChannel;
  const uint32_t packed_x_zero_point = replicateInt8Byte(x_zero_point);
  const uint32_t packed_w_zero_point = replicateInt8Byte(w_zero_point);

OutputBlockLoop:
  for (int m_block = 0; m_block < m_size; m_block += kOutputBlock) {
    uint32_t weight_cache[kOutputBlock]
        [MYACCEL_CONV6X6_STEM_INT8_MAX_INPUT_CHANNELS]
        [kWeightWordsPerInputChannel];
    int32_t bias_cache[kOutputBlock];
#pragma HLS ARRAY_PARTITION variable = weight_cache complete dim = 1
#pragma HLS ARRAY_PARTITION variable = weight_cache complete dim = 2
#pragma HLS ARRAY_PARTITION variable = weight_cache complete dim = 3
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
      LoadWeightWordLoop:
        for (int word = 0; word < kWeightWordsPerInputChannel; ++word) {
#pragma HLS PIPELINE II = 1
          weight_cache[local_m][c][word] =
              m < m_size
                  ? weight[(uint32_t)m * weight_words_per_output +
                           c * kWeightWordsPerInputChannel + word]
                  : packed_w_zero_point;
        }
      }
    }

  BatchLoop:
    for (int n = 0; n < n_size; ++n) {
    OutputRowLoop:
      for (int oh = 0; oh < oh_size; ++oh) {
      OutputWidthTileLoop:
        for (int ow_base = 0; ow_base < ow_size; ow_base += kWidthTile) {
          uint32_t input_patch[2][kKernelSize][kPatchWordsPerRow];
          int32_t accum[kOutputBlock][kWidthTile];
#pragma HLS ARRAY_PARTITION variable = input_patch complete dim = 1
#pragma HLS ARRAY_PARTITION variable = input_patch complete dim = 2
#pragma HLS ARRAY_PARTITION variable = input_patch complete dim = 3
#pragma HLS ARRAY_PARTITION variable = accum complete dim = 1
#pragma HLS ARRAY_PARTITION variable = accum complete dim = 2

        InitAccumOutputSubblockLoop:
          for (int output_base = 0; output_base < kOutputBlock;
               output_base += kOutputParallel) {
          InitAccumWidthLoop:
            for (int ow_lane = 0; ow_lane < kWidthTile; ++ow_lane) {
#pragma HLS PIPELINE II = 1
            InitAccumLaneLoop:
              for (int output_lane = 0; output_lane < kOutputParallel;
                   ++output_lane) {
#pragma HLS UNROLL
                accum[output_base + output_lane][ow_lane] = 0;
              }
            }
          }

          const int input_row_start = oh * stride_h - pad_top;
          const int input_col_start = ow_base * stride_w - pad_left;
          const int first_input_word = floorDivideByFour(input_col_start);
          const int patch_byte_offset =
              input_col_start - first_input_word * kInt8ValuesPerWord;

// Prime one patch bank. The overlap loop loads the next input-channel patch
// while the current patch feeds both four-output-channel subblocks.
        PreloadFirstPatchLoop:
          for (int load_index = 0; load_index < kPatchLoadIterations;
               ++load_index) {
#pragma HLS PIPELINE II = 1
            const int kh = load_index / kPatchWordsPerRow;
            const int patch_word = load_index % kPatchWordsPerRow;
            const int ih = input_row_start + kh;
            const int input_word = first_input_word + patch_word;
            uint32_t packed_input = packed_x_zero_point;
            if (ih >= 0 && ih < h_size && input_word >= 0 &&
                input_word < input_words_per_row) {
              const uint32_t x_index =
                  ((uint32_t)n * c_size * input_words_per_channel) +
                  (uint32_t)ih * input_words_per_row + input_word;
              packed_input = x[x_index];
            }
            input_patch[0][kh][patch_word] = packed_input;
          }

        InputChannelLoop:
          for (int c = 0; c < c_size; ++c) {
            const int current_buffer = c & 1;
            const int next_buffer = current_buffer ^ 1;
            const bool has_next = c + 1 < c_size;
            const int phase_iterations =
                has_next ? kOverlapIterations : kComputeIterations;

          OverlapPatchLoadComputeLoop:
            for (int phase = 0; phase < phase_iterations; ++phase) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = input_patch inter false
#pragma HLS DEPENDENCE variable = accum inter false
              if (has_next && phase < kPatchLoadIterations) {
                const int kh = phase / kPatchWordsPerRow;
                const int patch_word = phase % kPatchWordsPerRow;
                const int ih = input_row_start + kh;
                const int input_word = first_input_word + patch_word;
                uint32_t packed_input = packed_x_zero_point;
                if (ih >= 0 && ih < h_size && input_word >= 0 &&
                    input_word < input_words_per_row) {
                  const uint32_t x_index =
                      ((uint32_t)n * c_size + c + 1) *
                          input_words_per_channel +
                      (uint32_t)ih * input_words_per_row + input_word;
                  packed_input = x[x_index];
                }
                input_patch[next_buffer][kh][patch_word] = packed_input;
              }

              if (phase < kComputeIterations) {
                const int output_base =
                    (phase / kWidthTile) * kOutputParallel;
                const int ow_lane = phase % kWidthTile;
                centered_t patch_values[kKernelElements];
#pragma HLS ARRAY_PARTITION variable = patch_values complete

              ReadPatchRowLoop:
                for (int kh = 0; kh < kKernelSize; ++kh) {
#pragma HLS UNROLL
                ReadPatchColumnLoop:
                  for (int kw = 0; kw < kKernelSize; ++kw) {
#pragma HLS UNROLL
                    const int tap = kh * kKernelSize + kw;
                    const int packed_position =
                        patch_byte_offset + ow_lane * stride_w + kw;
                    const uint32_t packed_input =
                        input_patch[current_buffer][kh]
                                   [packed_position / kInt8ValuesPerWord];
                    patch_values[tap] = (centered_t)(
                        unpackLittleEndianInt8(
                            packed_input,
                            packed_position % kInt8ValuesPerWord) -
                        x_zero_point);
                  }
                }

              ComputeOutputLaneLoop:
                for (int output_lane = 0;
                     output_lane < kOutputParallel; ++output_lane) {
#pragma HLS UNROLL
                  const int local_m = output_base + output_lane;
                  int32_t products[kKernelElements];
#pragma HLS ARRAY_PARTITION variable = products complete
                ComputeTapLoop:
                  for (int tap = 0; tap < kKernelElements; ++tap) {
#pragma HLS UNROLL
                    const uint32_t packed_weight =
                        weight_cache[local_m][c]
                                    [tap / kInt8ValuesPerWord];
                    const centered_t weight_value = (centered_t)(
                        unpackLittleEndianInt8(
                            packed_weight, tap % kInt8ValuesPerWord) -
                        w_zero_point);
                    products[tap] =
                        dspMultiply(patch_values[tap], weight_value);
                  }
                  accum[local_m][ow_lane] += reduce36(products);
                }
              }
            }
          }

        StoreOutputLoop:
          for (int local_m = 0; local_m < kOutputBlock; ++local_m) {
            const int m = m_block + local_m;
            uint32_t packed_output = 0;
          StoreOutputWidthLoop:
            for (int ow_lane = 0; ow_lane < kWidthTile; ++ow_lane) {
#pragma HLS PIPELINE II = 1
              const int ow = ow_base + ow_lane;
              const bool valid = m < m_size && ow < ow_size;
              int8_t quantized = 0;
              if (valid) {
                quantized = myaccel_int8::requantize(
                    accum[local_m][ow_lane], bias_cache[local_m],
                    requant_multiplier_bits, output_zero_point);
              }
              const int byte_index = ow_lane % kInt8ValuesPerWord;
              const uint32_t next_packed_output = appendLittleEndianInt8(
                  packed_output, byte_index, quantized);
              if (byte_index == kInt8ValuesPerWord - 1 && valid) {
                const uint32_t output_index =
                    (((uint32_t)n * m_size + m) * oh_size + oh) *
                        output_words_per_row +
                    ow / kInt8ValuesPerWord;
                output[output_index] = next_packed_output;
              }
              packed_output = next_packed_output;
            }
          }
        }
      }
    }
  }
}
