#include "Conv1x1Int8Kernel.h"
#include "Int8Requantize.h"

#include <stdint.h>

#ifdef __SYNTHESIS__
#include <ap_int.h>
#endif

namespace {

constexpr int kInputParallel = 8;
constexpr int kOutputParallel = 8;
constexpr int kOutputBlock = 16;
constexpr int kPixelTile = 16;
constexpr int kInputLoadIterations = kInputParallel * kPixelTile;
constexpr int kComputeIterations =
    (kOutputBlock / kOutputParallel) * kPixelTile;

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

static int32_t reduce8(const int32_t value[kInputParallel]) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable = value complete
  const int32_t level1_0 = value[0] + value[1];
  const int32_t level1_1 = value[2] + value[3];
  const int32_t level1_2 = value[4] + value[5];
  const int32_t level1_3 = value[6] + value[7];
  const int32_t level2_0 = level1_0 + level1_1;
  const int32_t level2_1 = level1_2 + level1_3;
  return level2_0 + level2_1;
}

} // namespace

extern "C" void conv1x1_i8_kernel(const int8_t *x, const int8_t *weight,
    const int32_t *bias, int8_t *output, int n_size, int c_size, int h_size,
    int input_w_size, int m_size, int x_zero_point, int w_zero_point,
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
#pragma HLS INTERFACE s_axilite port = x_zero_point bundle = control
#pragma HLS INTERFACE s_axilite port = w_zero_point bundle = control
#pragma HLS INTERFACE s_axilite port = requant_multiplier_bits bundle = control
#pragma HLS INTERFACE s_axilite port = output_zero_point bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

  if (!x || !weight || !bias || !output || n_size <= 0 || c_size <= 0 ||
      c_size > MYACCEL_CONV1X1_INT8_MAX_INPUT_CHANNELS || h_size <= 0 ||
      input_w_size <= 0 || m_size <= 0 ||
      !myaccel_int8::isPositiveNormalMultiplier(requant_multiplier_bits))
    return;

  const int spatial_size = h_size * input_w_size;

OutputBlockLoop:
  for (int m_block = 0; m_block < m_size; m_block += kOutputBlock) {
    centered_t
        weight_cache[kOutputBlock][MYACCEL_CONV1X1_INT8_MAX_INPUT_CHANNELS];
    int32_t bias_cache[kOutputBlock];
#pragma HLS ARRAY_RESHAPE variable = weight_cache cyclic \
    factor = kOutputParallel dim = 1
#pragma HLS ARRAY_RESHAPE variable = weight_cache cyclic \
    factor = kInputParallel dim = 2
#pragma HLS BIND_STORAGE variable = weight_cache type = ram_2p impl = bram
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
#pragma HLS PIPELINE II = 1
        weight_cache[local_m][c] =
            m < m_size
                ? (centered_t)((int32_t)weight[(uint32_t)m * c_size + c] -
                               w_zero_point)
                : centered_t(0);
      }
    }

  BatchLoop:
    for (int n = 0; n < n_size; ++n) {
    PixelTileLoop:
      for (int pixel_base = 0; pixel_base < spatial_size;
           pixel_base += kPixelTile) {
        centered_t input_tile[2][kInputParallel][kPixelTile];
        int32_t accum[kOutputBlock][kPixelTile];
#pragma HLS ARRAY_PARTITION variable = input_tile complete dim = 1
#pragma HLS ARRAY_PARTITION variable = input_tile complete dim = 2
// Pack eight output lanes into one 256-bit word. Keeping the pixel dimension
// intact gives BRAM useful depth instead of creating one shallow RAM per pixel.
#pragma HLS ARRAY_RESHAPE variable = accum cyclic \
    factor = kOutputParallel dim = 1
#pragma HLS BIND_STORAGE variable = accum type = ram_2p impl = bram

      InitAccumOutputGroupLoop:
        for (int output_base = 0; output_base < kOutputBlock;
             output_base += kOutputParallel) {
        InitAccumPixelLoop:
          for (int pixel = 0; pixel < kPixelTile; ++pixel) {
#pragma HLS PIPELINE II = 1
          InitAccumLaneLoop:
            for (int output_lane = 0; output_lane < kOutputParallel;
                 ++output_lane) {
#pragma HLS UNROLL
              accum[output_base + output_lane][pixel] = 0;
            }
          }
        }

        const int input_channel_tiles =
            (c_size + kInputParallel - 1) / kInputParallel;

// Prime one bank. Each following phase consumes the current bank while the
// AXI port fills the other bank with the next input-channel tile.
      PreloadFirstInputTileLoop:
        for (int load_index = 0; load_index < kInputLoadIterations;
             ++load_index) {
#pragma HLS PIPELINE II = 1
          const int input_lane = load_index / kPixelTile;
          const int pixel = load_index % kPixelTile;
          const int c = input_lane;
          const int spatial = pixel_base + pixel;
          if (c < c_size && spatial < spatial_size) {
            const uint32_t x_index =
                ((uint32_t)n * c_size + c) * spatial_size + spatial;
            input_tile[0][input_lane][pixel] =
                (centered_t)((int32_t)x[x_index] - x_zero_point);
          } else {
            input_tile[0][input_lane][pixel] = 0;
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
              has_next ? kInputLoadIterations : kComputeIterations;

        OverlapInputLoadComputeLoop:
          for (int phase = 0; phase < phase_iterations; ++phase) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = input_tile inter false
#pragma HLS DEPENDENCE variable = accum inter false
            if (has_next) {
              const int input_lane = phase / kPixelTile;
              const int pixel = phase % kPixelTile;
              const int c = c_base + kInputParallel + input_lane;
              const int spatial = pixel_base + pixel;
              if (c < c_size && spatial < spatial_size) {
                const uint32_t x_index =
                    ((uint32_t)n * c_size + c) * spatial_size + spatial;
                input_tile[next_buffer][input_lane][pixel] =
                    (centered_t)((int32_t)x[x_index] - x_zero_point);
              } else {
                input_tile[next_buffer][input_lane][pixel] = 0;
              }
            }

            if (phase < kComputeIterations) {
              const int output_base =
                  (phase / kPixelTile) * kOutputParallel;
              const int pixel = phase % kPixelTile;
            ComputeOutputLaneLoop:
              for (int output_lane = 0;
                   output_lane < kOutputParallel; ++output_lane) {
#pragma HLS UNROLL
                const int local_m = output_base + output_lane;
                int32_t products[kInputParallel];
#pragma HLS ARRAY_PARTITION variable = products complete
              ComputeInputLaneLoop:
                for (int input_lane = 0; input_lane < kInputParallel;
                     ++input_lane) {
#pragma HLS UNROLL
                  const int c = c_base + input_lane;
                  const centered_t weight_value =
                      c < c_size ? weight_cache[local_m][c] : centered_t(0);
                  products[input_lane] = dspMultiply(
                      input_tile[current_buffer][input_lane][pixel],
                      weight_value);
                }
                accum[local_m][pixel] += reduce8(products);
              }
            }
          }
        }

      StoreOutputLoop:
        for (int local_m = 0; local_m < kOutputBlock; ++local_m) {
          const int m = m_block + local_m;
        StoreOutputPixelLoop:
          for (int pixel = 0; pixel < kPixelTile; ++pixel) {
#pragma HLS PIPELINE II = 1
            const int spatial = pixel_base + pixel;
            if (m < m_size && spatial < spatial_size) {
              const uint32_t output_index =
                  ((uint32_t)n * m_size + m) * spatial_size + spatial;
              output[output_index] = myaccel_int8::requantize(
                  accum[local_m][pixel], bias_cache[local_m],
                  requant_multiplier_bits, output_zero_point);
            }
          }
        }
      }
    }
  }
}
