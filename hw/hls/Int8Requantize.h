#ifndef MYACCEL_INT8_REQUANTIZE_H
#define MYACCEL_INT8_REQUANTIZE_H

#include <stdint.h>

#ifdef __SYNTHESIS__
#include <ap_int.h>
#endif

namespace myaccel_int8 {

// Return |value| without overflowing when value is INT64_MIN.
static inline uint64_t unsignedMagnitude(int64_t value) {
#pragma HLS INLINE
  return value < 0 ? (uint64_t)(-(value + 1)) + 1U : (uint64_t)value;
}

static inline uint64_t roundUnsignedShiftRightEven(
    uint64_t value, int shift) {
#pragma HLS INLINE
  if (value == 0 || shift >= 64)
    return 0;
  if (shift <= 0)
    return value;

  uint64_t quotient = value >> shift;
  const uint64_t remainderMask = ((uint64_t)1 << shift) - 1U;
  const uint64_t remainder = value & remainderMask;
  const uint64_t halfway = (uint64_t)1 << (shift - 1);
  if (remainder > halfway ||
      (remainder == halfway && (quotient & 1U) != 0))
    ++quotient;
  return quotient;
}

static inline int highestSetBit(uint64_t value) {
#pragma HLS INLINE
  // Six fixed comparisons synthesize into a compact priority tree and avoid
  // a variable-trip loop in the output pipeline.
  int highest = 0;
  if ((value >> 32) != 0) {
    value >>= 32;
    highest += 32;
  }
  if ((value >> 16) != 0) {
    value >>= 16;
    highest += 16;
  }
  if ((value >> 8) != 0) {
    value >>= 8;
    highest += 8;
  }
  if ((value >> 4) != 0) {
    value >>= 4;
    highest += 4;
  }
  if ((value >> 2) != 0) {
    value >>= 2;
    highest += 2;
  }
  if ((value >> 1) != 0)
    ++highest;
  return highest;
}

static inline int64_t saturatingMagnitudeShift(
    uint64_t magnitude, int shift, bool negative) {
#pragma HLS INLINE
  const uint64_t signedLimit =
      negative ? ((uint64_t)1 << 63) : (uint64_t)INT64_MAX;
  if (magnitude == 0)
    return 0;
  if (shift >= 63 || magnitude > (signedLimit >> shift))
    return negative ? INT64_MIN : INT64_MAX;

  const uint64_t shifted = magnitude << shift;
  if (!negative)
    return (int64_t)shifted;
  if (shifted == ((uint64_t)1 << 63))
    return INT64_MIN;
  return -(int64_t)shifted;
}

// Integer-only emulation of the current host fast path:
//
//   nearbyintf((float)(accumulator + bias) * multiplier)
//
// requantMultiplierBits contains the raw IEEE-754 binary32 bits of a positive,
// finite, normal multiplier. The two binary32 roundings (int64 conversion and
// multiplication) and nearbyintf's final FE_TONEAREST rounding are all
// round-to-nearest, ties-to-even.
static inline int64_t emulateFloatMultiplyRoundEven(
    int64_t accumulator, uint32_t requantMultiplierBits) {
#pragma HLS INLINE
  if (accumulator == 0)
    return 0;

  const bool negative = accumulator < 0;
  const uint64_t magnitude = unsignedMagnitude(accumulator);

  // Exact int64 -> binary32 conversion, represented as a 24-bit significand
  // and an unbiased exponent.
  int accumulatorExponent = highestSetBit(magnitude);
  uint32_t accumulatorSignificand;
  if (accumulatorExponent <= 23) {
    accumulatorSignificand =
        (uint32_t)(magnitude << (23 - accumulatorExponent));
  } else {
    accumulatorSignificand = (uint32_t)roundUnsignedShiftRightEven(
        magnitude, accumulatorExponent - 23);
    if (accumulatorSignificand == ((uint32_t)1 << 24)) {
      accumulatorSignificand >>= 1;
      ++accumulatorExponent;
    }
  }

  const int multiplierExponent =
      (int)((requantMultiplierBits >> 23) & 0xffU) - 127;
  const uint32_t multiplierSignificand =
      ((uint32_t)1 << 23) | (requantMultiplierBits & 0x7fffffU);

  // The significands are exactly 24 bits. Preserve that width during HLS so
  // this remains a 24x24 multiply instead of widening into a 64x64 operator.
  uint64_t significandProduct;
#ifdef __SYNTHESIS__
  const ap_uint<24> accumulatorSignificandHls = accumulatorSignificand;
  const ap_uint<24> multiplierSignificandHls = multiplierSignificand;
  ap_uint<48> significandProductHls;
#pragma HLS BIND_OP variable = significandProductHls op = mul impl = dsp
  significandProductHls =
      accumulatorSignificandHls * multiplierSignificandHls;
  significandProduct = (uint64_t)significandProductHls;
#else
  significandProduct =
      (uint64_t)accumulatorSignificand * multiplierSignificand;
#endif

  // Normalize and perform the binary32 multiplication's single rounding.
  int productExponent;
  uint32_t productSignificand;
  if (significandProduct >= ((uint64_t)1 << 47)) {
    productExponent = accumulatorExponent + multiplierExponent + 1;
    productSignificand = (uint32_t)roundUnsignedShiftRightEven(
        significandProduct, 24);
  } else {
    productExponent = accumulatorExponent + multiplierExponent;
    productSignificand = (uint32_t)roundUnsignedShiftRightEven(
        significandProduct, 23);
  }
  if (productSignificand == ((uint32_t)1 << 24)) {
    productSignificand >>= 1;
    ++productExponent;
  }

  // Round the represented binary32 product to an integer as nearbyintf does.
  if (productExponent < 23) {
    const uint64_t roundedMagnitude = roundUnsignedShiftRightEven(
        productSignificand, 23 - productExponent);
    return negative ? -(int64_t)roundedMagnitude
                    : (int64_t)roundedMagnitude;
  }
  return saturatingMagnitudeShift(
      productSignificand, productExponent - 23, negative);
}

static inline int8_t requantize(int32_t accumulator, int32_t bias,
    uint32_t requantMultiplierBits, int outputZeroPoint) {
#pragma HLS INLINE
  const int64_t biasedAccumulator =
      (int64_t)accumulator + (int64_t)bias;
  const int64_t scaled = emulateFloatMultiplyRoundEven(
      biasedAccumulator, requantMultiplierBits);

  // Compare before adding the zero point so a saturated int64 result cannot
  // overflow during the addition.
  const int64_t lower = -128LL - (int64_t)outputZeroPoint;
  const int64_t upper = 127LL - (int64_t)outputZeroPoint;
  if (scaled < lower)
    return (int8_t)-128;
  if (scaled > upper)
    return (int8_t)127;
  return (int8_t)(scaled + (int64_t)outputZeroPoint);
}

static inline bool isPositiveNormalMultiplier(uint32_t bits) {
#pragma HLS INLINE
  const uint32_t exponent = (bits >> 23) & 0xffU;
  return (bits >> 31) == 0 && exponent != 0 && exponent != 0xffU;
}

} // namespace myaccel_int8

#endif
