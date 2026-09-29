// GF5.h
//
// Arithmetic of GF(5) on the residues 0..4 stored as uint8_t, shared by
// CarrierChainMap.h and RelativeShiftClass.h.

#ifndef CMDB_GF5_H
#define CMDB_GF5_H

#include <stdint.h>

namespace gf5 {

inline uint8_t Add5 ( uint8_t a, uint8_t b ) {
  const uint8_t sum = static_cast<uint8_t> ( a + b );
  return sum >= 5 ? static_cast<uint8_t> ( sum - 5 ) : sum;
}

inline uint8_t Negate5 ( uint8_t a ) {
  return a == 0 ? 0 : static_cast<uint8_t> ( 5 - a );
}

inline uint8_t Multiply5 ( uint8_t a, uint8_t b ) {
  static const uint8_t table [ 5 ] [ 5 ] = {
    { 0, 0, 0, 0, 0 },
    { 0, 1, 2, 3, 4 },
    { 0, 2, 4, 1, 3 },
    { 0, 3, 1, 4, 2 },
    { 0, 4, 3, 2, 1 } };
  return table [ a ] [ b ];
}

inline uint8_t Inverse5 ( uint8_t a ) {
  static const uint8_t table [ 5 ] = { 0, 1, 3, 2, 4 };
  return table [ a ];
}

} // namespace gf5

#endif
