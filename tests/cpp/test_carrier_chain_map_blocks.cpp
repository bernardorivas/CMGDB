// test_carrier_chain_map_blocks.cpp
//
// The result of carrier_chain_map::ComputeCarrierChainMap does not depend on
// the block limits of its acyclicity check. Random inputs, many of them with
// an empty or non-acyclic carrier at varied cells, give equal results for
// blocks of one cell, of a few cells, of a few carrier vertices, and for a
// single block.
//
// Build, from the repository root:
//   c++ -std=c++17 -I src/CMGDB/_cmgdb/include/database
//       tests/cpp/test_carrier_chain_map_blocks.cpp -o blocks_test

#include <cstdio>
#include <set>
#include <string>
#include <vector>

#include "CarrierChainMap.h"

namespace ccm = carrier_chain_map;

static int failures = 0;

#define CHECK(condition)                                                    \
  do {                                                                      \
    if ( ! ( condition ) ) {                                                \
      std::printf ( "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition );   \
      ++ failures;                                                          \
    }                                                                       \
  } while ( 0 )

/// xorshift64; the standard distributions are implementation-defined.
struct Random {
  uint64_t state;
  uint64_t next ( void ) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
  }
  int64_t below ( int64_t bound ) {
    return static_cast<int64_t> ( next () % static_cast<uint64_t> ( bound ) );
  }
};

/// Increasing labels, not consecutive and partly negative.
std::vector<int32_t> Labels ( Random & random, int64_t count ) {
  std::vector<int32_t> labels;
  int32_t label = static_cast<int32_t> ( random . below ( 20 ) ) - 10;
  for ( int64_t i = 0; i < count; ++ i ) {
    labels . push_back ( label );
    label += 1 + static_cast<int32_t> ( random . below ( 3 ) );
  }
  return labels;
}

/// The closure of random simplices on the labels, in complex order.
std::vector<std::vector<int32_t> > Complex ( Random & random,
                                              const std::vector<int32_t> & labels,
                                              int64_t top,
                                              int64_t simplices ) {
  std::set<std::vector<int32_t> > cells;
  for ( int32_t label : labels ) cells . insert ( std::vector<int32_t> ( 1, label ) );
  for ( int64_t s = 0; s < simplices; ++ s ) {
    const int64_t largest = std::min<int64_t> ( top + 1, static_cast<int64_t> ( labels . size () ) );
    const int64_t size = 2 + random . below ( std::max<int64_t> ( 1, largest - 1 ) );
    std::set<int32_t> chosen;
    while ( static_cast<int64_t> ( chosen . size () ) < std::min<int64_t> ( size, largest ) ) {
      chosen . insert ( labels [ random . below ( static_cast<int64_t> ( labels . size () ) ) ] );
    }
    const std::vector<int32_t> simplex ( chosen . begin (), chosen . end () );
    const uint64_t subsets = uint64_t ( 1 ) << simplex . size ();
    for ( uint64_t mask = 1; mask < subsets; ++ mask ) {
      std::vector<int32_t> face;
      for ( size_t k = 0; k < simplex . size (); ++ k ) {
        if ( mask & ( uint64_t ( 1 ) << k ) ) face . push_back ( simplex [ k ] );
      }
      cells . insert ( face );
    }
  }
  std::vector<std::vector<int32_t> > degrees;
  for ( size_t d = 0; ; ++ d ) {
    std::vector<int32_t> flat;
    for ( const std::vector<int32_t> & cell : cells ) {
      if ( cell . size () == d + 1 ) flat . insert ( flat . end (), cell . begin (), cell . end () );
    }
    if ( flat . empty () && d > 0 ) break;
    degrees . push_back ( flat );
  }
  return degrees;
}

std::vector<int32_t> Subset ( Random & random, const std::vector<int32_t> & labels, int64_t largest ) {
  const int64_t size = random . below ( largest + 1 );
  std::vector<int32_t> subset;
  for ( int64_t i = 0; i < size; ++ i ) {
    subset . push_back ( labels [ random . below ( static_cast<int64_t> ( labels . size () ) ) ] );
  }
  return subset;
}

ccm::CarrierChainMapInput Input ( Random & random, int64_t number ) {
  ccm::CarrierChainMapInput input;
  const std::vector<int32_t> labels = Labels ( random, 2 + random . below ( 14 ) );
  input . source_simplices =
    Complex ( random, labels, 1 + random . below ( 4 ), 1 + random . below ( 16 ) );
  std::vector<int32_t> target_labels = labels;
  input . target_is_source = number % 4 != 0;
  if ( ! input . target_is_source ) {
    target_labels = Labels ( random, 2 + random . below ( 10 ) );
    input . target_simplices =
      Complex ( random, target_labels, 1 + random . below ( 4 ), 1 + random . below ( 12 ) );
  }
  // Images: random subsets, or the identity with a few enlarged images,
  // at times with an empty one.
  std::vector<std::vector<int32_t> > images ( labels . size () );
  for ( size_t v = 0; v < labels . size (); ++ v ) {
    if ( number % 3 == 0 || ! input . target_is_source ) {
      images [ v ] = Subset ( random, target_labels, 3 );
      if ( images [ v ] . empty () ) images [ v ] . push_back ( target_labels [ 0 ] );
    } else {
      images [ v ] . push_back ( labels [ v ] );
    }
  }
  if ( number % 3 != 0 ) {
    for ( int64_t k = random . below ( 3 ); k >= 0; -- k ) {
      std::vector<int32_t> & image = images [ random . below ( static_cast<int64_t> ( labels . size () ) ) ];
      const std::vector<int32_t> more = Subset ( random, target_labels, 4 );
      image . insert ( image . end (), more . begin (), more . end () );
    }
  }
  if ( random . below ( 6 ) == 0 ) {
    images [ random . below ( static_cast<int64_t> ( labels . size () ) ) ] . clear ();
  }
  input . vertex_image_indptr . push_back ( 0 );
  for ( const std::vector<int32_t> & image : images ) {
    input . vertex_image_indices . insert (
      input . vertex_image_indices . end (), image . begin (), image . end () );
    input . vertex_image_indptr . push_back (
      static_cast<int64_t> ( input . vertex_image_indices . size () ) );
  }
  for ( size_t v = 0; v < labels . size (); ++ v ) {
    input . source_exit . push_back ( random . below ( 4 ) == 0 ? 1 : 0 );
  }
  if ( ! input . target_is_source ) {
    for ( size_t v = 0; v < target_labels . size (); ++ v ) {
      input . target_exit . push_back ( random . below ( 3 ) == 0 ? 1 : 0 );
    }
  }
  return input;
}

bool Equal ( const ccm::CarrierChainMapResult & a, const ccm::CarrierChainMapResult & b ) {
  return a . status == b . status && a . failure_degree == b . failure_degree &&
         a . failure_row == b . failure_row && a . carrier_count == b . carrier_count &&
         a . carrier_ids == b . carrier_ids && a . has_chain_map == b . has_chain_map &&
         a . chain_map == b . chain_map && a . has_payload == b . has_payload &&
         a . cell_counts == b . cell_counts && a . boundary_entries == b . boundary_entries &&
         a . chain_map_entries == b . chain_map_entries;
}

int main ( void ) {
  const int64_t huge = int64_t ( 1 ) << 62;
  const int64_t limits [ ] [ 2 ] = {
    { 1, huge }, { 2, huge }, { 5, huge }, { huge, 1 }, { huge, 4 }, { 3, 6 }, { 0, 0 } };
  Random random = { 0x9E3779B97F4A7C15ULL };
  int64_t ok = 0;
  int64_t empty = 0;
  int64_t late_failures = 0;
  for ( int64_t number = 0; number < 3000; ++ number ) {
    ccm::CarrierChainMapInput input = Input ( random, number );
    input . block_cells = huge;
    input . block_vertices = huge;
    const ccm::CarrierChainMapResult single = ccm::ComputeCarrierChainMap ( input );
    ok += single . status == "ok";
    if ( single . status == "empty_carrier" || single . status == "not_acyclic" ) {
      // The cells before the failing cell are numbered, and no cell after it.
      int64_t position = single . failure_row;
      for ( int64_t d = 0; d < single . failure_degree; ++ d ) {
        position += static_cast<int64_t> ( input . source_simplices [ d ] . size () ) / ( d + 1 );
      }
      int64_t numbered = 0;
      for ( int64_t i = 0; i < static_cast<int64_t> ( single . carrier_ids . size () ); ++ i ) {
        const int64_t id = single . carrier_ids [ i ];
        CHECK ( i < position ? id >= 0 && id <= numbered : id == -1 );
        if ( i < position && id == numbered ) ++ numbered;
      }
      CHECK ( single . carrier_count == numbered );
    }
    empty += single . status == "empty_carrier";
    late_failures += single . status == "not_acyclic" && single . failure_degree > 0;
    for ( const int64_t * limit : limits ) {
      input . block_cells = limit [ 0 ];
      input . block_vertices = limit [ 1 ];
      const ccm::CarrierChainMapResult blocked = ccm::ComputeCarrierChainMap ( input );
      if ( ! Equal ( single, blocked ) ) {
        std::printf ( "FAIL input %lld with blocks of %lld cells and %lld vertices: "
                      "%s (%lld, %lld) against %s (%lld, %lld)\n",
                      static_cast<long long> ( number ),
                      static_cast<long long> ( limit [ 0 ] ),
                      static_cast<long long> ( limit [ 1 ] ),
                      blocked . status . c_str (),
                      static_cast<long long> ( blocked . failure_degree ),
                      static_cast<long long> ( blocked . failure_row ),
                      single . status . c_str (),
                      static_cast<long long> ( single . failure_degree ),
                      static_cast<long long> ( single . failure_row ) );
        ++ failures;
      }
    }
  }
  // The inputs exercise success, empty carriers, and failures past degree 0.
  CHECK ( ok >= 300 );
  CHECK ( empty >= 100 );
  CHECK ( late_failures >= 100 );
  std::printf ( "%lld ok, %lld empty_carrier, %lld not_acyclic past degree 0; %d failures\n",
                static_cast<long long> ( ok ), static_cast<long long> ( empty ),
                static_cast<long long> ( late_failures ), failures );
  return failures == 0 ? 0 : 1;
}
