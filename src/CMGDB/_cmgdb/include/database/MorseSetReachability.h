// MorseSetReachability.h
//
// Fixed-subdivision Morse-set reachability verification: CMGDB adapter.
//
// Verifies the reachability relation of an adaptive MorseGraph on the
// conceptual uniform TreeGrid at a caller-chosen subdivision depth,
// without materializing a complete TreeGrid or MapGraph. The supplied
// MorseGraph is never mutated.

#ifndef CMDB_MORSE_SET_REACHABILITY_H
#define CMDB_MORSE_SET_REACHABILITY_H

#include <stdint.h>
#include <algorithm>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "Geo.h"
#include "RectGeo.h"
#include "PrismGeo.h"
#include "UnionGeo.h"
#include "IntersectionGeo.h"
#include "Map.h"
#include "Model.h"
#include "TreeGrid.h"
#include "PointerGrid.h"
#include "CompressedTreeGrid.h"
#include "MorseGraph.h"
#include "MorseSetReachabilityCore.h"

#include <pybind11/pybind11.h>

namespace msr_detail {

/// Implicit uniform TreeGrid at a fixed subdivision depth. A fixed
/// GridElement is the root-to-leaf branch sequence, most-significant
/// branch first (encoding "tree_path_msb_uint64_v1"). Geometry and cover
/// reproduce TreeGrid's arithmetic exactly: split coordinate is
/// tree-depth mod dimension, bottom-up geometry arithmetic, normalized
/// 2^60 integer coordinates with 2^10 outward tolerance, weak
/// endpoint-inclusive intersection, clipping at nonperiodic boundaries,
/// and TreeGrid's periodic-image generation (including its endpoint
/// asymmetry). No complete TreeGrid or MapGraph is materialized.
class ImplicitFixedTreeGrid {
public:
  ImplicitFixedTreeGrid ( RectGeo const& bounds,
                          std::vector<bool> const& periodic,
                          uint32_t subdiv )
    : bounds_ ( bounds ),
      periodic_ ( periodic ),
      subdiv_ ( subdiv ) {
    dimension_ = static_cast<int> ( bounds_ . lower_bounds . size () );
  }

  int dimension ( void ) const { return dimension_; }
  uint32_t subdivision ( void ) const { return subdiv_; }

  /// Geometry of a fixed GridElement; identical arithmetic to
  /// TreeGrid::geometry on the complete tree.
  RectGeo geometry ( FixedGridElement element ) const {
    RectGeo rect ( dimension_, Real ( 0 ) );
    if ( dimension_ == 0 ) return rect;
    int division_dimension = static_cast<int> ( subdiv_ ) % dimension_;
    for ( uint32_t i = 0; i < subdiv_; ++ i ) {
      // Bit i is the branch at tree depth subdiv_ - 1 - i (climb from leaf).
      const bool right_child = ( element >> i ) & 1;
      -- division_dimension;
      if ( division_dimension < 0 ) division_dimension = dimension_ - 1;
      if ( not right_child ) {
        rect . upper_bounds [ division_dimension ] += Real ( 1 );
      } else {
        rect . lower_bounds [ division_dimension ] += Real ( 1 );
      }
      rect . lower_bounds [ division_dimension ] /= Real ( 2 );
      rect . upper_bounds [ division_dimension ] /= Real ( 2 );
    }
    for ( int d = 0; d < dimension_; ++ d ) {
      rect . lower_bounds [ d ] =
        rect . lower_bounds [ d ] * bounds_ . upper_bounds [ d ] +
        ( Real ( 1 ) - rect . lower_bounds [ d ] ) * bounds_ . lower_bounds [ d ];
      rect . upper_bounds [ d ] =
        rect . upper_bounds [ d ] * bounds_ . lower_bounds [ d ] +
        ( Real ( 1 ) - rect . upper_bounds [ d ] ) * bounds_ . upper_bounds [ d ];
    }
    return rect;
  }

  /// Sorted, unique cover of a Geo, with the same dispatch as
  /// Grid::cover / TreeGrid::cover.
  std::vector<FixedGridElement> cover ( std::shared_ptr<Geo> geo ) const {
    std::vector<FixedGridElement> results;
    coverDispatch ( geo, results );
    std::sort ( results . begin (), results . end () );
    results . erase ( std::unique ( results . begin (), results . end () ),
                      results . end () );
    return results;
  }

private:
  void coverDispatch ( std::shared_ptr<Geo> geo,
                       std::vector<FixedGridElement> & results ) const {
    std::shared_ptr<UnionGeo> union_geo =
      std::dynamic_pointer_cast<UnionGeo> ( geo );
    if ( union_geo ) {
      for ( size_t i = 0; i < union_geo -> elements . size (); ++ i ) {
        coverDispatch ( union_geo -> elements [ i ], results );
      }
      return;
    }
    std::shared_ptr<IntersectionGeo> intersect_geo =
      std::dynamic_pointer_cast<IntersectionGeo> ( geo );
    if ( intersect_geo ) {
      std::vector<FixedGridElement> intersection;
      bool first = true;
      for ( size_t i = 0; i < intersect_geo -> elements . size (); ++ i ) {
        std::vector<FixedGridElement> partial;
        coverDispatch ( intersect_geo -> elements [ i ], partial );
        std::sort ( partial . begin (), partial . end () );
        partial . erase ( std::unique ( partial . begin (), partial . end () ),
                          partial . end () );
        if ( first ) {
          intersection . swap ( partial );
          first = false;
        } else {
          std::vector<FixedGridElement> merged;
          std::set_intersection ( intersection . begin (), intersection . end (),
                                  partial . begin (), partial . end (),
                                  std::back_inserter ( merged ) );
          intersection . swap ( merged );
        }
      }
      results . insert ( results . end (), intersection . begin (),
                         intersection . end () );
      return;
    }
    Geo const* geo_ptr = geo . get ();
    if ( RectGeo const* rect_geo = dynamic_cast<RectGeo const*> ( geo_ptr ) ) {
      coverRect ( * rect_geo, results );
      return;
    }
    if ( PrismGeo const* prism_geo = dynamic_cast<PrismGeo const*> ( geo_ptr ) ) {
      coverPrism ( * prism_geo, results );
      return;
    }
    throw MorseSetReachabilityCoverError (
      "Bad Geo type in fixed-subdivision cover" );
  }

  void coverRect ( RectGeo const& geometric_region,
                   std::vector<FixedGridElement> & results ) const {
    if ( dimension_ == 0 ) {
      results . push_back ( 0 );
      return;
    }

    std::vector<double> width ( dimension_ );
    for ( int d = 0; d < dimension_; ++ d ) {
      width [ d ] = bounds_ . upper_bounds [ d ] - bounds_ . lower_bounds [ d ];
    }

    bool periodic_flag = false;
    for ( int d = 0; d < dimension_; ++ d ) {
      if ( periodic_ [ d ] == true ) periodic_flag = true;
    }

    std::vector<RectGeo> work_stack;
    if ( periodic_flag ) {
      RectGeo R = geometric_region;
      for ( int d = 0; d < dimension_; ++ d ) {
        if ( periodic_ [ d ] == false ) continue;
        if ( R . upper_bounds [ d ] > bounds_ . upper_bounds [ d ] ) {
          R . lower_bounds [ d ] -= width [ d ];
          R . upper_bounds [ d ] -= width [ d ];
        }
        if ( R . upper_bounds [ d ] - R . lower_bounds [ d ] > width [ d ] ) {
          R . upper_bounds [ d ] = R . lower_bounds [ d ] + width [ d ];
        }
      }
      long periodic_long = 0;
      for ( int d = 0; d < dimension_; ++ d ) {
        if ( periodic_ [ d ] ) periodic_long += ( 1 << d );
      }
      std::set<long> periodic_images;
      long hypercube = 2 << dimension_;
      for ( long k = 0; k < hypercube; ++ k ) {
        if ( periodic_images . count ( k & periodic_long ) ) continue;
        periodic_images . insert ( k & periodic_long );
        RectGeo r = R;
        for ( int d = 0; d < dimension_; ++ d ) {
          if ( periodic_ [ d ] == false ) continue;
          if ( k & ( 1 << d ) ) {
            r . lower_bounds [ d ] += width [ d ];
            r . upper_bounds [ d ] += width [ d ];
          }
        }
        work_stack . push_back ( r );
      }
    } else {
      work_stack . push_back ( geometric_region );
    }

    static const int64_t INT_PHASE_WIDTH = static_cast<int64_t> ( 1 ) << 60;
    static const int64_t TRUNCATION_ERROR = static_cast<int64_t> ( 1 ) << 10;
    const Real bignum ( static_cast<Real> ( INT_PHASE_WIDTH ) );

    RectGeo region ( dimension_ );
    std::vector<int64_t> LB ( dimension_ );
    std::vector<int64_t> UB ( dimension_ );
    std::vector<int64_t> NLB ( dimension_ );
    std::vector<int64_t> NUB ( dimension_ );

    while ( not work_stack . empty () ) {
      RectGeo GR = work_stack . back ();
      work_stack . pop_back ();

      bool out_of_bounds = false;
      for ( int d = 0; d < dimension_; ++ d ) {
        region . lower_bounds [ d ] =
          ( GR . lower_bounds [ d ] - bounds_ . lower_bounds [ d ] ) /
          ( bounds_ . upper_bounds [ d ] - bounds_ . lower_bounds [ d ] );
        region . upper_bounds [ d ] =
          ( GR . upper_bounds [ d ] - bounds_ . lower_bounds [ d ] ) /
          ( bounds_ . upper_bounds [ d ] - bounds_ . lower_bounds [ d ] );
        if ( region . upper_bounds [ d ] < Real ( 0 ) or
             region . lower_bounds [ d ] > Real ( 1 ) ) {
          out_of_bounds = true;
          break;
        }
        if ( region . lower_bounds [ d ] < Real ( 0 ) )
          region . lower_bounds [ d ] = Real ( 0 );
        if ( region . lower_bounds [ d ] > Real ( 1 ) )
          region . lower_bounds [ d ] = Real ( 1 );
        if ( region . upper_bounds [ d ] < Real ( 0 ) )
          region . upper_bounds [ d ] = Real ( 0 );
        if ( region . upper_bounds [ d ] > Real ( 1 ) )
          region . upper_bounds [ d ] = Real ( 1 );

        LB [ d ] = static_cast<int64_t> ( bignum * region . lower_bounds [ d ] )
          - TRUNCATION_ERROR;
        UB [ d ] = static_cast<int64_t> ( bignum * region . upper_bounds [ d ] )
          + TRUNCATION_ERROR;
        if ( LB [ d ] < 0 ) LB [ d ] = 0;
        if ( UB [ d ] > INT_PHASE_WIDTH ) UB [ d ] = INT_PHASE_WIDTH;
      }
      if ( out_of_bounds ) continue;

      for ( int d = 0; d < dimension_; ++ d ) {
        NLB [ d ] = 0;
        NUB [ d ] = INT_PHASE_WIDTH;
      }
      rectDFS ( 0, 0, LB, UB, NLB, NUB, results );
    }
  }

  void rectDFS ( uint32_t depth, FixedGridElement code,
                 std::vector<int64_t> const& LB,
                 std::vector<int64_t> const& UB,
                 std::vector<int64_t> & NLB,
                 std::vector<int64_t> & NUB,
                 std::vector<FixedGridElement> & results ) const {
    for ( int d = 0; d < dimension_; ++ d ) {
      if ( LB [ d ] > NUB [ d ] or UB [ d ] < NLB [ d ] ) return;
    }
    if ( depth == subdiv_ ) {
      results . push_back ( code );
      return;
    }
    const int div_dim = static_cast<int> ( depth ) % dimension_;
    const int64_t half = ( NUB [ div_dim ] - NLB [ div_dim ] ) >> 1;
    NUB [ div_dim ] -= half;
    rectDFS ( depth + 1, code << 1, LB, UB, NLB, NUB, results );
    NUB [ div_dim ] += half;
    NLB [ div_dim ] += half;
    rectDFS ( depth + 1, ( code << 1 ) | 1, LB, UB, NLB, NUB, results );
    NLB [ div_dim ] -= half;
  }

  void coverPrism ( PrismGeo const& prism,
                    std::vector<FixedGridElement> & results ) const {
    if ( dimension_ == 0 ) {
      results . push_back ( 0 );
      return;
    }
    // TreeGrid::coverAccept(PrismGeo) performs no periodic handling;
    // reproduce that behavior.
    static const uint64_t INT_PHASE_WIDTH = static_cast<uint64_t> ( 1 ) << 60;
    std::vector<uint64_t> NLB ( dimension_, 0 );
    std::vector<uint64_t> NUB ( dimension_, INT_PHASE_WIDTH );
    prismDFS ( 0, 0, prism, NLB, NUB, results );
  }

  void prismDFS ( uint32_t depth, FixedGridElement code,
                  PrismGeo const& prism,
                  std::vector<uint64_t> & NLB,
                  std::vector<uint64_t> & NUB,
                  std::vector<FixedGridElement> & results ) const {
    static const uint64_t INT_PHASE_WIDTH = static_cast<uint64_t> ( 1 ) << 60;
    RectGeo G ( dimension_ );
    for ( int d = 0; d < dimension_; ++ d ) {
      G . lower_bounds [ d ] = bounds_ . lower_bounds [ d ] +
        ( bounds_ . upper_bounds [ d ] - bounds_ . lower_bounds [ d ] ) *
        ( static_cast<Real> ( NLB [ d ] ) / static_cast<Real> ( INT_PHASE_WIDTH ) );
      G . upper_bounds [ d ] = bounds_ . lower_bounds [ d ] +
        ( bounds_ . upper_bounds [ d ] - bounds_ . lower_bounds [ d ] ) *
        ( static_cast<Real> ( NUB [ d ] ) / static_cast<Real> ( INT_PHASE_WIDTH ) );
    }
    if ( not prism . intersects ( G ) ) return;
    if ( depth == subdiv_ ) {
      results . push_back ( code );
      return;
    }
    const int div_dim = static_cast<int> ( depth ) % dimension_;
    const uint64_t half = ( NUB [ div_dim ] - NLB [ div_dim ] ) >> 1;
    NUB [ div_dim ] -= half;
    prismDFS ( depth + 1, code << 1, prism, NLB, NUB, results );
    NUB [ div_dim ] += half;
    NLB [ div_dim ] += half;
    prismDFS ( depth + 1, ( code << 1 ) | 1, prism, NLB, NUB, results );
    NLB [ div_dim ] -= half;
  }

  RectGeo bounds_;
  std::vector<bool> periodic_;
  int dimension_;
  uint32_t subdiv_;
};

/// Adjacency provider evaluating Model maps through the implicit fixed
/// grid: adjacency(source) == cover(Map(geometry(source))).
class ModelMapAdjacencyProvider : public FixedSubdivisionAdjacencyProvider {
public:
  ModelMapAdjacencyProvider ( std::shared_ptr<const Map> map,
                              ImplicitFixedTreeGrid const& grid )
    : map_ ( map ), grid_ ( grid ),
      map_evaluations_ ( 0 ), map_batches_ ( 0 ) {}

  virtual void adjacencies_batch ( std::vector<FixedGridElement> const& sources,
                                   EmitFn const& emit,
                                   CompleteFn const& source_complete ) {
    std::vector<std::shared_ptr<Geo>> geos;
    geos . reserve ( sources . size () );
    for ( size_t i = 0; i < sources . size (); ++ i ) {
      geos . push_back ( std::shared_ptr<Geo> (
        new RectGeo ( grid_ . geometry ( sources [ i ] ) ) ) );
    }

    map_evaluations_ += sources . size ();
    if ( map_ -> has_optimized_batch () ) ++ map_batches_;

    std::vector<std::shared_ptr<Geo>> images;
    try {
      images = map_ -> batch_map ( geos );
    } catch ( MorseSetReachabilityMapError const& ) {
      throw;
    } catch ( pybind11::error_already_set & error ) {
      // Propagate user interrupts; classify everything else as a map error.
      if ( error . matches ( PyExc_KeyboardInterrupt ) or
           error . matches ( PyExc_SystemExit ) ) {
        throw;
      }
      throw MorseSetReachabilityMapError ( error . what () );
    } catch ( std::exception const& error ) {
      throw MorseSetReachabilityMapError ( error . what () );
    }
    if ( images . size () != sources . size () ) {
      throw MorseSetReachabilityMapError (
        "map returned the wrong number of images" );
    }

    for ( size_t i = 0; i < sources . size (); ++ i ) {
      std::vector<FixedGridElement> targets;
      try {
        targets = grid_ . cover ( images [ i ] );
      } catch ( MorseSetReachabilityCoverError const& ) {
        throw;
      } catch ( std::exception const& error ) {
        throw MorseSetReachabilityCoverError ( error . what () );
      }
      for ( size_t j = 0; j < targets . size (); ++ j ) {
        if ( not emit ( i, targets [ j ] ) ) return;
      }
      source_complete ( i );
    }
  }

  virtual uint64_t map_evaluations_attempted ( void ) const {
    return map_evaluations_;
  }

  virtual uint64_t map_batches_attempted ( void ) const {
    return map_batches_;
  }

private:
  std::shared_ptr<const Map> map_;
  ImplicitFixedTreeGrid grid_;
  uint64_t map_evaluations_;
  uint64_t map_batches_;
};

/// Per-Morse-vertex provenance facts about the adaptive and fixed
/// representations.
struct MorseSetNodeProvenance {
  std::string adaptive_prefix_hash;
  std::string fixed_range_hash;
  uint64_t fixed_descendant_count;
  uint64_t adaptive_leaf_count;
  uint64_t max_adaptive_leaf_depth;
  std::vector<std::pair<FixedGridElement, FixedGridElement>> fixed_ranges;

  MorseSetNodeProvenance ( void )
    : fixed_descendant_count ( 0 ),
      adaptive_leaf_count ( 0 ),
      max_adaptive_leaf_depth ( 0 ) {}
};

/// Extract the canonical fixed-subdivision ranges of every Morse set.
/// Descendant ranges are derived arithmetically from adaptive leaf path
/// prefixes; cover is never used for this purpose.
inline void
ExtractMorseSetFixedRanges ( MorseGraph const& morse_graph,
                             uint32_t phase_subdiv,
                             std::vector<MorseSetFixedRanges> & morse_sets,
                             std::vector<MorseSetNodeProvenance> & nodes,
                             uint64_t & max_leaf_depth ) {
  const uint64_t m = morse_graph . NumVertices ();
  morse_sets . assign ( m, MorseSetFixedRanges () );
  nodes . assign ( m, MorseSetNodeProvenance () );
  max_leaf_depth = 0;

  for ( uint64_t v = 0; v < m; ++ v ) {
    std::shared_ptr<const TreeGrid> grid =
      std::dynamic_pointer_cast<const TreeGrid> ( morse_graph . grid ( v ) );
    if ( not grid ) {
      std::ostringstream message;
      message << "MorseGraph grid(" << v << ") is not TreeGrid-backed";
      throw std::invalid_argument ( message . str () );
    }

    // (depth, prefix) pairs, root-to-leaf branch bits MSB first.
    std::vector<std::pair<uint64_t, uint64_t>> prefixes;
    for ( Grid::iterator it = grid -> begin (); it != grid -> end (); ++ it ) {
      Tree::iterator node = grid -> GridToTree ( it );
      std::vector<bool> climb_bits;   // leaf-to-root
      while ( node != grid -> tree () . begin () ) {
        climb_bits . push_back ( not grid -> tree () . isLeft ( node ) );
        node = grid -> tree () . parent ( node );
      }
      const uint64_t depth = climb_bits . size ();
      if ( depth > max_leaf_depth ) max_leaf_depth = depth;
      if ( depth >= 64 ) {
        throw std::invalid_argument (
          "adaptive Morse-set leaf exceeds 63 subdivisions; unsupported in v1" );
      }
      uint64_t prefix = 0;
      for ( size_t i = climb_bits . size (); i > 0; -- i ) {
        prefix = ( prefix << 1 ) | ( climb_bits [ i - 1 ] ? 1 : 0 );
      }
      prefixes . push_back ( std::make_pair ( depth, prefix ) );
    }
    std::sort ( prefixes . begin (), prefixes . end () );

    Hasher prefix_hasher;
    prefix_hasher . u64 ( prefixes . size () );
    uint64_t node_max_depth = 0;
    for ( size_t i = 0; i < prefixes . size (); ++ i ) {
      prefix_hasher . u64 ( prefixes [ i ] . first );
      prefix_hasher . u64 ( prefixes [ i ] . second );
      if ( prefixes [ i ] . first > node_max_depth ) {
        node_max_depth = prefixes [ i ] . first;
      }
    }
    nodes [ v ] . adaptive_prefix_hash = prefix_hasher . hex ();
    nodes [ v ] . adaptive_leaf_count = prefixes . size ();
    nodes [ v ] . max_adaptive_leaf_depth = node_max_depth;

    // Convert to descendant ranges at phase_subdiv; requires
    // phase_subdiv >= depth for every leaf (checked by the caller
    // against max_leaf_depth before calling with final ranges).
    std::vector<std::pair<FixedGridElement, FixedGridElement>> ranges;
    for ( size_t i = 0; i < prefixes . size (); ++ i ) {
      const uint64_t depth = prefixes [ i ] . first;
      const uint64_t prefix = prefixes [ i ] . second;
      if ( depth > phase_subdiv ) {
        std::ostringstream message;
        message
          << "phase_subdiv=" << phase_subdiv << " is below the deepest "
          << "adaptive Morse-set leaf (TreeGrid depth " << depth << ")";
        throw std::invalid_argument ( message . str () );
      }
      const uint64_t shift = phase_subdiv - depth;
      ranges . push_back ( std::make_pair (
        prefix << shift, ( prefix + 1 ) << shift ) );
    }
    std::sort ( ranges . begin (), ranges . end () );

    // Canonicalize: merge adjacent/overlapping ranges of the same vertex.
    std::vector<std::pair<FixedGridElement, FixedGridElement>> merged;
    for ( size_t i = 0; i < ranges . size (); ++ i ) {
      if ( not merged . empty () and
           ranges [ i ] . first <= merged . back () . second ) {
        if ( ranges [ i ] . second > merged . back () . second ) {
          merged . back () . second = ranges [ i ] . second;
        }
      } else {
        merged . push_back ( ranges [ i ] );
      }
    }
    morse_sets [ v ] . ranges = merged;
    nodes [ v ] . fixed_range_hash = morse_sets [ v ] . hash ();
    nodes [ v ] . fixed_descendant_count = morse_sets [ v ] . element_count ();
    nodes [ v ] . fixed_ranges = merged;
  }
}

inline std::string
hash_adaptive_edges ( std::vector<std::pair<uint64_t, uint64_t>> edges ) {
  std::sort ( edges . begin (), edges . end () );
  Hasher h;
  h . u64 ( edges . size () );
  for ( size_t i = 0; i < edges . size (); ++ i ) {
    h . u64 ( edges [ i ] . first );
    h . u64 ( edges [ i ] . second );
  }
  return h . hex ();
}

inline std::string
double_bits_hex ( double value ) {
  uint64_t bits = 0;
  unsigned char const* raw = reinterpret_cast<unsigned char const*> ( &value );
  for ( int i = 0; i < 8; ++ i ) {
    bits |= static_cast<uint64_t> ( raw [ i ] ) << ( 8 * i );
  }
  static const char digits [] = "0123456789abcdef";
  std::string out ( 16, '0' );
  for ( int i = 0; i < 16; ++ i ) {
    out [ 15 - i ] = digits [ ( bits >> (4 * i) ) & 0xf ];
  }
  return out;
}

} // namespace msr_detail

/// Versioned provenance envelope (schema
/// CMGDB.MorseSetReachabilityProvenance, version 1).
struct MorseSetReachabilityProvenance {
  // model
  int param_dim;
  int phase_dim;
  std::vector<double> param_lower_bounds;
  std::vector<double> param_upper_bounds;
  std::vector<double> phase_lower_bounds;
  std::vector<double> phase_upper_bounds;
  std::vector<bool> phase_periodic;
  int phase_subdiv_init;
  int phase_subdiv_min;
  int phase_subdiv_max;
  int phase_subdiv_limit;
  std::string configuration_hash;     // FNV-1a-64 (not SHA-256)
  std::string map_kind;
  std::string map_fingerprint;
  std::string map_fingerprint_kind;   // caller_supplied | unavailable
  std::string evaluation_mode;        // scalar | optimized_batch
  // verification
  uint32_t phase_subdiv;
  std::string fixed_element_encoding; // tree_path_msb_uint64_v1
  bool has_max_visited_grid_elements;
  uint64_t max_visited_grid_elements;
  bool has_max_adjacencies_examined;
  uint64_t max_adjacencies_examined;
  uint64_t batch_size;
  std::string traversal;              // fifo_sorted_streaming_v1
  std::vector<std::string> warnings;
  // structural instrumentation: this implementation never materializes
  // a complete TreeGrid or MapGraph, so these are constant zeros.
  uint64_t complete_treegrid_materializations;
  uint64_t mapgraph_constructions;
  uint64_t complete_vertex_array_bytes;
  uint64_t complete_edge_array_bytes;
  // morse graph
  uint64_t num_vertices;
  std::vector<msr_detail::MorseSetNodeProvenance> nodes;
  std::string morse_sets_hash;
  std::string adaptive_unreduced_edge_hash;
  std::string combined_hash;

  MorseSetReachabilityProvenance ( void )
    : param_dim ( 0 ), phase_dim ( 0 ),
      phase_subdiv_init ( 0 ), phase_subdiv_min ( 0 ),
      phase_subdiv_max ( 0 ), phase_subdiv_limit ( 0 ),
      phase_subdiv ( 0 ),
      fixed_element_encoding ( "tree_path_msb_uint64_v1" ),
      has_max_visited_grid_elements ( false ),
      max_visited_grid_elements ( 0 ),
      has_max_adjacencies_examined ( false ),
      max_adjacencies_examined ( 0 ),
      batch_size ( 0 ),
      traversal ( "fifo_sorted_streaming_v1" ),
      complete_treegrid_materializations ( 0 ),
      mapgraph_constructions ( 0 ),
      complete_vertex_array_bytes ( 0 ),
      complete_edge_array_bytes ( 0 ),
      num_vertices ( 0 ) {}
};

/// Fixed-subdivision Morse-set reachability verification.
///
/// For each Morse vertex v the forward closure of its fixed-subdivision
/// descendants is exhausted independently under the adjacency
/// cover(Map(geometry(x))) on the conceptual uniform TreeGrid at
/// phase_subdiv total subdivisions. The supplied MorseGraph is not
/// mutated and no complete TreeGrid or MapGraph is constructed.
inline MorseSetReachabilityResult
ComputeMorseSetReachability (
    Model const& model,
    MorseGraph const& morse_graph,
    uint32_t phase_subdiv,
    MorseSetReachabilityOptions const& options = MorseSetReachabilityOptions (),
    MorseSetReachabilityProvenance * provenance_out = 0 ) {

  // ---- Preflight (raises before any Map invocation) ----
  std::shared_ptr<const Map> map = model . map ();
  if ( not map ) {
    throw std::invalid_argument (
      "ComputeMorseSetReachability requires a Model with a map" );
  }

  std::shared_ptr<const Grid> phase_grid = morse_graph . phaseSpace ();
  if ( not phase_grid ) {
    throw std::invalid_argument (
      "MorseGraph has no phase space grid (was clearGrids called?)" );
  }
  std::shared_ptr<const TreeGrid> phase_tree_grid =
    std::dynamic_pointer_cast<const TreeGrid> ( phase_grid );
  if ( not phase_tree_grid ) {
    throw std::invalid_argument (
      "MorseGraph phase space is not TreeGrid-backed" );
  }

  const int phase_dim = model . phase_dim ();
  if ( phase_tree_grid -> dimension () != phase_dim ) {
    throw std::invalid_argument (
      "Model and MorseGraph phase-space dimensions differ" );
  }
  for ( int d = 0; d < phase_dim; ++ d ) {
    if ( phase_tree_grid -> bounds () . lower_bounds [ d ] !=
           model . phase_lower_bounds () [ d ] or
         phase_tree_grid -> bounds () . upper_bounds [ d ] !=
           model . phase_upper_bounds () [ d ] ) {
      throw std::invalid_argument (
        "Model and MorseGraph phase-space bounds differ" );
    }
    if ( phase_tree_grid -> periodicity () [ d ] !=
         model . phase_periodic () [ d ] ) {
      throw std::invalid_argument (
        "Model and MorseGraph phase-space periodicity differ" );
    }
  }

  if ( phase_subdiv >= 64 ) {
    throw std::invalid_argument (
      "phase_subdiv must satisfy 0 <= phase_subdiv < 64 in v1" );
  }

  std::vector<MorseSetFixedRanges> morse_sets;
  std::vector<msr_detail::MorseSetNodeProvenance> nodes;
  uint64_t max_leaf_depth = 0;
  msr_detail::ExtractMorseSetFixedRanges (
    morse_graph, phase_subdiv, morse_sets, nodes, max_leaf_depth );

  std::vector<std::pair<uint64_t, uint64_t>> adaptive_edges =
    morse_graph . edges_unreduced ();
  std::sort ( adaptive_edges . begin (), adaptive_edges . end () );

  // Configuration hash over the model facts recorded in provenance.
  msr_detail::Hasher config_hasher;
  config_hasher . u64 ( model . param_dim () );
  config_hasher . u64 ( phase_dim );
  for ( int d = 0; d < phase_dim; ++ d ) {
    config_hasher . str (
      msr_detail::double_bits_hex ( model . phase_lower_bounds () [ d ] ) );
    config_hasher . str (
      msr_detail::double_bits_hex ( model . phase_upper_bounds () [ d ] ) );
    config_hasher . byte ( model . phase_periodic () [ d ] ? 1 : 0 );
  }
  config_hasher . u64 ( model . phase_subdiv_init () );
  config_hasher . u64 ( model . phase_subdiv_min () );
  config_hasher . u64 ( model . phase_subdiv_max () );
  config_hasher . u64 ( model . phase_subdiv_limit () );

  MorseSetReachabilityOptions core_options = options;
  core_options . phase_subdiv = phase_subdiv;
  core_options . configuration_sha = config_hasher . hex ();
  core_options . adaptive_edges_hash =
    msr_detail::hash_adaptive_edges ( adaptive_edges );

  msr_detail::ImplicitFixedTreeGrid fixed_grid (
    phase_tree_grid -> bounds (), phase_tree_grid -> periodicity (),
    phase_subdiv );
  msr_detail::ModelMapAdjacencyProvider provider ( map, fixed_grid );

  MorseSetReachabilityResult result =
    ComputeMorseSetReachabilityCore ( morse_sets, provider, core_options );
  result . set_adaptive_edges ( adaptive_edges );

  if ( provenance_out ) {
    MorseSetReachabilityProvenance & prov = * provenance_out;
    prov . param_dim = model . param_dim ();
    prov . phase_dim = phase_dim;
    prov . param_lower_bounds = model . param_lower_bounds ();
    prov . param_upper_bounds = model . param_upper_bounds ();
    prov . phase_lower_bounds = model . phase_lower_bounds ();
    prov . phase_upper_bounds = model . phase_upper_bounds ();
    prov . phase_periodic = model . phase_periodic ();
    prov . phase_subdiv_init = model . phase_subdiv_init ();
    prov . phase_subdiv_min = model . phase_subdiv_min ();
    prov . phase_subdiv_max = model . phase_subdiv_max ();
    prov . phase_subdiv_limit = model . phase_subdiv_limit ();
    prov . configuration_hash = core_options . configuration_sha;
    prov . map_kind = map -> has_optimized_batch ()
      ? "python_rectangle_map_with_optimized_batch"
      : "python_rectangle_map";
    prov . map_fingerprint = options . map_fingerprint;
    prov . map_fingerprint_kind = options . map_fingerprint . empty ()
      ? "unavailable" : "caller_supplied";
    prov . evaluation_mode = map -> has_optimized_batch ()
      ? "optimized_batch" : "scalar";
    prov . phase_subdiv = phase_subdiv;
    prov . has_max_visited_grid_elements =
      options . max_visited_grid_elements !=
      MorseSetReachabilityOptions::NO_LIMIT;
    prov . max_visited_grid_elements = options . max_visited_grid_elements;
    prov . has_max_adjacencies_examined =
      options . max_adjacencies_examined !=
      MorseSetReachabilityOptions::NO_LIMIT;
    prov . max_adjacencies_examined = options . max_adjacencies_examined;
    prov . batch_size = options . batch_size;
    if ( static_cast<int> ( phase_subdiv ) > model . phase_subdiv_max () ) {
      std::ostringstream warning;
      warning
        << "phase_subdiv=" << phase_subdiv << " exceeds "
        << "model.phase_subdiv_max()=" << model . phase_subdiv_max ()
        << "; phase_subdiv_max controls adaptive decomposition, not the "
        << "valid domain of the Map";
      prov . warnings . push_back ( warning . str () );
    }
    prov . num_vertices = morse_sets . size ();
    prov . nodes = nodes;
    prov . morse_sets_hash = msr_detail::hash_morse_sets ( morse_sets );
    prov . adaptive_unreduced_edge_hash = core_options . adaptive_edges_hash;
    msr_detail::Hasher combined;
    combined . str ( prov . morse_sets_hash );
    combined . str ( prov . adaptive_unreduced_edge_hash );
    prov . combined_hash = combined . hex ();
  }

  return result;
}

/// Python Bindings

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

/// Python-facing result wrapper: the core result plus optional adapter
/// provenance.
struct PyMorseSetReachabilityResult {
  MorseSetReachabilityResult core;
  std::shared_ptr<MorseSetReachabilityProvenance> provenance;
};

namespace msr_detail {

inline std::string
stop_reason_name ( MorseSetReachabilityStopReason reason ) {
  switch ( reason ) {
    case MorseSetReachabilityStopReason::NONE: return "NONE";
    case MorseSetReachabilityStopReason::MAX_VISITED_GRID_ELEMENTS:
      return "MAX_VISITED_GRID_ELEMENTS";
    case MorseSetReachabilityStopReason::MAX_ADJACENCIES_EXAMINED:
      return "MAX_ADJACENCIES_EXAMINED";
    case MorseSetReachabilityStopReason::MAP_ERROR: return "MAP_ERROR";
    case MorseSetReachabilityStopReason::COVER_ERROR: return "COVER_ERROR";
    case MorseSetReachabilityStopReason::CANCELLED: return "CANCELLED";
  }
  return "NONE";
}

inline std::string
diagnostic_name ( MorseSetRelationDiagnostic diagnostic ) {
  switch ( diagnostic ) {
    case MorseSetRelationDiagnostic::VALID_PARTIAL_ORDER:
      return "VALID_PARTIAL_ORDER";
    case MorseSetRelationDiagnostic::INCOMPLETE: return "INCOMPLETE";
    case MorseSetRelationDiagnostic::COALESCING_REQUIRED:
      return "COALESCING_REQUIRED";
    case MorseSetRelationDiagnostic::MORSE_SET_SPLITTING_REQUIRED:
      return "MORSE_SET_SPLITTING_REQUIRED";
  }
  return "VALID_PARTIAL_ORDER";
}

inline py::object
error_to_python ( MorseSetReachabilitySourceStats const& stats ) {
  if ( stats . error_category . empty () ) return py::none ();
  py::dict error;
  error [ "category" ] = stats . error_category;
  error [ "type" ] = stats . error_type;
  error [ "message" ] = stats . error_message;
  if ( stats . has_error_element ) {
    error [ "fixed_element" ] = stats . error_element;
  } else {
    error [ "fixed_element" ] = py::none ();
  }
  return error;
}

inline py::dict
stats_to_python ( MorseSetReachabilitySourceStats const& stats ) {
  py::dict out;
  out [ "frontier_exhausted" ] = stats . frontier_exhausted;
  out [ "stop_reason" ] = static_cast<int> ( stats . stop_reason );
  out [ "error_category" ] = stats . error_category;
  out [ "error_type" ] = stats . error_type;
  out [ "error_message" ] = stats . error_message;
  out [ "has_error_element" ] = stats . has_error_element;
  out [ "error_element" ] = stats . error_element;
  out [ "fixed_seed_count" ] = stats . fixed_seed_count;
  out [ "visited_grid_elements" ] = stats . visited_grid_elements;
  out [ "grid_elements_expanded" ] = stats . grid_elements_expanded;
  out [ "adjacencies_examined" ] = stats . adjacencies_examined;
  out [ "map_evaluations_attempted" ] = stats . map_evaluations_attempted;
  out [ "map_batches_attempted" ] = stats . map_batches_attempted;
  out [ "frontier_count" ] = stats . frontier_count;
  out [ "visited_hash" ] = stats . visited_hash;
  out [ "frontier_hash" ] = stats . frontier_hash;
  return out;
}

inline MorseSetReachabilitySourceStats
stats_from_python ( py::dict const& in ) {
  MorseSetReachabilitySourceStats stats;
  stats . frontier_exhausted = in [ "frontier_exhausted" ] . cast<bool> ();
  stats . stop_reason = static_cast<MorseSetReachabilityStopReason> (
    in [ "stop_reason" ] . cast<int> () );
  stats . error_category = in [ "error_category" ] . cast<std::string> ();
  stats . error_type = in [ "error_type" ] . cast<std::string> ();
  stats . error_message = in [ "error_message" ] . cast<std::string> ();
  stats . has_error_element = in [ "has_error_element" ] . cast<bool> ();
  stats . error_element = in [ "error_element" ] . cast<uint64_t> ();
  stats . fixed_seed_count = in [ "fixed_seed_count" ] . cast<uint64_t> ();
  stats . visited_grid_elements =
    in [ "visited_grid_elements" ] . cast<uint64_t> ();
  stats . grid_elements_expanded =
    in [ "grid_elements_expanded" ] . cast<uint64_t> ();
  stats . adjacencies_examined =
    in [ "adjacencies_examined" ] . cast<uint64_t> ();
  stats . map_evaluations_attempted =
    in [ "map_evaluations_attempted" ] . cast<uint64_t> ();
  stats . map_batches_attempted =
    in [ "map_batches_attempted" ] . cast<uint64_t> ();
  stats . frontier_count = in [ "frontier_count" ] . cast<uint64_t> ();
  stats . visited_hash = in [ "visited_hash" ] . cast<std::string> ();
  stats . frontier_hash = in [ "frontier_hash" ] . cast<std::string> ();
  return stats;
}

inline py::dict
checkpoint_to_python ( MorseSetReachabilityCheckpoint const& cp ) {
  py::dict out;
  out [ "schema_version" ] = cp . schema_version;
  out [ "algorithm_version" ] = cp . algorithm_version;
  out [ "configuration_sha" ] = cp . configuration_sha;
  out [ "map_fingerprint" ] = cp . map_fingerprint;
  out [ "phase_subdiv" ] = cp . phase_subdiv;
  out [ "morse_sets_hash" ] = cp . morse_sets_hash;
  out [ "adaptive_edges_hash" ] = cp . adaptive_edges_hash;
  out [ "num_vertices" ] = cp . num_vertices;
  out [ "status" ] = cp . status;
  out [ "source_finalized" ] = cp . source_finalized;
  py::list stats;
  for ( size_t i = 0; i < cp . stats . size (); ++ i ) {
    stats . append ( stats_to_python ( cp . stats [ i ] ) );
  }
  out [ "stats" ] = stats;
  py::dict partials;
  for ( std::map<uint64_t, MorseSetReachabilityPartialState>::const_iterator
          it = cp . partial_states . begin ();
        it != cp . partial_states . end (); ++ it ) {
    py::dict partial;
    partial [ "seed_range_index" ] = it -> second . seed_range_index;
    partial [ "seed_offset" ] = it -> second . seed_offset;
    partial [ "visited_sorted" ] = it -> second . visited_sorted;
    partial [ "frontier" ] = it -> second . frontier;
    partial [ "reached_targets" ] = it -> second . reached_targets;
    partials [ py::int_ ( it -> first ) ] = partial;
  }
  out [ "partial_states" ] = partials;
  out [ "payload_checksum" ] = cp . payload_checksum;
  return out;
}

inline MorseSetReachabilityCheckpoint
checkpoint_from_python ( py::dict const& in ) {
  MorseSetReachabilityCheckpoint cp;
  cp . schema_version = in [ "schema_version" ] . cast<uint32_t> ();
  cp . algorithm_version = in [ "algorithm_version" ] . cast<std::string> ();
  cp . configuration_sha = in [ "configuration_sha" ] . cast<std::string> ();
  cp . map_fingerprint = in [ "map_fingerprint" ] . cast<std::string> ();
  cp . phase_subdiv = in [ "phase_subdiv" ] . cast<uint32_t> ();
  cp . morse_sets_hash = in [ "morse_sets_hash" ] . cast<std::string> ();
  cp . adaptive_edges_hash = in [ "adaptive_edges_hash" ] . cast<std::string> ();
  cp . num_vertices = in [ "num_vertices" ] . cast<uint64_t> ();
  cp . status = in [ "status" ] . cast<std::vector<uint8_t>> ();
  cp . source_finalized =
    in [ "source_finalized" ] . cast<std::vector<uint8_t>> ();
  py::list stats = in [ "stats" ] . cast<py::list> ();
  for ( size_t i = 0; i < stats . size (); ++ i ) {
    cp . stats . push_back (
      stats_from_python ( stats [ i ] . cast<py::dict> () ) );
  }
  py::dict partials = in [ "partial_states" ] . cast<py::dict> ();
  for ( std::pair<py::handle, py::handle> item : partials ) {
    uint64_t source = item . first . cast<uint64_t> ();
    py::dict partial = item . second . cast<py::dict> ();
    MorseSetReachabilityPartialState state;
    state . seed_range_index = partial [ "seed_range_index" ] . cast<uint64_t> ();
    state . seed_offset = partial [ "seed_offset" ] . cast<uint64_t> ();
    state . visited_sorted =
      partial [ "visited_sorted" ] . cast<std::vector<uint64_t>> ();
    state . frontier = partial [ "frontier" ] . cast<std::vector<uint64_t>> ();
    state . reached_targets =
      partial [ "reached_targets" ] . cast<std::vector<uint8_t>> ();
    cp . partial_states [ source ] = state;
  }
  cp . payload_checksum = in [ "payload_checksum" ] . cast<std::string> ();
  cp . validate_checksum ();
  return cp;
}

/// Build a PointerGrid whose valid leaves are exactly the supplied
/// root-to-leaf branch paths ('0' = left, '1' = right). Used to
/// construct synthetic adaptive MorseGraphs in tests.
inline std::shared_ptr<Grid>
build_treegrid_from_paths ( RectGeo const& bounds,
                            std::vector<bool> const& periodic,
                            std::vector<std::string> const& paths ) {
  struct TrieNode {
    int64_t child [ 2 ];
    bool terminal;
    TrieNode ( void ) : terminal ( false ) {
      child [ 0 ] = -1;
      child [ 1 ] = -1;
    }
  };
  std::vector<TrieNode> trie ( 1 );
  for ( size_t i = 0; i < paths . size (); ++ i ) {
    int64_t current = 0;
    for ( size_t j = 0; j < paths [ i ] . size (); ++ j ) {
      const char c = paths [ i ] [ j ];
      if ( c != '0' and c != '1' ) {
        throw std::invalid_argument ( "leaf paths must consist of '0'/'1'" );
      }
      if ( trie [ current ] . terminal ) {
        throw std::invalid_argument ( "a leaf path is a prefix of another" );
      }
      const int branch = ( c == '1' ) ? 1 : 0;
      if ( trie [ current ] . child [ branch ] < 0 ) {
        trie [ current ] . child [ branch ] =
          static_cast<int64_t> ( trie . size () );
        trie . push_back ( TrieNode () );
      }
      current = trie [ current ] . child [ branch ];
    }
    if ( trie [ current ] . child [ 0 ] >= 0 or
         trie [ current ] . child [ 1 ] >= 0 ) {
      throw std::invalid_argument ( "a leaf path is a prefix of another" );
    }
    trie [ current ] . terminal = true;
  }

  std::shared_ptr<CompressedTreeGrid> compressed ( new CompressedTreeGrid );
  compressed -> bounds () = bounds;
  compressed -> periodicity () = periodic;
  std::vector<bool> & leaf_sequence = compressed -> tree () -> leaf_sequence;
  std::vector<bool> & valid_sequence = compressed -> tree () -> valid_sequence;
  leaf_sequence . clear ();
  valid_sequence . clear ();

  // Preorder emission over the full binary tree induced by the trie:
  // a missing sibling becomes an invalid leaf.
  std::vector<std::pair<int64_t, bool>> stack;   // (node, is_real)
  stack . push_back ( std::make_pair ( static_cast<int64_t> ( 0 ), true ) );
  while ( not stack . empty () ) {
    std::pair<int64_t, bool> frame = stack . back ();
    stack . pop_back ();
    if ( not frame . second ) {
      leaf_sequence . push_back ( false );
      valid_sequence . push_back ( false );
      continue;
    }
    TrieNode const& node = trie [ frame . first ];
    const bool interior = node . child [ 0 ] >= 0 or node . child [ 1 ] >= 0;
    if ( interior ) {
      leaf_sequence . push_back ( true );
      // Push right first so the left subtree is emitted first (preorder).
      if ( node . child [ 1 ] >= 0 ) {
        stack . push_back ( std::make_pair ( node . child [ 1 ], true ) );
      } else {
        stack . push_back ( std::make_pair ( static_cast<int64_t> ( -1 ),
                                             false ) );
      }
      if ( node . child [ 0 ] >= 0 ) {
        stack . push_back ( std::make_pair ( node . child [ 0 ], true ) );
      } else {
        stack . push_back ( std::make_pair ( static_cast<int64_t> ( -1 ),
                                             false ) );
      }
    } else {
      leaf_sequence . push_back ( false );
      valid_sequence . push_back ( node . terminal );
    }
  }

  std::shared_ptr<TreeGrid> grid ( new PointerGrid );
  grid -> assign ( compressed );
  return std::dynamic_pointer_cast<Grid> ( grid );
}

inline uint64_t
optional_limit_from_python ( py::object const& value, char const* name ) {
  if ( value . is_none () ) return MorseSetReachabilityOptions::NO_LIMIT;
  int64_t limit = value . cast<int64_t> ();
  if ( limit < 0 ) {
    std::ostringstream message;
    message << name << " must be nonnegative or None";
    throw std::invalid_argument ( message . str () );
  }
  return static_cast<uint64_t> ( limit );
}

inline py::dict
provenance_to_python ( PyMorseSetReachabilityResult const& wrapper ) {
  MorseSetReachabilityResult const& core = wrapper . core;
  py::dict out;
  out [ "schema_name" ] = "CMGDB.MorseSetReachabilityProvenance";
  out [ "schema_version" ] = 1;
  out [ "algorithm_version" ] = "fifo_sorted_streaming_v1";

  if ( wrapper . provenance ) {
    MorseSetReachabilityProvenance const& prov = * wrapper . provenance;
    py::dict model;
    model [ "param_dim" ] = prov . param_dim;
    model [ "phase_dim" ] = prov . phase_dim;
    model [ "param_lower_bounds" ] = prov . param_lower_bounds;
    model [ "param_upper_bounds" ] = prov . param_upper_bounds;
    model [ "phase_lower_bounds" ] = prov . phase_lower_bounds;
    model [ "phase_upper_bounds" ] = prov . phase_upper_bounds;
    py::list lower_bits, upper_bits;
    for ( size_t d = 0; d < prov . phase_lower_bounds . size (); ++ d ) {
      lower_bits . append ( double_bits_hex ( prov . phase_lower_bounds [ d ] ) );
      upper_bits . append ( double_bits_hex ( prov . phase_upper_bounds [ d ] ) );
    }
    model [ "phase_lower_bounds_ieee754" ] = lower_bits;
    model [ "phase_upper_bounds_ieee754" ] = upper_bits;
    model [ "phase_periodic" ] = prov . phase_periodic;
    model [ "phase_subdiv_init" ] = prov . phase_subdiv_init;
    model [ "phase_subdiv_min" ] = prov . phase_subdiv_min;
    model [ "phase_subdiv_max" ] = prov . phase_subdiv_max;
    model [ "phase_subdiv_limit" ] = prov . phase_subdiv_limit;
    model [ "configuration_hash" ] = prov . configuration_hash;
    model [ "map_kind" ] = prov . map_kind;
    if ( prov . map_fingerprint . empty () ) {
      model [ "map_fingerprint" ] = py::none ();
    } else {
      model [ "map_fingerprint" ] = prov . map_fingerprint;
    }
    model [ "map_fingerprint_kind" ] = prov . map_fingerprint_kind;
    model [ "evaluation_mode" ] = prov . evaluation_mode;
    out [ "model" ] = model;

    py::dict verification;
    verification [ "phase_subdiv" ] = prov . phase_subdiv;
    verification [ "fixed_element_encoding" ] = prov . fixed_element_encoding;
    if ( prov . has_max_visited_grid_elements ) {
      verification [ "max_visited_grid_elements" ] =
        prov . max_visited_grid_elements;
    } else {
      verification [ "max_visited_grid_elements" ] = py::none ();
    }
    if ( prov . has_max_adjacencies_examined ) {
      verification [ "max_adjacencies_examined" ] =
        prov . max_adjacencies_examined;
    } else {
      verification [ "max_adjacencies_examined" ] = py::none ();
    }
    verification [ "batch_size" ] = prov . batch_size;
    verification [ "traversal" ] = prov . traversal;
    verification [ "warnings" ] = prov . warnings;
    py::dict instrumentation;
    instrumentation [ "complete_treegrid_materializations" ] =
      prov . complete_treegrid_materializations;
    instrumentation [ "mapgraph_constructions" ] =
      prov . mapgraph_constructions;
    instrumentation [ "complete_vertex_array_bytes" ] =
      prov . complete_vertex_array_bytes;
    instrumentation [ "complete_edge_array_bytes" ] =
      prov . complete_edge_array_bytes;
    verification [ "instrumentation" ] = instrumentation;
    out [ "verification" ] = verification;

    py::dict morse;
    morse [ "num_vertices" ] = prov . num_vertices;
    py::list node_list;
    for ( size_t v = 0; v < prov . nodes . size (); ++ v ) {
      py::dict node;
      node [ "adaptive_prefix_hash" ] = prov . nodes [ v ] . adaptive_prefix_hash;
      node [ "fixed_range_hash" ] = prov . nodes [ v ] . fixed_range_hash;
      node [ "fixed_descendant_count" ] =
        prov . nodes [ v ] . fixed_descendant_count;
      node [ "adaptive_leaf_count" ] = prov . nodes [ v ] . adaptive_leaf_count;
      node [ "max_adaptive_leaf_depth" ] =
        prov . nodes [ v ] . max_adaptive_leaf_depth;
      node [ "fixed_ranges" ] = prov . nodes [ v ] . fixed_ranges;
      node_list . append ( node );
    }
    morse [ "nodes" ] = node_list;
    morse [ "morse_sets_hash" ] = prov . morse_sets_hash;
    morse [ "adaptive_unreduced_edge_hash" ] =
      prov . adaptive_unreduced_edge_hash;
    morse [ "combined_hash" ] = prov . combined_hash;
    out [ "morse_graph" ] = morse;
  } else {
    out [ "model" ] = py::none ();
    out [ "verification" ] = py::none ();
    out [ "morse_graph" ] = py::none ();
  }

  py::list sources;
  for ( uint64_t v = 0; v < core . num_vertices (); ++ v ) {
    MorseSetReachabilitySourceStats const& stats = core . source_stats ( v );
    py::dict source;
    source [ "source" ] = v;
    source [ "frontier_exhausted" ] = stats . frontier_exhausted;
    source [ "stop_reason" ] = stop_reason_name ( stats . stop_reason );
    py::list reached;
    for ( uint64_t w = 0; w < core . num_vertices (); ++ w ) {
      if ( core . status ( v, w ) == MorseSetReachabilityStatus::REACHABLE ) {
        reached . append ( w );
      }
    }
    source [ "reached_targets" ] = reached;
    source [ "fixed_seed_count" ] = stats . fixed_seed_count;
    source [ "visited_grid_elements" ] = stats . visited_grid_elements;
    if ( stats . frontier_exhausted ) {
      source [ "closure_size" ] = stats . visited_grid_elements;
    } else {
      source [ "closure_size" ] = py::none ();
    }
    source [ "closure_size_lower_bound" ] = stats . visited_grid_elements;
    source [ "grid_elements_expanded" ] = stats . grid_elements_expanded;
    source [ "adjacencies_examined" ] = stats . adjacencies_examined;
    source [ "map_evaluations_attempted" ] = stats . map_evaluations_attempted;
    source [ "map_batches_attempted" ] = stats . map_batches_attempted;
    source [ "visited_hash" ] = stats . visited_hash;
    source [ "frontier_count" ] = stats . frontier_count;
    source [ "frontier_hash" ] = stats . frontier_hash;
    source [ "error" ] = error_to_python ( stats );
    sources . append ( source );
  }
  out [ "sources" ] = sources;

  py::dict relation;
  relation [ "completed" ] = core . complete ();
  relation [ "diagnostic" ] = diagnostic_name ( core . diagnostics () );
  relation [ "certified_unreduced_edge_hash" ] =
    core . certified_unreduced_edge_hash ();
  std::string reduced_hash = core . reduced_edge_hash ();
  if ( reduced_hash . empty () ) {
    relation [ "reduced_edge_hash" ] = py::none ();
  } else {
    relation [ "reduced_edge_hash" ] = reduced_hash;
  }
  relation [ "coalescing_groups" ] = core . coalescing_groups ();
  relation [ "nontransitive_witnesses" ] = core . nontransitive_witnesses ();
  relation [ "absent_adaptive_edges" ] = core . absent_adaptive_edges ();
  relation [ "retained_adaptive_edges" ] = core . retained_adaptive_edges ();
  out [ "relation" ] = relation;

  return out;
}

} // namespace msr_detail

inline void
MorseSetReachabilityBinding ( py::module & m ) {
  py::register_exception<IncompleteMorseSetReachability> (
    m, "IncompleteMorseSetReachability" );
  py::register_exception<MorseSetCoalescingRequired> (
    m, "MorseSetCoalescingRequired" );
  py::register_exception<MorseSetSplittingRequired> (
    m, "MorseSetSplittingRequired" );

  py::enum_<MorseSetReachabilityStatus> ( m, "MorseSetReachabilityStatus" )
    . value ( "REACHABLE", MorseSetReachabilityStatus::REACHABLE )
    . value ( "NOT_REACHABLE", MorseSetReachabilityStatus::NOT_REACHABLE )
    . value ( "INCOMPLETE", MorseSetReachabilityStatus::INCOMPLETE );

  py::enum_<MorseSetReachabilityStopReason> (
      m, "MorseSetReachabilityStopReason" )
    . value ( "NONE", MorseSetReachabilityStopReason::NONE )
    . value ( "MAX_VISITED_GRID_ELEMENTS",
              MorseSetReachabilityStopReason::MAX_VISITED_GRID_ELEMENTS )
    . value ( "MAX_ADJACENCIES_EXAMINED",
              MorseSetReachabilityStopReason::MAX_ADJACENCIES_EXAMINED )
    . value ( "MAP_ERROR", MorseSetReachabilityStopReason::MAP_ERROR )
    . value ( "COVER_ERROR", MorseSetReachabilityStopReason::COVER_ERROR )
    . value ( "CANCELLED", MorseSetReachabilityStopReason::CANCELLED );

  py::enum_<MorseSetRelationDiagnostic> ( m, "MorseSetRelationDiagnostic" )
    . value ( "VALID_PARTIAL_ORDER",
              MorseSetRelationDiagnostic::VALID_PARTIAL_ORDER )
    . value ( "INCOMPLETE", MorseSetRelationDiagnostic::INCOMPLETE )
    . value ( "COALESCING_REQUIRED",
              MorseSetRelationDiagnostic::COALESCING_REQUIRED )
    . value ( "MORSE_SET_SPLITTING_REQUIRED",
              MorseSetRelationDiagnostic::MORSE_SET_SPLITTING_REQUIRED );

  py::class_<MorseSetReachabilityCheckpoint,
             std::shared_ptr<MorseSetReachabilityCheckpoint>> (
      m, "MorseSetReachabilityCheckpoint" )
    . def_readonly ( "schema_version",
                     &MorseSetReachabilityCheckpoint::schema_version )
    . def_readonly ( "algorithm_version",
                     &MorseSetReachabilityCheckpoint::algorithm_version )
    . def_readonly ( "map_fingerprint",
                     &MorseSetReachabilityCheckpoint::map_fingerprint )
    . def_readonly ( "phase_subdiv",
                     &MorseSetReachabilityCheckpoint::phase_subdiv )
    . def_readonly ( "morse_sets_hash",
                     &MorseSetReachabilityCheckpoint::morse_sets_hash )
    . def_readonly ( "adaptive_edges_hash",
                     &MorseSetReachabilityCheckpoint::adaptive_edges_hash )
    . def_readonly ( "num_vertices",
                     &MorseSetReachabilityCheckpoint::num_vertices )
    . def_readonly ( "payload_checksum",
                     &MorseSetReachabilityCheckpoint::payload_checksum )
    . def ( "to_dict",
            [] ( MorseSetReachabilityCheckpoint const& cp ) {
              return msr_detail::checkpoint_to_python ( cp );
            } )
    . def_static ( "from_dict",
            [] ( py::dict const& data ) {
              return std::shared_ptr<MorseSetReachabilityCheckpoint> (
                new MorseSetReachabilityCheckpoint (
                  msr_detail::checkpoint_from_python ( data ) ) );
            } )
    . def ( py::pickle (
        [] ( MorseSetReachabilityCheckpoint const& cp ) {
          return msr_detail::checkpoint_to_python ( cp );
        },
        [] ( py::dict const& data ) {
          return std::shared_ptr<MorseSetReachabilityCheckpoint> (
            new MorseSetReachabilityCheckpoint (
              msr_detail::checkpoint_from_python ( data ) ) );
        } ) );

  py::class_<PyMorseSetReachabilityResult,
             std::shared_ptr<PyMorseSetReachabilityResult>> (
      m, "MorseSetReachabilityResult" )
    . def ( "num_vertices",
            [] ( PyMorseSetReachabilityResult const& r ) {
              return r . core . num_vertices ();
            } )
    . def ( "status",
            [] ( PyMorseSetReachabilityResult const& r,
                 uint64_t source, uint64_t target ) {
              return r . core . status ( source, target );
            },
            py::arg ( "source" ), py::arg ( "target" ) )
    . def ( "adjacencies_unreduced",
            [] ( PyMorseSetReachabilityResult const& r, uint64_t source ) {
              return r . core . adjacencies_unreduced ( source );
            },
            py::arg ( "source" ) )
    . def ( "adjacencies",
            [] ( PyMorseSetReachabilityResult const& r, uint64_t source ) {
              return r . core . adjacencies ( source );
            },
            py::arg ( "source" ) )
    . def ( "complete",
            [] ( PyMorseSetReachabilityResult const& r ) {
              return r . core . complete ();
            } )
    . def ( "frontier_exhausted",
            [] ( PyMorseSetReachabilityResult const& r, uint64_t source ) {
              return r . core . frontier_exhausted ( source );
            },
            py::arg ( "source" ) )
    . def ( "visited_grid_elements",
            [] ( PyMorseSetReachabilityResult const& r, uint64_t source ) {
              return r . core . visited_grid_elements ( source );
            },
            py::arg ( "source" ) )
    . def ( "grid_elements_expanded",
            [] ( PyMorseSetReachabilityResult const& r, uint64_t source ) {
              return r . core . grid_elements_expanded ( source );
            },
            py::arg ( "source" ) )
    . def ( "adjacencies_examined",
            [] ( PyMorseSetReachabilityResult const& r, uint64_t source ) {
              return r . core . adjacencies_examined ( source );
            },
            py::arg ( "source" ) )
    . def ( "map_evaluations_attempted",
            [] ( PyMorseSetReachabilityResult const& r, uint64_t source ) {
              return r . core . map_evaluations_attempted ( source );
            },
            py::arg ( "source" ) )
    . def ( "map_batches_attempted",
            [] ( PyMorseSetReachabilityResult const& r, uint64_t source ) {
              return r . core . map_batches_attempted ( source );
            },
            py::arg ( "source" ) )
    . def ( "stop_reason",
            [] ( PyMorseSetReachabilityResult const& r, uint64_t source ) {
              return r . core . stop_reason ( source );
            },
            py::arg ( "source" ) )
    . def ( "error",
            [] ( PyMorseSetReachabilityResult const& r, uint64_t source ) {
              return msr_detail::error_to_python (
                r . core . source_stats ( source ) );
            },
            py::arg ( "source" ) )
    . def ( "coalescing_required",
            [] ( PyMorseSetReachabilityResult const& r ) {
              return r . core . coalescing_required ();
            } )
    . def ( "coalescing_groups",
            [] ( PyMorseSetReachabilityResult const& r ) {
              return r . core . coalescing_groups ();
            } )
    . def ( "nontransitive_witnesses",
            [] ( PyMorseSetReachabilityResult const& r ) {
              return r . core . nontransitive_witnesses ();
            } )
    . def ( "diagnostics",
            [] ( PyMorseSetReachabilityResult const& r ) {
              return r . core . diagnostics ();
            } )
    . def ( "absent_adaptive_edges",
            [] ( PyMorseSetReachabilityResult const& r ) {
              return r . core . absent_adaptive_edges ();
            } )
    . def ( "retained_adaptive_edges",
            [] ( PyMorseSetReachabilityResult const& r ) {
              return r . core . retained_adaptive_edges ();
            } )
    . def ( "provenance",
            [] ( PyMorseSetReachabilityResult const& r ) {
              return msr_detail::provenance_to_python ( r );
            } )
    . def ( "checkpoint",
            [] ( PyMorseSetReachabilityResult const& r ) {
              return std::shared_ptr<MorseSetReachabilityCheckpoint> (
                new MorseSetReachabilityCheckpoint ( * r . core . checkpoint () ) );
            } );

  m . def (
    "ComputeMorseSetReachability",
    [] ( std::shared_ptr<Model> model,
         std::shared_ptr<MorseGraph> morse_graph,
         uint32_t phase_subdiv,
         py::object max_visited_grid_elements,
         py::object max_adjacencies_examined,
         uint64_t batch_size,
         py::object map_fingerprint,
         py::object resume_from ) {
      MorseSetReachabilityOptions options;
      options . max_visited_grid_elements =
        msr_detail::optional_limit_from_python (
          max_visited_grid_elements, "max_visited_grid_elements" );
      options . max_adjacencies_examined =
        msr_detail::optional_limit_from_python (
          max_adjacencies_examined, "max_adjacencies_examined" );
      options . batch_size = batch_size;
      if ( not map_fingerprint . is_none () ) {
        options . map_fingerprint = map_fingerprint . cast<std::string> ();
      }
      if ( not resume_from . is_none () ) {
        options . resume_from =
          resume_from . cast<std::shared_ptr<MorseSetReachabilityCheckpoint>> ();
      }
      std::shared_ptr<PyMorseSetReachabilityResult> wrapper (
        new PyMorseSetReachabilityResult () );
      wrapper -> provenance . reset ( new MorseSetReachabilityProvenance () );
      wrapper -> core = ComputeMorseSetReachability (
        * model, * morse_graph, phase_subdiv, options,
        wrapper -> provenance . get () );
      return wrapper;
    },
    py::arg ( "model" ),
    py::arg ( "morse_graph" ),
    py::kw_only (),
    py::arg ( "phase_subdiv" ),
    py::arg ( "max_visited_grid_elements" ) = py::none (),
    py::arg ( "max_adjacencies_examined" ) = py::none (),
    py::arg ( "batch_size" ) = 4096,
    py::arg ( "map_fingerprint" ) = py::none (),
    py::arg ( "resume_from" ) = py::none () );

  // Core-level entry point over a deterministic in-memory adjacency,
  // used by the executable test plan (graph-relation tests).
  m . def (
    "_ComputeMorseSetReachabilityCoreInMemory",
    [] ( std::map<uint64_t, std::vector<uint64_t>> const& adjacency,
         std::vector<std::vector<uint64_t>> const& morse_set_cells,
         py::object max_visited_grid_elements,
         py::object max_adjacencies_examined,
         uint64_t batch_size,
         py::object map_fingerprint,
         py::object resume_from,
         py::object adaptive_edges ) {
      MorseSetReachabilityOptions options;
      options . max_visited_grid_elements =
        msr_detail::optional_limit_from_python (
          max_visited_grid_elements, "max_visited_grid_elements" );
      options . max_adjacencies_examined =
        msr_detail::optional_limit_from_python (
          max_adjacencies_examined, "max_adjacencies_examined" );
      options . batch_size = batch_size;
      if ( not map_fingerprint . is_none () ) {
        options . map_fingerprint = map_fingerprint . cast<std::string> ();
      }
      if ( not resume_from . is_none () ) {
        options . resume_from =
          resume_from . cast<std::shared_ptr<MorseSetReachabilityCheckpoint>> ();
      }

      std::vector<MorseSetFixedRanges> morse_sets ( morse_set_cells . size () );
      for ( size_t v = 0; v < morse_set_cells . size (); ++ v ) {
        std::vector<uint64_t> cells = morse_set_cells [ v ];
        std::sort ( cells . begin (), cells . end () );
        cells . erase ( std::unique ( cells . begin (), cells . end () ),
                        cells . end () );
        for ( size_t i = 0; i < cells . size (); ++ i ) {
          if ( not morse_sets [ v ] . ranges . empty () and
               cells [ i ] == morse_sets [ v ] . ranges . back () . second ) {
            ++ morse_sets [ v ] . ranges . back () . second;
          } else {
            morse_sets [ v ] . ranges . push_back (
              std::make_pair ( cells [ i ], cells [ i ] + 1 ) );
          }
        }
      }

      std::vector<std::pair<uint64_t, uint64_t>> edges;
      if ( not adaptive_edges . is_none () ) {
        edges = adaptive_edges
          . cast<std::vector<std::pair<uint64_t, uint64_t>>> ();
      }
      options . adaptive_edges_hash = msr_detail::hash_adaptive_edges ( edges );

      InMemoryAdjacencyProvider provider ( adjacency );
      std::shared_ptr<PyMorseSetReachabilityResult> wrapper (
        new PyMorseSetReachabilityResult () );
      wrapper -> core =
        ComputeMorseSetReachabilityCore ( morse_sets, provider, options );
      wrapper -> core . set_adaptive_edges ( edges );
      return wrapper;
    },
    py::arg ( "adjacency" ),
    py::arg ( "morse_sets" ),
    py::kw_only (),
    py::arg ( "max_visited_grid_elements" ) = py::none (),
    py::arg ( "max_adjacencies_examined" ) = py::none (),
    py::arg ( "batch_size" ) = 4096,
    py::arg ( "map_fingerprint" ) = py::none (),
    py::arg ( "resume_from" ) = py::none (),
    py::arg ( "adaptive_edges" ) = py::none () );

  // Test helper: build a synthetic adaptive MorseGraph from explicit
  // root-to-leaf branch paths per Morse vertex.
  m . def (
    "_BuildTestMorseGraph",
    [] ( std::vector<double> const& lower_bounds,
         std::vector<double> const& upper_bounds,
         std::vector<bool> const& periodic,
         std::vector<std::vector<std::string>> const& morse_set_paths,
         std::vector<std::pair<uint64_t, uint64_t>> const& edges ) {
      RectGeo bounds ( lower_bounds . size (), lower_bounds, upper_bounds );
      std::vector<std::string> all_paths;
      for ( size_t v = 0; v < morse_set_paths . size (); ++ v ) {
        all_paths . insert ( all_paths . end (),
                             morse_set_paths [ v ] . begin (),
                             morse_set_paths [ v ] . end () );
      }
      std::shared_ptr<MorseGraph> morse_graph ( new MorseGraph (
        msr_detail::build_treegrid_from_paths ( bounds, periodic,
                                                all_paths ) ) );
      for ( size_t v = 0; v < morse_set_paths . size (); ++ v ) {
        MorseGraph::Vertex vertex = morse_graph -> AddVertex ();
        morse_graph -> grid ( vertex ) =
          msr_detail::build_treegrid_from_paths ( bounds, periodic,
                                                  morse_set_paths [ v ] );
      }
      for ( size_t i = 0; i < edges . size (); ++ i ) {
        morse_graph -> AddEdge (
          static_cast<MorseGraph::Vertex> ( edges [ i ] . first ),
          static_cast<MorseGraph::Vertex> ( edges [ i ] . second ) );
      }
      return morse_graph;
    },
    py::arg ( "lower_bounds" ),
    py::arg ( "upper_bounds" ),
    py::arg ( "periodic" ),
    py::arg ( "morse_set_paths" ),
    py::arg ( "edges" ) );
}

#endif
