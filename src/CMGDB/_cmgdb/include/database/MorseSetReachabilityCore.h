// MorseSetReachabilityCore.h
//
// Fixed-subdivision Morse-set reachability verification: core search.
//
// This header is deliberately independent of CMGDB geometry, grids and
// Python. It operates on abstract fixed-subdivision GridElements
// (uint64_t path codes) supplied as canonical sorted half-open ranges,
// and on an abstract adjacency provider. The CMGDB adapter lives in
// MorseSetReachability.h.

#ifndef CMDB_MORSE_SET_REACHABILITY_CORE_H
#define CMDB_MORSE_SET_REACHABILITY_CORE_H

#include <stdint.h>
#include <algorithm>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

typedef uint64_t FixedGridElement;

enum class MorseSetReachabilityStatus : uint8_t {
  REACHABLE = 0,
  NOT_REACHABLE = 1,
  INCOMPLETE = 2
};

enum class MorseSetReachabilityStopReason : uint8_t {
  NONE = 0,
  MAX_VISITED_GRID_ELEMENTS = 1,
  MAX_ADJACENCIES_EXAMINED = 2,
  MAP_ERROR = 3,
  COVER_ERROR = 4,
  CANCELLED = 5
};

enum class MorseSetRelationDiagnostic : uint8_t {
  VALID_PARTIAL_ORDER = 0,
  INCOMPLETE = 1,
  COALESCING_REQUIRED = 2,
  MORSE_SET_SPLITTING_REQUIRED = 3
};

/// Exceptions raised by MorseSetReachabilityResult::adjacencies when a
/// transitive reduction over the original labels is unavailable.
struct IncompleteMorseSetReachability : public std::runtime_error {
  explicit IncompleteMorseSetReachability ( std::string const& message )
    : std::runtime_error ( message ) {}
};

struct MorseSetCoalescingRequired : public std::runtime_error {
  explicit MorseSetCoalescingRequired ( std::string const& message )
    : std::runtime_error ( message ) {}
};

struct MorseSetSplittingRequired : public std::runtime_error {
  explicit MorseSetSplittingRequired ( std::string const& message )
    : std::runtime_error ( message ) {}
};

/// Exceptions thrown by adjacency providers to classify runtime failures.
struct MorseSetReachabilityMapError : public std::runtime_error {
  explicit MorseSetReachabilityMapError ( std::string const& message )
    : std::runtime_error ( message ) {}
};

struct MorseSetReachabilityCoverError : public std::runtime_error {
  explicit MorseSetReachabilityCoverError ( std::string const& message )
    : std::runtime_error ( message ) {}
};

namespace msr_detail {

/// 64-bit FNV-1a accumulator used for all deterministic hashes/checksums.
class Hasher {
public:
  Hasher ( void ) : state_ ( 14695981039346656037ull ) {}
  void byte ( uint8_t b ) {
    state_ ^= static_cast<uint64_t> ( b );
    state_ *= 1099511628211ull;
  }
  void u64 ( uint64_t value ) {
    for ( int i = 0; i < 8; ++ i ) byte ( static_cast<uint8_t> ( value >> (8 * i) ) );
  }
  void str ( std::string const& s ) {
    u64 ( s . size () );
    for ( size_t i = 0; i < s . size (); ++ i ) byte ( static_cast<uint8_t> ( s [ i ] ) );
  }
  uint64_t value ( void ) const { return state_; }
  std::string hex ( void ) const {
    static const char digits [] = "0123456789abcdef";
    std::string out ( 16, '0' );
    for ( int i = 0; i < 16; ++ i ) {
      out [ 15 - i ] = digits [ ( state_ >> (4 * i) ) & 0xf ];
    }
    return out;
  }
private:
  uint64_t state_;
};

} // namespace msr_detail

/// Canonical fixed-subdivision representation of one Morse set:
/// sorted, disjoint, half-open ranges of FixedGridElements.
struct MorseSetFixedRanges {
  std::vector<std::pair<FixedGridElement, FixedGridElement>> ranges;

  uint64_t element_count ( void ) const {
    uint64_t count = 0;
    for ( size_t i = 0; i < ranges . size (); ++ i ) {
      count += ranges [ i ] . second - ranges [ i ] . first;
    }
    return count;
  }

  std::string hash ( void ) const {
    msr_detail::Hasher h;
    h . u64 ( ranges . size () );
    for ( size_t i = 0; i < ranges . size (); ++ i ) {
      h . u64 ( ranges [ i ] . first );
      h . u64 ( ranges [ i ] . second );
    }
    return h . hex ();
  }
};

/// Abstract adjacency provider over the conceptual fixed-subdivision grid.
///
/// Contract: for each source (in the given order) the provider yields that
/// source's complete adjacency, sorted and unique, via emit(batch_index,
/// target). If emit returns false the provider must abort immediately
/// (remaining targets and remaining sources are dropped). After a source's
/// adjacency is fully consumed the provider calls source_complete(batch_index).
/// Runtime failures are reported by throwing MorseSetReachabilityMapError or
/// MorseSetReachabilityCoverError.
class FixedSubdivisionAdjacencyProvider {
public:
  typedef std::function<bool(size_t, FixedGridElement)> EmitFn;
  typedef std::function<void(size_t)> CompleteFn;

  virtual ~FixedSubdivisionAdjacencyProvider ( void ) {}

  virtual void adjacencies_batch ( std::vector<FixedGridElement> const& sources,
                                   EmitFn const& emit,
                                   CompleteFn const& source_complete ) = 0;

  /// Cumulative counters (monotone over the provider's lifetime).
  virtual uint64_t map_evaluations_attempted ( void ) const { return 0; }
  virtual uint64_t map_batches_attempted ( void ) const { return 0; }
};

/// Deterministic in-memory provider for core-level tests.
class InMemoryAdjacencyProvider : public FixedSubdivisionAdjacencyProvider {
public:
  explicit InMemoryAdjacencyProvider (
      std::map<FixedGridElement, std::vector<FixedGridElement>> const& adjacency )
    : adjacency_ ( adjacency ) {
    for ( std::map<FixedGridElement, std::vector<FixedGridElement>>::iterator
            it = adjacency_ . begin (); it != adjacency_ . end (); ++ it ) {
      std::sort ( it -> second . begin (), it -> second . end () );
      it -> second . erase (
        std::unique ( it -> second . begin (), it -> second . end () ),
        it -> second . end () );
    }
  }

  virtual void adjacencies_batch ( std::vector<FixedGridElement> const& sources,
                                   EmitFn const& emit,
                                   CompleteFn const& source_complete ) {
    for ( size_t i = 0; i < sources . size (); ++ i ) {
      std::map<FixedGridElement, std::vector<FixedGridElement>>::const_iterator
        found = adjacency_ . find ( sources [ i ] );
      if ( found != adjacency_ . end () ) {
        for ( size_t j = 0; j < found -> second . size (); ++ j ) {
          if ( not emit ( i, found -> second [ j ] ) ) return;
        }
      }
      source_complete ( i );
    }
  }

private:
  std::map<FixedGridElement, std::vector<FixedGridElement>> adjacency_;
};

/// Per-source partial traversal state retained for resumable checkpoints.
struct MorseSetReachabilityPartialState {
  uint64_t seed_range_index;
  uint64_t seed_offset;
  std::vector<FixedGridElement> visited_sorted;
  std::vector<FixedGridElement> frontier;
  std::vector<uint8_t> reached_targets;

  MorseSetReachabilityPartialState ( void )
    : seed_range_index ( 0 ), seed_offset ( 0 ) {}
};

/// Per-source counters and outcome.
struct MorseSetReachabilitySourceStats {
  bool frontier_exhausted;
  MorseSetReachabilityStopReason stop_reason;
  std::string error_category;   // empty | "map" | "cover"
  std::string error_type;
  std::string error_message;
  bool has_error_element;
  FixedGridElement error_element;
  uint64_t fixed_seed_count;
  uint64_t visited_grid_elements;
  uint64_t grid_elements_expanded;
  uint64_t adjacencies_examined;
  uint64_t map_evaluations_attempted;
  uint64_t map_batches_attempted;
  uint64_t frontier_count;
  std::string visited_hash;
  std::string frontier_hash;

  MorseSetReachabilitySourceStats ( void )
    : frontier_exhausted ( false ),
      stop_reason ( MorseSetReachabilityStopReason::NONE ),
      has_error_element ( false ),
      error_element ( 0 ),
      fixed_seed_count ( 0 ),
      visited_grid_elements ( 0 ),
      grid_elements_expanded ( 0 ),
      adjacencies_examined ( 0 ),
      map_evaluations_attempted ( 0 ),
      map_batches_attempted ( 0 ),
      frontier_count ( 0 ) {}
};

/// Resumable checkpoint of a (possibly partial) computation.
struct MorseSetReachabilityCheckpoint {
  uint32_t schema_version;               // 1
  std::string algorithm_version;         // "fifo_sorted_streaming_v1"
  std::string configuration_sha;         // adapter-supplied model/config hash
  std::string map_fingerprint;           // caller-supplied map identity
  uint32_t phase_subdiv;
  std::string morse_sets_hash;
  std::string adaptive_edges_hash;
  uint64_t num_vertices;
  std::vector<uint8_t> status;           // m*m row-major, finalized sources only
  std::vector<uint8_t> source_finalized; // 1 if row is final (exhausted or error)
  std::vector<MorseSetReachabilitySourceStats> stats;
  std::map<uint64_t, MorseSetReachabilityPartialState> partial_states;
  std::string payload_checksum;

  MorseSetReachabilityCheckpoint ( void )
    : schema_version ( 1 ),
      algorithm_version ( "fifo_sorted_streaming_v1" ),
      phase_subdiv ( 0 ),
      num_vertices ( 0 ) {}

  std::string compute_checksum ( void ) const {
    msr_detail::Hasher h;
    h . u64 ( schema_version );
    h . str ( algorithm_version );
    h . str ( configuration_sha );
    h . str ( map_fingerprint );
    h . u64 ( phase_subdiv );
    h . str ( morse_sets_hash );
    h . str ( adaptive_edges_hash );
    h . u64 ( num_vertices );
    h . u64 ( status . size () );
    for ( size_t i = 0; i < status . size (); ++ i ) h . byte ( status [ i ] );
    h . u64 ( source_finalized . size () );
    for ( size_t i = 0; i < source_finalized . size (); ++ i ) {
      h . byte ( source_finalized [ i ] );
    }
    h . u64 ( stats . size () );
    for ( size_t i = 0; i < stats . size (); ++ i ) {
      MorseSetReachabilitySourceStats const& s = stats [ i ];
      h . byte ( s . frontier_exhausted ? 1 : 0 );
      h . byte ( static_cast<uint8_t> ( s . stop_reason ) );
      h . str ( s . error_category );
      h . str ( s . error_message );
      h . u64 ( s . fixed_seed_count );
      h . u64 ( s . visited_grid_elements );
      h . u64 ( s . grid_elements_expanded );
      h . u64 ( s . adjacencies_examined );
      h . u64 ( s . map_evaluations_attempted );
      h . u64 ( s . map_batches_attempted );
      h . u64 ( s . frontier_count );
    }
    h . u64 ( partial_states . size () );
    for ( std::map<uint64_t, MorseSetReachabilityPartialState>::const_iterator
            it = partial_states . begin (); it != partial_states . end (); ++ it ) {
      h . u64 ( it -> first );
      h . u64 ( it -> second . seed_range_index );
      h . u64 ( it -> second . seed_offset );
      h . u64 ( it -> second . visited_sorted . size () );
      for ( size_t i = 0; i < it -> second . visited_sorted . size (); ++ i ) {
        h . u64 ( it -> second . visited_sorted [ i ] );
      }
      h . u64 ( it -> second . frontier . size () );
      for ( size_t i = 0; i < it -> second . frontier . size (); ++ i ) {
        h . u64 ( it -> second . frontier [ i ] );
      }
      h . u64 ( it -> second . reached_targets . size () );
      for ( size_t i = 0; i < it -> second . reached_targets . size (); ++ i ) {
        h . byte ( it -> second . reached_targets [ i ] );
      }
    }
    return h . hex ();
  }

  void seal ( void ) { payload_checksum = compute_checksum (); }

  void validate_checksum ( void ) const {
    if ( payload_checksum != compute_checksum () ) {
      throw std::invalid_argument (
        "MorseSetReachabilityCheckpoint payload checksum mismatch" );
    }
  }
};

struct MorseSetReachabilityOptions {
  static const uint64_t NO_LIMIT = ~ static_cast<uint64_t> ( 0 );

  uint64_t max_visited_grid_elements;
  uint64_t max_adjacencies_examined;
  uint64_t batch_size;
  std::string map_fingerprint;
  std::shared_ptr<const MorseSetReachabilityCheckpoint> resume_from;

  // Context recorded into checkpoints and validated on resume; the
  // adapter fills these, the core treats them as opaque identity.
  uint32_t phase_subdiv;
  std::string configuration_sha;
  std::string adaptive_edges_hash;

  MorseSetReachabilityOptions ( void )
    : max_visited_grid_elements ( NO_LIMIT ),
      max_adjacencies_examined ( NO_LIMIT ),
      batch_size ( 4096 ),
      phase_subdiv ( 0 ) {}
};

/// Result of a fixed-subdivision Morse-set reachability verification.
/// This is reachability information, not a MorseGraph; the relation may
/// be incomplete, cyclic, or non-transitive.
class MorseSetReachabilityResult {
public:
  MorseSetReachabilityResult ( void ) : num_vertices_ ( 0 ), completed_ ( false ),
    coalescing_required_ ( false ),
    diagnostic_ ( MorseSetRelationDiagnostic::VALID_PARTIAL_ORDER ) {}

  uint64_t num_vertices ( void ) const { return num_vertices_; }

  MorseSetReachabilityStatus
  status ( uint64_t source, uint64_t target ) const {
    check_vertex ( source );
    check_vertex ( target );
    return static_cast<MorseSetReachabilityStatus> (
      status_ [ source * num_vertices_ + target ] );
  }

  /// Sorted off-diagonal targets certified REACHABLE from source. For an
  /// incomplete source this is a sound lower bound on the exact relation.
  std::vector<uint64_t> adjacencies_unreduced ( uint64_t source ) const {
    check_vertex ( source );
    std::vector<uint64_t> result;
    for ( uint64_t target = 0; target < num_vertices_; ++ target ) {
      if ( target == source ) continue;
      if ( status ( source, target ) == MorseSetReachabilityStatus::REACHABLE ) {
        result . push_back ( target );
      }
    }
    return result;
  }

  /// Transitive reduction over the original labels. Available only when
  /// every source closure exhausted and the exact strict relation is a
  /// partial order; otherwise raises the matching typed exception.
  std::vector<uint64_t> adjacencies ( uint64_t source ) const {
    check_vertex ( source );
    switch ( diagnostic_ ) {
      case MorseSetRelationDiagnostic::INCOMPLETE:
        throw IncompleteMorseSetReachability (
          "reduction unavailable: one or more source closures are incomplete" );
      case MorseSetRelationDiagnostic::COALESCING_REQUIRED:
        throw MorseSetCoalescingRequired (
          "reduction unavailable: mutually reachable Morse sets must be "
          "coalesced to represent a partial order" );
      case MorseSetRelationDiagnostic::MORSE_SET_SPLITTING_REQUIRED:
        throw MorseSetSplittingRequired (
          "reduction unavailable: the exact relation is not transitive; "
          "Morse-set splitting is required" );
      case MorseSetRelationDiagnostic::VALID_PARTIAL_ORDER:
        break;
    }
    return reduced_ [ source ];
  }

  bool complete ( void ) const { return completed_; }

  bool frontier_exhausted ( uint64_t source ) const {
    check_vertex ( source );
    return stats_ [ source ] . frontier_exhausted;
  }

  uint64_t visited_grid_elements ( uint64_t source ) const {
    check_vertex ( source );
    return stats_ [ source ] . visited_grid_elements;
  }

  uint64_t grid_elements_expanded ( uint64_t source ) const {
    check_vertex ( source );
    return stats_ [ source ] . grid_elements_expanded;
  }

  uint64_t adjacencies_examined ( uint64_t source ) const {
    check_vertex ( source );
    return stats_ [ source ] . adjacencies_examined;
  }

  uint64_t map_evaluations_attempted ( uint64_t source ) const {
    check_vertex ( source );
    return stats_ [ source ] . map_evaluations_attempted;
  }

  uint64_t map_batches_attempted ( uint64_t source ) const {
    check_vertex ( source );
    return stats_ [ source ] . map_batches_attempted;
  }

  MorseSetReachabilityStopReason stop_reason ( uint64_t source ) const {
    check_vertex ( source );
    return stats_ [ source ] . stop_reason;
  }

  MorseSetReachabilitySourceStats const& source_stats ( uint64_t source ) const {
    check_vertex ( source );
    return stats_ [ source ];
  }

  bool coalescing_required ( void ) const { return coalescing_required_; }

  std::vector<std::vector<uint64_t>> const& coalescing_groups ( void ) const {
    return coalescing_groups_;
  }

  std::vector<std::vector<uint64_t>> const& nontransitive_witnesses ( void ) const {
    return witnesses_;
  }

  MorseSetRelationDiagnostic diagnostics ( void ) const { return diagnostic_; }

  std::vector<std::pair<uint64_t, uint64_t>> const&
  absent_adaptive_edges ( void ) const { return absent_adaptive_edges_; }

  std::vector<std::pair<uint64_t, uint64_t>> const&
  retained_adaptive_edges ( void ) const { return retained_adaptive_edges_; }

  std::shared_ptr<const MorseSetReachabilityCheckpoint>
  checkpoint ( void ) const { return checkpoint_; }

  std::string certified_unreduced_edge_hash ( void ) const {
    msr_detail::Hasher h;
    uint64_t count = 0;
    for ( uint64_t v = 0; v < num_vertices_; ++ v ) {
      std::vector<uint64_t> targets = adjacencies_unreduced ( v );
      count += targets . size ();
    }
    h . u64 ( count );
    for ( uint64_t v = 0; v < num_vertices_; ++ v ) {
      std::vector<uint64_t> targets = adjacencies_unreduced ( v );
      for ( size_t i = 0; i < targets . size (); ++ i ) {
        h . u64 ( v );
        h . u64 ( targets [ i ] );
      }
    }
    return h . hex ();
  }

  std::string reduced_edge_hash ( void ) const {
    if ( diagnostic_ != MorseSetRelationDiagnostic::VALID_PARTIAL_ORDER ) {
      return std::string ();
    }
    msr_detail::Hasher h;
    uint64_t count = 0;
    for ( uint64_t v = 0; v < num_vertices_; ++ v ) count += reduced_ [ v ] . size ();
    h . u64 ( count );
    for ( uint64_t v = 0; v < num_vertices_; ++ v ) {
      for ( size_t i = 0; i < reduced_ [ v ] . size (); ++ i ) {
        h . u64 ( v );
        h . u64 ( reduced_ [ v ] [ i ] );
      }
    }
    return h . hex ();
  }

  // --- construction interface (used by the core computation and adapter) ---

  void initialize ( uint64_t num_vertices ) {
    num_vertices_ = num_vertices;
    status_ . assign ( num_vertices * num_vertices,
      static_cast<uint8_t> ( MorseSetReachabilityStatus::INCOMPLETE ) );
    stats_ . assign ( num_vertices, MorseSetReachabilitySourceStats () );
  }

  void set_status ( uint64_t source, uint64_t target,
                    MorseSetReachabilityStatus status ) {
    status_ [ source * num_vertices_ + target ] = static_cast<uint8_t> ( status );
  }

  MorseSetReachabilitySourceStats & mutable_stats ( uint64_t source ) {
    return stats_ [ source ];
  }

  std::vector<uint8_t> const& raw_status ( void ) const { return status_; }
  std::vector<uint8_t> & raw_status ( void ) { return status_; }
  std::vector<MorseSetReachabilitySourceStats> & raw_stats ( void ) { return stats_; }
  std::vector<MorseSetReachabilitySourceStats> const& raw_stats ( void ) const {
    return stats_;
  }

  void set_checkpoint ( std::shared_ptr<const MorseSetReachabilityCheckpoint> cp ) {
    checkpoint_ = cp;
  }

  void set_adaptive_edges (
      std::vector<std::pair<uint64_t, uint64_t>> const& edges ) {
    adaptive_edges_ = edges;
    absent_adaptive_edges_ . clear ();
    retained_adaptive_edges_ . clear ();
    for ( size_t i = 0; i < edges . size (); ++ i ) {
      uint64_t source = edges [ i ] . first;
      uint64_t target = edges [ i ] . second;
      if ( source >= num_vertices_ or target >= num_vertices_ ) {
        throw std::invalid_argument (
          "adaptive edge references a vertex outside the MorseGraph" );
      }
      MorseSetReachabilityStatus pair_status = status ( source, target );
      if ( pair_status == MorseSetReachabilityStatus::NOT_REACHABLE ) {
        absent_adaptive_edges_ . push_back ( edges [ i ] );
      } else if ( pair_status == MorseSetReachabilityStatus::REACHABLE ) {
        retained_adaptive_edges_ . push_back ( edges [ i ] );
      }
    }
    std::sort ( absent_adaptive_edges_ . begin (), absent_adaptive_edges_ . end () );
    std::sort ( retained_adaptive_edges_ . begin (),
                retained_adaptive_edges_ . end () );
  }

  /// Derive completed_, coalescing groups, non-transitivity witnesses,
  /// the diagnostic, and (when valid) the transitive reduction.
  void analyze_relation ( void ) {
    const uint64_t m = num_vertices_;
    completed_ = true;
    for ( uint64_t v = 0; v < m; ++ v ) {
      if ( not stats_ [ v ] . frontier_exhausted ) completed_ = false;
    }

    // Mutually reachable components: connected components of the
    // undirected graph of certified mutual pairs.
    coalescing_groups_ . clear ();
    coalescing_required_ = false;
    std::vector<uint64_t> component ( m );
    for ( uint64_t v = 0; v < m; ++ v ) component [ v ] = v;
    for ( uint64_t v = 0; v < m; ++ v ) {
      for ( uint64_t w = v + 1; w < m; ++ w ) {
        if ( is_reachable ( v, w ) and is_reachable ( w, v ) ) {
          uint64_t cv = find_root ( component, v );
          uint64_t cw = find_root ( component, w );
          if ( cv != cw ) component [ std::max ( cv, cw ) ] = std::min ( cv, cw );
        }
      }
    }
    std::map<uint64_t, std::vector<uint64_t>> groups;
    for ( uint64_t v = 0; v < m; ++ v ) {
      groups [ find_root ( component, v ) ] . push_back ( v );
    }
    for ( std::map<uint64_t, std::vector<uint64_t>>::const_iterator
            it = groups . begin (); it != groups . end (); ++ it ) {
      if ( it -> second . size () >= 2 ) {
        coalescing_required_ = true;
        coalescing_groups_ . push_back ( it -> second );
      }
    }

    // Non-transitivity witnesses over definitive statuses:
    // u ~> v REACHABLE, v ~> w REACHABLE, u ~> w NOT_REACHABLE.
    witnesses_ . clear ();
    for ( uint64_t u = 0; u < m; ++ u ) {
      for ( uint64_t v = 0; v < m; ++ v ) {
        if ( v == u or not is_reachable ( u, v ) ) continue;
        for ( uint64_t w = 0; w < m; ++ w ) {
          if ( w == v or w == u or not is_reachable ( v, w ) ) continue;
          if ( status ( u, w ) == MorseSetReachabilityStatus::NOT_REACHABLE ) {
            std::vector<uint64_t> witness;
            witness . push_back ( u );
            witness . push_back ( v );
            witness . push_back ( w );
            witnesses_ . push_back ( witness );
          }
        }
      }
    }

    if ( not completed_ ) {
      diagnostic_ = MorseSetRelationDiagnostic::INCOMPLETE;
    } else if ( coalescing_required_ ) {
      diagnostic_ = MorseSetRelationDiagnostic::COALESCING_REQUIRED;
    } else if ( not witnesses_ . empty () ) {
      diagnostic_ = MorseSetRelationDiagnostic::MORSE_SET_SPLITTING_REQUIRED;
    } else {
      diagnostic_ = MorseSetRelationDiagnostic::VALID_PARTIAL_ORDER;
    }

    reduced_ . assign ( m, std::vector<uint64_t> () );
    if ( diagnostic_ == MorseSetRelationDiagnostic::VALID_PARTIAL_ORDER ) {
      // R - R^2 over the strict certified relation, which here equals the
      // exact relation and is transitive and antisymmetric.
      for ( uint64_t u = 0; u < m; ++ u ) {
        for ( uint64_t w = 0; w < m; ++ w ) {
          if ( w == u or not is_reachable ( u, w ) ) continue;
          bool two_step = false;
          for ( uint64_t v = 0; v < m; ++ v ) {
            if ( v == u or v == w ) continue;
            if ( is_reachable ( u, v ) and is_reachable ( v, w ) ) {
              two_step = true;
              break;
            }
          }
          if ( not two_step ) reduced_ [ u ] . push_back ( w );
        }
      }
    }
  }

private:
  void check_vertex ( uint64_t vertex ) const {
    if ( vertex >= num_vertices_ ) {
      std::ostringstream message;
      message << "Morse vertex " << vertex << " is outside [0, "
              << num_vertices_ << ")";
      throw std::out_of_range ( message . str () );
    }
  }

  bool is_reachable ( uint64_t source, uint64_t target ) const {
    return status_ [ source * num_vertices_ + target ] ==
      static_cast<uint8_t> ( MorseSetReachabilityStatus::REACHABLE );
  }

  static uint64_t find_root ( std::vector<uint64_t> & component, uint64_t v ) {
    while ( component [ v ] != v ) {
      component [ v ] = component [ component [ v ] ];
      v = component [ v ];
    }
    return v;
  }

  uint64_t num_vertices_;
  std::vector<uint8_t> status_;
  std::vector<MorseSetReachabilitySourceStats> stats_;
  bool completed_;
  bool coalescing_required_;
  std::vector<std::vector<uint64_t>> coalescing_groups_;
  std::vector<std::vector<uint64_t>> witnesses_;
  MorseSetRelationDiagnostic diagnostic_;
  std::vector<std::vector<uint64_t>> reduced_;
  std::vector<std::pair<uint64_t, uint64_t>> adaptive_edges_;
  std::vector<std::pair<uint64_t, uint64_t>> absent_adaptive_edges_;
  std::vector<std::pair<uint64_t, uint64_t>> retained_adaptive_edges_;
  std::shared_ptr<const MorseSetReachabilityCheckpoint> checkpoint_;
};

namespace msr_detail {

/// Interval index mapping a FixedGridElement to its owning Morse vertex.
class TargetMembershipIndex {
public:
  explicit TargetMembershipIndex (
      std::vector<MorseSetFixedRanges> const& morse_sets ) {
    for ( size_t v = 0; v < morse_sets . size (); ++ v ) {
      for ( size_t i = 0; i < morse_sets [ v ] . ranges . size (); ++ i ) {
        Entry entry;
        entry . start = morse_sets [ v ] . ranges [ i ] . first;
        entry . end = morse_sets [ v ] . ranges [ i ] . second;
        entry . vertex = v;
        if ( entry . start < entry . end ) entries_ . push_back ( entry );
      }
    }
    std::sort ( entries_ . begin (), entries_ . end () );
    for ( size_t i = 1; i < entries_ . size (); ++ i ) {
      if ( entries_ [ i ] . start < entries_ [ i - 1 ] . end ) {
        throw std::invalid_argument (
          "Morse sets overlap at the requested subdivision; "
          "the input MorseGraph is invalid" );
      }
    }
  }

  /// Returns the owning vertex, or -1 if the element is in no Morse set.
  int64_t owner ( FixedGridElement element ) const {
    if ( entries_ . empty () ) return -1;
    size_t lo = 0;
    size_t hi = entries_ . size ();
    while ( lo < hi ) {
      size_t mid = ( lo + hi ) / 2;
      if ( entries_ [ mid ] . start <= element ) lo = mid + 1; else hi = mid;
    }
    if ( lo == 0 ) return -1;
    Entry const& candidate = entries_ [ lo - 1 ];
    if ( element < candidate . end ) {
      return static_cast<int64_t> ( candidate . vertex );
    }
    return -1;
  }

private:
  struct Entry {
    FixedGridElement start;
    FixedGridElement end;
    uint64_t vertex;
    bool operator < ( Entry const& other ) const {
      if ( start != other . start ) return start < other . start;
      if ( end != other . end ) return end < other . end;
      return vertex < other . vertex;
    }
  };
  std::vector<Entry> entries_;
};

inline std::string
hash_morse_sets ( std::vector<MorseSetFixedRanges> const& morse_sets ) {
  Hasher h;
  h . u64 ( morse_sets . size () );
  for ( size_t v = 0; v < morse_sets . size (); ++ v ) {
    h . str ( morse_sets [ v ] . hash () );
  }
  return h . hex ();
}

inline std::string
hash_element_sequence ( std::vector<FixedGridElement> const& elements ) {
  Hasher h;
  h . u64 ( elements . size () );
  for ( size_t i = 0; i < elements . size (); ++ i ) h . u64 ( elements [ i ] );
  return h . hex ();
}

/// Signals raised internally to unwind a source's traversal.
struct SourceStopSignal {
  MorseSetReachabilityStopReason reason;
  explicit SourceStopSignal ( MorseSetReachabilityStopReason r ) : reason ( r ) {}
};

} // namespace msr_detail

/// Core fixed-subdivision Morse-set reachability computation.
///
/// morse_sets[v].ranges must be sorted, disjoint, half-open ranges of
/// FixedGridElements; ranges of distinct vertices must not overlap.
inline MorseSetReachabilityResult
ComputeMorseSetReachabilityCore (
    std::vector<MorseSetFixedRanges> const& morse_sets,
    FixedSubdivisionAdjacencyProvider & adjacency_provider,
    MorseSetReachabilityOptions const& options = MorseSetReachabilityOptions () ) {

  const uint64_t m = morse_sets . size ();
  if ( options . batch_size == 0 ) {
    throw std::invalid_argument ( "batch_size must be positive" );
  }
  for ( size_t v = 0; v < morse_sets . size (); ++ v ) {
    std::vector<std::pair<FixedGridElement, FixedGridElement>> const& ranges =
      morse_sets [ v ] . ranges;
    for ( size_t i = 0; i < ranges . size (); ++ i ) {
      if ( ranges [ i ] . first >= ranges [ i ] . second ) {
        throw std::invalid_argument ( "Morse-set ranges must be nonempty" );
      }
      if ( i > 0 and ranges [ i ] . first < ranges [ i - 1 ] . second ) {
        throw std::invalid_argument (
          "Morse-set ranges must be sorted and disjoint" );
      }
    }
  }

  msr_detail::TargetMembershipIndex membership ( morse_sets );
  const std::string morse_sets_hash = msr_detail::hash_morse_sets ( morse_sets );

  // Validate any resume checkpoint before doing work.
  std::shared_ptr<const MorseSetReachabilityCheckpoint> resume =
    options . resume_from;
  if ( resume ) {
    resume -> validate_checksum ();
    if ( resume -> schema_version != 1 or
         resume -> algorithm_version != "fifo_sorted_streaming_v1" ) {
      throw std::invalid_argument (
        "checkpoint schema/algorithm version mismatch" );
    }
    if ( resume -> num_vertices != m or
         resume -> morse_sets_hash != morse_sets_hash ) {
      throw std::invalid_argument (
        "checkpoint does not match the supplied Morse sets" );
    }
    if ( resume -> phase_subdiv != options . phase_subdiv ) {
      throw std::invalid_argument (
        "checkpoint phase_subdiv does not match the requested phase_subdiv" );
    }
    if ( resume -> adaptive_edges_hash != options . adaptive_edges_hash ) {
      throw std::invalid_argument (
        "checkpoint does not match the supplied adaptive edges" );
    }
    if ( resume -> map_fingerprint != options . map_fingerprint ) {
      throw std::invalid_argument (
        "checkpoint map_fingerprint does not match; resuming requires the "
        "same caller-supplied map identity" );
    }
  }

  MorseSetReachabilityResult result;
  result . initialize ( m );

  std::shared_ptr<MorseSetReachabilityCheckpoint> checkpoint (
    new MorseSetReachabilityCheckpoint () );
  checkpoint -> configuration_sha = options . configuration_sha;
  checkpoint -> map_fingerprint = options . map_fingerprint;
  checkpoint -> phase_subdiv = options . phase_subdiv;
  checkpoint -> morse_sets_hash = morse_sets_hash;
  checkpoint -> adaptive_edges_hash = options . adaptive_edges_hash;
  checkpoint -> num_vertices = m;
  checkpoint -> status . assign ( m * m,
    static_cast<uint8_t> ( MorseSetReachabilityStatus::INCOMPLETE ) );
  checkpoint -> source_finalized . assign ( m, 0 );
  checkpoint -> stats . assign ( m, MorseSetReachabilitySourceStats () );

  for ( uint64_t source_vertex = 0; source_vertex < m; ++ source_vertex ) {
    MorseSetReachabilitySourceStats & stats =
      result . mutable_stats ( source_vertex );
    stats . fixed_seed_count = morse_sets [ source_vertex ] . element_count ();

    // Reuse a finalized row from the checkpoint when resuming.
    bool resumed_partial = false;
    uint64_t seed_range_index = 0;
    uint64_t seed_offset = 0;
    std::unordered_set<FixedGridElement> visited;
    std::deque<FixedGridElement> frontier;
    std::vector<uint8_t> reached ( m, 0 );

    if ( resume ) {
      if ( resume -> source_finalized [ source_vertex ] ) {
        for ( uint64_t target = 0; target < m; ++ target ) {
          result . set_status ( source_vertex, target,
            static_cast<MorseSetReachabilityStatus> (
              resume -> status [ source_vertex * m + target ] ) );
          checkpoint -> status [ source_vertex * m + target ] =
            resume -> status [ source_vertex * m + target ];
        }
        result . mutable_stats ( source_vertex ) =
          resume -> stats [ source_vertex ];
        checkpoint -> stats [ source_vertex ] = resume -> stats [ source_vertex ];
        checkpoint -> source_finalized [ source_vertex ] = 1;
        continue;
      }
      std::map<uint64_t, MorseSetReachabilityPartialState>::const_iterator
        partial = resume -> partial_states . find ( source_vertex );
      if ( partial != resume -> partial_states . end () ) {
        resumed_partial = true;
        seed_range_index = partial -> second . seed_range_index;
        seed_offset = partial -> second . seed_offset;
        visited . insert ( partial -> second . visited_sorted . begin (),
                           partial -> second . visited_sorted . end () );
        frontier . assign ( partial -> second . frontier . begin (),
                            partial -> second . frontier . end () );
        reached = partial -> second . reached_targets;
        stats = resume -> stats [ source_vertex ];
        stats . stop_reason = MorseSetReachabilityStopReason::NONE;
        stats . frontier_exhausted = false;
        stats . error_category . clear ();
        stats . error_type . clear ();
        stats . error_message . clear ();
        stats . has_error_element = false;
      }
    }

    (void) resumed_partial;
    const uint64_t provider_evals_base =
      adjacency_provider . map_evaluations_attempted ();
    const uint64_t provider_batches_base =
      adjacency_provider . map_batches_attempted ();
    const uint64_t stats_evals_base = stats . map_evaluations_attempted;
    const uint64_t stats_batches_base = stats . map_batches_attempted;

    MorseSetReachabilityStopReason stop_reason =
      MorseSetReachabilityStopReason::NONE;
    bool has_error_element = false;
    FixedGridElement error_element = 0;
    bool restart_element_known = false;
    FixedGridElement restart_element = 0;
    std::string error_category, error_type, error_message;

    // Admit a fixed element to the source-local visited set.
    // Throws SourceStopSignal when the visited budget is exhausted.
    struct Admit {
      std::unordered_set<FixedGridElement> & visited;
      std::deque<FixedGridElement> & frontier;
      MorseSetReachabilitySourceStats & stats;
      uint64_t limit;
      Admit ( std::unordered_set<FixedGridElement> & v,
              std::deque<FixedGridElement> & f,
              MorseSetReachabilitySourceStats & s,
              uint64_t l ) : visited ( v ), frontier ( f ), stats ( s ),
                             limit ( l ) {}
      void operator () ( FixedGridElement element ) {
        if ( visited . count ( element ) ) return;
        if ( stats . visited_grid_elements >= limit ) {
          throw msr_detail::SourceStopSignal (
            MorseSetReachabilityStopReason::MAX_VISITED_GRID_ELEMENTS );
        }
        visited . insert ( element );
        ++ stats . visited_grid_elements;
        frontier . push_back ( element );
      }
    } admit ( visited, frontier, stats,
              options . max_visited_grid_elements );

    try {
      // Phase 1: lazily enumerate and admit seeds in ascending order.
      std::vector<std::pair<FixedGridElement, FixedGridElement>> const& ranges =
        morse_sets [ source_vertex ] . ranges;
      while ( seed_range_index < ranges . size () ) {
        FixedGridElement seed =
          ranges [ seed_range_index ] . first + seed_offset;
        // An enumerated seed marks the source Morse set encountered
        // (zero-edge paths are allowed), before admission.
        reached [ source_vertex ] = 1;
        admit ( seed );
        ++ seed_offset;
        if ( ranges [ seed_range_index ] . first + seed_offset >=
             ranges [ seed_range_index ] . second ) {
          ++ seed_range_index;
          seed_offset = 0;
        }
      }

      // Phase 2: FIFO breadth-first closure in deterministic batches.
      while ( not frontier . empty () ) {
        std::vector<FixedGridElement> batch;
        const size_t batch_count = std::min (
          static_cast<size_t> ( options . batch_size ), frontier . size () );
        for ( size_t i = 0; i < batch_count; ++ i ) {
          batch . push_back ( frontier . front () );
          frontier . pop_front ();
        }

        size_t completed_in_batch = 0;
        size_t current_in_batch = 0;
        try {
          adjacency_provider . adjacencies_batch (
            batch,
            [&] ( size_t batch_index, FixedGridElement target ) -> bool {
              current_in_batch = batch_index;
              ++ stats . adjacencies_examined;
              int64_t owner = membership . owner ( target );
              if ( owner >= 0 ) reached [ owner ] = 1;
              if ( stats . adjacencies_examined >
                   options . max_adjacencies_examined ) {
                throw msr_detail::SourceStopSignal (
                  MorseSetReachabilityStopReason::MAX_ADJACENCIES_EXAMINED );
              }
              admit ( target );
              return true;
            },
            [&] ( size_t batch_index ) {
              current_in_batch = batch_index;
              completed_in_batch = batch_index + 1;
              ++ stats . grid_elements_expanded;
            } );
        } catch ( ... ) {
          // Reinsert unexpanded batch elements at the frontier head so a
          // checkpoint restart marker re-covers them from scratch.
          for ( size_t i = batch . size (); i > completed_in_batch; -- i ) {
            frontier . push_front ( batch [ i - 1 ] );
          }
          restart_element_known = true;
          restart_element = batch [ std::min ( current_in_batch,
                                               batch . size () - 1 ) ];
          throw;
        }
      }

      stats . frontier_exhausted = true;
    } catch ( msr_detail::SourceStopSignal const& signal ) {
      stop_reason = signal . reason;
    } catch ( MorseSetReachabilityMapError const& error ) {
      stop_reason = MorseSetReachabilityStopReason::MAP_ERROR;
      error_category = "map";
      error_type = "MorseSetReachabilityMapError";
      error_message = error . what ();
      has_error_element = restart_element_known;
      error_element = restart_element;
    } catch ( MorseSetReachabilityCoverError const& error ) {
      stop_reason = MorseSetReachabilityStopReason::COVER_ERROR;
      error_category = "cover";
      error_type = "MorseSetReachabilityCoverError";
      error_message = error . what ();
      has_error_element = restart_element_known;
      error_element = restart_element;
    }

    stats . stop_reason = stop_reason;
    stats . error_category = error_category;
    stats . error_type = error_type;
    stats . error_message = error_message;
    stats . has_error_element = has_error_element;
    stats . error_element = error_element;
    stats . frontier_count = frontier . size ();
    stats . map_evaluations_attempted = stats_evals_base +
      ( adjacency_provider . map_evaluations_attempted () - provider_evals_base );
    stats . map_batches_attempted = stats_batches_base +
      ( adjacency_provider . map_batches_attempted () - provider_batches_base );

    std::vector<FixedGridElement> visited_sorted ( visited . begin (),
                                                   visited . end () );
    std::sort ( visited_sorted . begin (), visited_sorted . end () );
    stats . visited_hash = msr_detail::hash_element_sequence ( visited_sorted );
    std::vector<FixedGridElement> frontier_vector ( frontier . begin (),
                                                    frontier . end () );
    stats . frontier_hash = msr_detail::hash_element_sequence ( frontier_vector );

    // Finalize the status row.
    for ( uint64_t target = 0; target < m; ++ target ) {
      MorseSetReachabilityStatus pair_status;
      if ( reached [ target ] ) {
        pair_status = MorseSetReachabilityStatus::REACHABLE;
      } else if ( stats . frontier_exhausted ) {
        pair_status = MorseSetReachabilityStatus::NOT_REACHABLE;
      } else {
        pair_status = MorseSetReachabilityStatus::INCOMPLETE;
      }
      result . set_status ( source_vertex, target, pair_status );
      checkpoint -> status [ source_vertex * m + target ] =
        static_cast<uint8_t> ( pair_status );
    }
    checkpoint -> stats [ source_vertex ] = stats;

    const bool finalized = stats . frontier_exhausted or
      stop_reason == MorseSetReachabilityStopReason::MAP_ERROR or
      stop_reason == MorseSetReachabilityStopReason::COVER_ERROR;
    checkpoint -> source_finalized [ source_vertex ] = finalized ? 1 : 0;
    if ( not stats . frontier_exhausted and not finalized ) {
      // Retain resumable state (limits only; errors restart the source).
      MorseSetReachabilityPartialState partial;
      partial . seed_range_index = seed_range_index;
      partial . seed_offset = seed_offset;
      partial . visited_sorted = visited_sorted;
      partial . frontier = frontier_vector;
      partial . reached_targets = reached;
      checkpoint -> partial_states [ source_vertex ] = partial;
    }
  }

  checkpoint -> seal ();
  result . set_checkpoint ( checkpoint );
  result . analyze_relation ();
  return result;
}

#endif
