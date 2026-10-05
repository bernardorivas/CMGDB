// MapGraph.h

#ifndef CMDP_MAPGRAPH_H
#define CMDP_MAPGRAPH_H

#include <cstdint>
#include <exception>
#include <vector>
#include <iterator>
#include <iostream>
#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
// #include <unistd.h>

#include "boost/unordered_map.hpp"
#include "boost/foreach.hpp"

#include "Grid.h"
#include "Map.h"
#include "RectGeo.h"

/// struct MapGraphOptions
///    Controls whether (and how) a MapGraph stores the transition graph.
///    With cache == true the full adjacency structure is computed once at
///    construction, in chunks of chunk_size rectangles, and stored in
///    compressed sparse row (CSR) form; subsequent adjacency queries never
///    evaluate the map again. max_cached_edges bounds that cache: as soon as
///    the edge count would exceed it, the cache is abandoned and the
///    MapGraph falls back to on-demand evaluation (0 means unlimited). It
///    does not bound a later, explicit MapGraph::build_cache.
///    reserve_edges controls the up-front reservation of the flat edge
///    array. 0 (the default) reserves twice the final edge count projected
///    from the chunks seen so far, which avoids the repeated-doubling
///    reallocation of multi-gigabyte edge arrays (a transient ~3x memory
///    peak on deep uniform grids); a positive value reserves exactly that
///    many edges instead. Reservation only engages once the projected edge
///    count reaches reserve_min_edges, so small graphs -- including the
///    per-Morse-set subgraphs built during adaptive runs -- never
///    over-allocate.
///    The environment variables read by build_cache (see cmgdb_detail
///    below) add opt-in hard limits, which raise instead of falling back,
///    and an up-front reservation hint.
struct MapGraphOptions {
  bool cache;
  uint64_t chunk_size;
  uint64_t max_cached_edges;
  uint64_t reserve_edges;
  uint64_t reserve_min_edges;
  MapGraphOptions ( void )
    : cache ( false ), chunk_size ( 65536 ), max_cached_edges ( 0 ),
      reserve_edges ( 0 ), reserve_min_edges ( uint64_t ( 1 ) << 24 ) {}
  explicit MapGraphOptions ( bool cache_,
                             uint64_t chunk_size_ = 65536,
                             uint64_t max_cached_edges_ = 0,
                             uint64_t reserve_edges_ = 0,
                             uint64_t reserve_min_edges_ = uint64_t ( 1 ) << 24 )
    : cache ( cache_ ), chunk_size ( chunk_size_ ),
      max_cached_edges ( max_cached_edges_ ),
      reserve_edges ( reserve_edges_ ),
      reserve_min_edges ( reserve_min_edges_ ) {}
};

/// class MapGraph
///    This class is used to created an object suitable for graph algorithms
///    given a grid and a map object. The transition graph is stored in CSR
///    form when the options ask for it (each rectangle's image is then
///    computed exactly once); otherwise "adjacencies" is computed on demand
///    in order to avoid storing the adjacency lists. The two-argument
///    constructor caches unless CMGDB_MAPGRAPH_CACHE=0.
class MapGraph {
public:
  // Typedefs
  typedef Grid::size_type size_type;
  typedef Grid::GridElement Vertex;

  /// AdjacencySpan
  ///    Non-owning view over the out-edges of a vertex. When the CSR cache
  ///    is present it points into the flat edge array; otherwise it points
  ///    into a scratch buffer that is overwritten by the next adjacency
  ///    query, so a span must be consumed before the next query is made.
  struct AdjacencySpan {
    typedef const Vertex * iterator;
    typedef const Vertex * const_iterator;
    const Vertex * begin_;
    const Vertex * end_;
    const Vertex * begin ( void ) const { return begin_; }
    const Vertex * end ( void ) const { return end_; }
    size_t size ( void ) const { return end_ - begin_; }
    bool empty ( void ) const { return begin_ == end_; }
  };
  /// AdjacencyView: the fork's name for AdjacencySpan.
  typedef AdjacencySpan AdjacencyView;

  // Constructor. Requires Grid and Map. Caches the transition graph unless
  // CMGDB_MAPGRAPH_CACHE=0.
  MapGraph ( std::shared_ptr<const Grid> grid,
             std::shared_ptr<const Map> f );

  MapGraph ( std::shared_ptr<const Grid> grid,
             std::shared_ptr<const Map> f,
             MapGraphOptions options );

  void initialize ( void );

  /// adjacencies
  ///   Return vector of Vertices which are out-edge adjacencies of input v
  std::vector<Vertex> adjacencies ( const Vertex & v ) const;

  /// adjacency_span
  ///   Return a non-owning view of the out-edge adjacencies of input v.
  ///   Avoids a copy when the transition graph is cached. See AdjacencySpan
  ///   for the lifetime rule in the uncached case.
  AdjacencySpan adjacency_span ( const Vertex & v ) const;

  /// adjacencies_view: the fork's name for adjacency_span.
  AdjacencyView adjacencies_view ( const Vertex & v ) const {
    return adjacency_span ( v );
  }

  /// num_vertices
  ///   Return number of vertices
  size_type num_vertices ( void ) const;

  /// has_cache
  ///   Return true if the transition graph is stored in the CSR cache.
  bool has_cache ( void ) const { return cached_; }

  /// num_cached_edges
  ///   Return the number of edges held in the CSR cache (0 if uncached).
  uint64_t num_cached_edges ( void ) const {
    return cached_ ? (uint64_t) csr_edges_ . size () : 0;
  }

  /// build_cache
  ///   Build the CSR transition-graph cache now (a no-op if it is already
  ///   built). Lets a lazy MapGraph -- e.g. the map_graph returned by
  ///   ComputeMorseGraph with cache_map_graph=False, or one whose cache was
  ///   abandoned at the options' max_cached_edges -- be upgraded to a cached
  ///   one after the fact, at the cost of one full (batched, if available)
  ///   pass of map evaluations over the grid. This explicit request is
  ///   bounded by its own max_cached_edges (0, the default, means
  ///   unlimited), not by the options' one: a graph with more edges throws
  ///   std::runtime_error and stays lazy. A call made while a build of this
  ///   graph is running (from another thread, or from the map itself)
  ///   throws std::runtime_error and leaves that build alone.
  void build_cache ( uint64_t max_cached_edges = 0 );

  /// validate_cached_csr
  ///   Validate the invariants required by the read-only NumPy CSR view.
  ///   Rows are canonical: targets are in range, strictly increasing, and
  ///   therefore duplicate-free.  This does not evaluate the map.
  void validate_cached_csr ( void ) const;

  const uint64_t * csr_offsets_data ( void ) const {
    return csr_offsets_ . data ();
  }
  const Vertex * csr_edges_data ( void ) const {
    return csr_edges_ . data ();
  }

private:
  // Private methods
  std::vector<size_type> compute_adjacencies ( const size_type & v ) const;
  /// try_build_cache
  ///   Build the CSR cache unless the graph has more than max_cached_edges
  ///   edges (0: no limit). Return whether the graph is cached.
  bool try_build_cache ( uint64_t max_cached_edges );
  // Private data
  std::shared_ptr<const Grid> grid_;
  std::shared_ptr<const Map> f_;
  MapGraphOptions options_;
  bool cached_;
  bool building_;
  std::vector<uint64_t> csr_offsets_;
  std::vector<Vertex> csr_edges_;
  mutable std::vector<Vertex> scratch_;
};

namespace cmgdb_detail {

/// map_graph_size_from_env
///   Read a positive base-10 size hint from the environment. These are
///   allocation hints only: nothing here bounds what a run is allowed to
///   attempt. A malformed value is still an error, because silently ignoring
///   a typo would hide the hint rather than apply it.
inline size_t
map_graph_size_from_env ( const char * name, size_t default_value ) {
  const char * raw = std::getenv ( name );
  if ( raw == nullptr or raw [ 0 ] == '\0' ) {
    return default_value;
  }

  for ( const char * digit = raw; *digit != '\0'; ++ digit ) {
    if ( *digit < '0' or *digit > '9' ) {
      std::ostringstream message;
      message << name << " must be a positive base-10 integer; got '" << raw << "'";
      throw std::invalid_argument ( message . str () );
    }
  }

  errno = 0;
  char * end = nullptr;
  const unsigned long long parsed = std::strtoull ( raw, &end, 10 );
  if ( errno == ERANGE or end == raw or *end != '\0' or parsed == 0 or
       parsed > std::numeric_limits<size_t>::max () ) {
    std::ostringstream message;
    message << name << " must be a positive base-10 integer; got '" << raw << "'";
    throw std::invalid_argument ( message . str () );
  }
  return static_cast<size_t> ( parsed );
}

/// map_graph_hard_limit_from_env
///   Read an opt-in nonnegative cache limit.  An unset variable means no
///   application-level limit.  Unlike reserve hints, zero is meaningful: it
///   permits only an empty corresponding payload.
inline size_t
map_graph_hard_limit_from_env ( const char * name ) {
  const char * raw = std::getenv ( name );
  if ( raw == nullptr or raw [ 0 ] == '\0' ) {
    return std::numeric_limits<size_t>::max ();
  }
  for ( const char * digit = raw; *digit != '\0'; ++ digit ) {
    if ( *digit < '0' or *digit > '9' ) {
      std::ostringstream message;
      message << name << " must be a nonnegative base-10 integer; got '"
              << raw << "'";
      throw std::invalid_argument ( message . str () );
    }
  }
  errno = 0;
  char * end = nullptr;
  const unsigned long long parsed = std::strtoull ( raw, &end, 10 );
  if ( errno == ERANGE or end == raw or *end != '\0' or
       parsed > std::numeric_limits<size_t>::max () ) {
    std::ostringstream message;
    message << name << " must be a nonnegative base-10 integer; got '"
            << raw << "'";
    throw std::invalid_argument ( message . str () );
  }
  return static_cast<size_t> ( parsed );
}

/// MapGraphEnv
///   The CMGDB_MAPGRAPH_RESERVE_* allocation hints and the opt-in
///   CMGDB_MAPGRAPH_HARD_MAX_* limits that build_cache applies.
struct MapGraphEnv {
  size_t reserve_edges;          // 0: no up-front reservation
  size_t reserve_min_vertices;
  size_t hard_max_vertices;      // SIZE_MAX: no limit
  size_t hard_max_edges;
  size_t hard_max_cache_bytes;
};

/// map_graph_env
///   Read those variables; a malformed value throws std::invalid_argument.
///   build_cache reads them on every build. ComputeMorseGraph and
///   ComputeConleyMorseGraph (Model) also call this before computing
///   whenever a cache will be built, so that a malformed value fails before
///   the first map evaluation even when the first cache built is the
///   returned map_graph's (lazy transition graphs).
inline MapGraphEnv
map_graph_env ( void ) {
  MapGraphEnv env;
  env . reserve_edges = map_graph_size_from_env (
    "CMGDB_MAPGRAPH_RESERVE_EDGES", 0 );
  env . reserve_min_vertices = map_graph_size_from_env (
    "CMGDB_MAPGRAPH_RESERVE_MIN_VERTICES", size_t ( 1 ) << 24 );
  env . hard_max_vertices = map_graph_hard_limit_from_env (
    "CMGDB_MAPGRAPH_HARD_MAX_VERTICES" );
  env . hard_max_edges = map_graph_hard_limit_from_env (
    "CMGDB_MAPGRAPH_HARD_MAX_EDGES" );
  env . hard_max_cache_bytes = map_graph_hard_limit_from_env (
    "CMGDB_MAPGRAPH_HARD_MAX_CACHE_BYTES" );
  return env;
}

/// map_graph_cache_enabled
///   Whether to build the eager CSR adjacency cache when the caller did not
///   say. Defaults to enabled.
///
///   Setting CMGDB_MAPGRAPH_CACHE=0 selects the lazy path, which recomputes
///   adjacencies through the map on every query: far slower, but it never
///   materializes the edge array. This is the explicit way to ask for a
///   memory-lean run. An explicit request (cache_map_graph=True,
///   cache_transition_graph=True, MapGraph(grid, map, cache=True) or
///   build_cache()) still builds the cache.
inline bool
map_graph_cache_enabled ( void ) {
  const char * raw = std::getenv ( "CMGDB_MAPGRAPH_CACHE" );
  if ( raw == nullptr or raw [ 0 ] == '\0' ) {
    return true;
  }
  const std::string value ( raw );
  if ( value == "0" or value == "off" or value == "false" ) {
    return false;
  }
  if ( value == "1" or value == "on" or value == "true" ) {
    return true;
  }
  std::ostringstream message;
  message
    << "CMGDB_MAPGRAPH_CACHE must be one of 0/1/on/off/true/false; got '"
    << raw << "'";
  throw std::invalid_argument ( message . str () );
}

} // namespace cmgdb_detail

inline
MapGraph::MapGraph ( std::shared_ptr<const Grid> grid,
                     std::shared_ptr<const Map> f ) :
grid_ ( grid ),
f_ ( f ),
options_ ( cmgdb_detail::map_graph_cache_enabled () ),
cached_ ( false ),
building_ ( false ) {
  initialize ();
}

inline
MapGraph::MapGraph ( std::shared_ptr<const Grid> grid,
                     std::shared_ptr<const Map> f,
                     MapGraphOptions options ) :
grid_ ( grid ),
f_ ( f ),
options_ ( options ),
cached_ ( false ),
building_ ( false ) {
  initialize ();
}

inline void
MapGraph::initialize ( void ) {
  if ( not f_ ) {
    throw std::logic_error ( "MapGraph::MapGraph. Unable to construct with uninitialized Map f\n");
  }
  if ( options_ . cache ) {
    // The options' max_cached_edges is a soft limit: a graph over it stays
    // lazy.
    try_build_cache ( options_ . max_cached_edges );
  }
}

/// try_build_cache
///    Structural intent: evaluate the multivalued map F(v) = cover(f(geo(v)))
///    exactly once per grid element, storing the resulting digraph in CSR
///    form (csr_offsets_, csr_edges_). The graph algorithms (strongly
///    connected components for the recurrent sets, reachability for the
///    Morse graph partial order) then read this stored digraph instead of
///    re-evaluating the map on every pass. Evaluation proceeds in chunks so
///    that maps providing a batched interface (Map::has_batch) receive many
///    rectangles per call -- for Python-defined maps this means one Python
///    call per chunk instead of one per rectangle. The adjacency lists this
///    produces are identical, element for element, to the on-demand path.
///
///    Limits. max_cached_edges is a soft limit, checked before
///    each row is stored: the cache is abandoned (the call returns
///    false) and the graph stays lazy, and the edge array never
///    holds more edges than that, whatever the chunk size (the batch
///    map's input and output for one chunk are not counted). The
///    constructor passes the options' limit; build_cache passes its
///    argument and raises instead. The opt-in environment limits
///    CMGDB_MAPGRAPH_HARD_MAX_{VERTICES,EDGES,CACHE_BYTES} are hard: they
///    raise before the CSR grows past them (the edge array's capacity is
///    clipped to them too). CMGDB_MAPGRAPH_RESERVE_EDGES reserves that many
///    edges up front once the grid has CMGDB_MAPGRAPH_RESERVE_MIN_VERTICES
///    vertices, capped at max_cached_edges when that is set. All of these
///    are read, and malformed values rejected, on every call (see
///    cmgdb_detail::map_graph_env).
inline void
MapGraph::build_cache ( uint64_t max_cached_edges ) {
  if ( try_build_cache ( max_cached_edges ) ) return;
  std::ostringstream message;
  message << "MapGraph::build_cache: the transition graph has more than "
          << "max_cached_edges=" << max_cached_edges
          << " edges, so it was not cached (the graph stays lazy)";
  throw std::runtime_error ( message . str () );
}

inline bool
MapGraph::try_build_cache ( uint64_t max_cached_edges ) {
  if ( cached_ ) return true;
  // Every map evaluation can run Python, which may let another thread call
  // build_cache on this graph, or call it from the map itself. A second
  // build would then interleave its rows with this one's.
  if ( building_ ) {
    throw std::runtime_error (
      "MapGraph::build_cache is already running on this graph" );
  }
  struct BuildingGuard {
    bool & flag;
    explicit BuildingGuard ( bool & f ) : flag ( f ) { flag = true; }
    ~BuildingGuard ( void ) { flag = false; }
  } guard ( building_ );
  const uint64_t n = num_vertices ();

  // Opt-in hard limits and the up-front reservation hint.
  const cmgdb_detail::MapGraphEnv env = cmgdb_detail::map_graph_env ();
  const size_t env_reserve_edges = env . reserve_edges;
  const size_t env_reserve_min_vertices = env . reserve_min_vertices;
  const size_t hard_max_vertices = env . hard_max_vertices;
  const size_t hard_max_edges = env . hard_max_edges;
  const size_t hard_max_cache_bytes = env . hard_max_cache_bytes;
  if ( n > hard_max_vertices ) {
    std::ostringstream message;
    message << "MapGraph vertex count " << n
            << " exceeds CMGDB_MAPGRAPH_HARD_MAX_VERTICES="
            << hard_max_vertices;
    throw std::runtime_error ( message . str () );
  }
  if ( n >= std::numeric_limits<size_t>::max () ) {
    throw std::overflow_error ( "MapGraph vertex count cannot form V+1 offsets" );
  }
  const size_t offset_count = n + 1;
  if ( offset_count > std::numeric_limits<size_t>::max () / sizeof ( uint64_t ) ) {
    throw std::overflow_error ( "MapGraph CSR offset byte count overflows size_t" );
  }
  const size_t offset_bytes = offset_count * sizeof ( uint64_t );
  if ( offset_bytes > hard_max_cache_bytes ) {
    std::ostringstream message;
    message << "MapGraph CSR offsets require " << offset_bytes
            << " bytes, above CMGDB_MAPGRAPH_HARD_MAX_CACHE_BYTES="
            << hard_max_cache_bytes;
    throw std::runtime_error ( message . str () );
  }
  const size_t maximum_edge_capacity = std::min (
    hard_max_edges, ( hard_max_cache_bytes - offset_bytes ) / sizeof ( Vertex ) );
  const bool hard_limited =
    maximum_edge_capacity != std::numeric_limits<size_t>::max ();
  // The soft limit: past it the cache is abandoned, so the edge array never
  // needs more slots than that either.
  size_t edge_capacity_limit = maximum_edge_capacity;
  if ( max_cached_edges > 0 and max_cached_edges < (uint64_t) edge_capacity_limit ) {
    edge_capacity_limit = (size_t) max_cached_edges;
  }
  const bool capacity_limited =
    edge_capacity_limit != std::numeric_limits<size_t>::max ();

  // The CSR is built in local arrays and moved into the graph only once it
  // is complete, so an abandoned or failed build (a hard limit, a map that
  // raises) leaves the graph lazy and untouched.
  std::vector<uint64_t> offsets;
  std::vector<Vertex> edges;

  // Raise before a row would take the CSR past a hard limit, and return
  // false (abandon the cache) before it would take it past max_cached_edges,
  // so the edge array never holds more than that, whatever the chunk size.
  // The capacity is grown here, geometrically but clipped to these limits,
  // so that the vector's own doubling cannot overshoot them.
  const auto append_row = [ & ] ( const std::vector<Vertex> & targets ) {
    const size_t edge_count = edges . size ();
    if ( targets . size () > hard_max_edges or
         edge_count > hard_max_edges - targets . size () ) {
      std::ostringstream message;
      message << "MapGraph edge count would exceed "
              << "CMGDB_MAPGRAPH_HARD_MAX_EDGES=" << hard_max_edges;
      throw std::runtime_error ( message . str () );
    }
    const size_t required = edge_count + targets . size ();
    if ( hard_limited and required > maximum_edge_capacity ) {
      std::ostringstream message;
      message << "MapGraph CSR needs at least " << required
              << " edge slots, above the configured hard edge/cache-byte "
                 "limit of " << maximum_edge_capacity;
      throw std::runtime_error ( message . str () );
    }
    if ( max_cached_edges > 0 and (uint64_t) required > max_cached_edges ) {
      return false;
    }
    if ( capacity_limited and required > edges . capacity () ) {
      const size_t old_capacity = edges . capacity ();
      const size_t growth = std::max ( old_capacity, size_t ( 65536 ) );
      size_t grown = required;
      if ( old_capacity <= std::numeric_limits<size_t>::max () - growth ) {
        grown = std::max ( required, old_capacity + growth );
      }
      edges . reserve ( std::min ( grown, edge_capacity_limit ) );
    }
    edges . insert ( edges . end (), targets . begin (), targets . end () );
    offsets . push_back ( edges . size () );
    return true;
  };

  offsets . reserve ( n + 1 );
  offsets . push_back ( 0 );
  if ( env_reserve_edges > 0 and n >= env_reserve_min_vertices ) {
    // Capped like the projected reservation below: a cache that outgrows
    // max_cached_edges is abandoned.
    edges . reserve ( std::min ( env_reserve_edges, edge_capacity_limit ) );
  }

  // The flat-buffer batch path requires rectangle geometry; fall back to
  // per-element evaluation for grids with other geometry types.
  bool use_batch = f_ -> has_batch () && n > 0 &&
    ( std::dynamic_pointer_cast<RectGeo> ( grid_ -> geometry ( (Vertex) 0 ) ) != nullptr );

  const uint64_t chunk = options_ . chunk_size > 0 ? options_ . chunk_size : n;
  std::vector<double> rects;
  std::vector<double> images;

  for ( uint64_t start = 0; start < n; start += chunk ) {
    const uint64_t stop = std::min ( start + chunk, n );
    if ( use_batch ) {
      // Gather rectangle bounds for this chunk into a flat buffer.
      const uint64_t count = stop - start;
      uint64_t dim = 0;
      rects . clear ();
      for ( uint64_t v = start; v < stop; ++ v ) {
        std::shared_ptr<RectGeo> rect =
          std::dynamic_pointer_cast<RectGeo> ( grid_ -> geometry ( (Vertex) v ) );
        if ( not rect ) {
          throw std::logic_error ( "MapGraph::build_cache. Mixed geometry types in grid.\n" );
        }
        dim = rect -> dimension ();
        rects . insert ( rects . end (), rect -> lower_bounds . begin (),
                         rect -> lower_bounds . end () );
        rects . insert ( rects . end (), rect -> upper_bounds . begin (),
                         rect -> upper_bounds . end () );
      }
      // One map evaluation for the whole chunk.
      f_ -> batch_map ( rects, count, dim, images );
      // Cover each image rectangle to produce the adjacency lists.
      RectGeo image ( dim );
      for ( uint64_t i = 0; i < count; ++ i ) {
        const double * bounds = images . data () + i * 2 * dim;
        for ( uint64_t d = 0; d < dim; ++ d ) {
          image . lower_bounds [ d ] = bounds [ d ];
          image . upper_bounds [ d ] = bounds [ dim + d ];
        }
        if ( not append_row ( grid_ -> cover ( image ) ) ) return false;
      }
    } else {
      for ( uint64_t v = start; v < stop; ++ v ) {
        if ( not append_row ( compute_adjacencies ( (Vertex) v ) ) ) return false;
      }
    }
    if ( stop < n ) {
      // Project the final edge count from the edge density seen so far.
      // The projection sizes the up-front reservation of the flat edge
      // array -- a deep uniform grid's multi-gigabyte edge array is then
      // allocated (nearly) once instead of repeatedly doubled with a
      // transient ~3x memory peak. Chunks follow the tree order (a spatial
      // sweep), so early density can be biased; the projection is therefore
      // refreshed every chunk and the auto reservation doubles it for
      // headroom. It never abandons the cache: a sweep that is denser early
      // on can project several times the real edge count, so only the
      // measured count (append_row) decides.
      const double density = (double) edges . size () / (double) stop;
      const uint64_t projected = (uint64_t) ( density * (double) n ) + 1;
      if ( projected >= options_ . reserve_min_edges ) {
        uint64_t target = options_ . reserve_edges > 0 ?
          options_ . reserve_edges : 2 * projected;
        target = std::min ( target, (uint64_t) edge_capacity_limit );
        if ( target > (uint64_t) edges . capacity () ) {
          try {
            edges . reserve ( target );
          } catch ( std::bad_alloc const& ) {
            // The reservation is an optimization only; fall back to
            // ordinary vector growth if the allocator refuses it.
          }
        }
      }
    }
  }
  csr_offsets_ . swap ( offsets );
  csr_edges_ . swap ( edges );
  cached_ = true;
  return true;
}

inline std::vector<MapGraph::Vertex>
MapGraph::adjacencies ( const size_type & source ) const {
  if ( cached_ ) {
    return std::vector<Vertex> ( csr_edges_ . begin () + csr_offsets_ [ source ],
                                 csr_edges_ . begin () + csr_offsets_ [ source + 1 ] );
  }
  return compute_adjacencies ( source );
}

inline MapGraph::AdjacencySpan
MapGraph::adjacency_span ( const size_type & source ) const {
  if ( cached_ ) {
    const Vertex * base = csr_edges_ . data ();
    return { base + csr_offsets_ [ source ], base + csr_offsets_ [ source + 1 ] };
  }
  scratch_ = compute_adjacencies ( source );
  return { scratch_ . data (), scratch_ . data () + scratch_ . size () };
}

inline std::vector<MapGraph::Vertex>
MapGraph::compute_adjacencies ( const Vertex & source ) const {
  std::vector < Vertex > target =
    grid_ -> cover ( (*f_) ( grid_ -> geometry ( source ) ) ); // here is the work
  return target;
}

inline void
MapGraph::validate_cached_csr ( void ) const {
  if ( not cached_ ) {
    throw std::runtime_error (
      "MapGraph CSR export requires a cached MapGraph (CMGDB_MAPGRAPH_CACHE "
      "enabled or cache_map_graph=True, if the graph is within the "
      "computation's max_cached_edges; or map_graph.build_cache(), which "
      "that limit does not bound)" );
  }
  const size_t n = static_cast<size_t> ( num_vertices () );
  if ( csr_offsets_ . size () != n + 1 or csr_offsets_ . empty () or
       csr_offsets_ . front () != 0 or
       csr_offsets_ . back () != csr_edges_ . size () ) {
    throw std::logic_error ( "MapGraph cached CSR offsets are inconsistent" );
  }
  for ( size_t source = 0; source < n; ++ source ) {
    const size_t begin = csr_offsets_ [ source ];
    const size_t end = csr_offsets_ [ source + 1 ];
    if ( begin > end or end > csr_edges_ . size () ) {
      throw std::logic_error ( "MapGraph cached CSR row bounds are inconsistent" );
    }
    Vertex previous = 0;
    bool first = true;
    for ( size_t edge = begin; edge < end; ++ edge ) {
      const Vertex target = csr_edges_ [ edge ];
      if ( target >= n ) {
        throw std::logic_error ( "MapGraph cached CSR target is out of range" );
      }
      if ( not first and target <= previous ) {
        throw std::logic_error (
          "MapGraph cached CSR rows must be sorted and duplicate-free" );
      }
      first = false;
      previous = target;
    }
  }
}

inline MapGraph::size_type
MapGraph::num_vertices ( void ) const {
  return grid_ -> size ();
}

/// Python Bindings

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
namespace py = pybind11;

inline void
MapGraphBinding(py::module &m) {
  py::class_<MapGraph, std::shared_ptr<MapGraph>>(m, "MapGraph")
    .def(py::init<std::shared_ptr<const Grid>, std::shared_ptr<const Map>>())
    .def(py::init([](std::shared_ptr<const Grid> grid,
                     std::shared_ptr<const Map> f, bool cache) {
           return new MapGraph ( grid, f, MapGraphOptions ( cache ) );
         }),
         py::arg("grid"), py::arg("map"), py::arg("cache"))
    .def("num_vertices", &MapGraph::num_vertices)
    .def("has_cache", &MapGraph::has_cache)
    .def("num_cached_edges", &MapGraph::num_cached_edges)
    .def("build_cache", &MapGraph::build_cache,
         py::arg("max_cached_edges") = 0,
         "Build the CSR transition-graph cache now (no-op if already built). "
         "Upgrades a lazy map_graph (cache_map_graph=False, or a cache "
         "abandoned at the max_cached_edges of the computation) to a cached "
         "one at the cost of one full pass of map evaluations over the grid. "
         "Only this call's max_cached_edges (0 = unlimited, the default) "
         "bounds it: a graph with more edges raises RuntimeError and stays "
         "lazy. Raises RuntimeError if a build of this graph is already "
         "running.")
    .def(
      "csr_view",
      [] ( const std::shared_ptr<MapGraph> & graph ) {
        graph -> validate_cached_csr ();
        if ( sizeof ( MapGraph::Vertex ) != sizeof ( int64_t ) ) {
          throw std::runtime_error (
            "MapGraph zero-copy CSR view requires 64-bit native indices" );
        }
        if ( graph -> num_vertices () >
               static_cast<uint64_t> ( std::numeric_limits<int64_t>::max () - 1 ) or
             graph -> num_cached_edges () >
               static_cast<uint64_t> ( std::numeric_limits<int64_t>::max () ) ) {
          throw std::overflow_error (
            "MapGraph CSR does not fit signed int64 NumPy indexing" );
        }

        py::object owner = py::cast ( graph );
        py::array offsets (
          py::dtype::of<int64_t> (),
          { static_cast<py::ssize_t> ( graph -> num_vertices () + 1 ) },
          { static_cast<py::ssize_t> ( sizeof ( int64_t ) ) },
          graph -> csr_offsets_data (),
          owner );
        py::array targets (
          py::dtype::of<int64_t> (),
          { static_cast<py::ssize_t> ( graph -> num_cached_edges () ) },
          { static_cast<py::ssize_t> ( sizeof ( int64_t ) ) },
          graph -> csr_edges_data (),
          owner );
        offsets . attr ( "setflags" ) ( py::arg ( "write" ) = false );
        targets . attr ( "setflags" ) ( py::arg ( "write" ) = false );
        return py::make_tuple ( offsets, targets );
      },
      "Return read-only zero-copy int64 CSR arrays owned by this MapGraph."
    )
    .def("adjacencies", &MapGraph::adjacencies);
}

#endif
