// MapGraph.h

#ifndef CMDP_MAPGRAPH_H
#define CMDP_MAPGRAPH_H

#include <exception>
#include <vector>
#include <iterator>
#include <iostream>
#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <stdexcept>
// #include <unistd.h>

#include "boost/unordered_map.hpp"
#include "boost/foreach.hpp"

#include "Grid.h"

#ifdef CMDB_STORE_GRAPH
#include "ComputeGraph.h"
#endif

/// class MapGraph
///    This class is used to created an object suitable for graph algorithms
///    given a grid and a map object. By default "adjacencies" is computed on demand
///    in order to avoid storing the adjacency lists.
class MapGraph {
public:
  // Typedefs
  typedef Grid::size_type size_type;
  typedef Grid::GridElement Vertex;

  struct AdjacencyView {
    typedef const Vertex *iterator;
    typedef const Vertex *const_iterator;
    const Vertex *begin_;
    const Vertex *end_;
    size_t size_;
    const Vertex *begin() const { return begin_; }
    const Vertex *end() const { return end_; }
    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
  };

  // Constructor. Requires Grid and Map.
  MapGraph ( std::shared_ptr<const Grid> grid,
             std::shared_ptr<const Map> f );

  // MapGraph ( std::shared_ptr<const Grid> grid,
  //            std::shared_ptr<const Model> model );

  void initialize ( void );

  /// adjacencies
  ///   Return vector of Vertices which are out-edge adjacencies of input v
  std::vector<Vertex> adjacencies ( const Vertex & v ) const;

  /// adjacencies_view
  ///   Return a non-owning view over out-edge adjacencies. When the CSR
  ///   cache is populated, this points directly into the flat edge buffer.
  AdjacencyView adjacencies_view ( const Vertex & v ) const;
  
  /// num_vertices
  ///   Return number of vertices
  size_type num_vertices ( void ) const;

  bool has_cache ( void ) const { return stored_graph; }
  size_t num_cached_edges ( void ) const {
    return stored_graph ? csr_edges_ . size () : 0;
  }

private:
  // Private methods
  std::vector<size_type> compute_adjacencies ( const size_type & v ) const;
  std::vector<std::vector<size_type> > compute_adjacencies_batch (
    const std::vector<size_type> & sources ) const;
  void build_csr_from_staging ( std::vector<std::vector<Vertex> > & staging );
  // Private data
  std::shared_ptr<const Grid> grid_;
  std::shared_ptr<const Map> f_;
  // Variables used if graph is stored in memory. (See CMDB_STORE_GRAPH define)
  bool stored_graph;
  std::vector<std::vector<Vertex> > adjacency_lists_;
  std::vector<size_t> csr_offsets_;
  std::vector<Vertex> csr_edges_;
};

namespace cmgdb_detail {

inline size_t
map_graph_limit_from_env ( const char * name, size_t default_value ) {
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

inline size_t
map_graph_optional_limit_from_env ( const char * name ) {
  const char * raw = std::getenv ( name );
  if ( raw == nullptr or raw [ 0 ] == '\0' ) {
    return 0;
  }
  return map_graph_limit_from_env ( name, 0 );
}

inline std::runtime_error
map_graph_vertex_limit_error ( size_t vertices, size_t limit ) {
  std::ostringstream message;
  message
    << "MapGraph optimized batch cache requires " << vertices
    << " vertices, exceeding CMGDB_MAPGRAPH_MAX_VERTICES=" << limit
    << ". Increase CMGDB_MAPGRAPH_MAX_VERTICES explicitly; refusing to "
       "fall back to per-cell map callbacks.";
  return std::runtime_error ( message . str () );
}

inline std::runtime_error
map_graph_edge_limit_error ( size_t observed_edges, size_t limit ) {
  std::ostringstream message;
  message
    << "MapGraph optimized batch cache produced at least " << observed_edges
    << " edges, exceeding CMGDB_MAPGRAPH_MAX_EDGES=" << limit
    << ". Increase CMGDB_MAPGRAPH_MAX_EDGES explicitly (each cached edge "
       "uses " << sizeof ( MapGraph::Vertex )
    << " bytes); refusing to fall back to per-cell map callbacks.";
  return std::runtime_error ( message . str () );
}

} // namespace cmgdb_detail

inline
MapGraph::MapGraph ( std::shared_ptr<const Grid> grid,
                     std::shared_ptr<const Map> f ) :
grid_ ( grid ),
f_ ( f ),
stored_graph ( false ) {
  initialize ();
}

// inline
// MapGraph::MapGraph ( std::shared_ptr<const Grid> grid,
//                      std::shared_ptr<const Model> model ) :
// grid_ ( grid ),
// f_ ( model -> map () ),
// stored_graph ( false ) {
//   initialize ();
// }

inline void
MapGraph::initialize ( void ) {
  if ( not f_ ) {
    throw std::logic_error ( "MapGraph::MapGraph. Unable to construct with uninitialized Map f\n");
  }

  constexpr size_t DEFAULT_VERTEX_LIMIT = size_t ( 1 ) << 24;
  constexpr size_t DEFAULT_EDGE_LIMIT = 200000000;
  constexpr size_t BATCH_CHUNK = 100000;
  const size_t vertex_limit = cmgdb_detail::map_graph_limit_from_env (
    "CMGDB_MAPGRAPH_MAX_VERTICES", DEFAULT_VERTEX_LIMIT );
  const size_t edge_limit = cmgdb_detail::map_graph_limit_from_env (
    "CMGDB_MAPGRAPH_MAX_EDGES", DEFAULT_EDGE_LIMIT );
  const size_t edge_reserve = cmgdb_detail::map_graph_optional_limit_from_env (
    "CMGDB_MAPGRAPH_RESERVE_EDGES" );
  const size_t reserve_min_vertices = cmgdb_detail::map_graph_limit_from_env (
    "CMGDB_MAPGRAPH_RESERVE_MIN_VERTICES", DEFAULT_VERTEX_LIMIT );
  if ( edge_reserve > edge_limit ) {
    std::ostringstream message;
    message
      << "CMGDB_MAPGRAPH_RESERVE_EDGES=" << edge_reserve
      << " exceeds CMGDB_MAPGRAPH_MAX_EDGES=" << edge_limit;
    throw std::invalid_argument ( message . str () );
  }
  const size_t n = num_vertices ();

  if ( n > vertex_limit ) {
    if ( f_ -> has_optimized_batch () ) {
      throw cmgdb_detail::map_graph_vertex_limit_error ( n, vertex_limit );
    }
    stored_graph = false;
    return;
  }

  if ( f_ -> has_optimized_batch () ) {
    // Append each batch directly to CSR. The old full-size
    // vector<vector<Vertex>> staging area alone cost 384 MiB at 2^24
    // vertices, then duplicated all edge storage while flattening.
    csr_offsets_ . clear ();
    csr_edges_ . clear ();
    csr_offsets_ . reserve ( n + 1 );
    if ( edge_reserve > 0 and n >= reserve_min_vertices ) {
      csr_edges_ . reserve ( edge_reserve );
    }
    csr_offsets_ . push_back ( 0 );
    size_t edge_count = 0;

    for ( size_t start = 0; start < n; start += BATCH_CHUNK ) {
      const size_t end = std::min ( start + BATCH_CHUNK, n );
      std::vector<Vertex> sources;
      sources.reserve ( end - start );
      for ( size_t source = start; source < end; ++ source ) {
        sources.push_back ( source );
      }
      std::vector<std::vector<Vertex> > chunk_adjacencies =
        compute_adjacencies_batch ( sources );
      size_t chunk_edge_count = 0;
      for ( const auto & adjacency : chunk_adjacencies ) {
        if ( adjacency . size () >
             edge_limit - edge_count - chunk_edge_count ) {
          const size_t observed_edges =
            edge_limit == std::numeric_limits<size_t>::max ()
              ? edge_limit
              : edge_limit + 1;
          throw cmgdb_detail::map_graph_edge_limit_error (
            observed_edges, edge_limit );
        }
        chunk_edge_count += adjacency . size ();
      }

      const size_t required_capacity = edge_count + chunk_edge_count;
      if ( required_capacity > csr_edges_ . capacity () ) {
        const size_t old_capacity = csr_edges_ . capacity ();
        const size_t available_growth = edge_limit - old_capacity;
        const size_t growth = std::min (
          available_growth, std::max ( old_capacity, BATCH_CHUNK ) );
        const size_t grown_capacity = old_capacity + growth;
        csr_edges_ . reserve ( std::max ( required_capacity, grown_capacity ) );
      }

      for ( auto & adjacency : chunk_adjacencies ) {
        csr_edges_ . insert (
          csr_edges_ . end (), adjacency . begin (), adjacency . end () );
        edge_count += adjacency . size ();
        csr_offsets_ . push_back ( edge_count );
      }
    }
    stored_graph = true;
    return;
  }

  {
    std::vector<std::vector<Vertex> > staging ( n );
    size_t edge_count = 0;
    for ( size_type source = 0; source < n; ++ source ) {
      staging [ source ] = compute_adjacencies ( source );
      if ( staging [ source ] . size () > edge_limit - edge_count ) {
        stored_graph = false;
        csr_offsets_ . clear ();
        csr_edges_ . clear ();
        return;
      }
      edge_count += staging [ source ] . size ();
    }
    build_csr_from_staging ( staging );
    stored_graph = true;
    return;
  }
#ifdef CMDB_STORE_GRAPH
  
  // Determine whether it is efficient to use an MPI job to store the graph
  if ( num_vertices () < 10000 ) {
    stored_graph = false;
    return;
  }
  stored_graph = true;
  
  // Make a file with required integrations
  MapEvals evals;
  evals . parameter () = f . parameter ();
  for ( size_type source = 0; source < num_vertices (); ++ source ) {
    Vertex domain_cell = lookup ( source );
    evals . insert ( domain_cell );
  }
  
  std::cout << "Saving grid to file.\n";
  // Save the grid and a list of required evaluations to disk
  grid_ -> save ("grid.txt");
  evals . save ( "mapevals.txt" );
  
  // Call a program to compute the adjacency information
  std::cout << "Calling MPI program to evaluate map.\n";
  system("./COMPUTEGRAPHSCRIPT");
  std::cout << "MPI program returned.\n";

  // Load and store the adjacency information
  evals . load ( "mapevals.txt" );
  adjacency_lists_ . resize ( num_vertices () );
  for ( size_type source = 0; source < num_vertices (); ++ source ) {
    Vertex domain_cell = lookup ( source );    
    index ( &adjacency_lists_ [ source ], evals . val ( domain_cell ) );
  }
  std::cout << "Map stored.\n";
#endif
}

inline std::vector<MapGraph::Vertex>
MapGraph::adjacencies ( const size_type & source ) const {
  if ( stored_graph ) {
    size_t begin = csr_offsets_ [ source ];
    size_t end = csr_offsets_ [ source + 1 ];
    return std::vector<Vertex> ( csr_edges_ . begin () + begin,
                                 csr_edges_ . begin () + end );
  }
  return compute_adjacencies ( source );
}

inline MapGraph::AdjacencyView
MapGraph::adjacencies_view ( const size_type & source ) const {
  if ( stored_graph ) {
    size_t begin = csr_offsets_ [ source ];
    size_t end = csr_offsets_ [ source + 1 ];
    if ( begin == end ) {
      return { nullptr, nullptr, 0 };
    }
    const Vertex * base = csr_edges_ . data ();
    return { base + begin, base + end, end - begin };
  }
  thread_local std::vector<Vertex> lazy_adjacencies;
  lazy_adjacencies = compute_adjacencies ( source );
  if ( lazy_adjacencies . empty () ) {
    return { nullptr, nullptr, 0 };
  }
  return { lazy_adjacencies . data (),
           lazy_adjacencies . data () + lazy_adjacencies . size (),
           lazy_adjacencies . size () };
}

inline std::vector<MapGraph::Vertex>
MapGraph::compute_adjacencies ( const Vertex & source ) const {
  std::vector < Vertex > target = 
    grid_ -> cover ( (*f_) ( grid_ -> geometry ( source ) ) ); // here is the work
  return target;
}

inline std::vector<std::vector<MapGraph::Vertex> >
MapGraph::compute_adjacencies_batch ( const std::vector<size_type> & sources ) const {
  std::vector<std::shared_ptr<Geo> > geos;
  geos.reserve ( sources . size () );
  for ( const size_type & source : sources ) {
    geos.push_back ( grid_ -> geometry ( source ) );
  }

  std::vector<std::shared_ptr<Geo> > image_geos = f_ -> batch_map ( geos );
  if ( image_geos . size () != sources . size () ) {
    throw std::runtime_error ( "MapGraph::compute_adjacencies_batch received the wrong number of images" );
  }

  std::vector<std::vector<Vertex> > result;
  result.reserve ( sources . size () );
  for ( const auto & image_geo : image_geos ) {
    result.push_back ( grid_ -> cover ( image_geo ) );
  }
  return result;
}

inline void
MapGraph::build_csr_from_staging ( std::vector<std::vector<Vertex> > & staging ) {
  csr_offsets_ . clear ();
  csr_edges_ . clear ();
  csr_offsets_ . reserve ( staging . size () + 1 );
  csr_offsets_ . push_back ( 0 );

  size_t total_edges = 0;
  for ( const auto & adj : staging ) {
    total_edges += adj . size ();
    csr_offsets_ . push_back ( total_edges );
  }

  csr_edges_ . reserve ( total_edges );
  for ( auto & adj : staging ) {
    csr_edges_ . insert ( csr_edges_ . end (), adj . begin (), adj . end () );
    std::vector<Vertex> () . swap ( adj );
  }
}

inline MapGraph::size_type
MapGraph::num_vertices ( void ) const {
  return grid_ -> size ();
}

/// Python Bindings

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

inline void
MapGraphBinding(py::module &m) {
  py::class_<MapGraph, std::shared_ptr<MapGraph>>(m, "MapGraph")
    .def(py::init<std::shared_ptr<const Grid>, std::shared_ptr<const Map>>())
    // .def(py::init<std::shared_ptr<const Grid>, std::shared_ptr<const Model>>())
    .def("num_vertices", &MapGraph::num_vertices)
    .def("has_cache", &MapGraph::has_cache)
    .def("num_cached_edges", &MapGraph::num_cached_edges)
    .def("adjacencies", &MapGraph::adjacencies);
}

#endif
