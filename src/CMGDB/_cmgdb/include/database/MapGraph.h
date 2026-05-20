// MapGraph.h

#ifndef CMDP_MAPGRAPH_H
#define CMDP_MAPGRAPH_H

#include <exception>
#include <vector>
#include <iterator>
#include <iostream>
#include <algorithm>
#include <cstddef>
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

  constexpr size_t VERTEX_CAP = 16000000;
  constexpr size_t EDGE_BUDGET = 200000000;
  constexpr size_t EDGE_BUDGET_STAGE = EDGE_BUDGET / 2;

  if ( num_vertices () < VERTEX_CAP ) {
    const size_t n = num_vertices ();
    std::vector<std::vector<Vertex> > staging ( n );
    size_t edge_count = 0;

    if ( f_ -> has_optimized_batch () ) {
      constexpr size_t BATCH_CHUNK = 100000;
      for ( size_t start = 0; start < n; start += BATCH_CHUNK ) {
        size_t end = std::min ( start + BATCH_CHUNK, n );
        std::vector<Vertex> sources;
        sources.reserve ( end - start );
        for ( size_t source = start; source < end; ++ source ) {
          sources.push_back ( source );
        }
        std::vector<std::vector<Vertex> > chunk_adjacencies =
          compute_adjacencies_batch ( sources );
        for ( size_t i = 0; i < chunk_adjacencies . size (); ++ i ) {
          edge_count += chunk_adjacencies [ i ] . size ();
          staging [ start + i ] = std::move ( chunk_adjacencies [ i ] );
        }
        if ( edge_count > EDGE_BUDGET_STAGE ) {
          stored_graph = false;
          csr_offsets_ . clear ();
          csr_edges_ . clear ();
          return;
        }
      }
      build_csr_from_staging ( staging );
      stored_graph = true;
      return;
    }

    for ( size_type source = 0; source < n; ++ source ) {
      staging [ source ] = compute_adjacencies ( source );
      edge_count += staging [ source ] . size ();
      if ( edge_count > EDGE_BUDGET_STAGE ) {
        stored_graph = false;
        csr_offsets_ . clear ();
        csr_edges_ . clear ();
        return;
      }
    }
    build_csr_from_staging ( staging );
    stored_graph = true;
    return;
  }

  stored_graph = false;
  return;
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
    .def("adjacencies", &MapGraph::adjacencies);
}

#endif
