#include <iostream>
#include <fstream>
#include <ctime>
#include <cmath>
#include <vector>
#include <set>
#include <map>
#include <sstream>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <tuple>
#include <optional>

// #define CMG_VERBOSE
#define MEMORYBOOKKEEPING

// The AtlasModel bindings need joinImpl<Atlas>, which join.h compiles only
// under CMGDB_USE_ATLAS: without it the extension still builds, but every
// Atlas Morse graph computation throws at run time. CMakeLists.txt defines
// it. Atlas.h includes the vendored sdsl-lite (RankSelect.h), so this fork
// always compiles sdsl, even though SuccinctGrid stays optional (below).
#ifndef CMGDB_USE_ATLAS
#error "CMGDB_USE_ATLAS must be defined: the AtlasModel bindings join Atlas grids (see CMakeLists.txt)"
#endif

#include "Model.h"
#include "AtlasModel.h"

#include "Map.h"
#include "ChompMap.h"
#include "MorseGraph.h"
#include "Compute_Morse_Graph.h"
#include "RectGeo.h"
#include "MorseSetReachability.h"

#include "SingleOutput.h"
#include "simple_interval.h"

#include "Configuration.h"

#include "chomp/ConleyIndex.h"
#include "chomp/ExplicitChainComplex.h"
#include "conleyIndexString.h"
#include "CarrierChainMap.h"
#include "RelativeShiftClass.h"

namespace {

typedef std::tuple<uint64_t, uint64_t, int> SparseEntry;
typedef std::vector<std::vector<SparseEntry> > GradedSparseEntries;
typedef chomp::SparseMatrix<chomp::Ring> ChainMatrix;

struct RelativeHomologyShiftResult {
  std::vector<std::string> shift_class;
  std::vector<uint64_t> homology_dimensions;
  std::vector<std::vector<std::vector<int64_t> > > induced_maps;
};

std::vector<ChainMatrix>
BuildExplicitMatrices (
    const std::vector<uint64_t> & cell_counts,
    const GradedSparseEntries & entries,
    const bool boundary_matrices ) {
  if ( entries . size () != cell_counts . size () ) {
    throw std::invalid_argument (
      boundary_matrices
        ? "boundary_entries must have one list for every chain dimension"
        : "chain_map_entries must have one list for every chain dimension" );
  }

  std::vector<ChainMatrix> result ( cell_counts . size () );
  for ( size_t d = 0; d < cell_counts . size (); ++ d ) {
    const uint64_t rows = boundary_matrices
      ? ( d == 0 ? 0 : cell_counts [ d - 1 ] )
      : cell_counts [ d ];
    const uint64_t columns = cell_counts [ d ];
    if ( rows > static_cast<uint64_t> ( std::numeric_limits<int64_t>::max () ) ||
         columns > static_cast<uint64_t> ( std::numeric_limits<int64_t>::max () ) ) {
      throw std::overflow_error ( "chain group is too large for CHOMP matrices" );
    }
    result [ d ] . resize (
      static_cast<int64_t> ( rows ), static_cast<int64_t> ( columns ) );

    std::set<std::pair<uint64_t, uint64_t> > occupied;
    for ( const SparseEntry & entry : entries [ d ] ) {
      const uint64_t row = std::get<0> ( entry );
      const uint64_t column = std::get<1> ( entry );
      const int coefficient = std::get<2> ( entry );
      if ( row >= rows || column >= columns ) {
        std::ostringstream message;
        message
          << ( boundary_matrices ? "boundary" : "chain map" )
          << " entry (" << row << ", " << column << ") in dimension " << d
          << " is outside its " << rows << " x " << columns << " matrix";
        throw std::out_of_range ( message . str () );
      }
      if ( ! occupied . insert ( std::make_pair ( row, column ) ) . second ) {
        std::ostringstream message;
        message
          << "duplicate " << ( boundary_matrices ? "boundary" : "chain map" )
          << " entry (" << row << ", " << column << ") in dimension " << d;
        throw std::invalid_argument ( message . str () );
      }
      const chomp::Ring value ( coefficient );
      if ( value != chomp::Ring ( 0 ) ) {
        result [ d ] . write (
          static_cast<int64_t> ( row ), static_cast<int64_t> ( column ), value );
      }
    }
  }
  return result;
}

bool IsZeroMatrix ( const ChainMatrix & matrix ) {
  return matrix . size () == 0;
}

bool EqualMatrices ( const ChainMatrix & lhs, const ChainMatrix & rhs ) {
  if ( lhs . number_of_rows () != rhs . number_of_rows () ||
       lhs . number_of_columns () != rhs . number_of_columns () ) return false;
  if ( lhs . size () != rhs . size () ) return false;
  for ( int64_t row = 0; row < lhs . number_of_rows (); ++ row ) {
    for ( ChainMatrix::MatrixPosition entry = lhs . row_begin ( row );
          entry != lhs . end ();
          lhs . row_advance ( entry ) ) {
      if ( lhs . read ( entry ) !=
           rhs . read ( row, lhs . column ( entry ) ) ) return false;
    }
  }
  return true;
}

void ValidateExplicitChainData (
    const std::vector<ChainMatrix> & boundaries,
    const std::vector<ChainMatrix> & chain_map ) {
  for ( size_t d = 2; d < boundaries . size (); ++ d ) {
    if ( ! IsZeroMatrix ( boundaries [ d - 1 ] * boundaries [ d ] ) ) {
      std::ostringstream message;
      message << "boundary squared is nonzero from dimension " << d
              << " to dimension " << d - 2 << " (coefficients are in F_5)";
      throw std::invalid_argument ( message . str () );
    }
  }
  for ( size_t d = 1; d < boundaries . size (); ++ d ) {
    const ChainMatrix boundary_after_map = boundaries [ d ] * chain_map [ d ];
    const ChainMatrix map_after_boundary = chain_map [ d - 1 ] * boundaries [ d ];
    if ( ! EqualMatrices ( boundary_after_map, map_after_boundary ) ) {
      std::ostringstream message;
      message << "chain-map equation fails in dimension " << d
              << ": boundary[d] * map[d] != map[d-1] * boundary[d] over F_5";
      throw std::invalid_argument ( message . str () );
    }
  }
}

chomp::Chain ApplyChainMatrix (
    const ChainMatrix & matrix, const chomp::Chain & input ) {
  chomp::Chain output;
  output . dimension () = input . dimension ();
  for ( const chomp::Term & term : input () ) {
    for ( ChainMatrix::MatrixPosition entry =
            matrix . column_begin ( static_cast<int64_t> ( term . index () ) );
          entry != matrix . end ();
          matrix . column_advance ( entry ) ) {
      output += chomp::Term (
        static_cast<chomp::Index> ( matrix . row ( entry ) ),
        matrix . read ( entry ) * term . coef () );
    }
  }
  return chomp::simplify ( output );
}

RelativeHomologyShiftResult
ComputeRelativeHomologyShiftClass (
    const std::vector<uint64_t> & cell_counts,
    const GradedSparseEntries & boundary_entries,
    const GradedSparseEntries & chain_map_entries ) {
  if ( cell_counts . empty () ) {
    throw std::invalid_argument ( "cell_counts must contain at least dimension zero" );
  }

  std::vector<ChainMatrix> boundaries =
    BuildExplicitMatrices ( cell_counts, boundary_entries, true );
  std::vector<ChainMatrix> chain_map =
    BuildExplicitMatrices ( cell_counts, chain_map_entries, false );
  ValidateExplicitChainData ( boundaries, chain_map );

  chomp::ExplicitChainComplex complex ( cell_counts, boundaries );
  chomp::MorseComplex morse ( complex );
  chomp::Generators_t generators =
    chomp::SmithGenerators ( morse, complex . dimension () );

  chomp::ConleyIndex_t conley_index;
  RelativeHomologyShiftResult result;
  for ( int d = 0; d <= complex . dimension (); ++ d ) {
    if ( d > morse . dimension () ) {
      conley_index . data () . push_back ( ChainMatrix ( 0, 0 ) );
      result . homology_dimensions . push_back ( 0 );
      result . induced_maps . push_back ( {} );
      continue;
    }

    std::vector<chomp::Chain> images;
    images . reserve ( generators [ d ] . size () );
    for ( const std::pair<chomp::Chain, chomp::Ring> & generator : generators [ d ] ) {
      const chomp::Chain lifted = morse . lift ( generator . first );
      const chomp::Chain mapped = ApplyChainMatrix ( chain_map [ d ], lifted );
      images . push_back ( morse . lower ( mapped ) );
    }

    const ChainMatrix basis = chomp::chainsToMatrix ( generators [ d ], morse, d );
    const ChainMatrix mapped_basis = chomp::chainsToMatrix ( images, morse, d );
    ChainMatrix induced_map = chomp::SmithSolve ( basis, mapped_basis );
    if ( induced_map . number_of_rows () != induced_map . number_of_columns () ) {
      throw std::runtime_error (
        "internal error: induced self-map on homology is not square" );
    }
    result . homology_dimensions . push_back (
      static_cast<uint64_t> ( induced_map . number_of_rows () ) );
    std::vector<std::vector<int64_t> > dense (
      induced_map . number_of_rows (),
      std::vector<int64_t> ( induced_map . number_of_columns (), 0 ) );
    for ( int64_t row = 0; row < induced_map . number_of_rows (); ++ row ) {
      for ( int64_t column = 0; column < induced_map . number_of_columns (); ++ column ) {
        dense [ row ] [ column ] =
          induced_map . read ( row, column ) . balanced_value ();
      }
    }
    result . induced_maps . push_back ( std::move ( dense ) );
    conley_index . data () . push_back ( induced_map );
  }
  result . shift_class = conleyIndexString ( conley_index );
  return result;
}

} // namespace

#include <boost/serialization/export.hpp>
// The succinct (sdsl-backed) grid is optional: the phase grid is
// PointerGrid and nothing constructs a SuccinctGrid, so default builds do
// not compile against the vendored sdsl-lite at all. Define
// CMGDB_USE_SUCCINCT to enable it.
#ifdef CMGDB_USE_SUCCINCT
#include "SuccinctGrid.h"
BOOST_CLASS_EXPORT_IMPLEMENT(SuccinctGrid);
#endif
#include "PointerGrid.h"
BOOST_CLASS_EXPORT_IMPLEMENT(PointerGrid);

std::vector < std::string >
ComputeConleyIndex ( const std::vector < uint64_t > & X_cubes,
                     const std::vector < uint64_t > & A_cubes,
                     const std::vector < uint64_t > & sizes,
                     const std::vector < bool > & periodic,
                     const std::unordered_map < uint64_t, std::vector < uint64_t > > & F,
                     bool acyclic_check = true ) {
  // Compute the Conley index from a combinatorial index pair (X, A) and a map F
  chomp::ConleyIndex_t conley_index;
  chomp::CombinatorialConleyIndex ( &conley_index, X_cubes, A_cubes, sizes, periodic, F, acyclic_check );
  // Return Conley index strings
  return conleyIndexString ( conley_index );
}

// The map that ComputeConleyIndexForCells hands to chomp::ConleyIndex: the
// ChompMap, except that the evaluation on the rectangles `domain`, the
// cells of S, returns `images`. ConleyIndex builds its pair from that one
// evaluation, so with the images of the index-pair check it builds the
// pair that was checked, and it does not evaluate the map on S again.
class ChompMapWithImages {
public:
  ChompMapWithImages ( const ChompMap & map,
                       const std::vector < std::shared_ptr < Geo > > & domain,
                       const std::vector < std::shared_ptr < Geo > > & images )
    : map_ ( map ), domain_ ( domain ), images_ ( images ) {}
  chomp::Rect operator () ( const chomp::Rect & rect ) const {
    return map_ ( rect );
  }
  std::shared_ptr < Geo > operator () ( const std::shared_ptr < Geo > & geo ) const {
    return map_ ( geo );
  }
  std::vector < chomp::Rect >
  images ( const std::vector < chomp::Rect > & rects ) const {
    return map_ . images ( rects );
  }
  std::vector < std::shared_ptr < Geo > >
  images ( const std::vector < std::shared_ptr < Geo > > & geos ) const {
    if ( isDomain ( geos ) ) return images_;
    return map_ . images ( geos );
  }
private:
  bool isDomain ( const std::vector < std::shared_ptr < Geo > > & geos ) const {
    if ( geos . size () != domain_ . size () ) return false;
    for ( size_t k = 0; k < geos . size (); ++ k ) {
      std::shared_ptr < RectGeo > rect =
        std::dynamic_pointer_cast < RectGeo > ( geos [ k ] );
      std::shared_ptr < RectGeo > cell =
        std::dynamic_pointer_cast < RectGeo > ( domain_ [ k ] );
      if ( not rect or not cell or
           rect -> lower_bounds != cell -> lower_bounds or
           rect -> upper_bounds != cell -> upper_bounds ) {
        return false;
      }
    }
    return true;
  }
  const ChompMap & map_;
  const std::vector < std::shared_ptr < Geo > > & domain_;
  const std::vector < std::shared_ptr < Geo > > & images_;
};

std::vector < std::string >
ComputeConleyIndexForCells (
    const Model & model,
    MorseGraph & morse_graph,
    std::vector < uint64_t > cells,
    uint64_t batch_chunk_size = 65536 ) {
  // Conley index of a cell set S of the final phase-space grid, from the
  // pair X = cover(F(S)), A = X \ S that chomp::ConleyIndex builds for a
  // Morse set. Of the index-pair conditions, F(X \ A) \subset X holds for
  // every S, since X \ A is part of S, but F(A) \cap X \subset A fails
  // when a cell of A maps into X \ A; the proof in chomp/ConleyIndex.h
  // derives it from S being a Morse set. That condition is checked here;
  // chomp::ConleyIndex then computes the index, as for the annotations of
  // ComputeConleyMorseGraph, from the same images of S.
  // Mixed-depth (adaptive-grid) cell sets are supported: the relative
  // complex subdivides the cells to the finest depth present in S, and
  // replaces cells of A deeper than that by their ancestors at that depth.
  // The check is made on the cells, not on those cubes.
  std::shared_ptr < const Map > map = model . map ();
  if ( not map ) {
    throw std::invalid_argument (
      "ComputeConleyIndexForCells requires a Model with a map" );
  }
  std::shared_ptr < TreeGrid > phase_space_chomp =
    std::dynamic_pointer_cast<TreeGrid> ( morse_graph . phaseSpace () );
  if ( not phase_space_chomp ) {
    throw std::runtime_error (
      "ComputeConleyIndexForCells requires a TreeGrid-backed Morse graph" );
  }
  if ( model . phase_dim () != phase_space_chomp -> dimension () ) {
    std::ostringstream message;
    message
      << "ComputeConleyIndexForCells: the Model's phase space has dimension "
      << model . phase_dim () << " but the Morse graph's grid has dimension "
      << phase_space_chomp -> dimension ();
    throw std::invalid_argument ( message . str () );
  }

  std::sort ( cells . begin (), cells . end () );
  cells . erase ( std::unique ( cells . begin (), cells . end () ), cells . end () );
  for ( const uint64_t cell : cells ) {
    if ( cell >= phase_space_chomp -> size () ) {
      std::ostringstream message;
      message
        << "ComputeConleyIndexForCells cell " << cell
        << " is outside [0, " << phase_space_chomp -> size () << ")";
      throw std::out_of_range ( message . str () );
    }
  }

  const TreeGrid & grid = * phase_space_chomp;
  ChompMap chomp_map ( map, batch_chunk_size );
  std::vector < std::shared_ptr < Geo > > S_geometries;
  S_geometries . reserve ( cells . size () );
  for ( const uint64_t cell : cells ) {
    S_geometries . push_back ( grid . geometry ( cell ) );
  }
  const std::vector < std::shared_ptr < Geo > > S_images =
    chomp_map . images ( S_geometries );
  std::vector < uint64_t > X;
  for ( const std::shared_ptr < Geo > & image : S_images ) {
    const std::vector < Grid::GridElement > image_cells = grid . cover ( image );
    X . insert ( X . end (), image_cells . begin (), image_cells . end () );
  }
  std::sort ( X . begin (), X . end () );
  X . erase ( std::unique ( X . begin (), X . end () ), X . end () );
  std::vector < uint64_t > A;
  for ( const uint64_t cell : X ) {
    if ( not std::binary_search ( cells . begin (), cells . end (), cell ) ) {
      A . push_back ( cell );
    }
  }

  // No cell of A may map into X \ A, the cells of S in X.
  std::vector < std::shared_ptr < Geo > > A_geometries;
  A_geometries . reserve ( A . size () );
  for ( const uint64_t cell : A ) {
    A_geometries . push_back ( grid . geometry ( cell ) );
  }
  const std::vector < std::shared_ptr < Geo > > exit_images =
    chomp_map . images ( A_geometries );
  uint64_t reentering = 0;
  uint64_t exit_cell = 0;
  uint64_t target_cell = 0;
  for ( size_t k = 0; k < A . size (); ++ k ) {
    std::vector < Grid::GridElement > targets = grid . cover ( exit_images [ k ] );
    std::sort ( targets . begin (), targets . end () );
    for ( const Grid::GridElement target : targets ) {
      if ( std::binary_search ( cells . begin (), cells . end (), target ) and
           std::binary_search ( X . begin (), X . end (), target ) ) {
        if ( reentering == 0 ) {
          exit_cell = A [ k ];
          target_cell = target;
        }
        ++ reentering;
        break;
      }
    }
  }
  if ( reentering > 0 ) {
    std::ostringstream message;
    message
      << "ComputeConleyIndexForCells: the cells S do not give an index pair. "
      << "With X = cover(F(S)) and A = X \\ S, no cell of A may map into "
      << "X \\ A, but cell " << exit_cell << " of A maps into cell "
      << target_cell << " of S (cells of A that map into X \\ A: "
      << reentering << " of " << A . size () << "). Sets that contain every "
      << "cell on a path between two of their cells pass, for example the "
      << "Morse sets of a run with equal initial, minimum and maximum "
      << "subdivision and the cells from MorseDirectedPathCells.";
    throw std::invalid_argument ( message . str () );
  }

  chomp::ConleyIndex_t conley_index;
  ChompMapWithImages map_with_images ( chomp_map, S_geometries, S_images );
  chomp::ConleyIndex ( & conley_index, grid, cells, map_with_images );
  return conleyIndexString ( conley_index );
}

// Shared body of the Compute*MorseGraph entry points.
//
// `initial_phase_space`, when non-null, receives the grid pointer captured
// *before* the decomposition runs. That is deliberate and must not be replaced
// by `morsegraph.phaseSpace()`: Compute_Morse_Graph reassigns the graph's own
// phase space to a joined master grid, so after the call the two are different
// objects, and the historical MapGraph construction uses the original.
//
// `options` governs the per-level transition-graph cache; its chunk_size also
// chunks the batched map evaluations of the Conley phase.
static MorseGraph ComputeMorseGraphCore (
    Model const& model,
    bool compute_conley_index,
    MapGraphOptions const& options,
    std::shared_ptr < Grid > * initial_phase_space ) {
  std::shared_ptr<const Map> map = model . map ();
  MorseGraph morsegraph ( model . phaseSpace () );
  std::shared_ptr < Grid > phase_space = morsegraph . phaseSpace ();
  if ( initial_phase_space ) * initial_phase_space = phase_space;

  int phase_subdiv_init = model . phase_subdiv_init ();
  int phase_subdiv_min = model . phase_subdiv_min ();
  int phase_subdiv_max = model . phase_subdiv_max ();
  int phase_subdiv_limit = model . phase_subdiv_limit ();

  // Compute Morse graph
  Compute_Morse_Graph ( & morsegraph, phase_space, map, phase_subdiv_init,
                        phase_subdiv_min, phase_subdiv_max, phase_subdiv_limit,
                        options );

  if ( compute_conley_index ) {
    std::shared_ptr < TreeGrid > phase_space_chomp =
      std::dynamic_pointer_cast<TreeGrid> ( morsegraph . phaseSpace () );

    if ( not phase_space_chomp ) {
      throw std::runtime_error ( "Cannot interface with chomp for this grid type!" );
    }

    typedef std::vector < Grid::GridElement > Subset;
    for ( size_t v = 0; v < morsegraph . NumVertices (); ++ v) {
      Subset subset = phase_space_chomp -> subset ( * morsegraph . grid ( v ) );
      std::shared_ptr<chomp::ConleyIndex_t> conley ( new chomp::ConleyIndex_t );
      morsegraph . conleyIndex ( v ) = conley;
      // The Conley phase gathers its map evaluations and runs them through
      // the batched evaluator when the model provides one, chunked by the
      // same batch_chunk_size that governs the transition-graph passes.
      ChompMap chomp_map ( map, options . chunk_size );
      chomp::ConleyIndex ( conley . get (), *phase_space_chomp, subset, chomp_map );
    }
  }

  return morsegraph;
}

// cache_transition_graph and cache_map_graph: None (the default) caches
// unless CMGDB_MAPGRAPH_CACHE=0; an explicit True or False is obeyed.
static bool ResolveCacheFlag ( std::optional<bool> const& flag ) {
  return flag ? * flag : cmgdb_detail::map_graph_cache_enabled ();
}

static MapGraphOptions TransitionGraphOptions (
    std::optional<bool> cache_transition_graph,
    uint64_t batch_chunk_size,
    uint64_t max_cached_edges,
    uint64_t reserve_edges,
    uint64_t reserve_min_edges ) {
  return MapGraphOptions ( ResolveCacheFlag ( cache_transition_graph ),
                           batch_chunk_size, max_cached_edges,
                           reserve_edges, reserve_min_edges );
}

// Compute multi-valued map digraph on the final grid. A cached graph costs
// one extra full (batched, if available) map pass over the grid and then
// answers adjacency queries without the map; a lazy one evaluates the map
// on demand per adjacency query and can be upgraded with build_cache().
// The callers resolve cache_map_graph before the computation, so that a
// malformed CMGDB_MAPGRAPH_CACHE fails before the first map evaluation.
// When a cache will be built they also read the CMGDB_MAPGRAPH_RESERVE_*
// and CMGDB_MAPGRAPH_HARD_MAX_* variables first (MapGraphEnv): with lazy
// transition graphs this graph's build would otherwise be the first to read
// them, after all Morse and Conley work.
// max_cached_edges bounds this cache too. When the caller asked for it
// explicitly (cache_map_graph=True) and the graph is over that limit, the
// graph comes back lazy with a RuntimeWarning rather than an error, which
// would lose the whole computation; build_cache() can still cache it.
static MapGraph ReturnedMapGraph ( std::shared_ptr < Grid > phase_space,
                                   std::shared_ptr<const Map> map,
                                   MapGraphOptions options,
                                   bool cache_map_graph,
                                   bool cache_requested ) {
  options . cache = cache_map_graph;
  MapGraph map_graph ( phase_space, map, options );
  if ( cache_requested and not map_graph . has_cache () ) {
    std::ostringstream message;
    message << "cache_map_graph=True, but the returned map_graph has more "
            << "than max_cached_edges=" << options . max_cached_edges
            << " edges, so it was left uncached; map_graph.build_cache() "
            << "caches it";
    py::gil_scoped_acquire gil;
    if ( PyErr_WarnEx ( PyExc_RuntimeWarning, message . str () . c_str (), 1 ) != 0 ) {
      throw py::error_already_set ();
    }
  }
  return map_graph;
}

std::pair<MorseGraph, MapGraph> ComputeConleyMorseGraph ( Model const& model,
                                                          std::optional<bool> cache_transition_graph = std::nullopt,
                                                          uint64_t batch_chunk_size = 65536,
                                                          uint64_t max_cached_edges = 0,
                                                          uint64_t reserve_edges = 0,
                                                          uint64_t reserve_min_edges = uint64_t ( 1 ) << 24,
                                                          std::optional<bool> cache_map_graph = std::nullopt ) {
  MapGraphOptions options = TransitionGraphOptions (
    cache_transition_graph, batch_chunk_size, max_cached_edges,
    reserve_edges, reserve_min_edges );
  const bool cache_returned = ResolveCacheFlag ( cache_map_graph );
  if ( options . cache or cache_returned ) cmgdb_detail::map_graph_env ();
  std::shared_ptr < Grid > phase_space;
  MorseGraph morsegraph = ComputeMorseGraphCore ( model, true, options, & phase_space );
  MapGraph map_graph = ReturnedMapGraph ( phase_space, model . map (), options,
                                          cache_returned,
                                          cache_map_graph . value_or ( false ) );

  return std::make_pair ( morsegraph, map_graph );
}

// As ComputeConleyMorseGraph, but without building the returned MapGraph.
//
// That MapGraph is a full extra pass of the box map over the entire phase
// space, built after all Morse and Conley work is done. Callers that do not
// need it (for instance, anything not computing regions of attraction) were
// paying roughly half of all box-map evaluations for an object they discard.
MorseGraph ComputeConleyMorseGraphOnly ( Model const& model,
                                         std::optional<bool> cache_transition_graph = std::nullopt,
                                         uint64_t batch_chunk_size = 65536,
                                         uint64_t max_cached_edges = 0,
                                         uint64_t reserve_edges = 0,
                                         uint64_t reserve_min_edges = uint64_t ( 1 ) << 24 ) {
  return ComputeMorseGraphCore (
    model, true,
    TransitionGraphOptions ( cache_transition_graph, batch_chunk_size,
                             max_cached_edges, reserve_edges, reserve_min_edges ),
    nullptr );
}

std::pair<MorseGraph, MapGraph> ComputeMorseGraph ( Model const& model,
                                                    std::optional<bool> cache_transition_graph = std::nullopt,
                                                    uint64_t batch_chunk_size = 65536,
                                                    uint64_t max_cached_edges = 0,
                                                    uint64_t reserve_edges = 0,
                                                    uint64_t reserve_min_edges = uint64_t ( 1 ) << 24,
                                                    std::optional<bool> cache_map_graph = std::nullopt ) {
  MapGraphOptions options = TransitionGraphOptions (
    cache_transition_graph, batch_chunk_size, max_cached_edges,
    reserve_edges, reserve_min_edges );
  const bool cache_returned = ResolveCacheFlag ( cache_map_graph );
  if ( options . cache or cache_returned ) cmgdb_detail::map_graph_env ();
  std::shared_ptr < Grid > phase_space;
  MorseGraph morsegraph = ComputeMorseGraphCore ( model, false, options, & phase_space );
  MapGraph map_graph = ReturnedMapGraph ( phase_space, model . map (), options,
                                          cache_returned,
                                          cache_map_graph . value_or ( false ) );

  return std::make_pair ( morsegraph, map_graph );
}

// As ComputeMorseGraph, but without building the returned MapGraph.
MorseGraph ComputeMorseGraphOnly ( Model const& model,
                                   std::optional<bool> cache_transition_graph = std::nullopt,
                                   uint64_t batch_chunk_size = 65536,
                                   uint64_t max_cached_edges = 0,
                                   uint64_t reserve_edges = 0,
                                   uint64_t reserve_min_edges = uint64_t ( 1 ) << 24 ) {
  return ComputeMorseGraphCore (
    model, false,
    TransitionGraphOptions ( cache_transition_graph, batch_chunk_size,
                             max_cached_edges, reserve_edges, reserve_min_edges ),
    nullptr );
}

// Atlas-backed counterpart.  The graph construction itself is grid-generic;
// Atlas::clone/subgrid/subdivide/join preserve chart tags throughout the
// adaptive decomposition.  Its caches follow CMGDB_MAPGRAPH_CACHE (default
// on) and the CMGDB_MAPGRAPH_* limits.
static MorseGraph ComputeAtlasMorseGraphCore (
    AtlasModel const& model,
    std::shared_ptr < Grid > * initial_phase_space ) {
  std::shared_ptr<const Map> map = model . map ();
  MorseGraph morsegraph ( model . phaseSpace () );
  std::shared_ptr<Grid> phase_space = morsegraph . phaseSpace ();
  if ( initial_phase_space ) * initial_phase_space = phase_space;

  Compute_Morse_Graph (
    & morsegraph,
    phase_space,
    map,
    model . phase_subdiv_init (),
    model . phase_subdiv_min (),
    model . phase_subdiv_max (),
    model . phase_subdiv_limit (),
    MapGraphOptions ( cmgdb_detail::map_graph_cache_enabled () ) );
  return morsegraph;
}

std::pair<MorseGraph, MapGraph>
ComputeMorseGraph ( AtlasModel const& model ) {
  std::shared_ptr<Grid> phase_space;
  MorseGraph morsegraph = ComputeAtlasMorseGraphCore ( model, & phase_space );
  MapGraph map_graph ( phase_space, model . map () );
  return std::make_pair ( morsegraph, map_graph );
}

MorseGraph
ComputeMorseGraphOnly ( AtlasModel const& model ) {
  return ComputeAtlasMorseGraphCore ( model, nullptr );
}

std::pair<MorseGraph, MapGraph>
ComputeConleyMorseGraph ( AtlasModel const& ) {
  throw std::logic_error (
    "Conley-index computation is not available directly on AtlasModel: "
    "supply a valid suspension index pair and carrier/chain map instead" );
}

MorseGraph
ComputeConleyMorseGraphOnly ( AtlasModel const& ) {
  throw std::logic_error (
    "Conley-index computation is not available directly on AtlasModel: "
    "supply a valid suspension index pair and carrier/chain map instead" );
}

// Native reachability queries on a cached MapGraph.
//
// The answers are defined on the cells of the returned map_graph: a cell
// reaches a Morse node when some directed path (possibly of length zero) leads
// from it to a cell of that node's Morse set. The queries never consult the
// Morse graph's edges and assume nothing about where the cycles of map_graph
// lie. On a hierarchical run (phase_subdiv_init < phase_subdiv_min) the Morse
// graph is computed on coarser grids than the final one, so its edges need not
// match reachability in map_graph, and map_graph can have cycles through cells
// outside the Morse sets, or through the cells of several Morse sets.
//
// Each binding runs the Check* function of its query with the GIL held and
// releases the GIL only for the Compute* function. build_cache() runs with the
// GIL held, so the has_cache() read cannot race a concurrent build, and the CSR
// of a cached MapGraph never changes again.

namespace {

void CheckCachedMapGraph ( const MapGraph & map_graph, const char * query ) {
  if ( not map_graph . has_cache () ) {
    throw std::runtime_error (
      std::string ( query ) + " requires a cached MapGraph; refusing to use "
      "on-demand map callbacks." );
  }
  if ( map_graph . num_vertices () > std::numeric_limits<uint32_t>::max () ) {
    throw std::runtime_error (
      std::string ( query ) + " currently supports at most 2^32-1 map vertices" );
  }
}

void CheckQueryVertices ( const MapGraph & map_graph,
                          const std::vector<uint64_t> & query_vertices,
                          const char * query ) {
  const uint64_t n = map_graph . num_vertices ();
  for ( const uint64_t vertex : query_vertices ) {
    if ( vertex >= n ) {
      std::ostringstream message;
      message
        << query << " query vertex " << vertex
        << " is outside [0, " << n << ")";
      throw std::out_of_range ( message . str () );
    }
  }
}

/// ReachabilitySweep
///    Every cell v of the cached map graph carries a value, initially its
///    seed, in a join semilattice whose join is `merge`. Sweep ( start ) runs
///    an iterative Tarjan search from start over the cells not swept yet. When
///    it returns, every swept cell holds the join of the seeds of all the
///    cells reachable from it, itself included.
///
///    Tarjan closes the strongly connected components in reverse topological
///    order, so a component's value is complete when it closes: it joins the
///    seeds of its own cells and the values of the components its edges reach,
///    all closed before it. Along the way a cell joins the value of each swept
///    successor and of each tree child as the child finishes. Every value
///    joined into a cell belongs to a cell it reaches, so the partial values
///    of open components are joined soundly too.
template < class Value, class Merge >
class ReachabilitySweep {
 public:
  ReachabilitySweep ( const MapGraph & map_graph,
                      const char * query,
                      std::vector<Value> & value,
                      Merge merge )
    : map_graph_ ( map_graph ),
      query_ ( query ),
      n_ ( map_graph . num_vertices () ),
      rank_ ( static_cast<size_t> ( n_ ), UNSEEN ),
      value_ ( value ),
      merge_ ( merge ) {}

  bool swept ( uint64_t vertex ) const { return rank_ [ vertex ] != UNSEEN; }

  void Sweep ( uint32_t start ) {
    if ( rank_ [ start ] != UNSEEN ) return;
    Open ( start );
    while ( not frames_ . empty () ) {
      Frame & frame = frames_ . back ();
      const MapGraph::AdjacencySpan adjacency =
        map_graph_ . adjacency_span ( frame . vertex );
      if ( adjacency . size () > std::numeric_limits<uint32_t>::max () ) {
        throw std::runtime_error (
          std::string ( query_ ) + " found more than 2^32-1 outgoing edges "
          "at one vertex" );
      }
      bool descended = false;
      while ( frame . next_adjacency < adjacency . size () ) {
        const uint64_t successor =
          adjacency . begin () [ frame . next_adjacency ];
        ++ frame . next_adjacency;
        if ( successor >= n_ ) {
          throw std::runtime_error (
            std::string ( query_ ) + " found an adjacency outside the MapGraph" );
        }
        if ( rank_ [ successor ] == UNSEEN ) {
          // Open invalidates frame.
          Open ( static_cast<uint32_t> ( successor ) );
          descended = true;
          break;
        }
        frame . lowlink = std::min ( frame . lowlink, rank_ [ successor ] );
        value_ [ frame . vertex ] =
          merge_ ( value_ [ frame . vertex ], value_ [ successor ] );
      }
      if ( descended ) continue;

      const uint32_t vertex = frame . vertex;
      const uint32_t lowlink = frame . lowlink;
      frames_ . pop_back ();
      if ( lowlink == rank_ [ vertex ] ) {
        // vertex is the root of its component, which closes now.
        const Value component_value = value_ [ vertex ];
        uint32_t member;
        do {
          member = component_stack_ . back ();
          component_stack_ . pop_back ();
          rank_ [ member ] = CLOSED;
          value_ [ member ] = component_value;
        } while ( member != vertex );
      }
      if ( not frames_ . empty () ) {
        Frame & parent = frames_ . back ();
        parent . lowlink = std::min ( parent . lowlink, lowlink );
        value_ [ parent . vertex ] =
          merge_ ( value_ [ parent . vertex ], value_ [ vertex ] );
      }
    }
  }

 private:
  // rank_ [ v ] is UNSEEN until v is swept, then its preorder number (1, 2,
  // ...) while its component is open, then CLOSED. Taking the minimum with
  // CLOSED, the largest uint32, never lowers a lowlink, so closed cells drop
  // out of the lowlink computation as Tarjan requires. (A preorder number of
  // 2^32-1, possible only on a graph of 2^32-1 cells, compares the same way
  // as CLOSED would.)
  static constexpr uint32_t UNSEEN = 0;
  static constexpr uint32_t CLOSED = std::numeric_limits<uint32_t>::max ();

  struct Frame {
    uint32_t vertex;
    uint32_t next_adjacency;
    uint32_t lowlink;
  };

  void Open ( uint32_t vertex ) {
    ++ preorder_;
    rank_ [ vertex ] = preorder_;
    component_stack_ . push_back ( vertex );
    frames_ . push_back ( { vertex, 0, preorder_ } );
  }

  const MapGraph & map_graph_;
  const char * query_;
  const uint64_t n_;
  std::vector<uint32_t> rank_;
  std::vector<Value> & value_;
  Merge merge_;
  uint32_t preorder_ = 0;
  std::vector<uint32_t> component_stack_;
  std::vector<Frame> frames_;
};

} // namespace

void
CheckMorseDirectedPathCells (
    const MapGraph & map_graph,
    const MorseGraph & morse_graph,
    const std::vector<uint64_t> & source_nodes,
    const std::vector<uint64_t> & target_nodes ) {
  CheckCachedMapGraph ( map_graph, "MorseDirectedPathCells" );
  if ( source_nodes . empty () or target_nodes . empty () ) {
    throw std::invalid_argument (
      "MorseDirectedPathCells requires nonempty source_nodes and target_nodes" );
  }
  const size_t number_of_morse_sets = morse_graph . NumVertices ();
  for ( const uint64_t node : source_nodes ) {
    if ( node >= number_of_morse_sets ) {
      std::ostringstream message;
      message
        << "MorseDirectedPathCells source node " << node
        << " is outside [0, " << number_of_morse_sets << ")";
      throw std::out_of_range ( message . str () );
    }
  }
  for ( const uint64_t node : target_nodes ) {
    if ( node >= number_of_morse_sets ) {
      std::ostringstream message;
      message
        << "MorseDirectedPathCells target node " << node
        << " is outside [0, " << number_of_morse_sets << ")";
      throw std::out_of_range ( message . str () );
    }
  }
}

// The cells reachable from a cell of a source Morse set that also reach a cell
// of a target Morse set, in increasing order. One sweep from the source cells
// visits exactly the forward-reachable cells and, with the target cells seeded
// 1, marks those that reach a target, so no reverse CSR is built. Requires
// CheckMorseDirectedPathCells.
std::vector<uint64_t>
ComputeMorseDirectedPathCells (
    const MapGraph & map_graph,
    const MorseGraph & morse_graph,
    const std::vector<uint64_t> & source_nodes,
    const std::vector<uint64_t> & target_nodes ) {
  const uint64_t n = map_graph . num_vertices ();
  std::vector<uint8_t> reaches_target ( static_cast<size_t> ( n ), 0 );
  for ( const uint64_t node : target_nodes ) {
    for ( const uint64_t cell : morse_graph . morse_set ( node ) ) {
      if ( cell >= n ) {
        throw std::runtime_error (
          "MorseDirectedPathCells found a target Morse cell outside the "
          "MapGraph" );
      }
      reaches_target [ cell ] = 1;
    }
  }

  ReachabilitySweep sweep (
    map_graph, "MorseDirectedPathCells", reaches_target,
    [] ( uint8_t left, uint8_t right ) -> uint8_t { return left | right; } );
  for ( const uint64_t node : source_nodes ) {
    for ( const uint64_t cell : morse_graph . morse_set ( node ) ) {
      if ( cell >= n ) {
        throw std::runtime_error (
          "MorseDirectedPathCells found a source Morse cell outside the "
          "MapGraph" );
      }
      sweep . Sweep ( static_cast<uint32_t> ( cell ) );
    }
  }

  std::vector<uint64_t> result;
  for ( uint64_t vertex = 0; vertex < n; ++ vertex ) {
    if ( sweep . swept ( vertex ) and reaches_target [ vertex ] ) {
      result . push_back ( vertex );
    }
  }
  return result;
}

void
CheckMorseReachabilityMasks (
    const MapGraph & map_graph,
    const MorseGraph & morse_graph,
    const std::vector<uint64_t> & query_vertices ) {
  CheckCachedMapGraph ( map_graph, "MorseReachabilityMasks" );
  const size_t number_of_morse_sets = morse_graph . NumVertices ();
  if ( number_of_morse_sets > 64 ) {
    std::ostringstream message;
    message
      << "MorseReachabilityMasks cannot encode " << number_of_morse_sets
      << " Morse nodes in a uint64 mask";
    throw std::runtime_error ( message . str () );
  }
  CheckQueryVertices ( map_graph, query_vertices, "MorseReachabilityMasks" );
}

// Bit i of a query cell's mask is set when the cell reaches a cell of Morse
// set i. Each Morse cell is seeded with the bit of its own node only.
// Requires CheckMorseReachabilityMasks.
std::vector<uint64_t>
ComputeMorseReachabilityMasks (
    const MapGraph & map_graph,
    const MorseGraph & morse_graph,
    const std::vector<uint64_t> & query_vertices ) {
  const uint64_t n = map_graph . num_vertices ();
  const size_t number_of_morse_sets = morse_graph . NumVertices ();
  std::vector<uint64_t> reach_mask ( static_cast<size_t> ( n ), 0 );
  for ( size_t node = 0; node < number_of_morse_sets; ++ node ) {
    for ( const uint64_t cell : morse_graph . morse_set ( node ) ) {
      if ( cell >= n ) {
        throw std::runtime_error (
          "MorseReachabilityMasks found a Morse cell outside the MapGraph" );
      }
      reach_mask [ cell ] |= uint64_t ( 1 ) << node;
    }
  }

  ReachabilitySweep sweep (
    map_graph, "MorseReachabilityMasks", reach_mask,
    [] ( uint64_t left, uint64_t right ) -> uint64_t { return left | right; } );
  std::vector<uint64_t> result ( query_vertices . size (), 0 );
  for ( size_t query_index = 0;
        query_index < query_vertices . size ();
        ++ query_index ) {
    const uint32_t query = static_cast<uint32_t> ( query_vertices [ query_index ] );
    sweep . Sweep ( query );
    result [ query_index ] = reach_mask [ query ];
  }
  return result;
}

void
CheckMorseSingletonReachability (
    const MapGraph & map_graph,
    const MorseGraph & morse_graph,
    const std::vector<uint64_t> & query_vertices ) {
  CheckCachedMapGraph ( map_graph, "MorseSingletonReachability" );
  CheckQueryVertices ( map_graph, query_vertices, "MorseSingletonReachability" );
  if ( static_cast<size_t> ( morse_graph . NumVertices () ) >
       static_cast<size_t> ( std::numeric_limits<int32_t>::max () ) ) {
    throw std::runtime_error (
      "MorseSingletonReachability cannot encode the Morse-node ids in int32" );
  }
}

// A query cell's summary is the id of the only Morse node it reaches, -1 when
// it reaches none and -2 when it reaches several. Each Morse cell is seeded
// with its own node only. Requires CheckMorseSingletonReachability.
std::vector<int32_t>
ComputeMorseSingletonReachability (
    const MapGraph & map_graph,
    const MorseGraph & morse_graph,
    const std::vector<uint64_t> & query_vertices ) {
  constexpr int32_t NO_MORSE_NODE = -1;
  constexpr int32_t MULTIPLE_MORSE_NODES = -2;
  const auto merge_summary =
    [=] ( int32_t left, int32_t right ) -> int32_t {
      if ( left == NO_MORSE_NODE ) return right;
      if ( right == NO_MORSE_NODE ) return left;
      if ( left == right ) return left;
      return MULTIPLE_MORSE_NODES;
    };

  const uint64_t n = map_graph . num_vertices ();
  const size_t number_of_morse_sets = morse_graph . NumVertices ();
  std::vector<int32_t> reach_summary (
    static_cast<size_t> ( n ), NO_MORSE_NODE );
  for ( size_t node = 0; node < number_of_morse_sets; ++ node ) {
    for ( const uint64_t cell : morse_graph . morse_set ( node ) ) {
      if ( cell >= n ) {
        throw std::runtime_error (
          "MorseSingletonReachability found a Morse cell outside the MapGraph" );
      }
      reach_summary [ cell ] =
        merge_summary ( reach_summary [ cell ], static_cast<int32_t> ( node ) );
    }
  }

  ReachabilitySweep sweep (
    map_graph, "MorseSingletonReachability", reach_summary, merge_summary );
  std::vector<int32_t> result ( query_vertices . size (), NO_MORSE_NODE );
  for ( size_t query_index = 0;
        query_index < query_vertices . size ();
        ++ query_index ) {
    const uint32_t query = static_cast<uint32_t> ( query_vertices [ query_index ] );
    sweep . Sweep ( query );
    result [ query_index ] = reach_summary [ query ];
  }
  return result;
}

void computeMorseGraph ( MorseGraph & morsegraph,
                         std::shared_ptr<const Map> map,
                         const int SINGLECMG_INIT_PHASE_SUBDIVISIONS,
                         const int SINGLECMG_MIN_PHASE_SUBDIVISIONS,
                         const int SINGLECMG_MAX_PHASE_SUBDIVISIONS,
                         const int SINGLECMG_COMPLEXITY_LIMIT,
                         const char * outputfile ) {
#ifdef CMG_VERBOSE
  std::cout << "SingleCMG: computeMorseGraph.\n";
#endif
  std::shared_ptr < Grid > phase_space = morsegraph . phaseSpace ();
  Compute_Morse_Graph ( & morsegraph,
                        phase_space,
                        map,
                        SINGLECMG_INIT_PHASE_SUBDIVISIONS,
                        SINGLECMG_MIN_PHASE_SUBDIVISIONS,
                        SINGLECMG_MAX_PHASE_SUBDIVISIONS,
                        SINGLECMG_COMPLEXITY_LIMIT,
                        MapGraphOptions ( cmgdb_detail::map_graph_cache_enabled () ) );
  if ( outputfile != NULL ) {
    morsegraph . save ( outputfile );
  }
}

MorseGraph MorseGraphIntvalMap ( int phase_subdiv_min, int phase_subdiv_max,
                                 std::vector<double> const& phase_lower_bounds,
                                 std::vector<double> const& phase_upper_bounds,
                                 std::vector<double> const& params,
                                 std::string output_file_name ) {
  std::vector<double> param_lower_bounds = params;
  std::vector<double> param_upper_bounds = params;
  int param_dim = params . size();
  int phase_dim = phase_lower_bounds . size();
  std::vector<bool> phase_periodic ( phase_dim, false );
  int phase_subdiv_init = 0;
  int phase_subdiv_limit = 10000;

  Model model;
  model . initialize ( param_dim, phase_dim,
                       phase_subdiv_min, phase_subdiv_max,
                       phase_subdiv_init, phase_subdiv_limit,
                       param_lower_bounds, param_upper_bounds,
                       phase_lower_bounds, phase_upper_bounds,
                       phase_periodic );
  std::shared_ptr<const Map> map = model . map ();

  MorseGraph morsegraph ( model . phaseSpace () );

  // INITIALIZE THE PHASE SPACE SUBDIVISION PARAMETERS
  int SINGLECMG_INIT_PHASE_SUBDIVISIONS = phase_subdiv_init;
  int SINGLECMG_MIN_PHASE_SUBDIVISIONS = phase_subdiv_min;
  int SINGLECMG_MAX_PHASE_SUBDIVISIONS = phase_subdiv_max;
  int SINGLECMG_COMPLEXITY_LIMIT= phase_subdiv_limit;

  // COMPUTE MORSE GRAPH
  computeMorseGraph ( morsegraph, map,
                      SINGLECMG_INIT_PHASE_SUBDIVISIONS,
                      SINGLECMG_MIN_PHASE_SUBDIVISIONS,
                      SINGLECMG_MAX_PHASE_SUBDIVISIONS,
                      SINGLECMG_COMPLEXITY_LIMIT,
                      output_file_name . c_str () );

#ifdef CMG_VERBOSE
  std::cout << "Total Time for Finding Morse Sets ";
  std::cout << "and reachability relation: ";
  std::cout << ": ";
#endif

  // Always output the Morse Graph
  // std::cout << "Creating graphviz .dot file...\n";
  // CreateDotFile ( "morsegraph.gv", conleymorsegraph );

  return morsegraph;
}

MorseGraph MorseGraphMap ( int phase_subdiv_min, int phase_subdiv_max,
                           std::vector<double> const& phase_lower_bounds,
                           std::vector<double> const& phase_upper_bounds,
                           std::string output_file_name,
                           std::function<std::vector<double>(std::vector<double>)> const& F ) {
  std::vector<double> params {0.0};
  std::vector<double> param_lower_bounds = params;
  std::vector<double> param_upper_bounds = params;
  int param_dim = params . size();
  int phase_dim = phase_lower_bounds . size();
  std::vector<bool> phase_periodic ( phase_dim, false );
  int phase_subdiv_init = 0;
  int phase_subdiv_limit = 10000;

  Model model;
  model . initialize ( param_dim, phase_dim,
                       phase_subdiv_min, phase_subdiv_max,
                       phase_subdiv_init, phase_subdiv_limit,
                       param_lower_bounds, param_upper_bounds,
                       phase_lower_bounds, phase_upper_bounds,
                       phase_periodic, F );
  std::shared_ptr<const Map> map = model . map ();

  MorseGraph morsegraph ( model . phaseSpace () );

  // INITIALIZE THE PHASE SPACE SUBDIVISION PARAMETERS
  int SINGLECMG_INIT_PHASE_SUBDIVISIONS = phase_subdiv_init;
  int SINGLECMG_MIN_PHASE_SUBDIVISIONS = phase_subdiv_min;
  int SINGLECMG_MAX_PHASE_SUBDIVISIONS = phase_subdiv_max;
  int SINGLECMG_COMPLEXITY_LIMIT= phase_subdiv_limit;

  // COMPUTE MORSE GRAPH
  computeMorseGraph ( morsegraph, map,
                      SINGLECMG_INIT_PHASE_SUBDIVISIONS,
                      SINGLECMG_MIN_PHASE_SUBDIVISIONS,
                      SINGLECMG_MAX_PHASE_SUBDIVISIONS,
                      SINGLECMG_COMPLEXITY_LIMIT,
                      output_file_name . c_str () );

#ifdef CMG_VERBOSE
  std::cout << "Total Time for Finding Morse Sets ";
  std::cout << "and reachability relation: ";
  std::cout << ": ";
#endif

  // Always output the Morse Graph
  // std::cout << "Creating graphviz .dot file...\n";
  // CreateDotFile ( "morsegraph.gv", conleymorsegraph );

  return morsegraph;
}

/// Python Bindings

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {

/// Values of a NumPy-compatible integer array, and its shape. An empty array
/// may have any dtype (``np.asarray([])`` is float64).
std::vector<int64_t>
IntegerArrayValues ( py::handle object,
                     const std::string & name,
                     std::vector<py::ssize_t> & shape,
                     const bool allow_bool ) {
  py::array array = py::array::ensure ( object );
  if ( ! array ) {
    throw py::type_error ( name + " must be convertible to a NumPy array" );
  }
  shape . assign ( array . shape (), array . shape () + array . ndim () );
  std::vector<int64_t> values;
  if ( array . size () == 0 ) return values;
  const char kind = array . dtype () . kind ();
  if ( kind != 'i' && kind != 'u' && ! ( allow_bool && kind == 'b' ) ) {
    throw py::type_error ( name + " must have an integer dtype" );
  }
  values . resize ( static_cast<size_t> ( array . size () ) );
  if ( kind == 'u' && array . itemsize () == 8 ) {
    py::array_t<uint64_t, py::array::c_style | py::array::forcecast> converted ( array );
    const uint64_t * data = converted . data ();
    for ( size_t i = 0; i < values . size (); ++ i ) {
      if ( data [ i ] > static_cast<uint64_t> ( std::numeric_limits<int64_t>::max () ) ) {
        std::ostringstream message;
        message << name << " has value " << data [ i ] << " at flat index " << i
                << ", outside the int64 range";
        throw std::invalid_argument ( message . str () );
      }
      values [ i ] = static_cast<int64_t> ( data [ i ] );
    }
  } else {
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> converted ( array );
    std::copy ( converted . data (), converted . data () + values . size (),
                values . begin () );
  }
  return values;
}

std::vector<int32_t>
Int32Values ( const std::vector<int64_t> & values, const std::string & name ) {
  std::vector<int32_t> result ( values . size () );
  for ( size_t i = 0; i < values . size (); ++ i ) {
    if ( values [ i ] < std::numeric_limits<int32_t>::min () ||
         values [ i ] > std::numeric_limits<int32_t>::max () ) {
      std::ostringstream message;
      message << name << " has value " << values [ i ] << " at flat index " << i
              << ", outside the int32 range";
      throw std::invalid_argument ( message . str () );
    }
    result [ i ] = static_cast<int32_t> ( values [ i ] );
  }
  return result;
}

std::string ShapeString ( const std::vector<py::ssize_t> & shape ) {
  std::ostringstream text;
  text << "(";
  for ( size_t k = 0; k < shape . size (); ++ k ) {
    if ( k > 0 ) text << ", ";
    text << shape [ k ];
  }
  if ( shape . size () == 1 ) text << ",";
  text << ")";
  return text . str ();
}

/// Cells of a simplicial complex given as a list over degrees d of integer
/// arrays of shape (n_d, d + 1); flat labels per degree.
std::vector<std::vector<int32_t> >
SimplexArrays ( py::handle object, const std::string & name ) {
  if ( ! py::isinstance<py::list> ( object ) && ! py::isinstance<py::tuple> ( object ) ) {
    throw py::type_error (
      name + " must be a list of integer arrays, one per degree" );
  }
  py::sequence degrees = py::reinterpret_borrow<py::sequence> ( object );
  if ( degrees . size () == 0 ) {
    throw std::invalid_argument ( name + " must contain the array of 0-cells" );
  }
  std::vector<std::vector<int32_t> > result;
  for ( size_t d = 0; d < degrees . size (); ++ d ) {
    const std::string label = name + "[" + std::to_string ( d ) + "]";
    std::vector<py::ssize_t> shape;
    const std::vector<int64_t> values =
      IntegerArrayValues ( degrees [ d ], label, shape, false );
    const py::ssize_t width = static_cast<py::ssize_t> ( d + 1 );
    const bool matrix = shape . size () == 2 && shape [ 1 ] == width;
    const bool vertex_list = d == 0 && shape . size () == 1;
    const bool empty_list = shape . size () == 1 && shape [ 0 ] == 0;
    if ( ! matrix && ! vertex_list && ! empty_list ) {
      std::ostringstream message;
      message << label << " must have shape (n, " << d + 1 << "); got "
              << ShapeString ( shape );
      throw std::invalid_argument ( message . str () );
    }
    result . push_back ( Int32Values ( values, label ) );
  }
  return result;
}

std::vector<uint8_t>
ExitMask ( py::handle object, const std::string & name ) {
  std::vector<py::ssize_t> shape;
  const std::vector<int64_t> values = IntegerArrayValues ( object, name, shape, true );
  if ( shape . size () != 1 ) {
    throw std::invalid_argument (
      name + " must be one-dimensional; got shape " + ShapeString ( shape ) );
  }
  std::vector<uint8_t> result ( values . size () );
  for ( size_t i = 0; i < values . size (); ++ i ) {
    if ( values [ i ] != 0 && values [ i ] != 1 ) {
      std::ostringstream message;
      message << name << "[" << i << "] = " << values [ i ] << " must be 0 or 1";
      throw std::invalid_argument ( message . str () );
    }
    result [ i ] = static_cast<uint8_t> ( values [ i ] );
  }
  return result;
}

std::vector<int64_t>
OneDimensionalValues ( py::handle object, const std::string & name ) {
  std::vector<py::ssize_t> shape;
  std::vector<int64_t> values = IntegerArrayValues ( object, name, shape, false );
  if ( shape . size () != 1 ) {
    throw std::invalid_argument (
      name + " must be one-dimensional; got shape " + ShapeString ( shape ) );
  }
  return values;
}

py::dict
CarrierChainMapToPython ( const carrier_chain_map::CarrierChainMapResult & result,
                          const bool return_carriers ) {
  py::dict output;
  output [ "status" ] = result . status;
  output [ "failure_degree" ] = result . failure_degree;
  output [ "failure_row" ] = result . failure_row;
  output [ "carrier_count" ] = result . carrier_count;
  if ( result . status == "ok" ) {
    if ( result . has_chain_map ) {
      py::list chain_map;
      for ( const std::vector<int64_t> & entries : result . chain_map ) {
        const py::ssize_t rows = static_cast<py::ssize_t> ( entries . size () / 3 );
        py::array_t<int64_t> array ( std::vector<py::ssize_t> { rows, 3 } );
        std::copy ( entries . begin (), entries . end (), array . mutable_data () );
        chain_map . append ( array );
      }
      output [ "chain_map" ] = chain_map;
    }
    if ( result . has_payload ) {
      py::dict payload;
      payload [ "cell_counts" ] = result . cell_counts;
      payload [ "boundary_entries" ] = result . boundary_entries;
      payload [ "chain_map_entries" ] = result . chain_map_entries;
      output [ "payload" ] = payload;
    }
  }
  if ( return_carriers ) {
    py::array_t<int64_t> ids ( static_cast<py::ssize_t> ( result . carrier_ids . size () ) );
    std::copy ( result . carrier_ids . begin (), result . carrier_ids . end (),
                ids . mutable_data () );
    output [ "carrier_ids" ] = ids;
  }
  return output;
}

} // namespace

PYBIND11_MODULE(_cmgdb, m) {
  GridBinding(m);
  ModelBinding(m);
  AtlasModelBinding(m);
  MapGraphBinding(m);
  MorseGraphBinding(m);
  MorseSetReachabilityBinding(m);

  m.doc() = "Conley Morse Graph Database Module";

  m.def("ComputeConleyIndex", &ComputeConleyIndex);
  m.def(
    "ComputeRelativeHomologyShiftClass",
    [] ( const std::vector<uint64_t> & cell_counts,
         const GradedSparseEntries & boundary_entries,
         const GradedSparseEntries & chain_map_entries ) {
      RelativeHomologyShiftResult result;
      {
        py::gil_scoped_release release;
        result = ComputeRelativeHomologyShiftClass (
          cell_counts, boundary_entries, chain_map_entries );
      }
      py::dict output;
      py::dict validation;
      validation [ "matrix_shapes_and_entries" ] = true;
      validation [ "boundary_squared_zero" ] = true;
      validation [ "chain_map_equation" ] = true;
      output [ "coefficient_field" ] = 5;
      output [ "cell_counts" ] = cell_counts;
      output [ "validation" ] = validation;
      output [ "homology_dimensions" ] = result . homology_dimensions;
      output [ "induced_maps" ] = result . induced_maps;
      output [ "shift_class" ] = result . shift_class;
      return output;
    },
    py::arg ( "cell_counts" ),
    py::arg ( "boundary_entries" ),
    py::arg ( "chain_map_entries" ),
    R"doc(
Compute a relative-homology shift class from an explicit finite chain map.

``cell_counts[d]`` is the number of basis cells in degree ``d``.
``boundary_entries[d]`` contains sparse ``(row, column, coefficient)`` entries
for the boundary ``C_d -> C_{d-1}``; its degree-zero list must be empty.
``chain_map_entries[d]`` contains sparse entries for the endomorphism of
``C_d``. Coefficients are reduced in CMGDB's coefficient field F_5.

The function validates matrix bounds and uniqueness, ``boundary^2 = 0``, and
the chain-map equation before computing the induced maps on homology. The
returned dictionary contains the homology dimensions, dense induced matrices,
and the usual CMGDB Frobenius/shift-class strings, one item per degree.

For a Conley-index computation, the supplied complex must be the relative
cellular chain complex of a valid index pair and the supplied chain map must be
a quotient-compatible chain selector carried by the outer approximation. This
API validates the algebraic data, but cannot certify that topological carrier
obligation from matrices alone.
)doc" );
  m.def(
    "ComputeRelativeShiftClass",
    [] ( const std::vector<uint64_t> & cell_counts,
         const GradedSparseEntries & boundary_entries,
         const GradedSparseEntries & chain_map_entries ) {
      relative_shift_class::RelativeShiftClassResult result;
      {
        py::gil_scoped_release release;
        result = relative_shift_class::ComputeRelativeShiftClass (
          cell_counts, boundary_entries, chain_map_entries );
      }
      py::dict output;
      py::dict validation;
      validation [ "matrix_shapes_and_entries" ] = true;
      validation [ "boundary_squared_zero" ] = true;
      validation [ "chain_map_equation" ] = true;
      output [ "coefficient_field" ] = 5;
      output [ "cell_counts" ] = cell_counts;
      output [ "validation" ] = validation;
      output [ "homology_dimensions" ] = result . homology_dimensions;
      output [ "induced_maps" ] = result . induced_maps;
      output [ "shift_class" ] = result . shift_class;
      return output;
    },
    py::arg ( "cell_counts" ),
    py::arg ( "boundary_entries" ),
    py::arg ( "chain_map_entries" ),
    R"doc(
Compute a relative-homology shift class from an explicit finite chain map, by
plain linear algebra over F_5.

The arguments, the validation (with its exceptions and messages) and the keys
and formats of the returned dictionary are those of
``ComputeRelativeHomologyShiftClass``: ``cell_counts[d]`` is the number of
basis cells in degree ``d``, ``boundary_entries[d]`` holds the sparse
``(row, column, coefficient)`` entries of the boundary ``C_d -> C_{d-1}`` (its
degree-zero list must be empty), and ``chain_map_entries[d]`` those of the
endomorphism of ``C_d``. Coefficients are reduced in F_5.

The boundary matrices are column reduced over F_5, without a Morse reduction
or a Smith normal form. The homology basis in degree ``d`` is a set of cycle
representatives chosen by the reduction, and ``induced_maps[d]`` has in column
``k`` the coordinates of the class of the image of the ``k``-th basis cycle,
with entries in -2..2. An induced matrix is determined up to a change of basis;
for a complex with zero boundary the basis is the standard one, so the induced
matrix is the given one. ``shift_class[d]`` is computed from the invariant
factors of the induced matrix and written as CMGDB writes shift classes: the
invariant factors with their powers of ``x`` removed, those of positive degree
concatenated in divisibility order, or ``"0"``.

Every step terminates. The GIL is released during the computation.
)doc" );
  m.def(
    "ComputeCarrierChainMap",
    [] ( py::object source_simplices,
         py::object vertex_image_indptr,
         py::object vertex_image_indices,
         py::object source_exit,
         py::object target_simplices,
         py::object target_exit,
         int64_t modulus,
         bool return_carriers,
         bool return_chain_map ) {
      if ( modulus != 5 ) {
        throw std::invalid_argument (
          "ComputeCarrierChainMap supports only modulus=5; got modulus="
          + std::to_string ( modulus ) );
      }
      carrier_chain_map::CarrierChainMapInput input;
      input . source_simplices = SimplexArrays ( source_simplices, "source_simplices" );
      input . vertex_image_indptr =
        OneDimensionalValues ( vertex_image_indptr, "vertex_image_indptr" );
      input . vertex_image_indices = Int32Values (
        OneDimensionalValues ( vertex_image_indices, "vertex_image_indices" ),
        "vertex_image_indices" );
      input . source_exit = ExitMask ( source_exit, "source_exit" );
      input . target_is_source = target_simplices . is_none ();
      if ( input . target_is_source ) {
        if ( ! target_exit . is_none () ) {
          throw std::invalid_argument (
            "target_exit is given only together with target_simplices" );
        }
      } else {
        if ( target_exit . is_none () ) {
          throw std::invalid_argument (
            "target_exit is required when target_simplices is given" );
        }
        input . target_simplices = SimplexArrays ( target_simplices, "target_simplices" );
        input . target_exit = ExitMask ( target_exit, "target_exit" );
      }
      input . return_chain_map = return_chain_map;
      carrier_chain_map::CarrierChainMapResult result;
      {
        py::gil_scoped_release release;
        result = carrier_chain_map::ComputeCarrierChainMap ( input );
      }
      return CarrierChainMapToPython ( result, return_carriers );
    },
    py::arg ( "source_simplices" ),
    py::arg ( "vertex_image_indptr" ),
    py::arg ( "vertex_image_indices" ),
    py::arg ( "source_exit" ),
    py::kw_only (),
    py::arg ( "target_simplices" ) = py::none (),
    py::arg ( "target_exit" ) = py::none (),
    py::arg ( "modulus" ) = 5,
    py::arg ( "return_carriers" ) = false,
    py::arg ( "return_chain_map" ) = true,
    R"doc(
Compute the chain map induced by an acyclic carrier over F_5.

``source_simplices[d]`` is an integer array of shape ``(n_d, d + 1)`` holding
the d-cells of a simplicial complex in complex order: every row is a strictly
increasing tuple of vertex labels (arbitrary int32 values), and the rows are in
strictly increasing lexicographic order. The row index of a cell is its
position in this order. The 0-cells may also be given as a one-dimensional
array. The complex must be closed under faces. ``target_simplices`` has the
same format; ``None`` means the target is the source complex.

``vertex_image_indptr`` (length ``n_0 + 1``) and ``vertex_image_indices`` are a
CSR map from every source vertex row to a set of target vertex labels;
duplicates are allowed. ``source_exit`` and ``target_exit`` are 0/1 masks over
the vertex rows marking the exit vertices, and ``P0`` is the subcomplex induced
on them. ``target_exit`` is required with ``target_simplices`` and not allowed
without it.

The carrier of a source cell ``s`` is the target subcomplex induced on the
union ``T(s)`` of the vertex images of its vertices. The function checks, in
this order, that every carrier is nonempty, that every distinct carrier is
acyclic over F_5 (Betti numbers ``(1, 0, ..., 0)``), and that every cell of the
source ``P0`` has its carrier in the target ``P0``. Acyclicity is checked once
for every distinct carrier, in complex order of first use, and the function
stops at the first cell whose carrier is not acyclic. It then
constructs the canonical chain selector: a vertex ``v`` goes to the smallest
vertex of ``T(v)``; a d-cell goes to the unique chain of its carrier with the
required boundary that is supported on the greedy independent d-cells of the
carrier, taken in complex order. The chain map is validated: ``d phi = phi d``
over F_5, every image lies in its carrier, and ``P0`` goes into ``P0``.

Returns a dict with ``status`` (``"ok"``, ``"empty_carrier"``,
``"not_acyclic"``, ``"pair_violation"``, ``"no_solution"`` or
``"chain_map_invalid"``), ``failure_degree`` and ``failure_row`` (the first
failing source cell in complex order, ``-1`` when the status is ok), and
``carrier_count`` (the number of distinct nonempty carriers). When the status
is ok, ``chain_map[d]`` is an int64 array of ``(source_row, target_row,
coefficient)`` rows with coefficients in 1..4, and, when the target is the
source, ``payload`` holds ``cell_counts``, ``boundary_entries`` and
``chain_map_entries`` of the relative complex ``C(X) / C(P0)`` in the argument
format of ``ComputeRelativeHomologyShiftClass``: the basis of degree d is the
d-cells outside ``P0`` in complex order, and entries at cells of ``P0`` are
dropped. With ``return_chain_map=False`` the dict holds no ``chain_map``, and
the arrays are not formed; the payload is unchanged. With
``return_carriers=True`` the dict also holds ``carrier_ids``, the carrier
number of every source cell in complex order (degree-major), or ``-1`` for an
empty carrier; carriers are numbered in order of first use.

When the status is ``"empty_carrier"`` or ``"not_acyclic"``, the function
stops at the failing cell, so ``carrier_count`` and ``carrier_ids`` describe
only the source cells before it in complex order: ``carrier_count`` is the
number of distinct carriers of those cells, and ``carrier_ids`` is ``-1`` from
the failing cell on.

Only ``modulus=5`` is supported. Invalid input raises ``ValueError``,
``IndexError`` or ``TypeError`` naming the offending position. The GIL is
released during the computation.
)doc" );
  const char * compute_kwargs_doc =
    "cache_transition_graph: cache the per-level transition graph used "
    "internally by the SCC/reachability passes during the computation "
    "(halves the map evaluations per level). None (default) = on unless "
    "CMGDB_MAPGRAPH_CACHE=0.\n"
    "batch_chunk_size: rectangles per batched map call (0 = whole grid).\n"
    "max_cached_edges: abandon a cache as soon as it would exceed this many "
    "edges and fall back to on-demand evaluation (0 = unlimited). This "
    "includes the returned map_graph's cache (cache_map_graph=True then "
    "warns), but not a later map_graph.build_cache(). The opt-in "
    "CMGDB_MAPGRAPH_HARD_MAX_{VERTICES,EDGES,CACHE_BYTES} limits raise "
    "instead.\n"
    "reserve_edges: reservation for the flat edge array, made after the "
    "first chunk; 0 = automatic (2x the projected edge count, renewed when "
    "the projection outgrows the array).\n"
    "reserve_min_edges: reservation engages only when the projected edge "
    "count reaches this size.\n"
    "cache_map_graph: cache the *returned* map_graph (one extra full "
    "batched map pass; required by the native reachability queries, "
    "csr_view and fast Python-side adjacency sweeps). None (default) = on "
    "unless CMGDB_MAPGRAPH_CACHE=0; False returns a lazy map_graph that "
    "evaluates the map per adjacency query, and map_graph.build_cache() "
    "upgrades it later. The *Only variants skip the returned map_graph.";
  m.def("ComputeConleyMorseGraph",
        py::overload_cast<Model const&, std::optional<bool>, uint64_t, uint64_t,
                          uint64_t, uint64_t, std::optional<bool>> ( &ComputeConleyMorseGraph ),
        py::arg("model"),
        py::arg("cache_transition_graph") = py::none(),
        py::arg("batch_chunk_size") = 65536,
        py::arg("max_cached_edges") = 0,
        py::arg("reserve_edges") = 0,
        py::arg("reserve_min_edges") = uint64_t ( 1 ) << 24,
        py::arg("cache_map_graph") = py::none(),
        compute_kwargs_doc);
  m.def("ComputeConleyMorseGraph",
        py::overload_cast<AtlasModel const&> ( &ComputeConleyMorseGraph ));
  m.def("ComputeMorseGraph",
        py::overload_cast<Model const&, std::optional<bool>, uint64_t, uint64_t,
                          uint64_t, uint64_t, std::optional<bool>> ( &ComputeMorseGraph ),
        py::arg("model"),
        py::arg("cache_transition_graph") = py::none(),
        py::arg("batch_chunk_size") = 65536,
        py::arg("max_cached_edges") = 0,
        py::arg("reserve_edges") = 0,
        py::arg("reserve_min_edges") = uint64_t ( 1 ) << 24,
        py::arg("cache_map_graph") = py::none(),
        compute_kwargs_doc);
  m.def("ComputeMorseGraph",
        py::overload_cast<AtlasModel const&> ( &ComputeMorseGraph ));
  m.def("ComputeConleyMorseGraphOnly",
        py::overload_cast<Model const&, std::optional<bool>, uint64_t, uint64_t,
                          uint64_t, uint64_t> ( &ComputeConleyMorseGraphOnly ),
        py::arg("model"),
        py::arg("cache_transition_graph") = py::none(),
        py::arg("batch_chunk_size") = 65536,
        py::arg("max_cached_edges") = 0,
        py::arg("reserve_edges") = 0,
        py::arg("reserve_min_edges") = uint64_t ( 1 ) << 24,
        "Conley-Morse graph without the extra returned MapGraph. Skips a full "
        "box-map pass over the phase space; use when the MapGraph is unused.");
  m.def("ComputeConleyMorseGraphOnly",
        py::overload_cast<AtlasModel const&> ( &ComputeConleyMorseGraphOnly ));
  m.def("ComputeMorseGraphOnly",
        py::overload_cast<Model const&, std::optional<bool>, uint64_t, uint64_t,
                          uint64_t, uint64_t> ( &ComputeMorseGraphOnly ),
        py::arg("model"),
        py::arg("cache_transition_graph") = py::none(),
        py::arg("batch_chunk_size") = 65536,
        py::arg("max_cached_edges") = 0,
        py::arg("reserve_edges") = 0,
        py::arg("reserve_min_edges") = uint64_t ( 1 ) << 24,
        "Morse graph without the extra returned MapGraph. Skips a full "
        "box-map pass over the phase space; use when the MapGraph is unused.");
  m.def("ComputeMorseGraphOnly",
        py::overload_cast<AtlasModel const&> ( &ComputeMorseGraphOnly ));
  m.def(
    "ComputeConleyIndexForCells",
    [] ( const Model & model,
         MorseGraph & morse_graph,
         std::vector<uint64_t> cells,
         uint64_t batch_chunk_size ) {
      std::vector<std::string> result;
      {
        py::gil_scoped_release release;
        result = ComputeConleyIndexForCells (
          model, morse_graph, std::move ( cells ), batch_chunk_size );
      }
      return result;
    },
    py::arg ( "model" ),
    py::arg ( "morse_graph" ),
    py::arg ( "cells" ),
    py::arg ( "batch_chunk_size" ) = 65536,
    R"doc(
Compute the homological Conley index of a set S of cells of the final
phase-space grid (``cells``, indices as in ``morse_graph.morse_set``) as
ComputeConleyMorseGraph computes it for its Morse sets, from the pair
X = cover(F(S)), A = X \ S.

The pair is an index pair exactly when no cell of A maps into X \ A, and
for cells of one depth the result is then the Conley index of the invariant
set in S. Otherwise the function raises ``ValueError``, naming a cell of A
and the cell of S it maps into. A set that contains every cell on a path
between two of its cells passes: the Morse sets of a run with equal initial,
minimum and maximum subdivision, and the cells returned by
``MorseDirectedPathCells``. A Morse set of an adaptive run can fail, since
the image of a small box under a sampled box map need not lie in the image
of the larger box that contains it.

For cells of different depths the homology is computed at the finest depth
among the cells of S: coarser cells are subdivided to it, and cells of A
deeper than it are replaced by their ancestors at that depth. The check is
made on the cells, not on these cubes. With a batch map attached
(``model.set_batch_map``), each of its calls takes at most
``batch_chunk_size`` rectangles, as in ``ComputeConleyMorseGraph`` (``0``
means no limit). Raises ``ValueError`` for a Model without a map or of
another dimension than the grid. The GIL is released during the computation.
)doc" );
  m.def(
    "MorseDirectedPathCells",
    [] ( const MapGraph & map_graph,
         const MorseGraph & morse_graph,
         const std::vector<uint64_t> & source_nodes,
         const std::vector<uint64_t> & target_nodes ) {
      CheckMorseDirectedPathCells (
        map_graph, morse_graph, source_nodes, target_nodes );
      std::vector<uint64_t> values;
      {
        py::gil_scoped_release release;
        values = ComputeMorseDirectedPathCells (
          map_graph, morse_graph, source_nodes, target_nodes );
      }
      py::array_t<uint64_t> result ( values . size () );
      auto output = result . mutable_unchecked<1> ();
      for ( size_t i = 0; i < values . size (); ++ i ) {
        output ( i ) = values [ i ];
      }
      return result;
    },
    py::arg ( "map_graph" ),
    py::arg ( "morse_graph" ),
    py::arg ( "source_nodes" ),
    py::arg ( "target_nodes" ),
    "Cells on some directed path in map_graph from a cell of a source Morse "
    "set to a cell of a target Morse set, in increasing order. Requires a "
    "cached map_graph (cache_map_graph=True or map_graph.build_cache())." );
  m.def(
    "MorseReachabilityMasks",
    [] ( const MapGraph & map_graph,
         const MorseGraph & morse_graph,
         const std::vector<uint64_t> & query_vertices ) {
      CheckMorseReachabilityMasks ( map_graph, morse_graph, query_vertices );
      std::vector<uint64_t> values;
      {
        py::gil_scoped_release release;
        values = ComputeMorseReachabilityMasks (
          map_graph, morse_graph, query_vertices );
      }
      py::array_t<uint64_t> result ( values . size () );
      auto output = result . mutable_unchecked<1> ();
      for ( size_t i = 0; i < values . size (); ++ i ) {
        output ( i ) = values [ i ];
      }
      return result;
    },
    py::arg ( "map_graph" ),
    py::arg ( "morse_graph" ),
    py::arg ( "query_vertices" ),
    "Per-query-cell uint64 bitmasks of the Morse nodes reachable through the "
    "box dynamics, that is, whose Morse sets the cell reaches in map_graph "
    "(bit i = Morse node i). Requires a cached map_graph." );
  m.def(
    "MorseSingletonReachability",
    [] ( const MapGraph & map_graph,
         const MorseGraph & morse_graph,
         const std::vector<uint64_t> & query_vertices ) {
      CheckMorseSingletonReachability ( map_graph, morse_graph, query_vertices );
      std::vector<int32_t> values;
      {
        py::gil_scoped_release release;
        values = ComputeMorseSingletonReachability (
          map_graph, morse_graph, query_vertices );
      }
      py::array_t<int32_t> result ( values . size () );
      auto output = result . mutable_unchecked<1> ();
      for ( size_t i = 0; i < values . size (); ++ i ) {
        output ( i ) = values [ i ];
      }
      return result;
    },
    py::arg ( "map_graph" ),
    py::arg ( "morse_graph" ),
    py::arg ( "query_vertices" ),
    "Per-query-cell summary of the Morse nodes whose Morse sets the cell "
    "reaches in map_graph: the node id when exactly one is reachable, -1 "
    "when none, -2 when several. Requires a cached map_graph." );
  m.def("MorseGraphIntvalMap", &MorseGraphIntvalMap);
  m.def("MorseGraphMap", &MorseGraphMap);
}
