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

// #define CMG_VERBOSE
#define MEMORYBOOKKEEPING

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
#include "SuccinctGrid.h"
BOOST_CLASS_EXPORT_IMPLEMENT(SuccinctGrid);
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

std::vector < std::string >
ComputeConleyIndexForCells (
    const Model & model,
    MorseGraph & morse_graph,
    std::vector < uint64_t > cells ) {
  std::shared_ptr < TreeGrid > phase_space_chomp =
    std::dynamic_pointer_cast<TreeGrid> ( morse_graph . phaseSpace () );
  if ( not phase_space_chomp ) {
    throw std::runtime_error (
      "ComputeConleyIndexForCells requires a TreeGrid-backed Morse graph" );
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

  chomp::ConleyIndex_t conley_index;
  ChompMap chomp_map ( model . map () );
  chomp::ConleyIndex (
    & conley_index, * phase_space_chomp, cells, chomp_map );
  return conleyIndexString ( conley_index );
}

// Shared body of the four Compute*MorseGraph entry points.
//
// `initial_phase_space`, when non-null, receives the grid pointer captured
// *before* the decomposition runs. That is deliberate and must not be replaced
// by `morsegraph.phaseSpace()`: Compute_Morse_Graph reassigns the graph's own
// phase space to a joined master grid, so after the call the two are different
// objects, and the historical MapGraph construction uses the original.
static MorseGraph ComputeMorseGraphCore (
    Model const& model,
    bool compute_conley_index,
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
                        phase_subdiv_min, phase_subdiv_max, phase_subdiv_limit );

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
      ChompMap chomp_map ( map );
      chomp::ConleyIndex ( conley . get (), *phase_space_chomp, subset, chomp_map );
    }
  }

  return morsegraph;
}

std::pair<MorseGraph, MapGraph> ComputeConleyMorseGraph ( Model const& model ) {
  std::shared_ptr < Grid > phase_space;
  MorseGraph morsegraph = ComputeMorseGraphCore ( model, true, & phase_space );

  // Compute multi-valued map digraph
  MapGraph map_graph ( phase_space, model . map () );

  return std::make_pair ( morsegraph, map_graph );
}

// As ComputeConleyMorseGraph, but without building the returned MapGraph.
//
// That MapGraph is a full extra pass of the box map over the entire phase
// space, built after all Morse and Conley work is done. Callers that do not
// need it (for instance, anything not computing regions of attraction) were
// paying roughly half of all box-map evaluations for an object they discard.
MorseGraph ComputeConleyMorseGraphOnly ( Model const& model ) {
  return ComputeMorseGraphCore ( model, true, nullptr );
}

std::pair<MorseGraph, MapGraph> ComputeMorseGraph ( Model const& model ) {
  std::shared_ptr < Grid > phase_space;
  MorseGraph morsegraph = ComputeMorseGraphCore ( model, false, & phase_space );

  // Compute multi-valued map digraph
  MapGraph map_graph ( phase_space, model . map () );

  return std::make_pair ( morsegraph, map_graph );
}

// As ComputeMorseGraph, but without building the returned MapGraph.
MorseGraph ComputeMorseGraphOnly ( Model const& model ) {
  return ComputeMorseGraphCore ( model, false, nullptr );
}

// Atlas-backed counterpart.  The graph construction itself is grid-generic;
// Atlas::clone/subgrid/subdivide/join preserve chart tags throughout the
// adaptive decomposition.
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
    model . phase_subdiv_limit () );
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

std::vector<uint64_t>
ComputeMorseDirectedPathCells (
    const MapGraph & map_graph,
    const MorseGraph & morse_graph,
    const std::vector<uint64_t> & source_nodes,
    const std::vector<uint64_t> & target_nodes ) {
  if ( not map_graph . has_cache () ) {
    throw std::runtime_error (
      "MorseDirectedPathCells requires a cached MapGraph; refusing to use "
      "on-demand map callbacks." );
  }

  const uint64_t n = map_graph . num_vertices ();
  if ( n > std::numeric_limits<uint32_t>::max () ) {
    throw std::runtime_error (
      "MorseDirectedPathCells currently supports at most 2^32-1 map vertices" );
  }
  const size_t number_of_morse_sets = morse_graph . NumVertices ();
  if ( source_nodes . empty () or target_nodes . empty () ) {
    throw std::invalid_argument (
      "MorseDirectedPathCells requires nonempty source_nodes and target_nodes" );
  }

  std::vector<uint8_t> is_source_node ( number_of_morse_sets, 0 );
  std::vector<uint8_t> can_reach_target_node ( number_of_morse_sets, 0 );
  for ( const uint64_t node : source_nodes ) {
    if ( node >= number_of_morse_sets ) {
      std::ostringstream message;
      message
        << "MorseDirectedPathCells source node " << node
        << " is outside [0, " << number_of_morse_sets << ")";
      throw std::out_of_range ( message . str () );
    }
    is_source_node [ node ] = 1;
  }
  for ( const uint64_t node : target_nodes ) {
    if ( node >= number_of_morse_sets ) {
      std::ostringstream message;
      message
        << "MorseDirectedPathCells target node " << node
        << " is outside [0, " << number_of_morse_sets << ")";
      throw std::out_of_range ( message . str () );
    }
    can_reach_target_node [ node ] = 1;
  }

  // Recurrent cells are terminals for the backward dynamic program. Seed each
  // fine Morse component by whether its node can reach a requested target in
  // the Morse DAG. This preserves all downstream paths while breaking every
  // cell-level directed cycle at its recurrent component.
  const std::vector<std::pair<uint64_t, uint64_t>> morse_edges =
    morse_graph . edges_unreduced ();
  for ( size_t pass = 0; pass < number_of_morse_sets; ++ pass ) {
    bool changed = false;
    for ( const auto & edge : morse_edges ) {
      if ( edge . first >= number_of_morse_sets or
           edge . second >= number_of_morse_sets ) {
        throw std::runtime_error (
          "MorseDirectedPathCells received an invalid Morse-graph edge" );
      }
      if ( can_reach_target_node [ edge . second ] and
           not can_reach_target_node [ edge . first ] ) {
        can_reach_target_node [ edge . first ] = 1;
        changed = true;
      }
    }
    if ( not changed ) break;
  }

  std::vector<uint8_t> forward ( static_cast<size_t> ( n ), 0 );
  std::vector<uint32_t> vertex_stack;
  for ( size_t node = 0; node < number_of_morse_sets; ++ node ) {
    if ( not is_source_node [ node ] ) continue;
    const std::vector<uint64_t> cells = morse_graph . morse_set ( node );
    for ( const uint64_t cell : cells ) {
      if ( cell >= n ) {
        throw std::runtime_error (
          "MorseDirectedPathCells found a source Morse cell outside the "
          "MapGraph" );
      }
      if ( not forward [ cell ] ) {
        forward [ cell ] = 1;
        vertex_stack . push_back ( static_cast<uint32_t> ( cell ) );
      }
    }
  }

  while ( not vertex_stack . empty () ) {
    const uint32_t source = vertex_stack . back ();
    vertex_stack . pop_back ();
    const MapGraph::AdjacencyView adjacency =
      map_graph . adjacencies_view ( source );
    for ( const uint64_t successor : adjacency ) {
      if ( successor >= n ) {
        throw std::runtime_error (
          "MorseDirectedPathCells found an adjacency outside the MapGraph" );
      }
      if ( not forward [ successor ] ) {
        forward [ successor ] = 1;
        vertex_stack . push_back ( static_cast<uint32_t> ( successor ) );
      }
    }
  }

  enum VisitState : uint8_t { UNSEEN = 0, ACTIVE = 1, DONE = 2 };
  std::vector<uint8_t> state ( static_cast<size_t> ( n ), UNSEEN );
  std::vector<uint8_t> can_reach_target ( static_cast<size_t> ( n ), 0 );
  for ( size_t node = 0; node < number_of_morse_sets; ++ node ) {
    const std::vector<uint64_t> cells = morse_graph . morse_set ( node );
    for ( const uint64_t cell : cells ) {
      if ( cell >= n ) {
        throw std::runtime_error (
          "MorseDirectedPathCells found a recurrent cell outside the MapGraph" );
      }
      state [ cell ] = DONE;
      can_reach_target [ cell ] = can_reach_target_node [ node ];
    }
  }

  struct Frame {
    uint32_t vertex;
    uint32_t next_adjacency;
  };
  std::vector<Frame> stack;
  for ( uint64_t raw_vertex = 0; raw_vertex < n; ++ raw_vertex ) {
    if ( not forward [ raw_vertex ] or state [ raw_vertex ] != UNSEEN ) continue;
    state [ raw_vertex ] = ACTIVE;
    stack . push_back ( { static_cast<uint32_t> ( raw_vertex ), 0 } );
    while ( not stack . empty () ) {
      Frame & frame = stack . back ();
      const MapGraph::AdjacencyView adjacency =
        map_graph . adjacencies_view ( frame . vertex );
      if ( adjacency . size () > std::numeric_limits<uint32_t>::max () ) {
        throw std::runtime_error (
          "MorseDirectedPathCells found more than 2^32-1 outgoing edges "
          "at one vertex" );
      }
      if ( frame . next_adjacency == adjacency . size () ) {
        state [ frame . vertex ] = DONE;
        stack . pop_back ();
        continue;
      }

      const uint64_t successor =
        adjacency . begin () [ frame . next_adjacency ];
      if ( successor >= n ) {
        throw std::runtime_error (
          "MorseDirectedPathCells found an adjacency outside the MapGraph" );
      }
      if ( state [ successor ] == UNSEEN ) {
        state [ successor ] = ACTIVE;
        stack . push_back (
          { static_cast<uint32_t> ( successor ), 0 } );
        continue;
      }
      if ( state [ successor ] == ACTIVE ) {
        throw std::runtime_error (
          "MorseDirectedPathCells found a directed cycle not covered by "
          "the supplied Morse sets" );
      }

      can_reach_target [ frame . vertex ] =
        can_reach_target [ frame . vertex ] or
        can_reach_target [ successor ];
      ++ frame . next_adjacency;
    }
  }

  std::vector<uint64_t> result;
  for ( uint64_t vertex = 0; vertex < n; ++ vertex ) {
    if ( forward [ vertex ] and can_reach_target [ vertex ] ) {
      result . push_back ( vertex );
    }
  }
  return result;
}

std::vector<uint64_t>
ComputeMorseReachabilityMasks (
    const MapGraph & map_graph,
    const MorseGraph & morse_graph,
    const std::vector<uint64_t> & query_vertices ) {
  if ( not map_graph . has_cache () ) {
    throw std::runtime_error (
      "MorseReachabilityMasks requires a cached MapGraph; refusing to use "
      "on-demand map callbacks." );
  }

  const uint64_t n = map_graph . num_vertices ();
  if ( n > std::numeric_limits<uint32_t>::max () ) {
    throw std::runtime_error (
      "MorseReachabilityMasks currently supports at most 2^32-1 map vertices" );
  }

  const size_t number_of_morse_sets = morse_graph . NumVertices ();
  if ( number_of_morse_sets > 64 ) {
    std::ostringstream message;
    message
      << "MorseReachabilityMasks cannot encode " << number_of_morse_sets
      << " Morse nodes in a uint64 mask";
    throw std::runtime_error ( message . str () );
  }

  for ( const uint64_t query : query_vertices ) {
    if ( query >= n ) {
      std::ostringstream message;
      message
        << "MorseReachabilityMasks query vertex " << query
        << " is outside [0, " << n << ")";
      throw std::out_of_range ( message . str () );
    }
  }

  // Bit i denotes reachability to Morse node i. Seed each recurrent component
  // with the transitive closure of its node in the Morse DAG. Once seeded,
  // recurrent cells are terminals: their closure already contains every
  // downstream Morse node, and skipping their cell-level outgoing edges breaks
  // every directed cycle in the remaining graph.
  std::vector<uint64_t> morse_masks ( number_of_morse_sets, 0 );
  for ( size_t node = 0; node < number_of_morse_sets; ++ node ) {
    morse_masks [ node ] = uint64_t ( 1 ) << node;
  }
  const std::vector<std::pair<uint64_t, uint64_t>> morse_edges =
    morse_graph . edges_unreduced ();
  for ( size_t pass = 0; pass < number_of_morse_sets; ++ pass ) {
    bool changed = false;
    for ( const auto & edge : morse_edges ) {
      if ( edge . first >= number_of_morse_sets or
           edge . second >= number_of_morse_sets ) {
        throw std::runtime_error (
          "MorseReachabilityMasks received an invalid Morse-graph edge" );
      }
      const uint64_t updated =
        morse_masks [ edge . first ] | morse_masks [ edge . second ];
      if ( updated != morse_masks [ edge . first ] ) {
        morse_masks [ edge . first ] = updated;
        changed = true;
      }
    }
    if ( not changed ) break;
  }

  enum VisitState : uint8_t { UNSEEN = 0, ACTIVE = 1, DONE = 2 };
  std::vector<uint8_t> state ( static_cast<size_t> ( n ), UNSEEN );
  std::vector<uint64_t> reach_mask ( static_cast<size_t> ( n ), 0 );

  for ( size_t node = 0; node < number_of_morse_sets; ++ node ) {
    const std::vector<uint64_t> cells = morse_graph . morse_set ( node );
    for ( const uint64_t cell : cells ) {
      if ( cell >= n ) {
        throw std::runtime_error (
          "MorseReachabilityMasks found a Morse cell outside the MapGraph" );
      }
      state [ cell ] = DONE;
      reach_mask [ cell ] |= morse_masks [ node ];
    }
  }

  struct Frame {
    uint32_t vertex;
    uint32_t next_adjacency;
  };
  std::vector<Frame> stack;
  std::vector<uint64_t> result ( query_vertices . size (), 0 );

  for ( size_t query_index = 0;
        query_index < query_vertices . size ();
        ++ query_index ) {
    const uint32_t query = static_cast<uint32_t> ( query_vertices [ query_index ] );
    if ( state [ query ] == UNSEEN ) {
      state [ query ] = ACTIVE;
      stack . push_back ( { query, 0 } );
      while ( not stack . empty () ) {
        Frame & frame = stack . back ();
        const MapGraph::AdjacencyView adjacency =
          map_graph . adjacencies_view ( frame . vertex );
        if ( adjacency . size () > std::numeric_limits<uint32_t>::max () ) {
          throw std::runtime_error (
            "MorseReachabilityMasks found more than 2^32-1 outgoing edges "
            "at one vertex" );
        }

        if ( frame . next_adjacency == adjacency . size () ) {
          state [ frame . vertex ] = DONE;
          stack . pop_back ();
          continue;
        }

        const uint64_t successor =
          adjacency . begin () [ frame . next_adjacency ];
        if ( successor >= n ) {
          throw std::runtime_error (
            "MorseReachabilityMasks found an adjacency outside the MapGraph" );
        }
        if ( state [ successor ] == UNSEEN ) {
          state [ successor ] = ACTIVE;
          stack . push_back (
            { static_cast<uint32_t> ( successor ), 0 } );
          continue;
        }
        if ( state [ successor ] == ACTIVE ) {
          throw std::runtime_error (
            "MorseReachabilityMasks found a directed cycle not covered by "
            "the supplied Morse sets" );
        }

        reach_mask [ frame . vertex ] |= reach_mask [ successor ];
        ++ frame . next_adjacency;
      }
    }
    result [ query_index ] = reach_mask [ query ];
  }

  return result;
}

std::vector<int32_t>
ComputeMorseSingletonReachability (
    const MapGraph & map_graph,
    const MorseGraph & morse_graph,
    const std::vector<uint64_t> & query_vertices ) {
  if ( not map_graph . has_cache () ) {
    throw std::runtime_error (
      "MorseSingletonReachability requires a cached MapGraph; refusing to use "
      "on-demand map callbacks." );
  }

  const uint64_t n = map_graph . num_vertices ();
  if ( n > std::numeric_limits<uint32_t>::max () ) {
    throw std::runtime_error (
      "MorseSingletonReachability currently supports at most 2^32-1 map "
      "vertices" );
  }
  for ( const uint64_t query : query_vertices ) {
    if ( query >= n ) {
      std::ostringstream message;
      message
        << "MorseSingletonReachability query vertex " << query
        << " is outside [0, " << n << ")";
      throw std::out_of_range ( message . str () );
    }
  }

  constexpr int32_t NO_MORSE_NODE = -1;
  constexpr int32_t MULTIPLE_MORSE_NODES = -2;
  const size_t number_of_morse_sets = morse_graph . NumVertices ();
  if ( number_of_morse_sets >
       static_cast<size_t> ( std::numeric_limits<int32_t>::max () ) ) {
    throw std::runtime_error (
      "MorseSingletonReachability cannot encode the Morse-node ids in int32" );
  }

  // A recurrent node is singleton-reachable exactly when it has no outgoing
  // edge to a distinct Morse node. Otherwise its reachable set already
  // contains itself plus at least one other node, so MULTIPLE is sufficient;
  // the full set is not needed for Marcio's strict singleton-basin criterion.
  std::vector<int32_t> morse_summary ( number_of_morse_sets, NO_MORSE_NODE );
  for ( size_t node = 0; node < number_of_morse_sets; ++ node ) {
    morse_summary [ node ] = static_cast<int32_t> ( node );
  }
  for ( const auto & edge : morse_graph . edges_unreduced () ) {
    if ( edge . first >= number_of_morse_sets or
         edge . second >= number_of_morse_sets ) {
      throw std::runtime_error (
        "MorseSingletonReachability received an invalid Morse-graph edge" );
    }
    if ( edge . first != edge . second ) {
      morse_summary [ edge . first ] = MULTIPLE_MORSE_NODES;
    }
  }

  const auto merge_summary =
    [=] ( int32_t left, int32_t right ) -> int32_t {
      if ( left == NO_MORSE_NODE ) return right;
      if ( right == NO_MORSE_NODE ) return left;
      if ( left == right ) return left;
      return MULTIPLE_MORSE_NODES;
    };

  enum VisitState : uint8_t { UNSEEN = 0, ACTIVE = 1, DONE = 2 };
  std::vector<uint8_t> state ( static_cast<size_t> ( n ), UNSEEN );
  std::vector<int32_t> reach_summary (
    static_cast<size_t> ( n ), NO_MORSE_NODE );

  for ( size_t node = 0; node < number_of_morse_sets; ++ node ) {
    const std::vector<uint64_t> cells = morse_graph . morse_set ( node );
    for ( const uint64_t cell : cells ) {
      if ( cell >= n ) {
        throw std::runtime_error (
          "MorseSingletonReachability found a Morse cell outside the MapGraph" );
      }
      state [ cell ] = DONE;
      reach_summary [ cell ] =
        merge_summary ( reach_summary [ cell ], morse_summary [ node ] );
    }
  }

  struct Frame {
    uint32_t vertex;
    uint32_t next_adjacency;
  };
  std::vector<Frame> stack;
  std::vector<int32_t> result (
    query_vertices . size (), NO_MORSE_NODE );

  for ( size_t query_index = 0;
        query_index < query_vertices . size ();
        ++ query_index ) {
    const uint32_t query = static_cast<uint32_t> ( query_vertices [ query_index ] );
    if ( state [ query ] == UNSEEN ) {
      state [ query ] = ACTIVE;
      stack . push_back ( { query, 0 } );
      while ( not stack . empty () ) {
        Frame & frame = stack . back ();
        const MapGraph::AdjacencyView adjacency =
          map_graph . adjacencies_view ( frame . vertex );
        if ( adjacency . size () > std::numeric_limits<uint32_t>::max () ) {
          throw std::runtime_error (
            "MorseSingletonReachability found more than 2^32-1 outgoing "
            "edges at one vertex" );
        }

        if ( frame . next_adjacency == adjacency . size () ) {
          state [ frame . vertex ] = DONE;
          stack . pop_back ();
          continue;
        }

        const uint64_t successor =
          adjacency . begin () [ frame . next_adjacency ];
        if ( successor >= n ) {
          throw std::runtime_error (
            "MorseSingletonReachability found an adjacency outside the "
            "MapGraph" );
        }
        if ( state [ successor ] == UNSEEN ) {
          state [ successor ] = ACTIVE;
          stack . push_back (
            { static_cast<uint32_t> ( successor ), 0 } );
          continue;
        }
        if ( state [ successor ] == ACTIVE ) {
          throw std::runtime_error (
            "MorseSingletonReachability found a directed cycle not covered "
            "by the supplied Morse sets" );
        }

        reach_summary [ frame . vertex ] = merge_summary (
          reach_summary [ frame . vertex ], reach_summary [ successor ] );
        ++ frame . next_adjacency;
      }
    }
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
                        SINGLECMG_COMPLEXITY_LIMIT );
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
    "ComputeConleyIndexForCells",
    [] ( const Model & model,
         MorseGraph & morse_graph,
         std::vector<uint64_t> cells ) {
      std::vector<std::string> result;
      {
        py::gil_scoped_release release;
        result = ComputeConleyIndexForCells (
          model, morse_graph, std::move ( cells ) );
      }
      return result;
    },
    py::arg ( "model" ),
    py::arg ( "morse_graph" ),
    py::arg ( "cells" ),
    "Compute the Conley index of an arbitrary phase-space cell subset." );
  m.def("ComputeConleyMorseGraph",
        py::overload_cast<Model const&> ( &ComputeConleyMorseGraph ));
  m.def("ComputeConleyMorseGraph",
        py::overload_cast<AtlasModel const&> ( &ComputeConleyMorseGraph ));
  m.def("ComputeMorseGraph",
        py::overload_cast<Model const&> ( &ComputeMorseGraph ));
  m.def("ComputeMorseGraph",
        py::overload_cast<AtlasModel const&> ( &ComputeMorseGraph ));
  m.def("ComputeConleyMorseGraphOnly",
        py::overload_cast<Model const&> ( &ComputeConleyMorseGraphOnly ),
        "Conley-Morse graph without the extra returned MapGraph. Skips a full "
        "box-map pass over the phase space; use when the MapGraph is unused.");
  m.def("ComputeConleyMorseGraphOnly",
        py::overload_cast<AtlasModel const&> ( &ComputeConleyMorseGraphOnly ));
  m.def("ComputeMorseGraphOnly",
        py::overload_cast<Model const&> ( &ComputeMorseGraphOnly ),
        "Morse graph without the extra returned MapGraph. Skips a full "
        "box-map pass over the phase space; use when the MapGraph is unused.");
  m.def("ComputeMorseGraphOnly",
        py::overload_cast<AtlasModel const&> ( &ComputeMorseGraphOnly ));
  m.def(
    "MorseDirectedPathCells",
    [] ( const MapGraph & map_graph,
         const MorseGraph & morse_graph,
         const std::vector<uint64_t> & source_nodes,
         const std::vector<uint64_t> & target_nodes ) {
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
    py::arg ( "target_nodes" ) );
  m.def(
    "MorseReachabilityMasks",
    [] ( const MapGraph & map_graph,
         const MorseGraph & morse_graph,
         const std::vector<uint64_t> & query_vertices ) {
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
    py::arg ( "query_vertices" ) );
  m.def(
    "MorseSingletonReachability",
    [] ( const MapGraph & map_graph,
         const MorseGraph & morse_graph,
         const std::vector<uint64_t> & query_vertices ) {
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
    py::arg ( "query_vertices" ) );
  m.def("MorseGraphIntvalMap", &MorseGraphIntvalMap);
  m.def("MorseGraphMap", &MorseGraphMap);
}
