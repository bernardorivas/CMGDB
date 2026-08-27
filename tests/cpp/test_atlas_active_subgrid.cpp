// Deterministic native checks for Atlas selected-dyadic active grids.
//
// Build:
//   c++ -std=c++17 -I src/CMGDB/_cmgdb/include/database \
//     tests/cpp/test_atlas_active_subgrid.cpp -o atlas_active_test

#include <cmath>
#include <cstdio>
#include <deque>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "Atlas.h"
#include "AtlasModelMapF.h"
#include "MapGraph.h"
#include "join.h"

static int failures = 0;

#define CHECK(condition)                                                    \
  do {                                                                      \
    if ( not ( condition ) ) {                                              \
      std::printf ( "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition );   \
      ++ failures;                                                          \
    }                                                                       \
  } while ( 0 )

static bool
near ( double first, double second ) {
  return std::abs ( first - second ) < 1.0e-12;
}

static RectGeo
rectangle ( std::vector<double> lower, std::vector<double> upper ) {
  return RectGeo ( lower . size (), lower, upper );
}

static std::shared_ptr<AtlasGeo>
cell_geometry ( const Atlas & atlas, uint64_t cell ) {
  return std::dynamic_pointer_cast<AtlasGeo> ( atlas . geometry ( cell ) );
}

static Atlas
make_mixed_active_atlas ( void ) {
  Atlas atlas;
  atlas . add_chart ( 10, rectangle ( { 0.0, 0.0 }, { 1.0, 1.0 } ) );
  atlas . add_chart ( 20, rectangle ( { -1.0 }, { 1.0 } ) );
  atlas . add_chart ( 30, rectangle ( { 2.0 }, { 3.0 } ) );
  atlas . set_chart_active_dyadic_cells (
    10, { AtlasDyadicCell ( 2, { 0, 0 } ),
          AtlasDyadicCell ( 2, { 3, 3 } ) } );
  atlas . set_chart_active_dyadic_cells (
    20, { AtlasDyadicCell ( 3, { 3 } ) } );
  atlas . set_chart_active_dyadic_cells ( 30, {} );
  atlas . finalize ();
  return atlas;
}

static void
test_geometry_cover_and_chart_ids ( void ) {
  Atlas atlas = make_mixed_active_atlas ();
  CHECK ( atlas . numCharts () == 3 );
  CHECK ( atlas . size () == 3 );

  std::shared_ptr<AtlasGeo> first = cell_geometry ( atlas, 0 );
  std::shared_ptr<AtlasGeo> second = cell_geometry ( atlas, 1 );
  std::shared_ptr<AtlasGeo> third = cell_geometry ( atlas, 2 );
  CHECK ( first -> id () == 10 );
  CHECK ( second -> id () == 10 );
  CHECK ( third -> id () == 20 );
  CHECK ( near ( first -> rect () . lower_bounds [ 0 ], 0.0 ) );
  CHECK ( near ( first -> rect () . upper_bounds [ 0 ], 0.25 ) );
  CHECK ( near ( second -> rect () . lower_bounds [ 1 ], 0.75 ) );
  CHECK ( near ( second -> rect () . upper_bounds [ 1 ], 1.0 ) );
  CHECK ( near ( third -> rect () . lower_bounds [ 0 ], -0.25 ) );
  CHECK ( near ( third -> rect () . upper_bounds [ 0 ], 0.0 ) );

  AtlasGeo inside ( 10, rectangle ( { 0.1, 0.1 }, { 0.1, 0.1 } ) );
  AtlasGeo inactive_gap ( 10, rectangle ( { 0.5, 0.5 }, { 0.5, 0.5 } ) );
  CHECK ( atlas . cover ( inside ) == std::vector<uint64_t> ( { 0 } ) );
  CHECK ( atlas . cover ( inactive_gap ) . empty () );
}

static void
test_clone_subgrid_subdivide_and_join ( void ) {
  Atlas atlas = make_mixed_active_atlas ();
  std::shared_ptr<Atlas> clone ( atlas . clone () );
  CHECK ( clone -> size () == 3 );
  CHECK ( cell_geometry ( * clone, 2 ) -> id () == 20 );

  std::shared_ptr<Grid> left (
    atlas . subgrid ( std::deque<uint64_t> ( { 0 } ) ) );
  std::shared_ptr<Grid> right (
    atlas . subgrid ( std::deque<uint64_t> ( { 1, 2 } ) ) );
  std::shared_ptr<Atlas> left_atlas = std::dynamic_pointer_cast<Atlas> ( left );
  std::shared_ptr<Atlas> right_atlas = std::dynamic_pointer_cast<Atlas> ( right );
  CHECK ( left_atlas -> numCharts () == 3 );
  CHECK ( right_atlas -> numCharts () == 3 );
  CHECK ( left_atlas -> size () == 1 );
  CHECK ( right_atlas -> size () == 2 );
  CHECK ( cell_geometry ( * right_atlas, 1 ) -> id () == 20 );

  clone -> subdivide ();
  CHECK ( clone -> size () == 6 );
  CHECK ( cell_geometry ( * clone, 4 ) -> id () == 20 );

  std::vector<std::shared_ptr<Grid>> pieces = { left, right };
  std::shared_ptr<Atlas> joined ( new Atlas );
  join ( joined, pieces . begin (), pieces . end () );
  CHECK ( joined -> numCharts () == 3 );
  CHECK ( joined -> size () == 3 );
  CHECK ( cell_geometry ( * joined, 0 ) -> id () == 10 );
  CHECK ( cell_geometry ( * joined, 2 ) -> id () == 20 );
  CHECK ( joined -> chart ( 30 ) -> bounds () . dimension () == 1 );
  CHECK ( near ( joined -> chart ( 30 ) -> bounds () . lower_bounds [ 0 ], 2.0 ) );
  CHECK ( joined -> subset ( atlas ) . size () == 3 );
}

static void
test_deep_cell_is_direct_and_validation_is_transactional ( void ) {
  Atlas atlas;
  atlas . add_chart ( 7, rectangle ( { 0.0, 0.0 }, { 1.0, 1.0 } ) );
  atlas . set_chart_active_dyadic_cells (
    7, { AtlasDyadicCell ( 30, { 123456789, 987654321 } ) } );
  CHECK ( atlas . size () == 1 );
  // A full 2D depth-30 grid would contain 2^60 cells.  The direct prefix
  // representation stores only the selected path and its invalid siblings.
  CHECK ( atlas . memory () < 1000000 );

  bool raised = false;
  try {
    atlas . set_chart_active_dyadic_cells (
      7, { AtlasDyadicCell ( 0, { 0, 0 } ),
            AtlasDyadicCell ( 1, { 0, 0 } ) } );
  } catch ( const std::invalid_argument & ) {
    raised = true;
  }
  CHECK ( raised );
  CHECK ( atlas . size () == 1 );
}

static void
test_mapgraph_empty_image_and_boundary_exit ( void ) {
  std::shared_ptr<Atlas> atlas ( new Atlas ( make_mixed_active_atlas () ) );
  std::unordered_map<uint64_t, uint64_t> dimensions = { { 10, 2 }, { 20, 1 } };
  AtlasModelMapF::Callback callback =
    [] ( uint64_t chart, std::vector<double> bounds ) {
      if ( chart == 20 ) {
        // Explicit empty image.
        return std::vector<TaggedRectangle> ();
      }
      if ( bounds [ 0 ] < 0.5 ) {
        // Retained self target.
        return std::vector<TaggedRectangle> (
          { TaggedRectangle ( 10, { 0.1, 0.1, 0.1, 0.1 } ) } );
      }
      // A nonempty geometric target in the inactive gap.  Atlas::cover returns
      // no active vertex, so this is an explicit boundary exit in MapGraph.
      return std::vector<TaggedRectangle> (
        { TaggedRectangle ( 10, { 0.5, 0.5, 0.5, 0.5 } ) } );
    };
  std::shared_ptr<Map> map ( new AtlasModelMapF ( dimensions, callback ) );
  MapGraph graph ( atlas, map );
  CHECK ( graph . num_vertices () == 3 );
  CHECK ( graph . adjacencies ( 0 ) == std::vector<uint64_t> ( { 0 } ) );
  CHECK ( graph . adjacencies ( 1 ) . empty () ); // boundary exit
  CHECK ( graph . adjacencies ( 2 ) . empty () ); // explicit empty image
}

int main ( void ) {
  test_geometry_cover_and_chart_ids ();
  test_clone_subgrid_subdivide_and_join ();
  test_deep_cell_is_direct_and_validation_is_transactional ();
  test_mapgraph_empty_image_and_boundary_exit ();
  if ( failures != 0 ) {
    std::printf ( "%d check(s) failed\n", failures );
    return 1;
  }
  std::printf ( "all Atlas active-subgrid checks passed\n" );
  return 0;
}
