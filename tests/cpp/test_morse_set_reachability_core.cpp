// test_morse_set_reachability_core.cpp
//
// Deterministic C++ tests of the fixed-subdivision Morse-set
// reachability core against an in-memory adjacency provider,
// independent of CMGDB geometry and Python callbacks.
//
// Build: c++ -std=c++11 -I src/CMGDB/_cmgdb/include/database \
//            tests/cpp/test_morse_set_reachability_core.cpp -o core_test

#include <cstdio>
#include <cstdlib>
#include <map>
#include <vector>

#include "MorseSetReachabilityCore.h"

static int failures = 0;

#define CHECK(condition)                                                    \
  do {                                                                      \
    if ( not ( condition ) ) {                                              \
      std::printf ( "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition );   \
      ++ failures;                                                          \
    }                                                                       \
  } while ( 0 )

typedef MorseSetReachabilityStatus Status;
typedef MorseSetReachabilityStopReason StopReason;
typedef MorseSetRelationDiagnostic Diagnostic;

static MorseSetFixedRanges
singleton ( FixedGridElement cell ) {
  MorseSetFixedRanges ranges;
  ranges . ranges . push_back ( std::make_pair ( cell, cell + 1 ) );
  return ranges;
}

static std::map<FixedGridElement, std::vector<FixedGridElement>>
test1_adjacency ( void ) {
  std::map<FixedGridElement, std::vector<FixedGridElement>> adjacency;
  adjacency [ 0 ] = { 0, 1 };
  adjacency [ 1 ] = { 2 };
  adjacency [ 2 ] = { 2, 3 };
  adjacency [ 3 ] = { 4 };
  adjacency [ 4 ] = { 4 };
  return adjacency;
}

static std::vector<MorseSetFixedRanges>
test1_morse_sets ( void ) {
  std::vector<MorseSetFixedRanges> morse_sets;
  morse_sets . push_back ( singleton ( 0 ) );
  morse_sets . push_back ( singleton ( 2 ) );
  morse_sets . push_back ( singleton ( 4 ) );
  return morse_sets;
}

static void
test_relation_and_reduction ( void ) {
  InMemoryAdjacencyProvider provider ( test1_adjacency () );
  MorseSetReachabilityResult result =
    ComputeMorseSetReachabilityCore ( test1_morse_sets (), provider );

  const Status expected [ 3 ] [ 3 ] = {
    { Status::REACHABLE, Status::REACHABLE, Status::REACHABLE },
    { Status::NOT_REACHABLE, Status::REACHABLE, Status::REACHABLE },
    { Status::NOT_REACHABLE, Status::NOT_REACHABLE, Status::REACHABLE } };
  for ( uint64_t v = 0; v < 3; ++ v ) {
    for ( uint64_t w = 0; w < 3; ++ w ) {
      CHECK ( result . status ( v, w ) == expected [ v ] [ w ] );
    }
  }

  const uint64_t expected_visited [ 3 ] = { 5, 3, 1 };
  const uint64_t expected_expanded [ 3 ] = { 5, 3, 1 };
  const uint64_t expected_examined [ 3 ] = { 7, 4, 1 };
  for ( uint64_t v = 0; v < 3; ++ v ) {
    CHECK ( result . frontier_exhausted ( v ) );
    CHECK ( result . visited_grid_elements ( v ) == expected_visited [ v ] );
    CHECK ( result . grid_elements_expanded ( v ) == expected_expanded [ v ] );
    CHECK ( result . adjacencies_examined ( v ) == expected_examined [ v ] );
    CHECK ( result . stop_reason ( v ) == StopReason::NONE );
  }

  CHECK ( result . complete () );
  CHECK ( result . diagnostics () == Diagnostic::VALID_PARTIAL_ORDER );
  CHECK ( result . adjacencies_unreduced ( 0 ) ==
          std::vector<uint64_t> ( { 1, 2 } ) );
  CHECK ( result . adjacencies_unreduced ( 1 ) ==
          std::vector<uint64_t> ( { 2 } ) );
  CHECK ( result . adjacencies_unreduced ( 2 ) . empty () );
  CHECK ( result . adjacencies ( 0 ) == std::vector<uint64_t> ( { 1 } ) );
  CHECK ( result . adjacencies ( 1 ) == std::vector<uint64_t> ( { 2 } ) );
  CHECK ( result . adjacencies ( 2 ) . empty () );
}

static void
test_disconnected_morse_set ( void ) {
  std::map<FixedGridElement, std::vector<FixedGridElement>> adjacency =
    test1_adjacency ();
  adjacency [ 5 ] = { 5 };
  std::vector<MorseSetFixedRanges> morse_sets = test1_morse_sets ();
  morse_sets . push_back ( singleton ( 5 ) );

  InMemoryAdjacencyProvider provider ( adjacency );
  MorseSetReachabilityResult result =
    ComputeMorseSetReachabilityCore ( morse_sets, provider );

  for ( uint64_t v = 0; v < 3; ++ v ) {
    CHECK ( result . status ( v, 3 ) == Status::NOT_REACHABLE );
    CHECK ( result . status ( 3, v ) == Status::NOT_REACHABLE );
  }
  CHECK ( result . status ( 3, 3 ) == Status::REACHABLE );
  CHECK ( result . visited_grid_elements ( 3 ) == 1 );
  CHECK ( result . grid_elements_expanded ( 3 ) == 1 );
  CHECK ( result . adjacencies_examined ( 3 ) == 1 );
}

static void
test_resource_limit ( void ) {
  InMemoryAdjacencyProvider provider ( test1_adjacency () );
  MorseSetReachabilityOptions options;
  options . max_visited_grid_elements = 1;
  MorseSetReachabilityResult result =
    ComputeMorseSetReachabilityCore ( test1_morse_sets (), provider, options );

  CHECK ( result . status ( 0, 0 ) == Status::REACHABLE );
  CHECK ( result . status ( 0, 1 ) == Status::INCOMPLETE );
  CHECK ( result . status ( 0, 2 ) == Status::INCOMPLETE );
  CHECK ( result . visited_grid_elements ( 0 ) == 1 );
  CHECK ( result . grid_elements_expanded ( 0 ) == 0 );
  CHECK ( result . adjacencies_examined ( 0 ) == 2 );
  CHECK ( not result . frontier_exhausted ( 0 ) );
  CHECK ( result . stop_reason ( 0 ) ==
          StopReason::MAX_VISITED_GRID_ELEMENTS );
  CHECK ( not result . complete () );
  CHECK ( result . diagnostics () == Diagnostic::INCOMPLETE );

  bool raised = false;
  try {
    result . adjacencies ( 0 );
  } catch ( IncompleteMorseSetReachability const& ) {
    raised = true;
  }
  CHECK ( raised );

  // Limit 2: one row containing both REACHABLE and INCOMPLETE.
  InMemoryAdjacencyProvider provider2 ( test1_adjacency () );
  options . max_visited_grid_elements = 2;
  MorseSetReachabilityResult result2 =
    ComputeMorseSetReachabilityCore ( test1_morse_sets (), provider2, options );
  CHECK ( result2 . status ( 0, 0 ) == Status::REACHABLE );
  CHECK ( result2 . status ( 0, 1 ) == Status::REACHABLE );
  CHECK ( result2 . status ( 0, 2 ) == Status::INCOMPLETE );
  CHECK ( result2 . adjacencies_unreduced ( 0 ) ==
          std::vector<uint64_t> ( { 1 } ) );

  // Resume the limited computation to completion from its checkpoint.
  InMemoryAdjacencyProvider provider3 ( test1_adjacency () );
  MorseSetReachabilityOptions resume_options;
  resume_options . resume_from = result . checkpoint ();
  MorseSetReachabilityResult resumed =
    ComputeMorseSetReachabilityCore ( test1_morse_sets (), provider3,
                                      resume_options );
  CHECK ( resumed . complete () );
  CHECK ( resumed . status ( 0, 2 ) == Status::REACHABLE );
  CHECK ( resumed . status ( 2, 0 ) == Status::NOT_REACHABLE );
}

static void
test_mutual_reachability_and_nontransitivity ( void ) {
  std::map<FixedGridElement, std::vector<FixedGridElement>> adjacency;
  adjacency [ 0 ] = { 0, 1 };
  adjacency [ 1 ] = { 0, 1 };
  std::vector<MorseSetFixedRanges> morse_sets;
  morse_sets . push_back ( singleton ( 0 ) );
  morse_sets . push_back ( singleton ( 1 ) );

  InMemoryAdjacencyProvider provider ( adjacency );
  MorseSetReachabilityResult result =
    ComputeMorseSetReachabilityCore ( morse_sets, provider );
  for ( uint64_t v = 0; v < 2; ++ v ) {
    for ( uint64_t w = 0; w < 2; ++ w ) {
      CHECK ( result . status ( v, w ) == Status::REACHABLE );
    }
  }
  CHECK ( result . coalescing_required () );
  CHECK ( result . coalescing_groups () . size () == 1 );
  CHECK ( result . coalescing_groups () [ 0 ] ==
          std::vector<uint64_t> ( { 0, 1 } ) );
  bool raised = false;
  try {
    result . adjacencies ( 0 );
  } catch ( MorseSetCoalescingRequired const& ) {
    raised = true;
  }
  CHECK ( raised );

  // Non-transitivity regression: split recurrent middle set.
  std::map<FixedGridElement, std::vector<FixedGridElement>> adjacency2;
  adjacency2 [ 0 ] = { 1 };
  adjacency2 [ 1 ] = { 1 };
  adjacency2 [ 2 ] = { 3 };
  adjacency2 [ 3 ] = { 3 };
  std::vector<MorseSetFixedRanges> morse_sets2;
  morse_sets2 . push_back ( singleton ( 0 ) );
  MorseSetFixedRanges middle;
  middle . ranges . push_back ( std::make_pair ( 1, 3 ) );
  morse_sets2 . push_back ( middle );
  morse_sets2 . push_back ( singleton ( 3 ) );

  InMemoryAdjacencyProvider provider2 ( adjacency2 );
  MorseSetReachabilityResult result2 =
    ComputeMorseSetReachabilityCore ( morse_sets2, provider2 );
  CHECK ( result2 . status ( 0, 1 ) == Status::REACHABLE );
  CHECK ( result2 . status ( 1, 2 ) == Status::REACHABLE );
  CHECK ( result2 . status ( 0, 2 ) == Status::NOT_REACHABLE );
  CHECK ( result2 . diagnostics () ==
          Diagnostic::MORSE_SET_SPLITTING_REQUIRED );
  CHECK ( result2 . nontransitive_witnesses () . size () == 1 );
  CHECK ( result2 . nontransitive_witnesses () [ 0 ] ==
          std::vector<uint64_t> ( { 0, 1, 2 } ) );
  raised = false;
  try {
    result2 . adjacencies ( 0 );
  } catch ( MorseSetSplittingRequired const& ) {
    raised = true;
  }
  CHECK ( raised );
}

int main ( void ) {
  test_relation_and_reduction ();
  test_disconnected_morse_set ();
  test_resource_limit ();
  test_mutual_reachability_and_nontransitivity ();
  if ( failures > 0 ) {
    std::printf ( "%d check(s) failed\n", failures );
    return 1;
  }
  std::printf ( "all core checks passed\n" );
  return 0;
}
