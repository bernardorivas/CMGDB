"""Executable test plan for fixed-subdivision Morse-set reachability.

Implements tests 1-9 of the design "Fixed-Subdivision Morse-Set
Reachability Verification": core graph-relation tests over a
deterministic in-memory adjacency, and public-API tests compared against
independently constructed complete MapGraph references.
"""

import math
import pickle
import shutil
import subprocess
import time
from pathlib import Path

import pytest

import CMGDB
from CMGDB._cmgdb import (
    _BuildTestMorseGraph,
    _ComputeMorseSetReachabilityCoreInMemory,
)

S = CMGDB.MorseSetReachabilityStatus
R, N, I = S.REACHABLE, S.NOT_REACHABLE, S.INCOMPLETE


# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------

TEST1_ADJACENCY = {0: [0, 1], 1: [2], 2: [2, 3], 3: [4], 4: [4]}
TEST1_MORSE_SETS = [[0], [2], [4]]
TEST1_ADAPTIVE_EDGES = [(0, 1), (0, 2), (1, 2)]


def core(adjacency, morse_sets, **kwargs):
    return _ComputeMorseSetReachabilityCoreInMemory(
        adjacency, morse_sets, **kwargs
    )


def status_matrix(result):
    m = result.num_vertices()
    return [[result.status(v, w) for w in range(m)] for v in range(m)]


def fixed_cells(result):
    """Per-vertex fixed descendant cell sets from provenance ranges."""
    nodes = result.provenance()["morse_graph"]["nodes"]
    return [
        set().union(*[set(range(a, b)) for a, b in node["fixed_ranges"]])
        if node["fixed_ranges"]
        else set()
        for node in nodes
    ]


def brute_force_reference(map_graph, seed_sets):
    """Exhaustive forward closures over a complete reference MapGraph.

    Returns per-source (visited_cells, adjacencies_examined).
    """
    out = []
    for seeds in seed_sets:
        visited = set(seeds)
        stack = list(seeds)
        examined = 0
        while stack:
            cell = stack.pop()
            for successor in map_graph.adjacencies(cell):
                examined += 1
                successor = int(successor)
                if successor not in visited:
                    visited.add(successor)
                    stack.append(successor)
        out.append((visited, examined))
    return out


def expected_reduction(matrix):
    """Transitive reduction R - R^2 of the strict certified relation."""
    m = len(matrix)
    strict = {
        (u, w)
        for u in range(m)
        for w in range(m)
        if u != w and matrix[u][w] == R
    }
    reduced = []
    for u in range(m):
        row = []
        for w in range(m):
            if (u, w) not in strict:
                continue
            if any(
                (u, v) in strict and (v, w) in strict
                for v in range(m)
                if v != u and v != w
            ):
                continue
            row.append(w)
        reduced.append(row)
    return reduced


def assert_matches_reference(model, morse_graph, phase_subdiv, reference_model):
    """Compare the verifier against a complete uniform MapGraph reference."""
    result = CMGDB.ComputeMorseSetReachability(
        model, morse_graph, phase_subdiv=phase_subdiv
    )
    _, reference = CMGDB.ComputeMorseGraph(reference_model)
    assert reference.num_vertices() == 2 ** phase_subdiv

    cells = fixed_cells(result)
    closures = brute_force_reference(reference, cells)
    m = result.num_vertices()
    for v in range(m):
        visited, examined = closures[v]
        assert result.frontier_exhausted(v)
        assert result.visited_grid_elements(v) == len(visited)
        assert result.grid_elements_expanded(v) == len(visited)
        assert result.adjacencies_examined(v) == examined
        for w in range(m):
            expected = R if (cells[w] & visited) else N
            assert result.status(v, w) == expected, (v, w)
    assert result.complete()

    matrix = status_matrix(result)
    for v in range(m):
        assert result.adjacencies_unreduced(v) == [
            w for w in range(m) if w != v and matrix[v][w] == R
        ]
    if result.diagnostics() == CMGDB.MorseSetRelationDiagnostic.VALID_PARTIAL_ORDER:
        reduction = expected_reduction(matrix)
        for v in range(m):
            assert result.adjacencies(v) == reduction[v]
    return result, reference


def morse_graph_signature(morse_graph):
    """Content signature used to prove the input MorseGraph is not mutated."""
    return (
        tuple(
            tuple(sorted(map(tuple, morse_graph.morse_set_boxes(v))))
            for v in range(morse_graph.num_vertices())
        ),
        tuple(sorted(morse_graph.edges_unreduced())),
    )


# ---------------------------------------------------------------------------
# Test 1: graph relation and reduction
# ---------------------------------------------------------------------------

def test_core_relation_and_reduction():
    result = core(
        TEST1_ADJACENCY, TEST1_MORSE_SETS,
        adaptive_edges=TEST1_ADAPTIVE_EDGES,
    )
    assert status_matrix(result) == [[R, R, R], [N, R, R], [N, N, R]]

    expected_metrics = {0: (5, 5, 7), 1: (3, 3, 4), 2: (1, 1, 1)}
    for v, (visited, expanded, examined) in expected_metrics.items():
        assert result.visited_grid_elements(v) == visited
        assert result.grid_elements_expanded(v) == expanded
        assert result.adjacencies_examined(v) == examined
        assert result.frontier_exhausted(v)
        assert result.stop_reason(v) == CMGDB.MorseSetReachabilityStopReason.NONE
        assert result.error(v) is None

    assert result.adjacencies_unreduced(0) == [1, 2]
    assert result.adjacencies_unreduced(1) == [2]
    assert result.adjacencies_unreduced(2) == []

    assert result.complete()
    assert (
        result.diagnostics()
        == CMGDB.MorseSetRelationDiagnostic.VALID_PARTIAL_ORDER
    )
    assert result.adjacencies(0) == [1]
    assert result.adjacencies(1) == [2]
    assert result.adjacencies(2) == []

    assert result.absent_adaptive_edges() == []
    assert sorted(result.retained_adaptive_edges()) == TEST1_ADAPTIVE_EDGES


# ---------------------------------------------------------------------------
# Test 2: disconnected Morse set
# ---------------------------------------------------------------------------

def test_core_disconnected_morse_set():
    adjacency = dict(TEST1_ADJACENCY)
    adjacency[5] = [5]
    morse_sets = TEST1_MORSE_SETS + [[5]]
    result = core(adjacency, morse_sets)

    assert status_matrix(result) == [
        [R, R, R, N],
        [N, R, R, N],
        [N, N, R, N],
        [N, N, N, R],
    ]
    assert result.visited_grid_elements(3) == 1
    assert result.grid_elements_expanded(3) == 1
    assert result.adjacencies_examined(3) == 1
    assert result.adjacencies_unreduced(0) == [1, 2]
    assert result.adjacencies_unreduced(1) == [2]
    assert result.adjacencies_unreduced(2) == []
    assert result.adjacencies_unreduced(3) == []


# ---------------------------------------------------------------------------
# Test 3: resource limit
# ---------------------------------------------------------------------------

def test_core_visited_limit_one():
    result = core(
        TEST1_ADJACENCY, TEST1_MORSE_SETS,
        max_visited_grid_elements=1,
        adaptive_edges=TEST1_ADAPTIVE_EDGES,
    )
    assert result.status(0, 0) == R
    assert result.status(0, 1) == I
    assert result.status(0, 2) == I
    assert result.visited_grid_elements(0) == 1
    assert result.grid_elements_expanded(0) == 0
    assert result.adjacencies_examined(0) == 2
    assert not result.frontier_exhausted(0)
    assert (
        result.stop_reason(0)
        == CMGDB.MorseSetReachabilityStopReason.MAX_VISITED_GRID_ELEMENTS
    )
    # An incomplete row never contains NOT_REACHABLE, so no adaptive edge
    # from that row may appear absent.
    assert result.absent_adaptive_edges() == []
    assert not result.complete()
    assert result.diagnostics() == CMGDB.MorseSetRelationDiagnostic.INCOMPLETE
    with pytest.raises(CMGDB.IncompleteMorseSetReachability):
        result.adjacencies(0)
    # The certified lower bound stays available.
    assert result.adjacencies_unreduced(0) == []


def test_core_visited_limit_two_mixed_row():
    result = core(
        TEST1_ADJACENCY, TEST1_MORSE_SETS,
        max_visited_grid_elements=2,
    )
    row = [result.status(0, w) for w in range(3)]
    assert row == [R, R, I]
    assert result.visited_grid_elements(0) == 2
    assert not result.frontier_exhausted(0)
    assert R in row and I in row and N not in row
    # Certified evidence recorded immediately before the cutoff survives.
    assert result.adjacencies_unreduced(0) == [1]


def test_core_adjacencies_examined_limit():
    result = core(
        TEST1_ADJACENCY, TEST1_MORSE_SETS,
        max_adjacencies_examined=1,
    )
    assert not result.frontier_exhausted(0)
    assert (
        result.stop_reason(0)
        == CMGDB.MorseSetReachabilityStopReason.MAX_ADJACENCIES_EXAMINED
    )
    # The cutoff-triggering entry is included in the count.
    assert result.adjacencies_examined(0) == 2
    # A source that needs exactly the limit completes normally.
    result2 = core(TEST1_ADJACENCY, TEST1_MORSE_SETS,
                   max_adjacencies_examined=7)
    assert result2.frontier_exhausted(0)
    assert result2.adjacencies_examined(0) == 7


def test_core_checkpoint_resume_and_pickle():
    limited = core(
        TEST1_ADJACENCY, TEST1_MORSE_SETS,
        max_visited_grid_elements=1,
        adaptive_edges=TEST1_ADAPTIVE_EDGES,
    )
    checkpoint = limited.checkpoint()
    assert checkpoint.payload_checksum

    resumed = core(
        TEST1_ADJACENCY, TEST1_MORSE_SETS,
        resume_from=checkpoint,
        adaptive_edges=TEST1_ADAPTIVE_EDGES,
    )
    assert resumed.complete()
    assert status_matrix(resumed) == [[R, R, R], [N, R, R], [N, N, R]]

    unpickled = pickle.loads(pickle.dumps(checkpoint))
    assert unpickled.payload_checksum == checkpoint.payload_checksum
    resumed2 = core(
        TEST1_ADJACENCY, TEST1_MORSE_SETS,
        resume_from=unpickled,
        adaptive_edges=TEST1_ADAPTIVE_EDGES,
    )
    assert resumed2.complete()

    # Resume validates identity: mismatched Morse sets are rejected.
    with pytest.raises(ValueError):
        core(TEST1_ADJACENCY, [[0], [2]], resume_from=checkpoint)
    # Persistent resume requires a matching caller-supplied fingerprint.
    fingerprinted = core(
        TEST1_ADJACENCY, TEST1_MORSE_SETS,
        max_visited_grid_elements=1, map_fingerprint="map-v1",
    )
    with pytest.raises(ValueError):
        core(
            TEST1_ADJACENCY, TEST1_MORSE_SETS,
            resume_from=fingerprinted.checkpoint(), map_fingerprint="map-v2",
        )


# ---------------------------------------------------------------------------
# Test 4: mutual reachability and non-transitivity
# ---------------------------------------------------------------------------

def test_core_mutual_reachability():
    result = core({0: [0, 1], 1: [0, 1]}, [[0], [1]])
    assert status_matrix(result) == [[R, R], [R, R]]
    assert result.coalescing_required()
    assert result.coalescing_groups() == [[0, 1]]
    assert result.adjacencies_unreduced(0) == [1]
    assert result.adjacencies_unreduced(1) == [0]
    assert (
        result.diagnostics()
        == CMGDB.MorseSetRelationDiagnostic.COALESCING_REQUIRED
    )
    with pytest.raises(CMGDB.MorseSetCoalescingRequired):
        result.adjacencies(0)


def test_core_nontransitive_completed_relation():
    result = core(
        {0: [1], 1: [1], 2: [3], 3: [3]},
        [[0], [1, 2], [3]],
    )
    assert result.complete()
    assert result.status(0, 1) == R
    assert result.status(1, 2) == R
    assert result.status(0, 2) == N
    assert (
        result.diagnostics()
        == CMGDB.MorseSetRelationDiagnostic.MORSE_SET_SPLITTING_REQUIRED
    )
    assert result.nontransitive_witnesses() == [[0, 1, 2]]
    with pytest.raises(CMGDB.MorseSetSplittingRequired):
        result.adjacencies(0)


# ---------------------------------------------------------------------------
# Test 5: complete MapGraph equivalence
# ---------------------------------------------------------------------------

def cubic_interval_hull(rect):
    """Exact interval hull of f(x) = 4.2x - 11.2x^2 + 8x^3."""
    lo, hi = rect

    def f(x):
        return 4.2 * x - 11.2 * x * x + 8 * x ** 3

    values = [f(lo), f(hi)]
    disc = 22.4 ** 2 - 4 * 24 * 4.2
    for root in (
        (22.4 - math.sqrt(disc)) / 48.0,
        (22.4 + math.sqrt(disc)) / 48.0,
    ):
        if lo <= root <= hi:
            values.append(f(root))
    return [min(values), max(values)]


def rotate_quarter(rect):
    lo, hi = rect
    return [lo + 0.25, hi + 0.25]


def _point_map_2d(x):
    return [x[i] / (2.0 - x[i]) for i in range(2)]


def box_map_2d(rect):
    return CMGDB.BoxMap(_point_map_2d, rect, padding=False)


def with_counting(f):
    calls = {"n": 0}

    def wrapped(rect):
        calls["n"] += 1
        return f(rect)

    return wrapped, calls


@pytest.mark.parametrize("use_batch", [False, True])
def test_equivalence_cubic_nonperiodic(use_batch):
    scalar, scalar_calls = with_counting(cubic_interval_hull)
    model = CMGDB.Model(3, 4, 2, 10000, [0.0], [1.0], scalar)
    morse_graph, _ = CMGDB.ComputeMorseGraph(model)
    scalar_calls["n"] = 0
    batch_calls = {"n": 0}
    if use_batch:
        def batch(rects):
            batch_calls["n"] += 1
            return [cubic_interval_hull(rect) for rect in rects]
        model.set_batch_map(batch)

    reference_model = CMGDB.Model(5, [0.0], [1.0], cubic_interval_hull)
    result, _ = assert_matches_reference(model, morse_graph, 5, reference_model)
    if use_batch:
        assert batch_calls["n"] > 0
        assert scalar_calls["n"] == 0
        assert sum(
            result.map_batches_attempted(v)
            for v in range(result.num_vertices())
        ) > 0
    else:
        assert scalar_calls["n"] > 0


@pytest.mark.parametrize("use_batch", [False, True])
def test_equivalence_periodic_quarter_rotation(use_batch):
    scalar, scalar_calls = with_counting(rotate_quarter)
    model = CMGDB.Model(2, 4, [0.0], [1.0], [True], scalar)
    morse_graph, _ = CMGDB.ComputeMorseGraph(model)
    # The quarter rotation is minimal: a single Morse set.
    assert morse_graph.num_vertices() == 1
    scalar_calls["n"] = 0
    batch_calls = {"n": 0}
    if use_batch:
        def batch(rects):
            batch_calls["n"] += 1
            return [rotate_quarter(rect) for rect in rects]
        model.set_batch_map(batch)

    reference_model = CMGDB.Model(4, [0.0], [1.0], [True], rotate_quarter)
    result, reference = assert_matches_reference(
        model, morse_graph, 4, reference_model
    )
    assert result.visited_grid_elements(0) == 16
    assert result.grid_elements_expanded(0) == 16
    assert result.adjacencies_examined(0) == 47
    # Explicit wraparound: the top cell's image [1.1875, 1.25] wraps to
    # the low end of the domain.
    assert set(map(int, reference.adjacencies(15))) == {2, 3, 4}
    if use_batch:
        assert batch_calls["n"] > 0
        assert scalar_calls["n"] == 0


@pytest.mark.parametrize("use_batch", [False, True])
def test_equivalence_point_map_2d(use_batch):
    model = CMGDB.Model(6, 10, 4, 10000, [0.0, 0.0], [1.2, 1.2], box_map_2d)
    morse_graph, _ = CMGDB.ComputeMorseGraph(model)
    scalar_calls = {"n": 0}

    def scalar(rect):
        scalar_calls["n"] += 1
        return box_map_2d(rect)

    model = CMGDB.Model(6, 10, 4, 10000, [0.0, 0.0], [1.2, 1.2], scalar)
    batch_calls = {"n": 0}
    if use_batch:
        def batch(rects):
            batch_calls["n"] += 1
            return [box_map_2d(rect) for rect in rects]
        model.set_batch_map(batch)

    reference_model = CMGDB.Model(10, [0.0, 0.0], [1.2, 1.2], box_map_2d)
    assert_matches_reference(model, morse_graph, 10, reference_model)
    if use_batch:
        assert batch_calls["n"] > 0
        assert scalar_calls["n"] == 0
    else:
        assert scalar_calls["n"] > 0


def shrink_map_1d(rect):
    lo, hi = rect
    width = hi - lo
    return [lo + 0.25 * width, hi - 0.25 * width]


@pytest.mark.parametrize(
    ("map_fn", "periodic", "expected_cover_of_first_cell"),
    [
        # Image exactly on an internal cell boundary covers both sides.
        (lambda rect: [0.5, 0.5], [False], {3, 4}),
        # Periodic image exactly at the lower domain bound covers both
        # the first and the last cell.
        (lambda rect: [0.0, 0.0], [True], {0, 7}),
        # Periodic image exactly at the upper domain bound covers only
        # the last cell (TreeGrid's periodic endpoint asymmetry).
        (lambda rect: [1.0, 1.0], [True], {7}),
    ],
)
def test_equivalence_exact_endpoints(map_fn, periodic, expected_cover_of_first_cell):
    subdiv = 3
    if periodic[0]:
        reference_model = CMGDB.Model(subdiv, [0.0], [1.0], periodic, map_fn)
        model = CMGDB.Model(1, subdiv, [0.0], [1.0], periodic, map_fn)
    else:
        reference_model = CMGDB.Model(subdiv, [0.0], [1.0], map_fn)
        model = CMGDB.Model(1, subdiv, [0.0], [1.0], map_fn)
    _, reference = CMGDB.ComputeMorseGraph(reference_model)
    assert (
        set(map(int, reference.adjacencies(0)))
        == expected_cover_of_first_cell
    )

    # Verify the fixed provider agrees with the reference cover through a
    # synthetic single-leaf Morse set at cell 0.
    morse_graph = _BuildTestMorseGraph(
        [0.0], [1.0], periodic, [["0" * subdiv]], []
    )
    result = CMGDB.ComputeMorseSetReachability(
        model, morse_graph, phase_subdiv=subdiv
    )
    closures = brute_force_reference(reference, [{0}])
    visited, examined = closures[0]
    assert result.visited_grid_elements(0) == len(visited)
    assert result.adjacencies_examined(0) == examined


# ---------------------------------------------------------------------------
# Test 6: mixed-depth adaptive Morse sets
# ---------------------------------------------------------------------------

def shrink_map_2d(rect):
    lower, upper = rect[:2], rect[2:]
    new_lower = [lo + 0.25 * (hi - lo) for lo, hi in zip(lower, upper)]
    new_upper = [hi - 0.25 * (hi - lo) for lo, hi in zip(lower, upper)]
    return new_lower + new_upper


def build_mixed_depth_fixture():
    return _BuildTestMorseGraph(
        [0.0, 0.0], [1.0, 1.0], [False, False],
        [["0", "100"], ["11"]], [],
    )


def test_mixed_depth_descendants_and_geometry():
    morse_graph = build_mixed_depth_fixture()
    model = CMGDB.Model(1, 4, [0.0, 0.0], [1.0, 1.0], shrink_map_2d)
    result = CMGDB.ComputeMorseSetReachability(
        model, morse_graph, phase_subdiv=4
    )
    nodes = result.provenance()["morse_graph"]["nodes"]

    # Exact descendant identifiers, no duplicates, exact counts.
    assert nodes[0]["fixed_ranges"] == [(0, 10)]
    assert nodes[0]["fixed_descendant_count"] == 10
    assert nodes[1]["fixed_ranges"] == [(12, 16)]
    assert nodes[1]["fixed_descendant_count"] == 4

    # Geometric unions preserved: "0" and "100" for M0, "11" for M1.
    assert sorted(map(tuple, morse_graph.morse_set_boxes(0))) == [
        (0.0, 0.0, 0.5, 1.0),
        (0.5, 0.0, 0.75, 0.5),
    ]
    assert sorted(map(tuple, morse_graph.morse_set_boxes(1))) == [
        (0.5, 0.5, 1.0, 1.0),
    ]
    # Volumes 10/16 and 4/16 (each fixed cell has volume 1/16).
    assert nodes[0]["fixed_descendant_count"] / 16.0 == 10 / 16
    assert nodes[1]["fixed_descendant_count"] / 16.0 == 4 / 16

    # Strict-interior map: one self-adjacency per fixed element.
    assert result.visited_grid_elements(0) == 10
    assert result.adjacencies_examined(0) == 10
    assert result.visited_grid_elements(1) == 4
    assert result.adjacencies_examined(1) == 4
    assert result.status(0, 1) == N
    assert result.status(1, 0) == N

    # Range hashes are stable across recomputation.
    result2 = CMGDB.ComputeMorseSetReachability(
        model, morse_graph, phase_subdiv=4
    )
    nodes2 = result2.provenance()["morse_graph"]["nodes"]
    for a, b in zip(nodes, nodes2):
        assert a["fixed_range_hash"] == b["fixed_range_hash"]
        assert a["adaptive_prefix_hash"] == b["adaptive_prefix_hash"]


def test_mixed_depth_rejects_shallow_subdivision():
    morse_graph = build_mixed_depth_fixture()
    model = CMGDB.Model(1, 4, [0.0, 0.0], [1.0, 1.0], shrink_map_2d)
    with pytest.raises(ValueError, match="TreeGrid depth 3"):
        CMGDB.ComputeMorseSetReachability(model, morse_graph, phase_subdiv=2)


# ---------------------------------------------------------------------------
# Test 7: adaptive-edge pruning regression
# ---------------------------------------------------------------------------

def test_adaptive_edge_pruning_cubic():
    model = CMGDB.Model(3, 4, 2, 10000, [0.0], [1.0], cubic_interval_hull)
    morse_graph, _ = CMGDB.ComputeMorseGraph(model)
    assert morse_graph.num_vertices() == 3

    # Label adaptive Morse vertices by geometry.
    def vertex_with_box(lo, hi):
        for v in range(morse_graph.num_vertices()):
            boxes = morse_graph.morse_set_boxes(v)
            if min(b[0] for b in boxes) == lo and max(b[1] for b in boxes) == hi:
                return v
        raise AssertionError("expected Morse vertex not found")

    middle = vertex_with_box(0.25, 0.5)
    left = vertex_with_box(0.0, 0.125)
    right = vertex_with_box(0.875, 1.0)

    assert sorted(morse_graph.edges_unreduced()) == sorted(
        [(left, middle), (right, middle), (right, left)]
    )

    before = morse_graph_signature(morse_graph)
    result = CMGDB.ComputeMorseSetReachability(
        model, morse_graph, phase_subdiv=5
    )
    assert morse_graph_signature(morse_graph) == before

    # Frozen full-MapGraph reference at s=5.
    cells = fixed_cells(result)
    assert cells[middle] == set(range(8, 16))
    assert cells[left] == set(range(0, 4))
    assert cells[right] == set(range(28, 32))

    expected = {
        middle: (8, 14, {middle}),
        left: (16, 37, {left, middle}),
        right: (26, 61, {right, middle}),
    }
    for v, (visited, examined, targets) in expected.items():
        assert result.visited_grid_elements(v) == visited
        assert result.adjacencies_examined(v) == examined
        reached = {
            w for w in range(3) if result.status(v, w) == R
        }
        assert reached == targets
    # Frozen visited-cell identifiers for left and right.
    reference_model = CMGDB.Model(5, [0.0], [1.0], cubic_interval_hull)
    _, reference = CMGDB.ComputeMorseGraph(reference_model)
    closures = brute_force_reference(
        reference, [cells[left], cells[right]]
    )
    assert closures[0][0] == set(range(0, 16))
    assert closures[1][0] == set(range(6, 32))

    assert result.status(right, left) == N
    assert result.absent_adaptive_edges() == [(right, left)]
    assert sorted(result.retained_adaptive_edges()) == sorted(
        [(left, middle), (right, middle)]
    )
    strict = {
        (v, w)
        for v in range(3)
        for w in range(3)
        if v != w and result.status(v, w) == R
    }
    assert strict == {(left, middle), (right, middle)}
    assert result.adjacencies(left) == [middle]
    assert result.adjacencies(right) == [middle]
    assert result.adjacencies(middle) == []


# ---------------------------------------------------------------------------
# Test 8: no complete graph materialization
# ---------------------------------------------------------------------------

def test_no_complete_graph_materialization_subdiv_40():
    calls = {"n": 0}

    def counting_shrink(rect):
        calls["n"] += 1
        return shrink_map_1d(rect)

    model = CMGDB.Model(1, 4, [0.0], [1.0], counting_shrink)
    morse_graph = _BuildTestMorseGraph(
        [0.0], [1.0], [False], [["0" * 40], ["1" * 40]], []
    )
    start = time.monotonic()
    result = CMGDB.ComputeMorseSetReachability(
        model, morse_graph, phase_subdiv=40
    )
    elapsed = time.monotonic() - start
    # 2^40 conceptual cells: any materialization would be unmissable.
    assert elapsed < 5.0

    assert status_matrix(result) == [[R, N], [N, R]]
    for v in range(2):
        assert result.visited_grid_elements(v) == 1
        assert result.grid_elements_expanded(v) == 1
        assert result.adjacencies_examined(v) == 1
        assert result.map_evaluations_attempted(v) == 1
    assert calls["n"] == 2

    instrumentation = result.provenance()["verification"]["instrumentation"]
    assert instrumentation == {
        "complete_treegrid_materializations": 0,
        "mapgraph_constructions": 0,
        "complete_vertex_array_bytes": 0,
        "complete_edge_array_bytes": 0,
    }
    # Exceeding model.phase_subdiv_max is a warning, not a rejection.
    assert any(
        "phase_subdiv_max" in warning
        for warning in result.provenance()["verification"]["warnings"]
    )


# ---------------------------------------------------------------------------
# Test 9: subdivision instability
# ---------------------------------------------------------------------------

def instability_box_map(rect):
    lo, hi = rect
    if lo >= 0.5:
        return [0.75, 0.75]
    return [0.25, 0.25 + 0.6 * (hi - lo)]


def test_subdivision_instability():
    model = CMGDB.Model(1, [0.0], [1.0], instability_box_map)
    morse_graph, _ = CMGDB.ComputeMorseGraph(model)
    assert morse_graph.num_vertices() == 2

    def vertex_with_box(lo, hi):
        for v in range(morse_graph.num_vertices()):
            box = morse_graph.morse_set_boxes(v)[0]
            if box[0] == lo and box[1] == hi:
                return v
        raise AssertionError("expected Morse vertex not found")

    low = vertex_with_box(0.0, 0.5)
    high = vertex_with_box(0.5, 1.0)
    assert morse_graph.edges_unreduced() == [(low, high)]

    result_s1 = CMGDB.ComputeMorseSetReachability(
        model, morse_graph, phase_subdiv=1
    )
    assert result_s1.visited_grid_elements(high) == 1
    assert result_s1.adjacencies_examined(high) == 1
    assert result_s1.visited_grid_elements(low) == 2
    assert result_s1.adjacencies_examined(low) == 3
    assert result_s1.status(low, high) == R

    result_s2 = CMGDB.ComputeMorseSetReachability(
        model, morse_graph, phase_subdiv=2
    )
    assert result_s2.visited_grid_elements(high) == 2
    assert result_s2.adjacencies_examined(high) == 4
    assert result_s2.visited_grid_elements(low) == 2
    assert result_s2.adjacencies_examined(low) == 4
    assert result_s2.status(low, high) == N
    assert result_s2.absent_adaptive_edges() == [(low, high)]

    study = CMGDB.ComputeMorseSetReachabilityStudy(
        model, morse_graph, phase_subdivisions=[1, 2]
    )
    assert study.classification(low, high) == CMGDB.UNSTABLE
    assert study.unstable_pairs() == [(low, high)]
    # No silent selection of the finest subdivision: an unstable pair is
    # never prunable.
    assert study.prunable_adaptive_edges() == []
    assert study.classification(high, low) == CMGDB.AGREE_NOT_REACHABLE
    assert study.classification(low, low) == CMGDB.AGREE_REACHABLE


# ---------------------------------------------------------------------------
# Provenance and input-validation coverage
# ---------------------------------------------------------------------------

def test_provenance_schema_and_hashes():
    model = CMGDB.Model(3, 4, 2, 10000, [0.0], [1.0], cubic_interval_hull)
    morse_graph, _ = CMGDB.ComputeMorseGraph(model)
    result = CMGDB.ComputeMorseSetReachability(
        model, morse_graph, phase_subdiv=5, map_fingerprint="cubic-v1"
    )
    prov = result.provenance()
    assert prov["schema_name"] == "CMGDB.MorseSetReachabilityProvenance"
    assert prov["schema_version"] == 1
    assert prov["model"]["map_fingerprint"] == "cubic-v1"
    assert prov["model"]["map_fingerprint_kind"] == "caller_supplied"
    assert prov["model"]["evaluation_mode"] == "scalar"
    assert prov["verification"]["phase_subdiv"] == 5
    assert (
        prov["verification"]["fixed_element_encoding"]
        == "tree_path_msb_uint64_v1"
    )
    assert prov["verification"]["traversal"] == "fifo_sorted_streaming_v1"
    assert len(prov["model"]["phase_lower_bounds_ieee754"]) == 1
    assert prov["morse_graph"]["num_vertices"] == 3
    assert prov["relation"]["completed"] is True
    assert prov["relation"]["reduced_edge_hash"] is not None
    for source in prov["sources"]:
        assert source["closure_size"] == source["visited_grid_elements"]
        assert source["error"] is None

    # Deterministic across recomputation.
    result2 = CMGDB.ComputeMorseSetReachability(
        model, morse_graph, phase_subdiv=5, map_fingerprint="cubic-v1"
    )
    prov2 = result2.provenance()
    assert prov["morse_graph"] == prov2["morse_graph"]
    assert prov["relation"] == prov2["relation"]
    assert prov["sources"] == prov2["sources"]


def test_preflight_rejects_mismatched_model():
    morse_graph = build_mixed_depth_fixture()
    wrong_bounds = CMGDB.Model(1, 4, [0.0, 0.0], [2.0, 1.0], shrink_map_2d)
    with pytest.raises(ValueError, match="bounds differ"):
        CMGDB.ComputeMorseSetReachability(
            wrong_bounds, morse_graph, phase_subdiv=4
        )
    wrong_dim = CMGDB.Model(1, 4, [0.0], [1.0], shrink_map_1d)
    with pytest.raises(ValueError, match="dimensions differ"):
        CMGDB.ComputeMorseSetReachability(
            wrong_dim, morse_graph, phase_subdiv=4
        )
    model = CMGDB.Model(1, 4, [0.0, 0.0], [1.0, 1.0], shrink_map_2d)
    with pytest.raises(ValueError, match="phase_subdiv"):
        CMGDB.ComputeMorseSetReachability(model, morse_graph, phase_subdiv=64)


def test_preflight_rejects_overlapping_morse_sets():
    morse_graph = _BuildTestMorseGraph(
        [0.0], [1.0], [False], [["0"], ["0"]], []
    )
    model = CMGDB.Model(1, 4, [0.0], [1.0], shrink_map_1d)
    with pytest.raises(ValueError, match="overlap"):
        CMGDB.ComputeMorseSetReachability(model, morse_graph, phase_subdiv=3)


def test_map_error_marks_source_incomplete_and_continues():
    failures = {"armed": True}

    def flaky(rect):
        lo, hi = rect
        if failures["armed"] and hi <= 0.5:
            raise RuntimeError("synthetic map failure")
        return shrink_map_1d(rect)

    model = CMGDB.Model(1, 4, [0.0], [1.0], flaky)
    morse_graph = _BuildTestMorseGraph(
        [0.0], [1.0], [False], [["00"], ["11"]], []
    )
    result = CMGDB.ComputeMorseSetReachability(
        model, morse_graph, phase_subdiv=2, batch_size=1
    )
    assert (
        result.stop_reason(0)
        == CMGDB.MorseSetReachabilityStopReason.MAP_ERROR
    )
    error = result.error(0)
    assert error["category"] == "map"
    assert "synthetic map failure" in error["message"]
    assert result.status(0, 0) == R      # seeds were encountered
    assert result.status(0, 1) == I
    # Processing continued with the other source.
    assert result.frontier_exhausted(1)
    assert result.status(1, 1) == R
    assert result.status(1, 0) == N


# ---------------------------------------------------------------------------
# C++ core test (tests/cpp/test_morse_set_reachability_core.cpp)
# ---------------------------------------------------------------------------

def test_cpp_core_standalone():
    compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("no C++ compiler available")
    repo = Path(__file__).resolve().parents[1]
    source = repo / "tests" / "cpp" / "test_morse_set_reachability_core.cpp"
    include = repo / "src" / "CMGDB" / "_cmgdb" / "include" / "database"
    binary = source.with_suffix("")
    subprocess.run(
        [compiler, "-std=c++11", "-O1", "-I", str(include),
         str(source), "-o", str(binary)],
        check=True,
    )
    try:
        completed = subprocess.run(
            [str(binary)], capture_output=True, text=True, check=False
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
    finally:
        binary.unlink(missing_ok=True)
