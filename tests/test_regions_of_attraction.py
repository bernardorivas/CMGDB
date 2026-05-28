"""Smoke tests for morse_graph_parser and cmgdb_roa modules."""

import dataclasses
import tempfile
from pathlib import Path

import numpy as np
import pytest

from CMGDB.cmgdb_roa import (
    BOUNDARY,
    ESCAPE,
    MULTI,
    CellROA,
    LatentBounds,
    collapse_roa_to_lca,
    compute_exact_roa,
    load_exact_roa,
    save_exact_roa,
)
from CMGDB.morse_graph_parser import MorseGraph


def create_minimal_morse_graph_dot(path: Path) -> None:
    """Write a minimal 3-node Morse DAG with two sinks.

    Structure: 0 -> 1, 0 -> 2. Nodes 1 and 2 are minimal (sinks).
    """
    dot_content = """digraph G {
    0 [label="0", fillcolor="#FF0000"];
    1 [label="1", fillcolor="#00FF00"];
    2 [label="2", fillcolor="#0000FF"];
    0 -> 1;
    0 -> 2;
}
"""
    path.write_text(dot_content)


class MockMapGraph:
    """Minimal mock CMGDB MapGraph for testing."""

    def __init__(self, n_vertices: int, adjacencies_list: list[list[int]]):
        self.n_vertices_val = n_vertices
        self.adjacencies_list = adjacencies_list

    def num_vertices(self) -> int:
        return self.n_vertices_val

    def adjacencies(self, src: int) -> list[int]:
        if 0 <= src < len(self.adjacencies_list):
            return self.adjacencies_list[src]
        return []

    def phase_space_box(self, vertex_id: int) -> np.ndarray:
        """Return a dummy box [0, 1] x [0, 1] x ... for each vertex."""
        dim = 2
        return np.array([vertex_id / 10.0, vertex_id / 10.0 + 0.5] * dim, dtype=np.float64)


class MockCMGDBMorseGraph:
    """Minimal mock CMGDB morse graph object."""

    def __init__(self, morse_sets: dict[int, list[int]]):
        self.morse_sets_dict = morse_sets

    def morse_set(self, node_id: int) -> list[int]:
        return self.morse_sets_dict.get(node_id, [])

    def phase_space_box(self, vertex_id: int) -> np.ndarray:
        """Return a dummy box."""
        dim = 2
        return np.array([vertex_id / 10.0, vertex_id / 10.0 + 0.5] * dim, dtype=np.float64)


def test_morse_graph_parser_basic():
    """Parse a minimal DOT file and verify the parsed graph structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dot_path = Path(tmpdir) / "morse_graph.dot"
        create_minimal_morse_graph_dot(dot_path)

        morse_graph = MorseGraph.from_dot(dot_path)

        # Verify nodes and edges
        assert morse_graph.nodes == [0, 1, 2]
        assert 0 in morse_graph.edges
        assert set(morse_graph.edges[0]) == {1, 2}
        assert 1 not in morse_graph.edges or morse_graph.edges.get(1) is None
        assert 2 not in morse_graph.edges or morse_graph.edges.get(2) is None

        # Verify minimal set: nodes 1 and 2 have no outgoing edges
        assert morse_graph.minimal == {1, 2}

        # Verify reachable minimals
        # Node 0 can reach both minimal nodes 1 and 2
        assert morse_graph.reachable_minimals[0] == frozenset({1, 2})
        # Nodes 1 and 2 are minimal, so they include themselves as reachable minimals
        assert morse_graph.reachable_minimals[1] == frozenset({1})
        assert morse_graph.reachable_minimals[2] == frozenset({2})


def test_morse_graph_lca():
    """Test the LCA logic on the minimal 3-node DAG."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dot_path = Path(tmpdir) / "morse_graph.dot"
        create_minimal_morse_graph_dot(dot_path)
        morse_graph = MorseGraph.from_dot(dot_path)

        # LCA of just {1} should be 1 itself
        assert morse_graph.lca_of_minimals(frozenset({1})) == 1

        # LCA of just {2} should be 2 itself
        assert morse_graph.lca_of_minimals(frozenset({2})) == 2

        # LCA of {1, 2} — both are minimal and reachable from node 0
        # Node 0 is the unique node with reachable_minimals == {1, 2}
        lca = morse_graph.lca_of_minimals(frozenset({1, 2}))
        assert lca == 0

        # LCA of empty set
        assert morse_graph.lca_of_minimals(frozenset()) is None


def test_compute_exact_roa():
    """Test exact RoA computation on a minimal graph."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dot_path = Path(tmpdir) / "morse_graph.dot"
        create_minimal_morse_graph_dot(dot_path)
        morse_graph = MorseGraph.from_dot(dot_path)

        # Create a simple 3-vertex map graph: 0 -> 1, 0 -> 2, 1 -> 2
        # All paths lead into the sinks (nodes 1 and 2 from the Morse graph)
        adjacencies = [[1, 2], [2], []]
        map_graph = MockMapGraph(3, adjacencies)

        # Morse sets: node 1 owns cell 1, node 2 owns cell 2
        cmgdb_morse_graph = MockCMGDBMorseGraph({1: [1], 2: [2]})

        bounds = LatentBounds(
            lower=np.array([0.0, 0.0], dtype=np.float64),
            upper=np.array([1.0, 1.0], dtype=np.float64),
        )

        roa = compute_exact_roa(
            map_graph,
            cmgdb_morse_graph,
            morse_graph,
            bounds=bounds,
            collapse_to_lca=True,
        )

        # Verify the returned CellROA structure
        assert isinstance(roa, CellROA)
        assert roa.box_roa is not None
        assert roa.bounds_lower is not None
        assert roa.bounds_upper is not None
        assert roa.reach_mask is not None
        assert roa.minimal_order is not None

        # Cell 0 should reach both minimal nodes (1 and 2)
        # When collapsed to LCA, it should be assigned to node 0 (the LCA of {1, 2})
        assert roa.box_roa[0] == 0
        assert roa.box_roa[1] == 1
        assert roa.box_roa[2] == 2


def test_collapse_roa_to_lca():
    """Test collapsing multi-label cells to LCA."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dot_path = Path(tmpdir) / "morse_graph.dot"
        create_minimal_morse_graph_dot(dot_path)
        morse_graph = MorseGraph.from_dot(dot_path)

        adjacencies = [[1, 2], [2], []]
        map_graph = MockMapGraph(3, adjacencies)
        cmgdb_morse_graph = MockCMGDBMorseGraph({1: [1], 2: [2]})

        bounds = LatentBounds(
            lower=np.array([0.0, 0.0], dtype=np.float64),
            upper=np.array([1.0, 1.0], dtype=np.float64),
        )

        # Compute without collapsing first
        roa_uncollapsed = compute_exact_roa(
            map_graph,
            cmgdb_morse_graph,
            morse_graph,
            bounds=bounds,
            collapse_to_lca=False,
        )

        # Now collapse
        collapsed_roa = collapse_roa_to_lca(roa_uncollapsed, morse_graph)

        # The result should be a 1D array of labels
        assert collapsed_roa.ndim == 1
        assert len(collapsed_roa) == 3


def test_save_and_load_exact_roa():
    """Test save and load of RoA data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dot_path = Path(tmpdir) / "morse_graph.dot"
        create_minimal_morse_graph_dot(dot_path)
        morse_graph = MorseGraph.from_dot(dot_path)

        adjacencies = [[1, 2], [2], []]
        map_graph = MockMapGraph(3, adjacencies)
        cmgdb_morse_graph = MockCMGDBMorseGraph({1: [1], 2: [2]})

        bounds = LatentBounds(
            lower=np.array([0.0, 0.0], dtype=np.float64),
            upper=np.array([1.0, 1.0], dtype=np.float64),
        )

        roa = compute_exact_roa(
            map_graph,
            cmgdb_morse_graph,
            morse_graph,
            bounds=bounds,
            collapse_to_lca=True,
        )

        out_dir = Path(tmpdir) / "roa_output"
        save_path = save_exact_roa(roa, out_dir)

        assert save_path.exists()

        # Load it back
        loaded_roa = load_exact_roa(save_path)

        assert np.array_equal(loaded_roa.box_roa, roa.box_roa)
        assert np.array_equal(loaded_roa.bounds_lower, roa.bounds_lower)
        assert np.array_equal(loaded_roa.bounds_upper, roa.bounds_upper)


def test_latent_bounds_dataclass():
    """Test the inlined LatentBounds dataclass."""
    lower = np.array([-1.0, -2.0], dtype=np.float64)
    upper = np.array([1.0, 2.0], dtype=np.float64)
    bounds = LatentBounds(lower=lower, upper=upper)

    assert bounds.dim == 2
    assert np.array_equal(bounds.lower, lower)
    assert np.array_equal(bounds.upper, upper)

    # Test immutability (frozen=True)
    with pytest.raises(dataclasses.FrozenInstanceError):
        bounds.dim = 3  # type: ignore


def test_collapse_roa_to_lca_stronger():
    """Test that collapse_roa_to_lca correctly assigns LCA labels to multi-label cells."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dot_path = Path(tmpdir) / "morse_graph.dot"
        create_minimal_morse_graph_dot(dot_path)
        morse_graph = MorseGraph.from_dot(dot_path)

        adjacencies = [[1, 2], [2], []]
        map_graph = MockMapGraph(3, adjacencies)
        cmgdb_morse_graph = MockCMGDBMorseGraph({1: [1], 2: [2]})

        bounds = LatentBounds(
            lower=np.array([0.0, 0.0], dtype=np.float64),
            upper=np.array([1.0, 1.0], dtype=np.float64),
        )

        # Compute without collapsing to get multi-label cells
        roa_uncollapsed = compute_exact_roa(
            map_graph,
            cmgdb_morse_graph,
            morse_graph,
            bounds=bounds,
            collapse_to_lca=False,
        )

        # Collapse the RoA
        collapsed_roa = collapse_roa_to_lca(roa_uncollapsed, morse_graph)

        # Verify shape and dimensionality
        assert collapsed_roa.ndim == 1
        assert len(collapsed_roa) == 3

        # Verify the actual collapsed labels:
        # Cell 0 reaches both minimal nodes {1, 2}, so it should collapse to their LCA (node 0)
        assert collapsed_roa[0] == 0
        # Cell 1 reaches only minimal node 1
        assert collapsed_roa[1] == 1
        # Cell 2 reaches only minimal node 2
        assert collapsed_roa[2] == 2


def test_compute_and_save_exact_roa_happy_path():
    """Test the orchestrating compute_and_save_exact_roa entry point."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dot_path = Path(tmpdir) / "morse_graph.dot"
        create_minimal_morse_graph_dot(dot_path)

        adjacencies = [[1, 2], [2], []]
        map_graph = MockMapGraph(3, adjacencies)
        cmgdb_morse_graph = MockCMGDBMorseGraph({1: [1], 2: [2]})

        bounds = LatentBounds(
            lower=np.array([0.0, 0.0], dtype=np.float64),
            upper=np.array([1.0, 1.0], dtype=np.float64),
        )

        out_dir = Path(tmpdir) / "roa_output"

        # Call compute_and_save_exact_roa
        from CMGDB.cmgdb_roa import compute_and_save_exact_roa

        save_path = compute_and_save_exact_roa(
            map_graph=map_graph,
            cmgdb_morse_graph=cmgdb_morse_graph,
            morse_graph_dot=dot_path,
            out_dir=out_dir,
            bounds=bounds,
            max_vertices=1_000_000,
            collapse_to_lca=True,
        )

        # Verify the file was written
        assert save_path.exists()

        # Load it back and verify round-trip
        loaded_roa = load_exact_roa(save_path)
        assert loaded_roa.box_roa is not None
        assert len(loaded_roa.box_roa) == 3


def test_compute_and_save_exact_roa_max_vertices_guard():
    """Test that compute_and_save_exact_roa raises when max_vertices is exceeded."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dot_path = Path(tmpdir) / "morse_graph.dot"
        create_minimal_morse_graph_dot(dot_path)

        adjacencies = [[1, 2], [2], []]
        map_graph = MockMapGraph(3, adjacencies)
        cmgdb_morse_graph = MockCMGDBMorseGraph({1: [1], 2: [2]})

        bounds = LatentBounds(
            lower=np.array([0.0, 0.0], dtype=np.float64),
            upper=np.array([1.0, 1.0], dtype=np.float64),
        )

        out_dir = Path(tmpdir) / "roa_output"

        from CMGDB.cmgdb_roa import compute_and_save_exact_roa

        # Call with max_vertices=1 to trigger the guard (map has 3 vertices)
        with pytest.raises(ValueError, match="exceeding"):
            compute_and_save_exact_roa(
                map_graph=map_graph,
                cmgdb_morse_graph=cmgdb_morse_graph,
                morse_graph_dot=dot_path,
                out_dir=out_dir,
                bounds=bounds,
                max_vertices=1,  # Too small; should raise
                collapse_to_lca=True,
            )
