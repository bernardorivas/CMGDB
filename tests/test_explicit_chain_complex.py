import pytest

import CMGDB


def compute(cell_counts, boundaries, chain_map):
    return CMGDB.ComputeRelativeHomologyShiftClass(
        cell_counts, boundaries, chain_map
    )


def test_point_identity_has_expected_structured_result():
    result = compute([1], [[]], [[(0, 0, 1)]])

    assert result == {
        "coefficient_field": 5,
        "cell_counts": [1],
        "validation": {
            "matrix_shapes_and_entries": True,
            "boundary_squared_zero": True,
            "chain_map_equation": True,
        },
        "homology_dimensions": [1],
        "induced_maps": [[[1]]],
        "shift_class": ["x-1"],
    }


def test_circle_degree_minus_one_map():
    # One vertex, one loop, and zero cellular boundary.  The map fixes H_0
    # and acts by -1 on H_1.
    result = compute(
        [1, 1],
        [[], []],
        [[(0, 0, 1)], [(0, 0, -1)]],
    )

    assert result["homology_dimensions"] == [1, 1]
    assert result["induced_maps"] == [[[1]], [[-1]]]
    assert result["shift_class"] == ["x-1", "x+1"]


def test_contractible_interval_reduces_before_inducing_map():
    # Boundary of the oriented edge is v_1-v_0.  This also exercises an empty
    # homology group in positive degree.
    result = compute(
        [2, 1],
        [[], [(0, 0, -1), (1, 0, 1)]],
        [[(0, 0, 1), (1, 1, 1)], [(0, 0, 1)]],
    )

    assert result["homology_dimensions"] == [1, 0]
    assert result["induced_maps"] == [[[1]], []]
    assert result["shift_class"] == ["x-1", "0"]


def test_zero_sized_intermediate_chain_group_is_preserved():
    result = compute(
        [1, 0, 1],
        [[], [], []],
        [[(0, 0, 1)], [], [(0, 0, 1)]],
    )

    assert result["homology_dimensions"] == [1, 0, 1]
    assert result["shift_class"] == ["x-1", "0", "x-1"]


def test_coefficients_are_reduced_modulo_five():
    result = compute([1], [[]], [[(0, 0, 6)]])
    assert result["induced_maps"] == [[[1]]]
    assert result["shift_class"] == ["x-1"]


def test_shift_class_discards_nilpotent_part():
    # The chain complex is concentrated in degree zero and the displayed map
    # is a nilpotent Jordan block.  Its shift-equivalence class is trivial.
    result = compute([2], [[]], [[(0, 1, 1)]])

    assert result["homology_dimensions"] == [2]
    assert result["induced_maps"] == [[[0, 1], [0, 0]]]
    assert result["shift_class"] == ["0"]


def test_rejects_nonzero_boundary_squared():
    with pytest.raises(ValueError, match="boundary squared is nonzero"):
        compute(
            [1, 1, 1],
            [[], [(0, 0, 1)], [(0, 0, 1)]],
            [[(0, 0, 1)], [(0, 0, 1)], [(0, 0, 1)]],
        )


def test_rejects_failure_of_chain_map_equation():
    with pytest.raises(ValueError, match="chain-map equation fails"):
        compute(
            [2, 1],
            [[], [(0, 0, -1), (1, 0, 1)]],
            [[(0, 0, 1)], [(0, 0, 1)]],
        )


def test_rejects_duplicate_sparse_coordinates():
    with pytest.raises(ValueError, match="duplicate chain map entry"):
        compute([1], [[]], [[(0, 0, 1), (0, 0, 2)]])


def test_rejects_out_of_bounds_sparse_coordinates():
    with pytest.raises(IndexError, match="outside its 1 x 1 matrix"):
        compute([1], [[]], [[(1, 0, 1)]])


def test_rejects_missing_degree_lists():
    with pytest.raises(ValueError, match="boundary_entries must have one list"):
        compute([1, 1], [[]], [[(0, 0, 1)], [(0, 0, 1)]])
