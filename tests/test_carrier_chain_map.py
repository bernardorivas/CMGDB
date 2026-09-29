"""Tests of ComputeCarrierChainMap, the acyclic-carrier chain-map kernel.

A pure-Python reference of the same specification is compared with the native
function on hand-built and randomized simplicial complexes, for exact equality
of the chain map and of the relative payload, including the order of entries.
The reference forms every carrier before it checks any, so the comparison also
shows that the native function, which stops at the first cell whose carrier is
not acyclic, reports the same failure.
"""

import itertools
import random

import numpy as np
import pytest

import CMGDB

P = 5

OK_KEYS = {"status", "failure_degree", "failure_row", "carrier_count", "chain_map"}
FAILURE_KEYS = {"status", "failure_degree", "failure_row", "carrier_count"}


# Complexes and arguments.


def closure(simplices):
    """All nonempty faces of the given simplices, as sorted tuples."""
    cells = set()
    for simplex in simplices:
        simplex = tuple(sorted(simplex))
        for size in range(1, len(simplex) + 1):
            cells.update(itertools.combinations(simplex, size))
    return cells


def complex_order(cells):
    """Cells grouped by degree, each degree in lexicographic order."""
    top = max(len(cell) for cell in cells) - 1
    return [sorted(cell for cell in cells if len(cell) == d + 1) for d in range(top + 1)]


def simplicial_complex(simplices):
    return complex_order(closure(simplices))


def as_arrays(degrees):
    return [
        np.array(group, dtype=np.int32).reshape(len(group), d + 1)
        for d, group in enumerate(degrees)
    ]


def image_csr(degrees, images):
    indptr = [0]
    indices = []
    for (vertex,) in degrees[0]:
        indices.extend(images[vertex])
        indptr.append(len(indices))
    return np.array(indptr, dtype=np.int64), np.array(indices, dtype=np.int32)


def exit_mask(degrees, exit_vertices):
    return np.array([vertex in exit_vertices for (vertex,) in degrees[0]], dtype=np.uint8)


def native(source, images, source_exit=(), *, target=None, target_exit=(), **kwargs):
    indptr, indices = image_csr(source, images)
    options = dict(kwargs)
    if target is not None:
        options["target_simplices"] = as_arrays(target)
        options["target_exit"] = exit_mask(target, set(target_exit))
    return CMGDB.ComputeCarrierChainMap(
        as_arrays(source), indptr, indices, exit_mask(source, set(source_exit)), **options
    )


def comparable(result):
    """The native result with its arrays turned into lists."""
    output = dict(result)
    if "chain_map" in output:
        output["chain_map"] = [
            [tuple(int(value) for value in row) for row in array] for array in output["chain_map"]
        ]
    if "carrier_ids" in output:
        output["carrier_ids"] = [int(value) for value in output["carrier_ids"]]
    return output


def shift_class(result):
    payload = result["payload"]
    return CMGDB.ComputeRelativeHomologyShiftClass(
        payload["cell_counts"], payload["boundary_entries"], payload["chain_map_entries"]
    )


def identity(degrees):
    return {vertex: [vertex] for (vertex,) in degrees[0]}


# Pure-Python reference of the specification.


def boundary(cell):
    """Faces of a simplex with incidence (-1)^i, in removal-index order."""
    if len(cell) == 1:
        return {}
    return {cell[:i] + cell[i + 1 :]: (-1) ** i for i in range(len(cell))}


def add_scaled(accumulator, vector, scale):
    """accumulator += scale * vector over GF(5), keeping dict insertion order."""
    scale %= P
    if not scale:
        return
    for key, value in vector.items():
        updated = (accumulator.get(key, 0) + scale * value) % P
        if updated:
            accumulator[key] = updated
        else:
            accumulator.pop(key, None)


def rank(columns):
    """Rank over GF(5) of sparse columns, with largest-row pivots."""
    pivots = {}
    for column in columns:
        vector = {row: value % P for row, value in column.items() if value % P}
        while vector:
            pivot = max(vector)
            if pivot not in pivots:
                inverse = pow(vector[pivot], -1, P)
                pivots[pivot] = {row: value * inverse % P for row, value in vector.items()}
                break
            add_scaled(vector, pivots[pivot], -vector[pivot])
    return len(pivots)


def eliminate(rows, columns):
    """Smallest-pivot column echelon form, with the combinations of columns."""
    row_index = {cell: index for index, cell in enumerate(rows)}
    pivots = {}
    for number, column in enumerate(columns):
        vector = {}
        for face, sign in boundary(column).items():
            add_scaled(vector, {row_index[face]: 1}, sign)
        combination = {number: 1}
        while vector:
            pivot = min(vector)
            if pivot in pivots:
                scale = -vector[pivot]
                add_scaled(vector, pivots[pivot][0], scale)
                add_scaled(combination, pivots[pivot][1], scale)
                continue
            inverse = pow(vector[pivot], -1, P)
            pivots[pivot] = (
                {key: value * inverse % P for key, value in vector.items()},
                {key: value * inverse % P for key, value in combination.items()},
            )
            break
    return row_index, pivots


def solve(row_index, pivots, columns, rhs):
    """The solution supported on the pivot columns, or None."""
    residual = {row_index[cell]: value for cell, value in rhs.items()}
    solution = {}
    while residual:
        pivot = min(residual)
        if pivot not in pivots:
            return None
        scale = residual[pivot]
        add_scaled(residual, pivots[pivot][0], -scale)
        add_scaled(solution, pivots[pivot][1], scale)
    return {columns[number]: value for number, value in solution.items()}


def reference(source, images, source_exit=(), *, target=None, target_exit=()):
    """ComputeCarrierChainMap on cells given as sorted label tuples."""

    same = target is None
    source_exit = set(source_exit)
    if same:
        target, target_exit = source, source_exit
    target_exit = set(target_exit)
    target_row = [{cell: row for row, cell in enumerate(group)} for group in target]

    def induced(vertices, d):
        if d >= len(target):
            return []
        return [cell for cell in target[d] if vertices.issuperset(cell)]

    carrier_of = {}
    number = {}
    first_use = []
    carrier_ids = []
    empty = None
    for d, group in enumerate(source):
        for row, cell in enumerate(group):
            vertices = frozenset(t for v in cell for t in images[v])
            if not vertices:
                empty = empty or (d, row)
                carrier_ids.append(-1)
                continue
            if vertices not in number:
                number[vertices] = len(first_use)
                first_use.append((d, row))
            carrier_of[cell] = vertices
            carrier_ids.append(number[vertices])
    result = {"carrier_count": len(first_use), "carrier_ids": carrier_ids}

    def fail(status, d, row):
        result.update(status=status, failure_degree=d, failure_row=row)
        if status in ("empty_carrier", "not_acyclic"):
            # The kernel stops at the failing cell: only the carriers of the
            # cells before it are numbered.
            before = sum(len(group) for group in source[:d]) + row
            kept = carrier_ids[:before]
            result["carrier_ids"] = kept + [-1] * (len(carrier_ids) - before)
            result["carrier_count"] = len({index for index in kept if index >= 0})
        return result

    if empty:
        return fail("empty_carrier", *empty)

    # Carriers are numbered by first use, so the first failing carrier in that
    # order is the carrier of the first failing cell.
    for vertices, index in sorted(number.items(), key=lambda item: item[1]):
        cells = [induced(vertices, d) for d in range(len(target))]
        ranks = [0] * (len(target) + 1)
        for d in range(1, len(target)):
            row_index = {cell: row for row, cell in enumerate(cells[d - 1])}
            ranks[d] = rank(
                [
                    {row_index[face]: sign for face, sign in boundary(cell).items()}
                    for cell in cells[d]
                ]
            )
        betti = [len(cells[d]) - ranks[d] - ranks[d + 1] for d in range(len(target))]
        if betti != [1] + [0] * (len(target) - 1):
            return fail("not_acyclic", *first_use[index])

    for d, group in enumerate(source):
        for row, cell in enumerate(group):
            if set(cell) <= source_exit and not carrier_of[cell] <= target_exit:
                return fail("pair_violation", d, row)

    phi = {}
    for d, group in enumerate(source):
        for row, cell in enumerate(group):
            vertices = carrier_of[cell]
            if d == 0:
                phi[cell] = {(min(vertices),): 1}
                continue
            rhs = {}
            for face, sign in boundary(cell).items():
                add_scaled(rhs, phi[face], sign)
            columns = induced(vertices, d)
            row_index, pivots = eliminate(induced(vertices, d - 1), columns)
            if not set(rhs) <= set(row_index):
                return fail("chain_map_invalid", d, row)
            image = solve(row_index, pivots, columns, rhs)
            if image is None:
                return fail("no_solution", d, row)
            phi[cell] = image

    # The chain-map equation, checked apart from the construction.
    for group in source[1:]:
        for cell in group:
            mapped_boundary = {}
            for face, sign in boundary(cell).items():
                add_scaled(mapped_boundary, phi[face], sign)
            boundary_of_image = {}
            for image_cell, value in phi[cell].items():
                add_scaled(boundary_of_image, boundary(image_cell), value)
            assert mapped_boundary == boundary_of_image

    result.update(status="ok", failure_degree=-1, failure_row=-1)
    result["chain_map"] = [
        [
            (row, target_row[d][image_cell], value)
            for row, cell in enumerate(group)
            for image_cell, value in phi[cell].items()
        ]
        for d, group in enumerate(source)
    ]
    if same:
        basis = [[cell for cell in group if not set(cell) <= source_exit] for group in source]
        row_of = [{cell: row for row, cell in enumerate(group)} for group in basis]
        boundary_entries = [[]]
        for d in range(1, len(basis)):
            boundary_entries.append(
                [
                    (row_of[d - 1][face], column, sign % P)
                    for column, cell in enumerate(basis[d])
                    for face, sign in boundary(cell).items()
                    if face in row_of[d - 1]
                ]
            )
        chain_map_entries = [
            [
                (row_of[d][image_cell], column, value)
                for column, cell in enumerate(basis[d])
                for image_cell, value in phi[cell].items()
                if image_cell in row_of[d]
            ]
            for d in range(len(basis))
        ]
        result["payload"] = {
            "cell_counts": [len(group) for group in basis],
            "boundary_entries": boundary_entries,
            "chain_map_entries": chain_map_entries,
        }
    return result


def assert_matches_reference(source, images, source_exit=(), *, target=None, target_exit=()):
    expected = reference(source, images, source_exit, target=target, target_exit=target_exit)
    result = comparable(
        native(
            source,
            images,
            source_exit,
            target=target,
            target_exit=target_exit,
            return_carriers=True,
        )
    )
    assert result == expected
    return result


# Hand-built complexes.

TRIANGLE = simplicial_complex([(0, 1, 2)])
CIRCLE = simplicial_complex([(0, 1), (1, 2), (0, 2)])
HEXAGON = simplicial_complex([(i, (i + 1) % 6) for i in range(6)])
# A triangulated annulus: outer ring 0..3, inner ring 4..7.
ANNULUS = simplicial_complex(
    [(j, (j + 1) % 4, 4 + j) for j in range(4)]
    + [((j + 1) % 4, 4 + j, 4 + (j + 1) % 4) for j in range(4)]
)


def test_complex_sizes():
    assert [len(group) for group in TRIANGLE] == [3, 3, 1]
    assert [len(group) for group in ANNULUS] == [8, 16, 8]


def test_identity_on_a_triangle():
    result = native(TRIANGLE, identity(TRIANGLE))

    assert set(result) == OK_KEYS | {"payload"}
    assert result["status"] == "ok"
    assert (result["failure_degree"], result["failure_row"]) == (-1, -1)
    assert result["carrier_count"] == 7
    for d, array in enumerate(result["chain_map"]):
        assert array.dtype == np.int64
        assert array.shape == (len(TRIANGLE[d]), 3)
        assert array.tolist() == [[row, row, 1] for row in range(len(TRIANGLE[d]))]
    assert result["payload"] == {
        "cell_counts": [3, 3, 1],
        "boundary_entries": [
            [],
            [(1, 0, 1), (0, 0, 4), (2, 1, 1), (0, 1, 4), (2, 2, 1), (1, 2, 4)],
            [(2, 0, 1), (1, 0, 4), (0, 0, 1)],
        ],
        "chain_map_entries": [
            [(0, 0, 1), (1, 1, 1), (2, 2, 1)],
            [(0, 0, 1), (1, 1, 1), (2, 2, 1)],
            [(0, 0, 1)],
        ],
    }
    assert shift_class(result)["shift_class"] == ["x-1", "0", "0"]


def test_identity_on_a_circle():
    result = assert_matches_reference(CIRCLE, identity(CIRCLE))
    assert result["payload"]["chain_map_entries"] == [
        [(0, 0, 1), (1, 1, 1), (2, 2, 1)],
        [(0, 0, 1), (1, 1, 1), (2, 2, 1)],
    ]
    assert shift_class(result)["shift_class"] == ["x-1", "x-1"]


@pytest.mark.parametrize(
    "vertex_map, expected",
    [
        (lambda i: (i + 1) % 6, ["x-1", "x-1"]),  # rotation
        (lambda i: (-i) % 6, ["x-1", "x+1"]),  # reflection
    ],
)
def test_simplicial_maps_of_a_hexagon(vertex_map, expected):
    images = {i: [vertex_map(i)] for i in range(6)}
    result = assert_matches_reference(HEXAGON, images)
    assert result["status"] == "ok"
    assert shift_class(result)["shift_class"] == expected


def test_enlarged_carriers_on_a_hexagon():
    # T(v_i) = {v_i, v_(i+1)}: vertex carriers are edges and edge carriers are
    # paths of two edges, so every carrier is acyclic.
    images = {i: [i, (i + 1) % 6] for i in range(6)}
    result = assert_matches_reference(HEXAGON, images)
    assert result["status"] == "ok"
    assert result["chain_map"][0][5] == (5, 0, 1)  # the smallest vertex of {5, 0}
    assert shift_class(result)["shift_class"] == ["x-1", "x-1"]


def test_identity_and_enlarged_carriers_on_an_annulus():
    result = assert_matches_reference(ANNULUS, identity(ANNULUS))
    assert shift_class(result)["shift_class"] == ["x-1", "x-1", "0"]

    def rotate(v):
        return (v + 1) % 4 if v < 4 else 4 + (v - 3) % 4

    rotation = assert_matches_reference(ANNULUS, {v: [rotate(v)] for v in range(8)})
    assert shift_class(rotation)["shift_class"] == ["x-1", "x-1", "0"]

    # Every carrier of T(v) = {v, rotate(v)} is a strip of at most three
    # triangles, so the carrier is acyclic and homotopic to the identity.
    enlarged = assert_matches_reference(ANNULUS, {v: [v, rotate(v)] for v in range(8)})
    assert enlarged["status"] == "ok"
    assert shift_class(enlarged)["shift_class"] == ["x-1", "x-1", "0"]


def test_relative_pair_drops_exit_cells():
    path = simplicial_complex([(0, 1), (1, 2)])
    result = assert_matches_reference(path, identity(path), {0, 2})
    assert result["payload"] == {
        "cell_counts": [1, 2],
        "boundary_entries": [[], [(0, 0, 1), (0, 1, 4)]],
        "chain_map_entries": [[(0, 0, 1)], [(0, 0, 1), (1, 1, 1)]],
    }
    output = shift_class(result)
    assert output["homology_dimensions"] == [0, 1]
    assert output["shift_class"] == ["0", "x-1"]


def test_labels_are_arbitrary_int32_values():
    labels = [-2**31, -7, 12, 2**31 - 1]
    cone = simplicial_complex([(labels[0], labels[1], labels[3]), (labels[1], labels[2], labels[3])])
    images = {labels[0]: [labels[1]], labels[1]: [labels[3], labels[2]], labels[2]: [labels[2]], labels[3]: [labels[3]]}
    result = assert_matches_reference(cone, images, {labels[2]})
    assert result["status"] == "ok"


def test_different_target_complex():
    # The boundary of a triangle mapped into the full triangle: no payload.
    result = assert_matches_reference(CIRCLE, identity(CIRCLE), target=TRIANGLE)
    assert result["status"] == "ok"
    assert "payload" not in result
    assert result["chain_map"][1] == [(0, 0, 1), (1, 1, 1), (2, 2, 1)]

    # A circle wrapped once around a hexagon, relabeled.
    target = [[tuple(10 + v for v in cell) for cell in group] for group in HEXAGON]
    images = {0: [10, 11], 1: [12, 13], 2: [14, 15]}
    wrapped = assert_matches_reference(CIRCLE, images, target=target)
    assert wrapped["status"] == "ok"


# Failures.


def test_first_non_acyclic_vertex_carrier():
    images = {i: [i] for i in range(6)}
    images[3] = list(range(6))
    result = assert_matches_reference(HEXAGON, images)
    assert set(result) == FAILURE_KEYS | {"carrier_ids"}
    assert (result["status"], result["failure_degree"], result["failure_row"]) == (
        "not_acyclic",
        0,
        3,
    )


def test_non_acyclic_edge_carrier_of_a_circle():
    # Vertex carriers are edges; every edge carrier is the whole circle.
    images = {i: [i, (i + 1) % 3] for i in range(3)}
    result = assert_matches_reference(CIRCLE, images)
    assert (result["status"], result["failure_degree"], result["failure_row"]) == (
        "not_acyclic",
        1,
        0,
    )


def test_first_failure_is_in_complex_order():
    # A square 0-1-2-3.  Only the carrier of edge (1, 2), row 2, is the whole
    # square; the carriers of the vertices and of the other edges are paths.
    square = simplicial_complex([(0, 1), (1, 2), (2, 3), (0, 3)])
    assert square[1] == [(0, 1), (0, 3), (1, 2), (2, 3)]
    images = {0: [0], 1: [1], 2: [0, 2, 3], 3: [3]}
    result = assert_matches_reference(square, images)
    assert (result["status"], result["failure_degree"], result["failure_row"]) == (
        "not_acyclic",
        1,
        2,
    )


def test_empty_carrier():
    images = {0: [0], 1: [], 2: [2]}
    result = comparable(native(TRIANGLE, images, return_carriers=True))
    assert (result["status"], result["failure_degree"], result["failure_row"]) == (
        "empty_carrier",
        0,
        1,
    )
    assert result["carrier_ids"][1] == -1
    assert result == reference(TRIANGLE, images)


def test_pair_violation():
    path = simplicial_complex([(0, 1), (1, 2)])
    images = {0: [1], 1: [1], 2: [2]}
    result = assert_matches_reference(path, images, {0})
    assert (result["status"], result["failure_degree"], result["failure_row"]) == (
        "pair_violation",
        0,
        0,
    )


def test_failure_stops_at_the_failing_cell():
    # Vertex 2 of the hexagon has the whole hexagon as its carrier.
    images = {i: [i, (i + 1) % 6] for i in range(6)}
    images[2] = list(range(6))
    result = assert_matches_reference(HEXAGON, images)
    assert (result["status"], result["failure_degree"], result["failure_row"]) == (
        "not_acyclic",
        0,
        2,
    )
    assert result["carrier_count"] == 2
    assert result["carrier_ids"] == [0, 1] + [-1] * 10


def test_empty_carrier_precedes_an_earlier_non_acyclic_carrier():
    # The carrier of vertex 0 is the whole circle; vertex 2 has no image.
    images = {0: [0, 1, 2], 1: [1], 2: []}
    result = assert_matches_reference(CIRCLE, images)
    assert (result["status"], result["failure_degree"], result["failure_row"]) == (
        "empty_carrier",
        0,
        2,
    )
    assert result["carrier_count"] == 2
    assert result["carrier_ids"] == [0, 1, -1, -1, -1, -1]


def test_later_failures_number_every_carrier():
    path = simplicial_complex([(0, 1), (1, 2)])
    result = assert_matches_reference(path, {0: [1], 1: [1], 2: [2]}, {0})
    assert result["status"] == "pair_violation"
    assert result["carrier_count"] == 3
    assert result["carrier_ids"] == [0, 0, 1, 0, 2]


def test_carrier_ids_are_returned_only_on_request():
    assert "carrier_ids" not in native(TRIANGLE, identity(TRIANGLE))
    ids = native(TRIANGLE, {0: [0], 1: [0], 2: [2]}, return_carriers=True)["carrier_ids"]
    assert ids.dtype == np.int64
    assert ids.tolist() == [0, 0, 1, 0, 2, 2, 2]


# Input validation.


def call(**changes):
    arguments = {
        "source_simplices": as_arrays(TRIANGLE),
        "vertex_image_indptr": np.array([0, 1, 2, 3], dtype=np.int64),
        "vertex_image_indices": np.array([0, 1, 2], dtype=np.int32),
        "source_exit": np.zeros(3, dtype=np.uint8),
    }
    arguments.update(changes)
    return CMGDB.ComputeCarrierChainMap(**arguments)


def test_valid_call_and_plain_lists():
    assert call()["status"] == "ok"
    result = CMGDB.ComputeCarrierChainMap(
        [[0, 1, 2], [[0, 1], [0, 2], [1, 2]], [[0, 1, 2]]], [0, 1, 2, 3], [0, 1, 2], [0, 0, 0]
    )
    assert result["status"] == "ok"
    assert call(source_exit=np.array([True, False, False]))["status"] == "ok"


@pytest.mark.parametrize(
    "changes, error, match",
    [
        ({"modulus": 3}, ValueError, "modulus=5"),
        ({"source_simplices": []}, ValueError, "0-cells"),
        ({"source_simplices": np.zeros((3, 1), dtype=np.int32)}, TypeError, "list"),
        (
            {"source_simplices": [np.array([[0], [1], [2]]), np.array([[0, 1, 2]])]},
            ValueError,
            r"source_simplices\[1\] must have shape \(n, 2\)",
        ),
        (
            {"source_simplices": [np.array([[0], [1], [2]], dtype=float)]},
            TypeError,
            "integer dtype",
        ),
        (
            {"source_simplices": [np.array([[0], [2], [1]])]},
            ValueError,
            r"source_simplices\[0\] rows 1 and 2",
        ),
        (
            {"source_simplices": [np.array([[0], [1], [2]]), np.array([[0, 2], [0, 1]])]},
            ValueError,
            r"source_simplices\[1\] rows 0 and 1",
        ),
        (
            {"source_simplices": [np.array([[0], [1], [2]]), np.array([[1, 0]])]},
            ValueError,
            r"source_simplices\[1\] row 0 is not a strictly increasing",
        ),
        (
            {"source_simplices": [np.array([[0], [1], [2]]), np.array([[0, 5]])]},
            ValueError,
            r"source_simplices\[1\] row 0 has vertex label 5",
        ),
        (
            {
                "source_simplices": [
                    np.array([[0], [1], [2]]),
                    np.array([[0, 1], [1, 2]]),
                    np.array([[0, 1, 2]]),
                ]
            },
            ValueError,
            r"face \(0, 2\) \(removal index 1\) of source_simplices\[2\] row 0",
        ),
        (
            {"source_simplices": [np.array([[0], [1], [2**40]])]},
            ValueError,
            "int32 range",
        ),
        ({"vertex_image_indptr": np.array([0, 1, 2])}, ValueError, "length"),
        ({"vertex_image_indptr": np.array([0, 1, 7, 3])}, IndexError, r"vertex_image_indptr\[2\] = 7"),
        ({"vertex_image_indptr": np.array([0, 2, 1, 3])}, ValueError, "decreases from position 1"),
        ({"vertex_image_indptr": np.array([1, 1, 2, 3])}, ValueError, r"vertex_image_indptr\[0\]"),
        ({"vertex_image_indptr": np.array([0, 1, 2, 2])}, ValueError, r"vertex_image_indptr\[3\] = 2"),
        (
            {"vertex_image_indices": np.array([0, 9, 2])},
            ValueError,
            r"vertex_image_indices\[1\] = 9 \(in the image of source vertex row 1\)",
        ),
        ({"source_exit": np.zeros(2, dtype=np.uint8)}, ValueError, "source_exit must have one entry"),
        ({"source_exit": np.array([0, 2, 0])}, ValueError, r"source_exit\[1\] = 2"),
        ({"target_exit": np.zeros(3, dtype=np.uint8)}, ValueError, "only together"),
        ({"target_simplices": as_arrays(TRIANGLE)}, ValueError, "target_exit is required"),
        (
            {
                "target_simplices": [np.array([[0], [1]])],
                "target_exit": np.zeros(3, dtype=np.uint8),
            },
            ValueError,
            "target_exit must have one entry",
        ),
        (
            {"target_simplices": [np.array([[0], [1]])], "target_exit": np.zeros(2, dtype=np.uint8)},
            ValueError,
            "vertex_image_indices\\[2\\] = 2 .* not a vertex label of target_simplices",
        ),
    ],
)
def test_input_validation(changes, error, match):
    with pytest.raises(error, match=match):
        call(**changes)


def test_modulus_five_is_accepted_by_keyword_only():
    assert call(modulus=5)["status"] == "ok"
    with pytest.raises(TypeError):
        CMGDB.ComputeCarrierChainMap(
            as_arrays(TRIANGLE), [0, 1, 2, 3], [0, 1, 2], [0, 0, 0], None, None, 5
        )


# Randomized comparison with the reference.


def random_labels(rng, count):
    return sorted(rng.sample(range(-40, 400), count))


def random_complex(rng, labels, top_dimension, simplex_count):
    simplices = [(label,) for label in labels]
    for _ in range(simplex_count):
        size = rng.randint(2, min(top_dimension + 1, len(labels)))
        simplices.append(tuple(rng.sample(labels, size)))
    return simplicial_complex(simplices)


def cone(rng, top_dimension):
    """A cone over a random complex, and its apex; every carrier that contains
    the apex is a cone, hence acyclic."""
    labels = random_labels(rng, rng.randint(3, 9))
    apex = rng.choice(labels)
    base = [label for label in labels if label != apex]
    simplices = [(label,) for label in base]
    for _ in range(rng.randint(1, 8)):
        size = rng.randint(1, min(top_dimension, len(base)))
        simplices.append(tuple(rng.sample(base, size)))
    return simplicial_complex([simplex + (apex,) for simplex in simplices]), apex


def random_subset(rng, labels, largest):
    return rng.sample(labels, rng.randint(0, min(largest, len(labels))))


@pytest.mark.parametrize("seed", range(80))
def test_random_cones_match_reference(seed):
    rng = random.Random(seed)
    complex_, apex = cone(rng, top_dimension=rng.randint(1, 5))
    labels = [vertex for (vertex,) in complex_[0]]
    exit_vertices = set(random_subset(rng, labels, 3)) | ({apex} if rng.random() < 0.5 else set())
    images = {}
    for vertex in labels:
        pool = sorted(exit_vertices) if vertex in exit_vertices else labels
        images[vertex] = [apex] + random_subset(rng, pool, 3) if apex in pool else random_subset(rng, pool, 2)
        rng.shuffle(images[vertex])
        if rng.random() < 0.3 and images[vertex]:
            images[vertex].append(images[vertex][0])  # a duplicate
    assert_matches_reference(complex_, images, exit_vertices)


@pytest.mark.parametrize("seed", range(80))
def test_random_complexes_match_reference(seed):
    # Random images that are often not acyclic, and simplicial images that
    # always are.
    rng = random.Random(1000 + seed)
    labels = random_labels(rng, rng.randint(2, 9))
    complex_ = random_complex(rng, labels, rng.randint(1, 5), rng.randint(1, 7))
    if seed % 2 == 0:
        images = {vertex: random_subset(rng, labels, 3) or [vertex] for vertex in labels}
    else:
        simplex = rng.choice(complex_[-1])
        images = {vertex: rng.sample(simplex, rng.randint(1, len(simplex))) for vertex in labels}
    exit_vertices = set(random_subset(rng, labels, 2)) if seed % 3 == 0 else set()
    assert_matches_reference(complex_, images, exit_vertices)


@pytest.mark.parametrize("seed", range(120))
def test_random_failures_match_reference(seed):
    # The identity with a few enlarged vertex images, and at times an empty
    # one, so that the first failing cell falls at varied positions.
    rng = random.Random(3000 + seed)
    labels = random_labels(rng, rng.randint(3, 12))
    complex_ = random_complex(rng, labels, rng.randint(1, 4), rng.randint(2, 12))
    images = {vertex: [vertex] for vertex in labels}
    for _ in range(rng.randint(1, 3)):
        vertex = rng.choice(labels)
        images[vertex] = sorted(set(images[vertex]) | set(random_subset(rng, labels, 4)))
    if seed % 4 == 0:
        images[rng.choice(labels)] = []
    exit_vertices = set(random_subset(rng, labels, 2)) if seed % 3 == 0 else set()
    assert_matches_reference(complex_, images, exit_vertices)


def test_top_dimension_five_is_exercised():
    rng = random.Random(7)
    complex_, apex = cone(rng, top_dimension=5)
    while len(complex_) < 6:
        complex_, apex = cone(rng, top_dimension=5)
    labels = [vertex for (vertex,) in complex_[0]]
    images = {vertex: [apex] + random_subset(rng, labels, 4) for vertex in labels}
    result = assert_matches_reference(complex_, images)
    assert result["status"] == "ok"
    assert len(result["payload"]["cell_counts"]) == 6
    assert shift_class(result)["shift_class"][0] == "x-1"


@pytest.mark.parametrize("seed", range(40))
def test_random_maps_to_another_complex_match_reference(seed):
    rng = random.Random(2000 + seed)
    labels = random_labels(rng, rng.randint(2, 7))
    source = random_complex(rng, labels, rng.randint(1, 4), rng.randint(1, 5))
    target, apex = cone(rng, top_dimension=rng.randint(1, 5))
    target_labels = [vertex for (vertex,) in target[0]]
    target_exit = set(random_subset(rng, target_labels, 3))
    source_exit = set(random_subset(rng, labels, 2))
    images = {}
    for vertex in labels:
        if vertex in source_exit and target_exit:
            images[vertex] = random_subset(rng, sorted(target_exit), 2) or [min(target_exit)]
        elif seed % 4 == 0:
            images[vertex] = random_subset(rng, target_labels, 2) or [target_labels[0]]
        else:
            images[vertex] = [apex] + random_subset(rng, target_labels, 3)
    assert_matches_reference(
        source, images, source_exit, target=target, target_exit=target_exit
    )


# The native name.


def test_native_function_is_not_shadowed():
    assert CMGDB.ComputeCarrierChainMap is CMGDB._cmgdb.ComputeCarrierChainMap
