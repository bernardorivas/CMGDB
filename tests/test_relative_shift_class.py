"""Tests of ComputeRelativeShiftClass, the linear-algebra shift class.

The function takes the arguments of ComputeRelativeHomologyShiftClass and
returns a dictionary with the same keys and formats.  The tests check:

- the same result as the existing function on complexes with zero boundary,
  where that function is exact (its homology basis is the standard one), and
  the same exceptions and messages on invalid input;
- the homotopy oracle: F = s I + dH + Hd is chain homotopic to s I, so its
  induced map is exactly s I in every basis;
- a split oracle: a complex H + K with K acyclic, in a random basis, and the
  chain map A + 0 plus a null-homotopic map, whose induced map is similar to A;
- a random simplicial complex and a dense complex on which the Morse complex
  of the existing function need not be minimal.  There the result of the
  existing function depends on the iteration order of Boost's unordered
  containers, so it is only run, and only the new function is checked.
"""

import itertools
import random

import pytest

import CMGDB

P = 5


def compute(cell_counts, boundaries, chain_map):
    return CMGDB.ComputeRelativeShiftClass(cell_counts, boundaries, chain_map)


def existing(cell_counts, boundaries, chain_map):
    return CMGDB.ComputeRelativeHomologyShiftClass(cell_counts, boundaries, chain_map)


# Dense matrices over F_5.


def zeros(rows, columns):
    return [[0] * columns for _ in range(rows)]


def identity(size, scalar=1):
    return [[scalar % P if i == j else 0 for j in range(size)] for i in range(size)]


def product(a, b, rows, inner, columns):
    result = zeros(rows, columns)
    for i in range(rows):
        for k in range(inner):
            if a[i][k]:
                for j in range(columns):
                    result[i][j] = (result[i][j] + a[i][k] * b[k][j]) % P
    return result


def add(a, b):
    return [[(x + y) % P for x, y in zip(row_a, row_b)] for row_a, row_b in zip(a, b)]


def rank(matrix):
    rows = [row[:] for row in matrix]
    result = 0
    columns = len(rows[0]) if rows else 0
    for column in range(columns):
        pivot = next((i for i in range(result, len(rows)) if rows[i][column] % P), None)
        if pivot is None:
            continue
        rows[result], rows[pivot] = rows[pivot], rows[result]
        inverse = pow(rows[result][column], P - 2, P)
        rows[result] = [value * inverse % P for value in rows[result]]
        for i in range(len(rows)):
            if i != result and rows[i][column] % P:
                factor = rows[i][column]
                rows[i] = [(x - factor * y) % P for x, y in zip(rows[i], rows[result])]
        result += 1
    return result


def kernel_basis(matrix, rows, columns):
    """A basis of the solutions of matrix x = 0."""
    reduced = [row[:] for row in matrix]
    pivots = []
    row = 0
    for column in range(columns):
        pivot = next((i for i in range(row, rows) if reduced[i][column] % P), None)
        if pivot is None:
            continue
        reduced[row], reduced[pivot] = reduced[pivot], reduced[row]
        inverse = pow(reduced[row][column], P - 2, P)
        reduced[row] = [value * inverse % P for value in reduced[row]]
        for i in range(rows):
            if i != row and reduced[i][column]:
                factor = reduced[i][column]
                reduced[i] = [(x - factor * y) % P for x, y in zip(reduced[i], reduced[row])]
        pivots.append(column)
        row += 1
    basis = []
    for free in (j for j in range(columns) if j not in pivots):
        vector = [0] * columns
        vector[free] = 1
        for i, column in enumerate(pivots):
            vector[column] = (-reduced[i][free]) % P
        basis.append(vector)
    return basis


def invertible_pair(rng, size):
    """A random invertible matrix and its inverse."""
    while True:
        matrix = [[rng.randrange(P) for _ in range(size)] for _ in range(size)]
        augmented = [row[:] + identity(size)[i] for i, row in enumerate(matrix)]
        for column in range(size):
            pivot = next((i for i in range(column, size) if augmented[i][column]), None)
            if pivot is None:
                break
            augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
            inverse = pow(augmented[column][column], P - 2, P)
            augmented[column] = [value * inverse % P for value in augmented[column]]
            for i in range(size):
                if i != column and augmented[i][column]:
                    factor = augmented[i][column]
                    augmented[i] = [
                        (x - factor * y) % P for x, y in zip(augmented[i], augmented[column])
                    ]
        else:
            return matrix, [row[size:] for row in augmented]


def sparse(matrix):
    return [
        (i, j, value)
        for i, row in enumerate(matrix)
        for j, value in enumerate(row)
        if value % P
    ]


def random_square(rng, size):
    """A random matrix of one of several kinds, to vary the invariant factors."""
    kind = rng.choice(["dense", "sparse", "nilpotent", "scalar", "jordan", "companion", "permutation"])
    if kind == "dense":
        return [[rng.randrange(P) for _ in range(size)] for _ in range(size)]
    if kind == "sparse":
        return [[rng.randrange(1, P) if rng.random() < 0.2 else 0 for _ in range(size)] for _ in range(size)]
    if kind == "nilpotent":
        return [[rng.randrange(P) if j > i else 0 for j in range(size)] for i in range(size)]
    if kind == "scalar":
        return identity(size, rng.randrange(P))
    if kind == "permutation":
        order = list(range(size))
        rng.shuffle(order)
        return [[1 if order[j] == i else 0 for j in range(size)] for i in range(size)]
    matrix = zeros(size, size)
    start = 0
    while start < size:
        block = rng.randint(1, size - start)
        if kind == "jordan":
            eigenvalue = rng.randrange(P)
            for k in range(block):
                matrix[start + k][start + k] = eigenvalue
                if k + 1 < block:
                    matrix[start + k][start + k + 1] = 1
        else:
            for k in range(1, block):
                matrix[start + k][start + k - 1] = 1
            for k in range(block):
                matrix[start + k][start + block - 1] = rng.randrange(P)
        start += block
    if size and rng.random() < 0.5:
        change, inverse = invertible_pair(rng, size)
        matrix = product(product(change, matrix, size, size, size), inverse, size, size, size)
    return matrix


def balanced(matrix):
    return [[value - P if value > P // 2 else value for value in row] for row in matrix]


def scalar_class(scalar, size):
    """The shift-class string of scalar * identity of the given size."""
    if size == 0 or scalar % P == 0:
        return "0"
    value = balanced([[scalar % P]])[0][0]
    return ("x-1", "x-2", "x+2", "x+1")[[1, 2, -2, -1].index(value)] * size


def homotopy_oracle(boundaries, counts, scalar, homotopy):
    """F = scalar I + d H + H d for H_d : C_d -> C_{d+1}."""
    top = len(counts) - 1
    maps = []
    for d, size in enumerate(counts):
        matrix = identity(size, scalar)
        if d < top:
            matrix = add(matrix, product(boundaries[d + 1], homotopy[d], size, counts[d + 1], size))
        if d > 0:
            matrix = add(matrix, product(homotopy[d - 1], boundaries[d], size, counts[d - 1], size))
        maps.append(matrix)
    return maps


def assert_scalar_induced_maps(result, counts, scalar, dimensions=None):
    if dimensions is not None:
        assert result["homology_dimensions"] == dimensions
    for d, matrix in enumerate(result["induced_maps"]):
        size = result["homology_dimensions"][d]
        assert matrix == balanced(identity(size, scalar)), d
        assert result["shift_class"][d] == scalar_class(scalar, size), d


# The structured result.


def test_point_identity_has_the_structured_result_of_the_existing_function():
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
    assert list(result) == list(existing([1], [[]], [[(0, 0, 1)]]))


HAND_BUILT = {
    "circle degree -1": ([1, 1], [[], []], [[(0, 0, 1)], [(0, 0, -1)]]),
    "interval": (
        [2, 1],
        [[], [(0, 0, -1), (1, 0, 1)]],
        [[(0, 0, 1), (1, 1, 1)], [(0, 0, 1)]],
    ),
    "zero-sized intermediate group": ([1, 0, 1], [[], [], []], [[(0, 0, 1)], [], [(0, 0, 1)]]),
    "zero-sized top groups": ([1, 0, 0], [[], [], []], [[(0, 0, 2)], [], []]),
    "coefficients mod 5": ([1], [[]], [[(0, 0, 6)]]),
    "negative coefficients": ([2], [[]], [[(0, 0, -7), (1, 1, -1), (0, 1, -3)]]),
    "zero coefficients are dropped": ([2], [[]], [[(0, 0, 5), (1, 1, 1)]]),
    "nilpotent block": ([2], [[]], [[(0, 1, 1)]]),
    "empty complex": ([0], [[]], [[]]),
}


@pytest.mark.parametrize("name", sorted(HAND_BUILT))
def test_hand_built_complexes_agree_with_the_existing_function(name):
    arguments = HAND_BUILT[name]
    assert compute(*arguments) == existing(*arguments)


def test_hand_built_results():
    assert compute(*HAND_BUILT["circle degree -1"])["shift_class"] == ["x-1", "x+1"]
    interval = compute(*HAND_BUILT["interval"])
    assert interval["homology_dimensions"] == [1, 0]
    assert interval["induced_maps"] == [[[1]], []]
    assert interval["shift_class"] == ["x-1", "0"]
    assert compute(*HAND_BUILT["zero-sized top groups"])["shift_class"] == ["x-2", "0", "0"]
    assert compute(*HAND_BUILT["nilpotent block"])["induced_maps"] == [[[0, 1], [0, 0]]]
    assert compute(*HAND_BUILT["nilpotent block"])["shift_class"] == ["0"]
    assert compute(*HAND_BUILT["empty complex"])["shift_class"] == ["0"]


INVALID = {
    "boundary squared": (
        ValueError,
        "boundary squared is nonzero from dimension 2 to dimension 0",
        ([1, 1, 1], [[], [(0, 0, 1)], [(0, 0, 1)]], [[(0, 0, 1)], [(0, 0, 1)], [(0, 0, 1)]]),
    ),
    "chain-map equation": (
        ValueError,
        "chain-map equation fails in dimension 1",
        ([2, 1], [[], [(0, 0, -1), (1, 0, 1)]], [[(0, 0, 1)], [(0, 0, 1)]]),
    ),
    "duplicate": (
        ValueError,
        r"duplicate chain map entry \(0, 0\) in dimension 0",
        ([1], [[]], [[(0, 0, 1), (0, 0, 2)]]),
    ),
    "first failure in input order is a duplicate": (
        ValueError,
        r"duplicate boundary entry \(0, 1\) in dimension 1",
        ([2, 2], [[], [(0, 1, 1), (1, 0, 1), (0, 1, 1), (0, 0, 1), (1, 0, 1), (5, 0, 1)]], [[], []]),
    ),
    "first failure in input order is out of bounds": (
        IndexError,
        r"boundary entry \(5, 0\) in dimension 1 is outside its 2 x 2 matrix",
        ([2, 2], [[], [(0, 1, 1), (5, 0, 1), (0, 1, 1)]], [[], []]),
    ),
    "out of bounds": (
        IndexError,
        r"chain map entry \(1, 0\) in dimension 0 is outside its 1 x 1 matrix",
        ([1], [[]], [[(1, 0, 1)]]),
    ),
    "degree-zero boundary": (
        IndexError,
        r"boundary entry \(0, 0\) in dimension 0 is outside its 0 x 2 matrix",
        ([2], [[(0, 0, 1)]], [[]]),
    ),
    "missing boundary degree": (
        ValueError,
        "boundary_entries must have one list for every chain dimension",
        ([1, 1], [[]], [[(0, 0, 1)], [(0, 0, 1)]]),
    ),
    "missing chain-map degree": (
        ValueError,
        "chain_map_entries must have one list for every chain dimension",
        ([1, 1], [[], []], [[(0, 0, 1)]]),
    ),
    "no degree": (ValueError, "cell_counts must contain at least dimension zero", ([], [], [])),
    "chain group above the int64 range": (
        OverflowError,
        "chain group is too large for CHOMP matrices",
        ([2**63], [[]], [[]]),
    ),
}


@pytest.mark.parametrize("name", sorted(INVALID))
def test_invalid_input_raises_as_the_existing_function(name):
    error, match, arguments = INVALID[name]
    with pytest.raises(error, match=match) as new:
        compute(*arguments)
    with pytest.raises(error) as old:
        existing(*arguments)
    assert str(new.value) == str(old.value)


def test_chain_groups_too_large_to_pack_are_refused():
    # Below the int64 range; the existing function cannot allocate its
    # matrices at this size, so it is not run.
    with pytest.raises(OverflowError, match="too large for ComputeRelativeShiftClass"):
        compute([2**60], [[]], [[]])


# Zero boundary: the existing function is exact there.


@pytest.mark.parametrize("seed", range(40))
def test_zero_boundary_agrees_with_the_existing_function(seed):
    rng = random.Random(seed)
    for _ in range(25):
        matrices = [random_square(rng, rng.randint(0, 8)) for _ in range(rng.randint(1, 4))]
        counts = [len(matrix) for matrix in matrices]
        arguments = (counts, [[] for _ in counts], [sparse(matrix) for matrix in matrices])
        result = compute(*arguments)
        assert result == existing(*arguments)
        assert result["induced_maps"] == [balanced(matrix) for matrix in matrices]


@pytest.mark.parametrize("size", [12, 20])
def test_zero_boundary_agrees_on_larger_matrices(size):
    rng = random.Random(size)
    for _ in range(10):
        matrix = random_square(rng, size)
        arguments = ([size], [[]], [sparse(matrix)])
        assert compute(*arguments) == existing(*arguments)


# The homotopy oracle.


def dense_oracle_case(rng):
    """A random complex with dense boundaries over F_5, and s I + dH + Hd."""
    top = rng.randint(1, 3)
    counts = [rng.randint(0, 6) for _ in range(top + 1)]
    boundaries = [zeros(0, counts[0])]
    for d in range(1, top + 1):
        if d == 1:
            boundary = [
                [rng.randrange(P) if rng.random() < 0.5 else 0 for _ in range(counts[1])]
                for _ in range(counts[0])
            ]
        else:
            cycles = kernel_basis(boundaries[d - 1], counts[d - 2], counts[d - 1])
            boundary = zeros(counts[d - 1], counts[d])
            for j in range(counts[d]):
                if cycles and rng.random() < 0.8:
                    for cycle in cycles:
                        coefficient = rng.randrange(P)
                        for i in range(counts[d - 1]):
                            boundary[i][j] = (boundary[i][j] + coefficient * cycle[i]) % P
        boundaries.append(boundary)
    scalar = rng.randrange(1, P)
    homotopy = [
        [[rng.randrange(P) if rng.random() < 0.4 else 0 for _ in range(counts[d])] for _ in range(counts[d + 1])]
        for d in range(top)
    ]
    maps = homotopy_oracle(boundaries, counts, scalar, homotopy)
    return counts, [sparse(boundary) for boundary in boundaries], [sparse(m) for m in maps], scalar


def simplicial_oracle_case(rng):
    """A random 2-dimensional simplicial complex with +-1 boundaries, and s I + dH + Hd."""
    vertices = rng.randint(4, 8)
    triangles = [t for t in itertools.combinations(range(vertices), 3) if rng.random() < 0.35]
    extra_edges = [e for e in itertools.combinations(range(vertices), 2) if rng.random() < 0.3]
    edges = sorted(set([e for t in triangles for e in itertools.combinations(t, 2)] + extra_edges))
    counts = [vertices, len(edges), len(triangles)]
    edge_index = {edge: i for i, edge in enumerate(edges)}
    first = zeros(counts[0], counts[1])
    second = zeros(counts[1], counts[2])
    for j, (a, b) in enumerate(edges):
        first[a][j] = P - 1
        first[b][j] = 1
    for j, (a, b, c) in enumerate(triangles):
        second[edge_index[(b, c)]][j] = 1
        second[edge_index[(a, c)]][j] = P - 1
        second[edge_index[(a, b)]][j] = 1
    boundaries = [zeros(0, counts[0]), first, second]
    scalar = rng.randrange(1, P)
    homotopy = [
        [[rng.choice([0, 0, 0, 1, 4]) for _ in range(counts[d])] for _ in range(counts[d + 1])]
        for d in range(2)
    ]
    maps = homotopy_oracle(boundaries, counts, scalar, homotopy)
    return counts, [sparse(boundary) for boundary in boundaries], [sparse(m) for m in maps], scalar


@pytest.mark.parametrize("seed", range(40))
def test_homotopy_oracle_on_dense_complexes(seed):
    rng = random.Random(seed)
    for _ in range(10):
        counts, boundaries, maps, scalar = dense_oracle_case(rng)
        assert_scalar_induced_maps(compute(counts, boundaries, maps), counts, scalar)


@pytest.mark.parametrize("seed", range(30))
def test_homotopy_oracle_on_simplicial_complexes(seed):
    rng = random.Random(100 + seed)
    for _ in range(10):
        counts, boundaries, maps, scalar = simplicial_oracle_case(rng)
        assert_scalar_induced_maps(compute(counts, boundaries, maps), counts, scalar)


def test_simplicial_case_where_the_morse_complex_is_not_minimal():
    # H_1 has dimension 4 and the map is 2 I on it.  The coreduction Morse
    # complex of the existing function can have more critical 1-cells than
    # b_1 here, and its result then depends on the iteration order of Boost's
    # unordered containers, which differs between Boost versions.  Only the
    # new function is checked; the existing one is only run.
    counts, boundaries, maps, scalar = simplicial_oracle_case(random.Random(10273))
    assert counts == [8, 28, 19]
    assert scalar == 2
    result = compute(counts, boundaries, maps)
    assert_scalar_induced_maps(result, counts, scalar, dimensions=[1, 4, 2])
    assert result["shift_class"] == ["x-2", "x-2x-2x-2x-2", "x-2x-2"]
    existing(counts, boundaries, maps)


def test_dense_case_with_a_scalar_map():
    # A dense complex with H = (0, 2, 1) and the map 3 I = -2 I on it.  The
    # result of the existing function on it depends on the iteration order
    # of Boost's unordered containers, as above, so that function is only
    # run.
    counts = [2, 6, 3]
    boundaries = [
        [],
        [(0, 0, 3), (0, 2, 3), (0, 3, 1), (0, 4, 1), (1, 0, 1), (1, 2, 1), (1, 4, 4)],
        [(0, 0, 4), (0, 1, 4), (1, 0, 1), (2, 0, 4), (2, 1, 4), (3, 0, 3), (3, 1, 3),
         (4, 0, 3), (4, 1, 3), (5, 0, 2)],
    ]
    maps = [
        [(0, 0, 2), (0, 1, 4), (1, 0, 3), (1, 1, 4)],
        [(0, 0, 1), (0, 4, 2), (0, 5, 3), (1, 1, 3), (1, 2, 3), (2, 0, 1), (2, 2, 1),
         (2, 3, 3), (2, 4, 1), (2, 5, 3), (3, 2, 4), (3, 3, 3), (3, 5, 1), (4, 0, 4),
         (4, 2, 3), (4, 4, 4), (4, 5, 1), (5, 0, 4), (5, 4, 1), (5, 5, 3)],
        [(0, 1, 2), (1, 0, 4), (1, 1, 3), (2, 1, 1), (2, 2, 3)],
    ]
    result = compute(counts, boundaries, maps)
    assert_scalar_induced_maps(result, counts, 3, dimensions=[0, 2, 1])
    assert result["shift_class"] == ["0", "x+2x+2", "x+2"]
    existing(counts, boundaries, maps)


# The split oracle: induced maps that are not scalar.


def split_oracle_case(rng, top):
    """C = H + K in a random basis, with zero boundary on H and K a sum of
    acyclic pairs, and F = (A + 0) + dG + Gd; F induces A on H(C) = H."""
    homology = [rng.randint(0, 3) for _ in range(top + 1)]
    pairs = [0] + [rng.randint(0, 3) for _ in range(top)]   # pairs[d]: C_d -> C_{d-1}
    counts = [homology[d] + pairs[d] + (pairs[d + 1] if d < top else 0) for d in range(top + 1)]
    boundaries = [zeros(0, counts[0])]
    for d in range(1, top + 1):
        boundary = zeros(counts[d - 1], counts[d])
        for k in range(pairs[d]):
            boundary[homology[d - 1] + k][homology[d] + (pairs[d + 1] if d < top else 0) + k] = 1
        boundaries.append(boundary)
    induced = [random_square(rng, size) for size in homology]
    maps = []
    for d in range(top + 1):
        matrix = zeros(counts[d], counts[d])
        for i in range(homology[d]):
            for j in range(homology[d]):
                matrix[i][j] = induced[d][i][j] % P
        maps.append(matrix)
    # A change of basis in every degree.
    changes = [invertible_pair(rng, size) if size else ([], []) for size in counts]
    for d in range(top + 1):
        change, inverse = changes[d]
        n = counts[d]
        maps[d] = product(product(change, maps[d], n, n, n), inverse, n, n, n) if n else []
        if d > 0:
            lower = changes[d - 1][0]
            boundaries[d] = product(
                product(lower, boundaries[d], counts[d - 1], counts[d - 1], n), inverse,
                counts[d - 1], n, n,
            ) if n and counts[d - 1] else zeros(counts[d - 1], n)
    homotopy = [
        [[rng.randrange(P) if rng.random() < 0.4 else 0 for _ in range(counts[d])] for _ in range(counts[d + 1])]
        for d in range(top)
    ]
    null = homotopy_oracle(boundaries, counts, 0, homotopy)
    maps = [add(m, n) for m, n in zip(maps, null)]
    return (
        counts,
        [sparse(boundary) for boundary in boundaries],
        [sparse(matrix) for matrix in maps],
        homology,
        induced,
    )


def power_ranks(matrix):
    size = len(matrix)
    power = identity(size)
    ranks = []
    for _ in range(size):
        power = product(power, [[v % P for v in row] for row in matrix], size, size, size)
        ranks.append(rank(power))
    return ranks


@pytest.mark.parametrize("seed", range(40))
def test_split_oracle(seed):
    # The shift class, with the ranks of the powers, determines the similarity
    # class; the reference shift class is that of the existing function on the
    # zero-boundary complex with the map A.
    rng = random.Random(3000 + seed)
    for _ in range(5):
        counts, boundaries, maps, homology, induced = split_oracle_case(rng, rng.randint(1, 5))
        result = compute(counts, boundaries, maps)
        assert result["homology_dimensions"] == homology
        reference = existing(homology, [[] for _ in homology], [sparse(matrix) for matrix in induced])
        assert result["shift_class"] == reference["shift_class"]
        for d, matrix in enumerate(result["induced_maps"]):
            assert power_ranks(matrix) == power_ranks(induced[d]), d


# The native name.


def test_native_function_is_not_shadowed():
    assert CMGDB.ComputeRelativeShiftClass is CMGDB._cmgdb.ComputeRelativeShiftClass
