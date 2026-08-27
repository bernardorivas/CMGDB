# Explicit relative chain maps

`CMGDB.ComputeRelativeHomologyShiftClass` is the noncubical entry point for a
finite cellular suspension model. It computes the shift-equivalence class of
an explicitly supplied endomorphism of a relative cellular chain complex.
It does not convert the complex to cubes.

## Input convention

```python
result = CMGDB.ComputeRelativeHomologyShiftClass(
    cell_counts,
    boundary_entries,
    chain_map_entries,
)
```

All coefficients lie in the same field used by CMGDB's existing Conley-index
code, currently `F_5`.

- `cell_counts = [n_0, n_1, ..., n_D]` gives the size of each based chain
  group. Include zero-sized intermediate dimensions.
- `boundary_entries` has exactly `D + 1` lists. An entry `(row, column,
  coefficient)` in `boundary_entries[d]` belongs to the `n_{d-1} x n_d`
  matrix of `boundary_d : C_d -> C_{d-1}`. `boundary_entries[0]` is empty.
- `chain_map_entries` also has exactly `D + 1` lists. Its degree-`d` triples
  describe the square `n_d x n_d` matrix of `F_d : C_d -> C_d`.
- Unlisted matrix coordinates are zero. Duplicate coordinates are rejected.
  Integer coefficients are reduced modulo five.

For example, this is a one-vertex, one-edge circle with a map that fixes the
vertex and reverses the oriented edge:

```python
import CMGDB

result = CMGDB.ComputeRelativeHomologyShiftClass(
    cell_counts=[1, 1],
    boundary_entries=[[], []],
    chain_map_entries=[[(0, 0, 1)], [(0, 0, -1)]],
)

assert result["homology_dimensions"] == [1, 1]
assert result["induced_maps"] == [[[1]], [[-1]]]
assert result["shift_class"] == ["x-1", "x+1"]
```

The result is a dictionary with:

- `coefficient_field`: `5`;
- `cell_counts`: the supplied chain-group sizes;
- `validation`: confirmation that sparse matrix entries, `boundary^2 = 0`, and
  the chain-map equation passed validation (a failure raises instead);
- `homology_dimensions`: the dimension of homology in every degree;
- `induced_maps`: dense matrices for the induced homology endomorphisms, in
  the bases chosen by CHOMP; and
- `shift_class`: the usual CMGDB Frobenius/shift-class string in every degree.

The implementation always checks matrix bounds, duplicate coordinates,
`boundary[d-1] * boundary[d] == 0`, and
`boundary[d] * F[d] == F[d-1] * boundary[d]` before computing homology.

## Topological obligation

This endpoint begins after the topological lifting step. For a Conley-index
calculation, the caller must establish that:

1. the supplied complex is the relative cellular complex `C_*(N, L; F_5)` of
   a valid index pair for the fixed-time map under study;
2. the supplied endomorphism descends to that quotient; and
3. it is a chain selector carried by the acyclic multivalued outer
   approximation (or is otherwise known to induce its homology map).

Those facts depend on the suspension cells, index pair, and carrier. They
cannot be inferred from the three matrices, so this function does not claim to
validate them. The algebraic validation here starts once a suspension adapter
has discharged that obligation.
