# MapGraph CSR checkpoints

`MapGraph` stores cached adjacency natively as compressed sparse row (CSR)
arrays. Calling `adjacencies(source)` is convenient for small graphs, but every
call creates a Python list. Retaining those lists, or converting their entries
to Python sets and JSON, can dominate memory on a dense Atlas relation.

The compact interface exposes the existing storage without changing any edge:

```python
offsets, targets = map_graph.csr_view()
```

Both arrays are read-only `int64` NumPy views. They do not own a second edge
buffer: their NumPy base retains the immutable native `MapGraph`, so the views
stay valid even if the caller drops its original graph variable. Export
requires the eager cache (`CMGDB_MAPGRAPH_CACHE=1`, the default). Before the
view is returned, CMGDB checks that offsets are consistent and every row is
in-range, strictly increasing, and duplicate-free.

The scalar MapGraph construction path now also appends rows directly to CSR.
This matters for `AtlasModel`, whose Python-backed tagged-union map does not
currently advertise an optimized native batch callback. The old scalar path
kept a `vector` for every source and then copied all edges into CSR; the direct
path has identical adjacency and avoids that temporary second edge store.

## Atomic, strict checkpoint

Use an explicit configuration record and explicit resource ceilings:

```python
import CMGDB

configuration = {
    "model": "garcia-passive-walker",
    "tau": 0.5,
    "active_family_sha256": "...",
    "box_map_revision": "...",
}
caps = CMGDB.MapGraphCSRCheckpointCaps(
    max_vertices=500_000,
    max_edges=100_000_000,
    max_payload_bytes=1_000_000_000,
)

CMGDB.write_map_graph_csr_checkpoint(
    map_graph,
    "relation.csr",
    configuration=configuration,
    caps=caps,
)
relation = CMGDB.load_map_graph_csr_checkpoint(
    "relation.csr",
    expected_configuration=configuration,
    caps=caps,
)
```

The writer checks vertex, edge, and payload-byte caps before requesting the
native view or allocating an output array. It refuses to overwrite an existing
path and atomically renames a completed sibling temporary directory. The
directory contains:

- `offsets.npy`: little-endian `int64`, length `V + 1`;
- `targets.npy`: little-endian `int32` when all vertex ids fit, otherwise
  `int64`, length `E`;
- `metadata.json`: the canonical configuration, write caps, exact dimensions,
  per-array SHA-256 hashes, and a configuration/content fingerprint.

Loading is deliberately strict. The caller must supply the expected
configuration and fresh read caps. The loader rejects unknown files, schema or
configuration drift, incomplete metadata, size/dtype/shape disagreement,
invalid CSR rows, hash mismatch, and an optional expected-fingerprint mismatch.
It memory-maps both arrays read-only rather than expanding edges.

`MapGraphCSR` implements a read-only mapping from every source id, including
empty rows, to a lazy `Collection[int]`. Iterative SCC code can traverse those
rows without retaining Python edge objects. `row.as_array()` provides the
underlying read-only mmap slice for vectorized consumers.

Checkpoint caps apply after the native graph exists. For a large computation,
also set CMGDB's opt-in native limits before constructing the `MapGraph`:

```bash
CMGDB_MAPGRAPH_HARD_MAX_VERTICES=1000000 \
CMGDB_MAPGRAPH_HARD_MAX_EDGES=300000000 \
CMGDB_MAPGRAPH_HARD_MAX_CACHE_BYTES=3000000000 \
python run.py
```

The native byte limit counts `int64` offsets and the capacity of the native
`uint64` target vector. It is checked before offset reserve and before every
edge-buffer reserve/append. It cannot include an individual callback row or
optimized batch until that callback has returned, and it does not cover the
grid, Morse graph, or application provenance. The three variables are unset by
default, accept nonnegative integers, and fail rather than silently switching
to the lazy graph. On-disk checkpoint limits are separate because automatic
`int32` targets can halve persisted edge storage.

## Persisted Garcia parity audit

The implementation was checked against the already-persisted passive-walker
relations; no dynamics or ODE endpoints were reevaluated. Every CSR row,
including every empty row, agreed exactly with the stored depth-8, depth-12,
and depth-16 image lists.

| total depth | vertices | edges | precomputed native replay | CSR payload | compact write | strict mmap load, validation, hashes |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 512 | 69,605 | 0.125 s | 282,524 B | 0.022 s | 0.011 s |
| 12 | 8,192 | 1,992,616 | 0.798 s | 8,036,008 B | 0.060 s | 0.019 s |
| 16 | 82,944 | 25,928,612 | 16.079 s | 104,378,008 B | 0.131 s | 0.073 s |

The native replays used the persisted tagged callback values and checked the
new native CSR row-for-row; they did not integrate the walker. These are local
warm-filesystem verification timings, not performance guarantees. All three
checkpoints selected `int32` targets. At depth 16, the
Garcia adapter's conservative estimate for its retained Python
dictionary/frozenset relation was 5,318,032,128 bytes; the mmap CSR payload is
about 50.95 times smaller. The existing gzip JSON-lines file is smaller on
disk because it is compressed, but resuming it recreates Python edge objects;
the CSR checkpoint is designed for compact live access.

## Scope

This checkpoint contains adjacency only. In particular, it does not contain
Atlas cells, chart geometry, sampled image pieces, open-exit witnesses, or
box-map diagnostics, and it adds no enclosure or Conley-index claim. A hybrid
application should persist that source-level provenance separately and bind it
to the same configuration/fingerprint. Equal adjacent offsets preserve an
empty image exactly; no cemetery state is introduced.
