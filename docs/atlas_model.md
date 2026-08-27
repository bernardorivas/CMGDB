# Tagged-union maps on an Atlas

`CMGDB.AtlasModel` runs CMGDB's ordinary adaptive Morse-graph algorithm on a
finite disjoint union of rectangular charts.  It changes the geometry accepted
at the box-map boundary; it does not replace the SCC or Morse-graph algorithm.

```python
import CMGDB

model = CMGDB.AtlasModel(
    phase_subdiv_min=8,
    phase_subdiv_max=10,
    phase_subdiv_init=4,
    phase_subdiv_limit=10_000,
)
model.add_chart(0, [0.0, 0.0], [1.0, 1.0])
model.add_chart(1, [0.0, 0.0], [1.0, 1.0])

def box_map(source_chart, source_bounds):
    # Bounds use the usual CMGDB layout [lower..., upper...].
    # Every returned piece is (target_chart, target_bounds).
    return [
        (0, [0.2, 0.3, 0.4, 0.5]),
        (1, [0.6, 0.1, 0.8, 0.2]),
    ]

model.set_map(box_map)
morse_graph, map_graph = CMGDB.ComputeMorseGraph(model)

for node in morse_graph.vertices():
    for chart_id, bounds in morse_graph.morse_set_chart_boxes(node):
        pass
```

The callback may return any finite iterable of `(chart_id, bounds)` pairs,
dictionaries with `chart_id` and `bounds`, or `CMGDB.TaggedRectangle` objects.
Return `[]` for an empty image.  CMGDB covers every piece separately; it never
forms a Euclidean hull across pieces or charts.

## Precomputed tagged-union callbacks

When the active source family is known, its callback values can be computed
once and replayed through the ordinary `AtlasModel.set_map` interface:

```python
sources = [
    (atlas.cell(index).chart_id, atlas.cell(index).bounds)
    for index in range(atlas.size())
]
cached_box_map = CMGDB.precompute_atlas_box_map(
    box_map,
    sources,
    batch_size=4096,
    batch_callback=box_map_batch,          # optional
    provenance_callback=source_record,    # optional
)
model.set_map(cached_box_map)
```

`box_map_batch` receives a bounded ordered sequence of `(chart_id, bounds)`
sources and returns one finite tagged union per source.  The precomputed map
uses the chart id and exact binary64 source bounds as its key.  Duplicate keys
are rejected.  A missing source raises an exception: it is never silently
converted to `[]`, because an explicit empty union has open-exit meaning.
`cached_box_map.provenance(chart_id, bounds)` returns any uninterpreted source
record captured by `source_record`, while `stats()` reports lookup hits and
misses.

This adapter only stores callback outputs; it does not certify that they
enclose the continuous image.  Its Python `batch` lookup is available for
future/native batch consumers, but the current `AtlasModel` callback boundary
is scalar.  A model-specific endpoint precomputer may still be preferable
when neighboring source boxes share costly sample points.

## Selected dyadic initial grids

`set_active_subgrid` replaces the full chart roots by an explicit finite family
of tagged dyadic cells before the map is installed:

```python
model = CMGDB.AtlasModel(0)  # no additional uniform subdivision
model.add_chart(10, [0.0, 0.0], [1.0, 1.0])
model.add_chart(20, [-1.0], [1.0])
model.set_active_subgrid([
    (10, 3, [1, 6]),
    {"chart_id": 10, "axis_depth": 3, "coordinates": [2, 6]},
    CMGDB.TaggedDyadicCell(20, 4, [7]),
])
model.set_map(box_map)
```

The middle integer is the dyadic depth in **each coordinate**. At axis depth
`q`, coordinate `k` denotes the closed interval
`[lower + k*(upper-lower)/2^q, lower + (k+1)*(upper-lower)/2^q]`.
Cells in one chart may use different depths, but the selected family must be an
antichain: no selected cell may contain another. Exact duplicates are removed.
Charts omitted from the family remain registered, with their chart id and
bounds, but have zero active cells.

The native implementation builds the compressed binary prefix tree along the
selected cell paths. It does not first construct the complete rectangle at the
largest depth. Thus a single selected depth-30 cell in a two-dimensional chart
does not allocate the hypothetical `2^60`-cell grid. Existing Atlas operations
(`clone`, `subgrid`, `subdivide`, `join`, geometry, and cover) continue to work
on the selected family and preserve chart ids. `ComputeMorseGraph` uses the
same MapGraph/SCC/Morse implementation as for a full Atlas.

`model.phaseSpace().cover(chart_id, bounds)` exposes the active cover directly.
A nonempty geometric target whose active cover is `[]` exits the chosen active
family and therefore contributes no MapGraph edge. This has the same empty
adjacency as a callback that explicitly returns `[]`; applications that need
to distinguish those cases must retain the callback value and report the
active-boundary exit, as the hybrid adapters do. CMGDB does not add a cemetery
vertex implicitly.

The active family can be replaced transactionally until `set_map` is called.
Chart additions are locked after `set_active_subgrid`, so the declared geometry
cannot silently change underneath the selected cell coordinates.

## Quotient charts are not glued by `Atlas`

`Atlas` stores disjoint tagged charts.  In particular, two faces that represent
the same point of a suspension quotient are not automatically incident.  The
box map must emit every tagged representative required at a glued target face.
For example, an image reaching an identified reset seam must return both the
base-chart face and the handle-chart face when both representatives belong to
the outer image.  Omitting one does not cause `Atlas` to infer it.

This duplicated target representation supplies map-graph edges across the
seam.  It still does not turn the disjoint Atlas into a quotient cell complex.
Any spatial connectedness check, cellular boundary calculation, or carrier
construction must use separately supplied quotient-face incidence data.

## Conley-index boundary

`ComputeMorseGraph` and `ComputeMorseGraphOnly` support `AtlasModel` because
CMGDB's adaptive graph construction is grid-generic.  The legacy CHOMP path is
specific to a cubical `TreeGrid`, so `ComputeConleyMorseGraph` and
`ComputeConleyMorseGraphOnly` deliberately reject an `AtlasModel`.

For a suspension Conley index, first construct and validate the quotient
cellular complex, an index pair, and a compatible carried chain map.  Then pass
the resulting relative chain complex and endomorphism to
`CMGDB.ComputeRelativeHomologyShiftClass`.  That function validates the
algebraic chain data; it does not certify the preceding topological
construction.
