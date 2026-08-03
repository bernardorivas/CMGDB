"""Graphviz rendering for :class:`CMGDB.morse_lattice.DerivedGraph`.

Renders the attractor/repeller lattices and the nontrivial Conley-Morse graph
produced by :mod:`CMGDB.morse_lattice`. Color-index assignment:

* ``sets`` present (a lattice): the empty-set bottom is light gray, a singleton
  element ``{m}`` gets Morse node ``m``'s color index, and join elements get
  indices beyond the Morse-node range so no join reuses an elementary
  attractor/repeller's index (colors can still repeat once the palette wraps).
* ``sets`` absent (nontrivial Conley-Morse graph): vertices are surviving
  original Morse-node ids and are colored by id.

Under the default palette or a user ``clist`` these indices reproduce
``CMGDB.PlotMorseGraph``'s colors for the same Morse nodes exactly. A user
``cmap`` is normalized over this graph's own index range, which on a pruned
graph need not coincide with ``PlotMorseGraph`` on the full graph.
"""

from __future__ import annotations

__all__ = ["plot_derived_graph"]

import matplotlib
import graphviz

from CMGDB.morse_lattice import DerivedGraph

# Same default palette as CMGDB.PlotMorseGraph so colors agree across plots.
_DEFAULT_CLIST = ['#1f77b4', '#e6550d', '#31a354', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                  '#bcbd22', '#80b1d3', '#ffffb3', '#fccde5', '#b3de69', '#fdae6b', '#6a3d9a', '#c49c94',
                  '#fb8072', '#dbdb8d', '#bc80bd', '#ffed6f', '#637939', '#c5b0d5', '#636363', '#c7c7c7',
                  '#8dd3c7', '#b15928', '#e8cb32', '#9e9ac8', '#74c476', '#ff7f0e', '#9edae5', '#90d743',
                  '#e7969c', '#17becf', '#7b4173', '#8ca252', '#ad494a', '#8c6d31', '#a55194', '#00cc49']

_EMPTY_SET_COLOR = '#cdcdcd'


def _color_indices(graph: DerivedGraph) -> dict[int, int | None]:
    """Map each vertex to a colormap index; ``None`` marks the empty-set bottom."""
    if graph.sets is None:
        return {v: v for v in graph.nodes}
    all_nodes = frozenset().union(*graph.sets.values()) if graph.sets else frozenset()
    join_index = (max(all_nodes) + 1) if all_nodes else 0
    indices: dict[int, int | None] = {}
    for v in graph.nodes:
        element = graph.sets[v]
        if not element:
            indices[v] = None
        elif len(element) == 1:
            indices[v] = next(iter(element))
        else:
            indices[v] = join_index
            join_index += 1
    return indices


def plot_derived_graph(graph: DerivedGraph, cmap=None, clist=None, shape='ellipse',
                       margin='0.11, 0.055'):
    """Render a :class:`DerivedGraph` as a ``graphviz.Source``.

    ``cmap``/``clist`` select the colormap exactly as in ``PlotMorseGraph``;
    ``shape`` and ``margin`` are passed through to graphviz node attributes.
    """
    indices = _color_indices(graph)
    num_indices = max((i for i in indices.values() if i is not None), default=0) + 1
    if cmap is None and clist is None:
        clist = _DEFAULT_CLIST
    if cmap is None:
        cmap = matplotlib.colors.ListedColormap(clist[:num_indices])
    try:
        num_colors = len(cmap.colors)
    except AttributeError:
        num_colors = 0  # Continuous colormap
    if 0 < num_colors < num_indices:
        # Cycle a too-small palette; otherwise spread indices across the colormap.
        cmap_norm = lambda k: k % num_colors
    else:
        cmap_norm = matplotlib.colors.Normalize(vmin=0, vmax=num_indices - 1)

    def vertex_color(v):
        if indices[v] is None:
            return _EMPTY_SET_COLOR
        return str(matplotlib.colors.to_hex(cmap(cmap_norm(indices[v])), keep_alpha=True))

    sinks = [v for v in graph.nodes if not graph.edges.get(v)]
    targets = {v for children in graph.edges.values() for v in children}
    sources = [v for v in graph.nodes if v not in targets]

    gv = 'digraph {\n'
    for v in graph.nodes:
        label = f'{v} : {graph.labels[v]}' if graph.labels.get(v) else str(v)
        gv += (f'{v} [label="{label}", shape={shape}, style=filled, '
               f'fillcolor="{vertex_color(v)}", margin="{margin}"];\n')
    gv += '{rank=same; ' + ' '.join(str(v) for v in sinks) + '};\n'
    gv += '{rank=same; ' + ' '.join(str(v) for v in sources) + '};\n'
    for u in graph.nodes:
        for v in graph.edges.get(u, []):
            gv += f'{u} -> {v};\n'
    gv += '}\n'
    return graphviz.Source(gv)
