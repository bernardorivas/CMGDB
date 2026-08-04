"""Graphviz rendering for :class:`CMGDB.morse_lattice.DerivedGraph`.

Renders the attractor/repeller lattices and the nontrivial Conley-Morse graph
produced by :mod:`CMGDB.morse_lattice`, and re-renders a parsed
:class:`CMGDB.morse_graph_parser.MorseGraph` (which shares the
``(nodes, edges, labels)`` shape). Color-index assignment:

* ``sets`` present (a lattice): the empty-set bottom is light gray, a singleton
  element ``{m}`` gets Morse node ``m``'s color index, and join elements get
  indices beyond the Morse-node range so no join reuses an elementary
  attractor/repeller's index.
* ``sets`` absent (nontrivial Conley-Morse graph, parsed ``MorseGraph``):
  vertices are original Morse-node ids and are colored by id.

CMGDB emits contiguous node ids ``0..n-1``, so these indices reproduce
``CMGDB.PlotMorseGraph``'s colors for the same Morse nodes under the default
palette or a user ``clist``. A user ``clist`` is indexed by color index (not
vertex position) and cycles, with a warning, when shorter than the index
range; a user ``cmap`` is normalized over this graph's own index range, which
need not coincide with ``PlotMorseGraph``'s normalization over the full graph.
"""

from __future__ import annotations

__all__ = ["plot_derived_graph"]

import re
import warnings

import matplotlib
import graphviz

from CMGDB.morse_graph_parser import MorseGraph
from CMGDB.morse_lattice import DerivedGraph
from CMGDB.PlotMorseGraph import _DEFAULT_CLIST

_EMPTY_SET_COLOR = '#cdcdcd'

_DOT_ESCAPES = str.maketrans({c: '\\' + c for c in '\\"{}|<>'})


def _dot_escape(text: str) -> str:
    """Escape DOT label specials (quotes, backslashes, record delimiters)."""
    return text.translate(_DOT_ESCAPES)


def _vertex_label(v: int, raw: str | None) -> str:
    if not raw:
        return str(v)
    # Parsed MorseGraph labels already carry the "v : ..." prefix.
    if re.match(rf"{v}\s*:", raw):
        return raw
    return f"{v} : {raw}"


def _color_indices(graph) -> dict[int, int | None]:
    """Map each vertex to a colormap index; ``None`` marks the empty-set bottom."""
    sets = getattr(graph, "sets", None)
    if sets is None:
        return {v: v for v in graph.nodes}
    all_nodes = frozenset().union(*sets.values()) if sets else frozenset()
    join_index = (max(all_nodes) + 1) if all_nodes else 0
    indices: dict[int, int | None] = {}
    for v in graph.nodes:
        element = sets[v]
        if not element:
            indices[v] = None
        elif len(element) == 1:
            indices[v] = next(iter(element))
        else:
            indices[v] = join_index
            join_index += 1
    return indices


def plot_derived_graph(graph: DerivedGraph | MorseGraph, cmap=None, clist=None,
                       shape=None, margin=None):
    """Render a :class:`DerivedGraph` or parsed :class:`MorseGraph` as ``graphviz.Source``.

    ``cmap``/``clist`` select the colormap as in ``PlotMorseGraph``; ``shape``
    and ``margin`` pass through to graphviz node attributes and default to
    ``ellipse`` and ``0.11, 0.055`` (``None`` selects the default).
    """
    if shape is None:
        shape = 'ellipse'
    if margin is None:
        margin = '0.11, 0.055'
    indices = _color_indices(graph)
    num_indices = max((i for i in indices.values() if i is not None), default=0) + 1
    user_palette = cmap is not None or bool(clist)
    if cmap is None and not clist:
        clist = _DEFAULT_CLIST
    if cmap is None:
        cmap = matplotlib.colors.ListedColormap(clist[:num_indices])
    try:
        num_colors = len(cmap.colors)
    except (AttributeError, TypeError):
        num_colors = 0  # Continuous colormap
    if 0 < num_colors < num_indices:
        # Cycle a too-small palette; otherwise spread indices across the colormap.
        if user_palette:
            warnings.warn(
                f"palette has {num_colors} colors but color indices reach "
                f"{num_indices - 1}; colors will repeat", stacklevel=2)
        cmap_norm = lambda k: k % num_colors
    else:
        cmap_norm = matplotlib.colors.Normalize(vmin=0, vmax=num_indices - 1)

    def vertex_color(v):
        if indices[v] is None:
            return _EMPTY_SET_COLOR
        return str(matplotlib.colors.to_hex(cmap(cmap_norm(indices[v])), keep_alpha=True))

    sinks = [v for v in graph.nodes if not graph.edges.get(v)]
    targets = {v for children in graph.edges.values() for v in children}
    # Keep isolated vertices (both sink and source) out of the source row: a
    # vertex shared by both rank groups would make dot merge the two ranks.
    sources = [v for v in graph.nodes if v not in targets and graph.edges.get(v)]

    gv = 'digraph {\n'
    for v in graph.nodes:
        label = _dot_escape(_vertex_label(v, graph.labels.get(v)))
        gv += (f'{v} [label="{label}", shape={shape}, style=filled, '
               f'fillcolor="{vertex_color(v)}", margin="{margin}"];\n')
    for group in (sinks, sources):
        if group:
            gv += '{rank=same; ' + ' '.join(str(v) for v in group) + '};\n'
    for u in graph.nodes:
        for v in graph.edges.get(u, []):
            gv += f'{u} -> {v};\n'
    gv += '}\n'
    return graphviz.Source(gv)
