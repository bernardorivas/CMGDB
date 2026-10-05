"""Shared test setup.

The MapGraph cache variables (CMGDB_MAPGRAPH_CACHE, CMGDB_MAPGRAPH_RESERVE_*
and CMGDB_MAPGRAPH_HARD_MAX_*) change what every computation caches, reserves
and refuses, so they are removed from the environment for the whole session:
the suite then runs on the defaults whatever the caller's shell exports. A
test that needs one sets it with monkeypatch.
"""

import os

import pytest


def _is_map_graph_variable(name):
    return (name == "CMGDB_MAPGRAPH_CACHE"
            or name.startswith(("CMGDB_MAPGRAPH_RESERVE_",
                                "CMGDB_MAPGRAPH_HARD_MAX_")))


@pytest.fixture(autouse=True, scope="session")
def _map_graph_environment():
    # Session scope, so that module-scoped fixtures see the defaults too.
    with pytest.MonkeyPatch.context() as patch:
        for name in list(os.environ):
            if _is_map_graph_variable(name):
                patch.delenv(name)
        yield
