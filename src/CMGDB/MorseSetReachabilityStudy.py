"""Multi-subdivision Morse-set reachability study.

Runs the fixed-subdivision reachability verifier independently at several
subdivision depths and classifies every ordered Morse-vertex pair by
agreement across the tested subdivisions. Each subdivision is computed
independently: no closures, checkpoints, or absence conclusions are
propagated between subdivisions, and there is no "finest wins" policy.
"""

from CMGDB._cmgdb import (
    ComputeMorseSetReachability,
    MorseSetReachabilityStatus,
)

__all__ = [
    "MorseSetReachabilityStudy",
    "ComputeMorseSetReachabilityStudy",
    "AGREE_REACHABLE",
    "AGREE_NOT_REACHABLE",
    "UNSTABLE",
    "UNRESOLVED",
]

AGREE_REACHABLE = "AGREE_REACHABLE"
AGREE_NOT_REACHABLE = "AGREE_NOT_REACHABLE"
UNSTABLE = "UNSTABLE"
UNRESOLVED = "UNRESOLVED"


class MorseSetReachabilityStudy:
    """Per-pair agreement classification over independently computed
    fixed-subdivision reachability results."""

    def __init__(self, phase_subdivisions, results):
        if len(phase_subdivisions) != len(results):
            raise ValueError("phase_subdivisions and results length mismatch")
        if not results:
            raise ValueError("at least one phase subdivision is required")
        self.phase_subdivisions = list(phase_subdivisions)
        self._results = dict(zip(self.phase_subdivisions, results))
        counts = {r.num_vertices() for r in results}
        if len(counts) != 1:
            raise ValueError("results disagree on the number of Morse vertices")
        self._num_vertices = counts.pop()

    def num_vertices(self):
        return self._num_vertices

    def result(self, phase_subdiv):
        """The independently computed result at one subdivision."""
        return self._results[phase_subdiv]

    def statuses(self, source, target):
        """Mapping phase_subdiv -> MorseSetReachabilityStatus."""
        return {
            s: self._results[s].status(source, target)
            for s in self.phase_subdivisions
        }

    def classification(self, source, target):
        statuses = set(self.statuses(source, target).values())
        reachable = MorseSetReachabilityStatus.REACHABLE in statuses
        not_reachable = MorseSetReachabilityStatus.NOT_REACHABLE in statuses
        incomplete = MorseSetReachabilityStatus.INCOMPLETE in statuses
        if reachable and not_reachable:
            return UNSTABLE
        if incomplete:
            return UNRESOLVED
        return AGREE_REACHABLE if reachable else AGREE_NOT_REACHABLE

    def unstable_pairs(self):
        return [
            (v, w)
            for v in range(self._num_vertices)
            for w in range(self._num_vertices)
            if v != w and self.classification(v, w) == UNSTABLE
        ]

    def unresolved_pairs(self):
        return [
            (v, w)
            for v in range(self._num_vertices)
            for w in range(self._num_vertices)
            if v != w and self.classification(v, w) == UNRESOLVED
        ]

    def prunable_adaptive_edges(self, policy="ALL_TESTED_NOT_REACHABLE"):
        """Adaptive edges removable under the requested pruning policy.

        ALL_TESTED_NOT_REACHABLE removes an adaptive edge only when every
        tested subdivision reports NOT_REACHABLE. This is agreement over
        the tested subdivisions, not a convergence theorem.
        """
        if policy != "ALL_TESTED_NOT_REACHABLE":
            raise ValueError("unknown pruning policy: %r" % (policy,))
        absent = None
        for s in self.phase_subdivisions:
            edges = set(map(tuple, self._results[s].absent_adaptive_edges()))
            absent = edges if absent is None else absent & edges
        return sorted(absent) if absent else []


def ComputeMorseSetReachabilityStudy(
    model, morse_graph, phase_subdivisions, **kwargs
):
    """Run ComputeMorseSetReachability independently at each subdivision.

    Keyword arguments (limits, batch_size, map_fingerprint) are passed
    through to every run. Checkpoints are not shared between runs.
    """
    if "resume_from" in kwargs:
        raise ValueError(
            "resume_from cannot be shared across subdivisions; resume "
            "individual runs with ComputeMorseSetReachability instead"
        )
    results = [
        ComputeMorseSetReachability(
            model, morse_graph, phase_subdiv=int(s), **kwargs
        )
        for s in phase_subdivisions
    ]
    return MorseSetReachabilityStudy(list(phase_subdivisions), results)
