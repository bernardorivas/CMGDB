"""End-to-end benchmark for CMGDB.

The default suite is intentionally quick and doubles as a correctness
check. Each scenario validates its Morse graph output before reporting
timings. Use ``--heavy`` for longer scenarios once a change looks
promising in the quick suite.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import gc
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple

import CMGDB


_libc = ctypes.CDLL(None)


@contextlib.contextmanager
def silenced_stdout():
    """Redirect C and Python stdout to /dev/null during a benchmark run."""
    sys.stdout.flush()
    _libc.fflush(None)
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(1)
    os.dup2(devnull, 1)
    try:
        yield
    finally:
        sys.stdout.flush()
        _libc.fflush(None)
        os.dup2(saved, 1)
        os.close(saved)
        os.close(devnull)


def leslie_like_2d(x):
    return [x[0] / (2.0 - x[0]), x[1] / (2.0 - x[1])]


def leslie_like_3d(x):
    return [x[i] / (2.0 - x[i]) for i in range(3)]


def leslie_like_4d(x):
    return [x[i] / (2.0 - x[i]) for i in range(4)]


def henon_3d(x):
    return [1.0 - 1.4 * x[0] * x[0] + 0.3 * x[1], x[0], x[1]]


def make_box_map(f):
    def F(rect):
        return CMGDB.BoxMap(f, rect, padding=False)

    return F


def make_batch_map(f):
    def F_batch(rects):
        out = []
        for rect in rects:
            dim = len(rect) // 2
            lower = rect[:dim]
            upper = rect[dim:]
            images = [f(lower), f(upper)]
            out_lower = [min(image[i] for image in images) for i in range(dim)]
            out_upper = [max(image[i] for image in images) for i in range(dim)]
            out.append(out_lower + out_upper)
        return out

    return F_batch


def compute_morse(model):
    return CMGDB.ComputeMorseGraph(model)


def compute_conley(model):
    return CMGDB.ComputeConleyMorseGraph(model)


def validate_vertices(expected: int):
    def check(morse_graph) -> Optional[str]:
        got = morse_graph.num_vertices()
        if got != expected:
            return f"expected {expected} Morse vertices, got {got}"
        return None

    return check


def validate_conley(expected_vertices: int, expected_annotations: List[List[str]]):
    def check(morse_graph) -> Optional[str]:
        got = morse_graph.num_vertices()
        if got != expected_vertices:
            return (
                f"expected {expected_vertices} Morse vertices, got {got} "
                "(annotation comparison skipped)"
            )
        actual = [morse_graph.annotations(i) for i in range(got)]
        if actual != expected_annotations:
            return (
                "Conley index annotations differ:\n"
                f"  expected: {expected_annotations}\n"
                f"  actual:   {actual}"
            )
        return None

    return check


@dataclass
class Scenario:
    name: str
    description: str
    build: Callable[[float, int], Any]
    compute: Callable[[Any], Tuple[Any, Any]] = compute_morse
    validate: Optional[Callable[[Any], Optional[str]]] = None
    tags: List[str] = field(default_factory=list)
    default: bool = True


@dataclass
class Result:
    name: str
    build_times: List[float]
    compute_times: List[float]
    num_morse_vertices: int


class ValidationFailed(Exception):
    pass


def _shift(n: int, scale: float, offset: int) -> int:
    return max(1, int(round(n * scale)) + offset)


def _model_supports_batch_map() -> bool:
    model = CMGDB.Model(
        2,
        2,
        2,
        100,
        [0.0, 0.0],
        [1.0, 1.0],
        make_box_map(leslie_like_2d),
    )
    return hasattr(model, "set_batch_map")


_CONLEY_2D_ANNOTATIONS = [
    ["x-1", "0", "0"],
    ["0", "x-1", "0"],
    ["0", "x-1", "0"],
    ["0", "0", "x-1"],
]

_CONLEY_3D_ANNOTATIONS = [
    ["x-1", "0", "0", "0"],
    ["0", "x-1", "0", "0"],
]


def scenarios(scale: float, include_heavy: bool, include_batch: bool) -> List[Scenario]:
    s: List[Scenario] = []

    def _py_small(scale, offset):
        return CMGDB.Model(
            _shift(6, scale, offset),
            _shift(10, scale, offset),
            _shift(4, scale, offset),
            10000,
            [0.0, 0.0],
            [1.2, 1.2],
            make_box_map(leslie_like_2d),
        )

    s.append(
        Scenario(
            name="py_small",
            description="2D adaptive 6/10/4 (matches test_basic)",
            build=_py_small,
            validate=validate_vertices(4),
            tags=["quick", "adaptive", "py-callback"],
        )
    )

    def _py_medium(scale, offset):
        return CMGDB.Model(
            _shift(10, scale, offset),
            _shift(14, scale, offset),
            _shift(8, scale, offset),
            10000,
            [0.0, 0.0],
            [1.2, 1.2],
            make_box_map(leslie_like_2d),
        )

    s.append(
        Scenario(
            name="py_medium",
            description="2D adaptive 10/14/8",
            build=_py_medium,
            validate=validate_vertices(4),
            tags=["quick", "adaptive", "py-callback"],
        )
    )

    if include_batch:
        def _batch_medium(scale, offset):
            model = CMGDB.Model(
                _shift(10, scale, offset),
                _shift(14, scale, offset),
                _shift(8, scale, offset),
                10000,
                [0.0, 0.0],
                [1.2, 1.2],
                make_box_map(leslie_like_2d),
            )
            model.set_batch_map(make_batch_map(leslie_like_2d))
            return model

        s.append(
            Scenario(
                name="batch_medium",
                description="Same as py_medium but routed through set_batch_map",
                build=_batch_medium,
                validate=validate_vertices(4),
                tags=["quick", "adaptive", "batch"],
                default=False,
            )
        )

    def _uniform_2d(scale, offset):
        depth = _shift(12, scale, offset)
        return CMGDB.Model(
            depth,
            depth,
            depth,
            10000,
            [0.0, 0.0],
            [1.2, 1.2],
            make_box_map(leslie_like_2d),
        )

    s.append(
        Scenario(
            name="uniform_2d",
            description="2D fixed-subdiv path",
            build=_uniform_2d,
            validate=validate_vertices(25),
            tags=["quick", "uniform", "py-callback"],
        )
    )

    def _reach_4d(scale, offset):
        return CMGDB.Model(
            _shift(16, scale, offset),
            _shift(18, scale, offset),
            _shift(14, scale, offset),
            10000,
            [0.0] * 4,
            [1.2] * 4,
            make_box_map(leslie_like_4d),
        )

    s.append(
        Scenario(
            name="reach_4d",
            description="4D adaptive case with >64 Morse sets for reachability",
            build=_reach_4d,
            validate=validate_vertices(225),
            tags=["quick", "adaptive", "reachability", "4d"],
            default=False,
        )
    )

    def _conley_2d(scale, offset):
        return CMGDB.Model(
            _shift(6, scale, offset),
            _shift(10, scale, offset),
            _shift(4, scale, offset),
            10000,
            [0.0, 0.0],
            [1.2, 1.2],
            make_box_map(leslie_like_2d),
        )

    s.append(
        Scenario(
            name="conley_2d",
            description="2D Conley index (matches test_basic)",
            build=_conley_2d,
            compute=compute_conley,
            validate=validate_conley(4, _CONLEY_2D_ANNOTATIONS),
            tags=["quick", "adaptive", "conley"],
        )
    )

    if include_heavy:
        def _py_3d(scale, offset):
            return CMGDB.Model(
                _shift(27, scale, offset),
                _shift(30, scale, offset),
                _shift(22, scale, offset),
                10000,
                [0.0, 0.0, 0.0],
                [1.2, 1.2, 1.2],
                make_box_map(leslie_like_3d),
            )

        s.append(
            Scenario(
                name="py_3d",
                description="3D adaptive 27/30/22",
                build=_py_3d,
                validate=validate_vertices(27),
                tags=["heavy", "adaptive", "py-callback", "3d"],
                default=False,
            )
        )

        def _py_4d(scale, offset):
            return CMGDB.Model(
                _shift(24, scale, offset),
                _shift(26, scale, offset),
                _shift(20, scale, offset),
                10000,
                [0.0] * 4,
                [1.2] * 4,
                make_box_map(leslie_like_4d),
            )

        s.append(
            Scenario(
                name="py_4d",
                description="4D adaptive 24/26/20",
                build=_py_4d,
                validate=validate_vertices(225),
                tags=["heavy", "adaptive", "py-callback", "4d"],
                default=False,
            )
        )

        def _conley_3d(scale, offset):
            return CMGDB.Model(
                _shift(26, scale, offset),
                _shift(28, scale, offset),
                _shift(22, scale, offset),
                10000,
                [-1.5, -1.5, -1.5],
                [1.5, 1.5, 1.5],
                make_box_map(henon_3d),
            )

        s.append(
            Scenario(
                name="conley_3d",
                description="3D Conley index on Henon attractor",
                build=_conley_3d,
                compute=compute_conley,
                validate=validate_conley(2, _CONLEY_3D_ANNOTATIONS),
                tags=["heavy", "adaptive", "conley", "3d"],
                default=False,
            )
        )

    return s


def _stats(samples: List[float]) -> dict:
    if not samples:
        return {"min": float("nan"), "median": float("nan"), "stdev": float("nan")}
    return {
        "min": min(samples),
        "median": statistics.median(samples),
        "stdev": statistics.stdev(samples) if len(samples) > 1 else 0.0,
    }


def run_scenario(
    sc: Scenario,
    scale: float,
    depth_offset: int,
    repeats: int,
    warmup: int,
    verbose: bool = False,
    enforce_validation: bool = True,
) -> Result:
    build_times: List[float] = []
    compute_times: List[float] = []
    num_vertices = -1
    quiet_factory = contextlib.nullcontext if verbose else silenced_stdout

    for i in range(warmup + repeats):
        gc.collect()
        t0 = time.perf_counter()
        with quiet_factory():
            model = sc.build(scale, depth_offset)
            t1 = time.perf_counter()
            morse_graph, _map_graph = sc.compute(model)
        t2 = time.perf_counter()

        if sc.validate is not None and enforce_validation:
            err = sc.validate(morse_graph)
            if err is not None:
                raise ValidationFailed(
                    f"scenario '{sc.name}' (run {i + 1}/{warmup + repeats}): {err}"
                )

        if i >= warmup:
            build_times.append(t1 - t0)
            compute_times.append(t2 - t1)
            num_vertices = morse_graph.num_vertices()

    return Result(sc.name, build_times, compute_times, num_vertices)


def format_table(results: List[Result]) -> str:
    header = (
        f"{'scenario':<16} {'verts':>6}   "
        f"{'build min':>10} {'build med':>10}   "
        f"{'compute min':>12} {'compute med':>12} {'compute stdev':>14}"
    )
    lines = [header, "-" * len(header)]
    for result in results:
        build = _stats(result.build_times)
        compute = _stats(result.compute_times)
        lines.append(
            f"{result.name:<16} {result.num_morse_vertices:>6}   "
            f"{build['min'] * 1000:>9.1f}ms {build['median'] * 1000:>9.1f}ms   "
            f"{compute['min'] * 1000:>11.1f}ms "
            f"{compute['median'] * 1000:>11.1f}ms "
            f"{compute['stdev'] * 1000:>13.1f}ms"
        )
    return "\n".join(lines)


def run_scale_depth(
    sc: Scenario,
    scale: float,
    repeats: int,
    warmup: int,
    verbose: bool = False,
) -> str:
    rows = []
    for offset in [-1, 0, 1]:
        result = run_scenario(
            sc,
            scale,
            offset,
            repeats,
            warmup,
            verbose=verbose,
            enforce_validation=(offset == 0),
        )
        stats = _stats(result.compute_times)
        rows.append((offset, result.num_morse_vertices, stats["min"], stats["median"]))

    out = [f"--scale-depth for '{sc.name}' (scale={scale})"]
    out.append(f"  {'offset':>7} {'verts':>8} {'compute min':>14} {'compute med':>14}")
    for offset, vertices, compute_min, compute_med in rows:
        out.append(
            f"  {offset:>+7d} {vertices:>8d} "
            f"{compute_min * 1000:>11.1f}ms {compute_med * 1000:>11.1f}ms"
        )

    xs = [offset for offset, _, _, median in rows if median > 0]
    ys = [math.log(median) for _, _, _, median in rows if median > 0]
    if len(xs) >= 2:
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        denominator = sum((x - mean_x) ** 2 for x in xs)
        if denominator != 0:
            slope = numerator / denominator
            growth = math.exp(slope)
            slope_bits = slope / math.log(2)
            out.append(
                f"  fit: time grows {growth:.2f}x per +1 subdiv "
                f"(about 2^{slope_bits:.2f}x cells)"
            )
        else:
            out.append("  fit: skipped (degenerate offsets)")
    else:
        out.append("  fit: skipped (need at least 2 finite timings)")
    return "\n".join(out)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run correctness-validating CMGDB benchmarks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--heavy", action="store_true")
    parser.add_argument("--scenarios", type=str, default=None)
    parser.add_argument("--scale-depth", type=str, default=None, metavar="SCENARIO")
    parser.add_argument("--profile", type=str, default=None, metavar="FILE")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)

    scale = float(os.environ.get("CMGDB_BENCH_SCALE", "1.0"))
    expose_all = args.scale_depth is not None or args.scenarios is not None or args.list
    all_scenarios = scenarios(
        scale=scale,
        include_heavy=args.heavy or expose_all,
        include_batch=_model_supports_batch_map(),
    )

    if args.list:
        for scenario in all_scenarios:
            tags = ",".join(scenario.tags) if scenario.tags else "-"
            default = "default" if scenario.default else "optional"
            print(f"{scenario.name:<16} [{tags}] [{default}]  {scenario.description}")
        return 0

    if args.scale_depth:
        match = [scenario for scenario in all_scenarios if scenario.name == args.scale_depth]
        if not match:
            print(f"unknown scenario: {args.scale_depth!r}", file=sys.stderr)
            return 2
        try:
            print(
                run_scale_depth(
                    match[0],
                    scale,
                    args.repeats,
                    args.warmup,
                    verbose=args.verbose,
                )
            )
        except ValidationFailed as exc:
            print(f"VALIDATION FAILED: {exc}", file=sys.stderr)
            return 3
        return 0

    selected = [scenario for scenario in all_scenarios if scenario.default]
    if args.scenarios:
        wanted = set(args.scenarios.split(","))
        selected = [scenario for scenario in all_scenarios if scenario.name in wanted]
        missing = wanted - {scenario.name for scenario in selected}
        if missing:
            print(f"unknown scenarios: {sorted(missing)}", file=sys.stderr)
            return 2

    print(f"CMGDB benchmark - scale={scale}, repeats={args.repeats}, warmup={args.warmup}")
    print(f"running {len(selected)} scenario(s): {', '.join(s.name for s in selected)}\n")

    def _run_all():
        return [
            run_scenario(
                scenario,
                scale,
                0,
                args.repeats,
                args.warmup,
                verbose=args.verbose,
            )
            for scenario in selected
        ]

    try:
        if args.profile:
            import cProfile
            import pstats

            profile = cProfile.Profile()
            profile.enable()
            results = _run_all()
            profile.disable()
            profile.dump_stats(args.profile)
            print(f"wrote cProfile stats to {args.profile}\n")
            pstats.Stats(profile).sort_stats("cumulative").print_stats(20)
        else:
            results = _run_all()
    except ValidationFailed as exc:
        print(f"VALIDATION FAILED: {exc}", file=sys.stderr)
        return 3

    print(format_table(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
