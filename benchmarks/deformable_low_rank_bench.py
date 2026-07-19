"""Compare deformable low-rank kernel construction methods.

Run from the repository root:
    PYTHONPATH=. python benchmarks/deformable_low_rank_bench.py
"""

import argparse
import time

import numpy as np

from biocpd import DeformableRegistration


def make_case(seed, point_count):
    generator = np.random.default_rng(seed)
    source = generator.normal(size=(point_count, 3)).astype(np.float32)
    source *= np.array([1.5, 0.9, 0.6], dtype=np.float32)
    displacement = np.column_stack(
        (
            np.sin(source[:, 1]) + 0.25 * np.cos(source[:, 2]),
            np.sin(source[:, 2]) - 0.20 * np.cos(source[:, 0]),
            np.sin(source[:, 0]) + 0.15 * np.cos(source[:, 1]),
        )
    ).astype(np.float32)
    target = source + 0.025 * displacement
    return source, target


def run_case(method, source, target, args):
    start = time.perf_counter()
    registration = DeformableRegistration(
        X=target,
        Y=source,
        alpha=args.alpha,
        beta=args.beta,
        low_rank=True,
        num_eig=args.rank,
        low_rank_method=method,
        use_kdtree=True,
        k=args.neighbors,
        dtype=np.float32,
        max_iterations=args.iterations,
        tolerance=0.0,
    )
    initialization_seconds = time.perf_counter() - start

    start = time.perf_counter()
    transformed, _ = registration.register(callback=None)
    registration_seconds = time.perf_counter() - start
    rms_error = float(np.sqrt(np.mean((transformed - target) ** 2)))
    return {
        "method": method,
        "initialization_seconds": initialization_seconds,
        "registration_seconds": registration_seconds,
        "rms_error": rms_error,
        "rank": registration.low_rank_diagnostics["rank"],
        "max_residual": registration.low_rank_diagnostics[
            "max_residual_diagonal"
        ],
    }


def warm_up(method, source, target, args):
    warm_count = min(len(source), 24)
    DeformableRegistration(
        X=target[:warm_count],
        Y=source[:warm_count],
        low_rank=True,
        num_eig=min(args.rank, 8),
        low_rank_method=method,
        use_kdtree=False,
        dtype=np.float32,
        max_iterations=0,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark deformable low-rank kernel construction.",
    )
    parser.add_argument("--points", type=int, default=2500)
    parser.add_argument("--rank", type=int, default=120)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--neighbors", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260719)
    args = parser.parse_args()

    source, target = make_case(args.seed, args.points)
    methods = ("randomized_svd", "pivoted_cholesky")
    for method in methods:
        warm_up(method, source, target, args)

    results = []
    for method in methods:
        repeated = [
            run_case(method, source, target, args)
            for _ in range(args.repeats)
        ]
        result = repeated[-1]
        result["initialization_seconds"] = float(
            np.median(
                [value["initialization_seconds"] for value in repeated]
            )
        )
        result["registration_seconds"] = float(
            np.median(
                [value["registration_seconds"] for value in repeated]
            )
        )
        results.append(result)

    print(
        f"{'method':<20} {'init_s':>10} {'fit_s':>10} "
        f"{'rms_error':>12} {'rank':>6} {'max_diag_res':>14}"
    )
    for result in results:
        residual = result["max_residual"]
        residual_text = "n/a" if residual is None else f"{residual:.6g}"
        print(
            f"{result['method']:<20} "
            f"{result['initialization_seconds']:>10.3f} "
            f"{result['registration_seconds']:>10.3f} "
            f"{result['rms_error']:>12.6g} "
            f"{result['rank']:>6d} "
            f"{residual_text:>14}"
        )


if __name__ == "__main__":
    main()
