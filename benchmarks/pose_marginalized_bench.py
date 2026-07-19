"""Local benchmark for pose-marginalized atlas initialization.

Run from the repository root:
    PYTHONPATH=. python benchmarks/pose_marginalized_bench.py
"""

import argparse
import time

import numpy as np
from scipy.spatial.transform import Rotation

from biocpd import pose_marginalized_initialization


def make_case(seed, point_count, rank):
    generator = np.random.default_rng(seed)
    source = generator.normal(size=(point_count, 3))
    source *= np.array([2.0, 1.2, 0.7])
    source[:, 0] += 0.2 * source[:, 1] ** 2
    orthogonal, _ = np.linalg.qr(
        generator.normal(size=(source.size, rank))
    )
    modes = orthogonal * np.linspace(0.5, 0.15, rank)
    eigenvalues = np.linspace(0.5, 0.08, rank)
    coefficients = np.linspace(0.25, -0.1, rank)
    deformed = source + (modes @ coefficients).reshape(source.shape)
    rotation = Rotation.from_euler(
        "xyz",
        [145.0, -70.0, 120.0],
        degrees=True,
    ).as_matrix()
    target = 1.08 * (deformed @ rotation.T) + np.array([0.8, -0.4, 0.2])
    return source, target, modes, eigenvalues


def run_case(name, source, target, modes, eigenvalues, **kwargs):
    start = time.perf_counter()
    result = pose_marginalized_initialization(
        source,
        target,
        modes,
        eigenvalues,
        seed=7,
        **kwargs,
    )
    elapsed = time.perf_counter() - start
    recovered = result.scale * (
        (
            source
            + (modes @ result.coefficients).reshape(source.shape)
        )
        @ result.rotation.T
    ) + result.translation
    diagonal = np.linalg.norm(np.ptp(target, axis=0))
    error = np.median(np.linalg.norm(recovered - target, axis=1)) / diagonal
    return name, elapsed, error, result


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark pose-marginalized initialization.",
    )
    parser.add_argument("--points", type=int, default=400)
    parser.add_argument("--modes", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--n-jobs", type=int, default=2)
    args = parser.parse_args()
    source, target, modes, eigenvalues = make_case(
        args.seed,
        args.points,
        args.modes,
    )

    cases = [
        (
            "default-full",
            dict(
                n_jobs=1,
            ),
        ),
        (
            "staged-serial",
            dict(
                coarse_screen_iterations=2,
                coarse_survivor_count=48,
                refine_source_count=min(1600, args.points),
                n_jobs=1,
            ),
        ),
        (
            "staged-parallel",
            dict(
                coarse_screen_iterations=2,
                coarse_survivor_count=48,
                refine_source_count=min(1600, args.points),
                n_jobs=args.n_jobs,
            ),
        ),
    ]
    results = [
        run_case(
            name,
            source,
            target,
            modes,
            eigenvalues,
            **configuration,
        )
        for name, configuration in cases
    ]

    print(f"{'configuration':<20} {'seconds':>10} {'error':>12} {'poses':>8}")
    for name, elapsed, error, result in results:
        poses = f"{result.hypotheses_evaluated}/{result.hypotheses_refined}"
        print(f"{name:<20} {elapsed:>10.3f} {error:>12.6g} {poses:>8}")


if __name__ == "__main__":
    main()
