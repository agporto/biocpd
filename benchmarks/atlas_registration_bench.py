import argparse
import time

import numpy as np

from biocpd import AtlasRegistration


def make_case(seed: int, num_points: int, dims: int, num_modes: int, noise: float):
    rng = np.random.default_rng(seed)
    Y = rng.normal(size=(num_points, dims))
    U = rng.normal(size=(num_points * dims, num_modes)) * 0.01
    L = np.linspace(2.0, 0.15, num_modes)

    b_true = rng.normal(scale=0.25, size=(num_modes, 1))
    Yb = Y + (U @ b_true).reshape(num_points, dims)

    A = rng.normal(size=(dims, dims))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, -1] *= -1

    X = 1.06 * (Yb @ Q.T) + np.array([[0.15, -0.08, 0.05]])[:, :dims] + noise * rng.normal(size=(num_points, dims))
    return X, Y, U, L


def run_case(name: str, X, Y, U, L, args, **kwargs):
    reg = AtlasRegistration(
        X=X,
        Y=Y,
        U=U,
        eigenvalues=L,
        lambda_reg=args.lambda_reg,
        normalize=True,
        max_iterations=args.max_iterations,
        tolerance=args.tolerance,
        optimize_similarity=True,
        with_scale=True,
        w=0.0,
        dense_block_size=args.dense_block_size,
        store_posterior=False,
        **kwargs,
    )

    start = time.perf_counter()
    TY, params = reg.register()
    elapsed = time.perf_counter() - start

    return {
        "name": name,
        "elapsed": elapsed,
        "iterations": reg.iteration,
        "sigma2": float(reg.sigma2),
        "TY": TY,
        "b": params["b"],
        "R": params["R"],
        "s": float(params["s"]),
        "t": params["t"],
    }


def quality_metrics(candidate, dense, args):
    point_delta = float(np.sqrt(np.mean((candidate["TY"] - dense["TY"]) ** 2)))
    b_delta = float(np.linalg.norm(candidate["b"] - dense["b"]))
    r_delta = float(np.max(np.abs(candidate["R"] - dense["R"])))
    s_delta = abs(candidate["s"] - dense["s"])
    t_delta = float(np.max(np.abs(candidate["t"] - dense["t"])))
    sigma2_delta = abs(candidate["sigma2"] - dense["sigma2"])
    quality_ok = (
        point_delta <= args.point_delta_tol
        and b_delta <= args.b_delta_tol
        and r_delta <= args.r_delta_tol
        and s_delta <= args.s_delta_tol
        and t_delta <= args.t_delta_tol
        and sigma2_delta <= args.sigma2_delta_tol
    )
    return point_delta, b_delta, r_delta, s_delta, t_delta, sigma2_delta, quality_ok

def main():
    parser = argparse.ArgumentParser(description="Benchmark AtlasRegistration backends.")
    parser.add_argument("--points", type=int, default=5000, help="Number of 3D points.")
    parser.add_argument("--dims", type=int, default=3, help="Point dimensionality.")
    parser.add_argument("--modes", type=int, default=40, help="Number of atlas modes.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--noise", type=float, default=1.5e-3, help="Observation noise.")
    parser.add_argument("--lambda-reg", type=float, default=0.15, help="Atlas prior weight.")
    parser.add_argument("--max-iterations", type=int, default=25, help="Maximum EM iterations.")
    parser.add_argument("--tolerance", type=float, default=1e-6, help="EM tolerance.")
    parser.add_argument("--dense-block-size", type=int, default=None, help="Optional dense E-step block size.")
    parser.add_argument("--point-delta-tol", type=float, default=0.10, help="Dense-vs-candidate RMS point delta tolerance.")
    parser.add_argument("--b-delta-tol", type=float, default=1.0, help="Dense-vs-candidate coefficient norm tolerance.")
    parser.add_argument("--r-delta-tol", type=float, default=0.10, help="Dense-vs-candidate rotation max-abs tolerance.")
    parser.add_argument("--s-delta-tol", type=float, default=0.10, help="Dense-vs-candidate scale tolerance.")
    parser.add_argument("--t-delta-tol", type=float, default=0.10, help="Dense-vs-candidate translation max-abs tolerance.")
    parser.add_argument("--sigma2-delta-tol", type=float, default=0.10, help="Dense-vs-candidate sigma2 tolerance.")
    args = parser.parse_args()

    X, Y, U, L = make_case(args.seed, args.points, args.dims, args.modes, args.noise)

    configs = [
        ("dense-exact", dict(use_kdtree=False)),
        ("knn-k10", dict(use_kdtree=True, k=10)),
    ]

    results = [run_case(name, X, Y, U, L, args, **kwargs) for name, kwargs in configs]
    dense = results[0]

    header = (
        f"{'config':<18} {'time_s':>8} {'iters':>5} {'sigma2':>10} {'sig2_d':>10} "
        f"{'point_d':>10} {'b_d':>10} {'R_d':>10} {'s_d':>10} {'t_d':>10} {'pass':>6}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        if result["name"] == "dense-exact":
            point_delta, b_delta, r_delta, s_delta, t_delta, sigma2_delta, quality_ok = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, True)
        else:
            point_delta, b_delta, r_delta, s_delta, t_delta, sigma2_delta, quality_ok = quality_metrics(result, dense, args)
        print(
            f"{result['name']:<18} {result['elapsed']:>8.3f} {result['iterations']:>5} {result['sigma2']:>10.6f} {sigma2_delta:>10.6f} "
            f"{point_delta:>10.6f} {b_delta:>10.6f} {r_delta:>10.6f} {s_delta:>10.6f} {t_delta:>10.6f} {str(quality_ok):>6}"
        )


if __name__ == "__main__":
    main()
