"""
benchmark.py
------------
Mide T_total, T_cómputo, T_comm simulando p procesos con ProcessPoolExecutor.
Varía: objetos × tamaños × número de procesos.
"""

import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from matrices import Rz


# ── Generadores de nubes ────────────────────────────────────────────────────

def make_cube(n=26):
    h = 1.0
    pts = [[x, y, z] for x in [-h,h] for y in [-h,h] for z in [-h,h]]
    pts += [[h,0,0],[-h,0,0],[0,h,0],[0,-h,0],[0,0,h],[0,0,-h]]
    cloud = np.array(pts, dtype=float)
    if n > len(cloud):
        rng = np.random.default_rng(0)
        extra = rng.uniform(-h, h, (n - len(cloud), 3))
        cloud = np.vstack([cloud, extra])
    return cloud[:n]

def make_sphere(n=500):
    golden = (1 + np.sqrt(5)) / 2
    i = np.arange(n)
    theta = 2*np.pi*i / golden
    phi   = np.arccos(1 - 2*(i+0.5)/n)
    return np.column_stack([np.sin(phi)*np.cos(theta),
                             np.sin(phi)*np.sin(theta),
                             np.cos(phi)])

def make_sklearn(n=500):
    try:
        from sklearn.datasets import make_swiss_roll
        X, _ = make_swiss_roll(n, noise=0.1, random_state=0)
        X = (X - X.min(0)) / (X.max(0) - X.min(0)) * 2 - 1
        return X
    except ImportError:
        return make_sphere(n)


# ── Worker (ejecutado en proceso separado) ──────────────────────────────────

def _rotate_chunk(args):
    chunk, R = args
    return (np.array(chunk) @ R.T)


# ── Experimento único ────────────────────────────────────────────────────────

def run_one(points, n_procs, R):
    t0 = time.perf_counter()

    # Scatter
    ts = time.perf_counter()
    chunks = np.array_split(points, n_procs)
    t_scatter = time.perf_counter() - ts

    # Cómputo paralelo
    tc = time.perf_counter()
    if n_procs == 1:
        results = [chunks[0] @ R.T]
    else:
        with ProcessPoolExecutor(max_workers=n_procs) as ex:
            results = list(ex.map(_rotate_chunk, [(c, R) for c in chunks]))
    t_comp = time.perf_counter() - tc

    # Gather
    tg = time.perf_counter()
    _ = np.vstack(results)
    t_gather = time.perf_counter() - tg

    t_total = time.perf_counter() - t0
    t_comm  = t_scatter + t_gather
    return t_total, t_comp, t_comm


# ── Benchmark completo ───────────────────────────────────────────────────────

def run_benchmark():
    R = Rz(45)
    sizes   = [26, 200, 1000, 5000]
    procs   = [1, 2, 4, 8]
    objects = [("cubo",   make_cube),
               ("esfera", make_sphere),
               ("sklearn",make_sklearn)]

    header = f"{'Objeto':<8} {'N':>5} {'p':>3}  {'T_tot(ms)':>10} {'T_comp(ms)':>11} {'T_comm(ms)':>11} {'Speedup':>8} {'Efic%':>7}"
    print("\n" + "=" * len(header))
    print("  BENCHMARK MPI SPMD — Rotación 3D")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    serial_ref = {}
    for obj_name, gen in objects:
        for n in sizes:
            pts = gen(n)
            t_total, t_comp, t_comm = run_one(pts, 1, R)
            serial_ref[(obj_name, n)] = t_total

    for obj_name, gen in objects:
        for n in sizes:
            pts = gen(n)
            t_s = serial_ref[(obj_name, n)]
            for p in procs:
                t_total, t_comp, t_comm = run_one(pts, p, R)
                speedup = t_s / t_total if p > 1 else 1.0
                effic   = speedup / p * 100
                print(f"{obj_name:<8} {n:>5} {p:>3}  "
                      f"{t_total*1e3:>10.3f} {t_comp*1e3:>11.3f} "
                      f"{t_comm*1e3:>11.3f} {speedup:>8.3f} {effic:>7.1f}")
        print()

    # Guardar CSV
    import csv
    with open("benchmark_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["objeto","N","p","T_total_ms","T_comp_ms","T_comm_ms","speedup","efic_pct"])
        for obj_name, gen in objects:
            for n in sizes:
                pts = gen(n)
                t_s = serial_ref[(obj_name, n)]
                for p in procs:
                    t_total, t_comp, t_comm = run_one(pts, p, R)
                    speedup = t_s / t_total if p > 1 else 1.0
                    w.writerow([obj_name, n, p,
                                 round(t_total*1e3,4), round(t_comp*1e3,4),
                                 round(t_comm*1e3,4), round(speedup,4),
                                 round(speedup/p*100,2)])
    print("Guardado: benchmark_results.csv")


if __name__ == "__main__":
    run_benchmark()
