"""
mpi_rotation.py
---------------
Rotación 3D paralela de nube de puntos usando MPI (modelo SPMD).

Diego Villalba - 2024-06
"""
import numpy as np
import argparse
import time
import os
import csv
import matplotlib.pyplot as plt
from mpi4py import MPI
from matrices import Rx, Ry, Rz, R_arbitrary
from clouds import make_cube, make_sphere, make_sklearn

def plot_comparison(original, rotated, title=""):
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    # Configuración de puntos
    ax1.scatter(original[:, 0], original[:, 1], original[:, 2], c='blue', s=20, alpha=0.6)
    ax1.set_title("Original")
    
    ax2.scatter(rotated[:, 0], rotated[:, 1], rotated[:, 2], c='red', s=20, alpha=0.6)
    ax2.set_title(f"Rotado {title}")

    for ax in [ax1, ax2]:
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        # Forzar ejes iguales para que el cubo no parezca un ladrillo
        ax.set_box_aspect([1,1,1]) 

    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--object",  choices=["cube","sphere","sklearn"], default="cube")
    parser.add_argument("--axis",    choices=["x","y","z","arbitrary"],   default="z")
    parser.add_argument("--angle",   type=float, default=45.0)
    parser.add_argument("--n",       type=int,   default=None)
    parser.add_argument("--performance", action="store_true", help="Activa métricas CSV")
    parser.add_argument("--plot", action="store_true", help="Muestra gráfica 3D")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # 1. Matriz
    R_map = {"x": Rx, "y": Ry, "z": Rz}
    R = R_arbitrary([1,1,0], args.angle) if args.axis == "arbitrary" else R_map[args.axis](args.angle)

    # 2. Setup inicial
    points = None
    original_copy = None
    t_start = 0

    if rank == 0:
        if args.object == "cube": points = make_cube()
        elif args.object == "sphere": points = make_sphere(args.n or 500)
        elif args.object == "sklearn": points = make_sklearn(args.n or 500)
        
        if args.plot:
            original_copy = points.copy()
            
        t_start = time.perf_counter()
    
    # 3. Distribución (Scatter)
    if rank == 0:
        chunks = np.array_split(points, size)
        local_data = chunks[0]
        for dst in range(1, size):
            comm.send(chunks[dst], dest=dst, tag=0)
    else:
        local_data = comm.recv(source=0, tag=0)

    # 4. Cómputo
    t_comp_start = time.perf_counter()
    local_rot = local_data @ R.T
    t_comp_end = time.perf_counter()

    # 5. Recolección (Gather)
    if rank == 0:
        results = [local_rot]
        for src in range(1, size):
            results.append(comm.recv(source=src, tag=1))
        final_result = np.vstack(results)
        t_end = time.perf_counter()

        # --- Lógica de Salida ---
        total_ms = (t_end - t_start) * 1000
        comp_ms = (t_comp_end - t_comp_start) * 1000
        print(f"[+] {args.object} completado en {total_ms:.2f}ms")

        if args.performance:
            file = "benchmark_results.csv"
            exists = os.path.isfile(file)
            with open(file, "a", newline="") as f:
                writer = csv.writer(f)
                if not exists:
                    writer.writerow(["objeto", "n_pts", "n_procs", "total_ms", "comp_ms"])
                writer.writerow([args.object, len(final_result), size, round(total_ms,3), round(comp_ms,3)])
            print(f"[CSV] Datos añadidos a {file}")

        if args.plot:
            plot_comparison(original_copy, final_result, f"{args.axis.upper()} {args.angle}°")

    else:
        comm.send(local_rot, dest=0, tag=1)

    MPI.Finalize()

if __name__ == "__main__":
    main()

# # ── SPMD ───────────────────────────────────────────────────────────────────
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# size = comm.Get_size()

# AXIS, ANGLE = "z", 45.0
# R = rot_z(ANGLE)  # todos calculan la misma R

# # Rank 0 genera los puntos
# points = make_cube() if rank == 0 else None

# # Scatter manual (acepta N no divisible por p)
# if rank == 0:
#     chunks = np.array_split(points, size)
#     for dst in range(1, size):
#         comm.send(chunks[dst], dest=dst, tag=0)
#     local = chunks[0]
# else:
#     local = comm.recv(source=0, tag=0)

# # Cada proceso rota su porción
# local_rot = local @ R.T

# # Gather
# if rank == 0:
#     parts = [local_rot]
#     for src in range(1, size):
#         parts.append(comm.recv(source=src, tag=1))
#     result = np.vstack(parts)
#     print(f"Rotación {ANGLE}° eje {AXIS.upper()} completada.")
#     print(f"Procesos: {size} | Puntos: {len(result)}")
#     print(f"Primeros 3 puntos rotados:\n{result[:3].round(4)}")
#     np.save("rotated_cube.npy", result)
# else:
#     comm.send(local_rot, dest=0, tag=1)

# MPI.Finalize()
