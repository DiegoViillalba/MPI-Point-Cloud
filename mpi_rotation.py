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
from mpi4py import MPI
from matrices import Rx, Ry, Rz, R_arbitrary
from clouds import make_cube, make_sphere, make_sklearn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--object",  choices=["cube","sphere","sklearn"], default="cube")
    parser.add_argument("--axis",    choices=["x","y","z","arbitrary"],   default="z")
    parser.add_argument("--angle",   type=float, default=45.0)
    parser.add_argument("--n",       type=int,   default=None)
    parser.add_argument("--performance", action="store_true", help="Guarda métricas en CSV")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # 1. Preparar Matriz (Todos los procesos)
    R_map = {"x": Rx, "y": Ry, "z": Rz}
    R = R_arbitrary([1, 1, 0], args.angle) if args.axis == "arbitrary" else R_map[args.axis](args.angle)

    # 2. Generación y Sincronización (Solo Rank 0 mide tiempo total)
    points = None
    t_start = 0
    
    if rank == 0:
        if args.object == "cube":
            points = make_cube()
        elif args.object == "sphere":
            points = make_sphere(args.n or 500)
        elif args.object == "sklearn":
            points = make_sklearn(args.n or 500)
        
        n_points = len(points)
        print(f"[*] Iniciando: {args.object} ({n_points} pts) con {size} procesos.")
        t_start = time.perf_counter()
    else:
        n_points = None

    # Compartir el número de puntos con todos (opcional, para logging)
    n_points = comm.bcast(n_points, root=0)

    # 3. Distribución (Scatter)
    t_comm_start = time.perf_counter()
    if rank == 0:
        chunks = np.array_split(points, size)
        local_data = chunks[0]
        for dst in range(1, size):
            comm.send(chunks[dst], dest=dst, tag=0)
    else:
        local_data = comm.recv(source=0, tag=0)
    
    # 4. Cómputo (Todos los procesos)
    t_comp_start = time.perf_counter()
    local_rot = local_data @ R.T
    t_comp_end = time.perf_counter()
    
    # 5. Recolección (Gather)
    if rank == 0:
        results_list = [local_rot]
        for src in range(1, size):
            results_list.append(comm.recv(source=src, tag=1))
        final_result = np.vstack(results_list)
        
        t_end = time.perf_counter()
        
        # --- Cálculos de Benchmark ---
        total_ms = (t_end - t_start) * 1000
        comp_ms = (t_comp_end - t_comp_start) * 1000
        comm_ms = total_ms - comp_ms # Tiempo que no fue cálculo puro

        print(f"[+] Finalizado en {total_ms:.3f}ms (Cómputo: {comp_ms:.3f}ms)")

        # 6. Guardar en CSV si el flag está activo
        if args.performance:
            csv_file = "benchmark_results.csv"
            file_exists = os.path.isfile(csv_file)
            
            with open(csv_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["objeto", "n_puntos", "n_procesos", "total_ms", "comp_ms", "comm_ms"])
                writer.writerow([args.object, n_points, size, round(total_ms, 4), round(comp_ms, 4), round(comm_ms, 4)])
            print(f"[!] Datos guardados en {csv_file}")
            
        np.save("rotated_output.npy", final_result)
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
