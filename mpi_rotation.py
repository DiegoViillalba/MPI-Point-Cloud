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

def plot_comparison(original, rotated, title="Rotación 3D"):
    fig = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={'projection': '3d'})
    fig, (ax1, ax2) = fig # Desempaquetado para claridad

    # Gráfica Antes
    ax1.scatter(original[:, 0], original[:, 1], original[:, 2], c='blue', alpha=0.6)
    ax1.set_title("Original")
    ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")
    
    # Gráfica Después
    ax2.scatter(rotated[:, 0], rotated[:, 1], rotated[:, 2], c='red', alpha=0.6)
    ax2.set_title(f"Resultado {title}")
    ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_zlabel("Z")

    # Mantener los ejes proporcionales para no deformar el cubo
    for ax in [ax1, ax2]:
        max_range = np.array([original.max()-original.min(), 
                             original.max()-original.min(), 
                             original.max()-original.min()]).max() / 2.0
        mid_x = (original[:,0].max()+original[:,0].min()) * 0.5
        mid_y = (original[:,1].max()+original[:,1].min()) * 0.5
        mid_z = (original[:,2].max()+original[:,2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--object",  choices=["cube","sphere","sklearn"], default="sklearn")
    parser.add_argument("--axis",    choices=["x","y","z","arbitrary"],   default="z")
    parser.add_argument("--angle",   type=float, default=45.0)
    parser.add_argument("--n",       type=int,   default=None)
    parser.add_argument("--performance", action="store_true")
    parser.add_argument("--plot", action="store_true", help="Mostrar gráfica al finalizar")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    R_map = {"x": Rx, "y": Ry, "z": Rz}
    R = R_arbitrary([1, 1, 0], args.angle) if args.axis == "arbitrary" else R_map[args.axis](args.angle)

    points = None
    original_full = None # Guardar copia completa para graficar
    
    if rank == 0:
        if args.object == "cube":
            points = make_cube()
        elif args.object == "sphere":
            points = make_sphere(args.n or 500)
        elif args.object == "sklearn":
            points = make_sklearn(args.n or 500)
        
        if args.plot:
            original_full = points.copy()
        
        print(f"[*] Procesando {args.object} con {size} procesos...")

    # Scatter
    if rank == 0:
        chunks = np.array_split(points, size)
        local_data = chunks[0]
        for dst in range(1, size):
            comm.send(chunks[dst], dest=dst, tag=0)
    else:
        local_data = comm.recv(source=0, tag=0)
    
    # Cómputo
    local_rot = local_data @ R.T
    
    # Gather
    if rank == 0:
        results_list = [local_rot]
        for src in range(1, size):
            results_list.append(comm.recv(source=src, tag=1))
        final_result = np.vstack(results_list)
        
        print("[+] Rotación completada.")

        if args.plot:
            print("[*] Abriendo visualización...")
            plot_comparison(original_full, final_result, f"({args.axis.upper()} {args.angle}°)")
            
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
