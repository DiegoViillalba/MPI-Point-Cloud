"""
visualize.py
------------
Visualiza los 3 objetos antes y después de la rotación.
Guarda: visualization.png
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matrices import Rx, Ry, Rz, R_arbitrary


# ── Generadores ─────────────────────────────────────────────────────────────

def make_cube():
    h = 1.0
    pts = [[x,y,z] for x in [-h,h] for y in [-h,h] for z in [-h,h]]
    pts += [[h,0,0],[-h,0,0],[0,h,0],[0,-h,0],[0,0,h],[0,0,-h]]
    return np.array(pts, dtype=float)

def make_sphere(n=200):
    golden = (1+np.sqrt(5))/2
    i = np.arange(n)
    theta = 2*np.pi*i/golden
    phi = np.arccos(1-2*(i+0.5)/n)
    return np.column_stack([np.sin(phi)*np.cos(theta),
                             np.sin(phi)*np.sin(theta),
                             np.cos(phi)])

def make_sklearn(n=300):
    try:
        from sklearn.datasets import make_swiss_roll
        X, _ = make_swiss_roll(n, noise=0.1, random_state=0)
        X = (X-X.min(0))/(X.max(0)-X.min(0))*2-1
        return X
    except ImportError:
        return make_sphere(n)


# ── Plot ─────────────────────────────────────────────────────────────────────

def add_scatter(ax, pts, color, title):
    ax.scatter(pts[:,0], pts[:,1], pts[:,2], c=color, s=12, alpha=0.85)
    ax.set_title(title, color="white", fontsize=9, pad=4)
    ax.set_facecolor("#12122a")
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor("#333355")
    ax.tick_params(colors="#666688", labelsize=6)
    ax.set_xlabel("X", color="#8888aa", fontsize=7)
    ax.set_ylabel("Y", color="#8888aa", fontsize=7)
    ax.set_zlabel("Z", color="#8888aa", fontsize=7)


configs = [
    ("Cubo",         make_cube(),       Rz(45),                  "Rz(45°)",   "#4fc3f7", "#ef5350"),
    ("Esfera",       make_sphere(),     Ry(60),                  "Ry(60°)",   "#81c784", "#ce93d8"),
    ("Swiss Roll",   make_sklearn(),    Rx(90),                  "Rx(90°)",   "#ffb74d", "#f48fb1"),
    ("Eje arbitrario\n[1,1,0] 135°", make_cube(), R_arbitrary([1,1,0],135), "R_arb(135°)", "#80cbc4", "#ffcc80"),
]

fig = plt.figure(figsize=(16, 9), facecolor="#0a0a1a")
fig.suptitle("Rotación 3D — SPMD con MPI", color="white",
             fontsize=15, fontweight="bold", y=0.98)

for col, (name, pts, R, label, c_orig, c_rot) in enumerate(configs):
    rot = pts @ R.T

    ax1 = fig.add_subplot(2, 4, col + 1,     projection="3d")
    ax2 = fig.add_subplot(2, 4, col + 4 + 1, projection="3d")

    add_scatter(ax1, pts, c_orig, f"{name}\nOriginal")
    add_scatter(ax2, rot, c_rot,  f"{label}")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("visualization.png", dpi=150, bbox_inches="tight", facecolor="#0a0a1a")
print("Guardado: visualization.png")
