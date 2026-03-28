"""
matrices.py
-----------
Matrices de rotación 3D fundamentales + rotación arbitraria (Rodrigues).
"""

import numpy as np


# ── Rotaciones fundamentales 

def Rx(deg):
    """Rotación alrededor del eje X."""
    t = np.radians(deg)
    return np.array([[1,          0,           0],
                     [0,  np.cos(t), -np.sin(t)],
                     [0,  np.sin(t),  np.cos(t)]])

def Ry(deg):
    """Rotación alrededor del eje Y."""
    t = np.radians(deg)
    return np.array([[ np.cos(t), 0, np.sin(t)],
                     [0,          1,          0],
                     [-np.sin(t), 0, np.cos(t)]])

def Rz(deg):
    """Rotación alrededor del eje Z."""
    t = np.radians(deg)
    return np.array([[np.cos(t), -np.sin(t), 0],
                     [np.sin(t),  np.cos(t), 0],
                     [0,          0,          1]])


# ── Rotación alrededor de eje arbitrario (fórmula de Rodrigues) 

def R_arbitrary(axis, deg):
    """
    Rotación θ grados alrededor de un eje arbitrario u (vector unitario).

    Fórmula de Rodrigues:
        R = I·cos θ + (1−cos θ)(u⊗u) + sin θ · [u]×

    donde [u]× es la matriz antisimétrica (skew-symmetric) de u:
        [u]× = | 0  -uz   uy |
               | uz   0  -ux |
               |-uy  ux    0 |
    """
    u = np.asarray(axis, dtype=float)
    u = u / np.linalg.norm(u)          # normalizar
    ux, uy, uz = u
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)

    skew = np.array([[ 0,  -uz,  uy],
                     [ uz,   0, -ux],
                     [-uy,  ux,   0]])

    return c * np.eye(3) + (1 - c) * np.outer(u, u) + s * skew


# ── tests básicos de las matrices

if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    print("=== Matrices de rotación fundamentales ===\n")
    print("Rx(90°):\n", Rx(90))
    print("\nRy(90°):\n", Ry(90))
    print("\nRz(90°):\n", Rz(90))

    print("\n=== Rotación arbitraria (Rodrigues) ===\n")
    # Eje diagonal [1,1,1], 120° — cicla los ejes coordenados
    R = R_arbitrary([1, 1, 1], 120)
    print("R([1,1,1], 120°):\n", R)

    # Verificar: debe ciclar x→y→z→x
    x = np.array([1, 0, 0])
    print(f"\nR·x = {R @ x.round(4)}  (esperado: y ≈ [0,1,0])")

    # Verificar propiedades
    for name, M in [("Rx(45)", Rx(45)), ("Ry(30)", Ry(30)),
                    ("Rz(60)", Rz(60)), ("R_arb", R)]:
        ok_orth = np.allclose(M.T @ M, np.eye(3))
        ok_det  = np.isclose(np.linalg.det(M), 1.0)
        print(f"{name}: ortogonal={ok_orth}, det=+1={ok_det}")
