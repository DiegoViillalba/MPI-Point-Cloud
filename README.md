# Rotación 3D Paralela con MPI — SPMD

Aplicación del modelo **Single Program, Multiple Data (SPMD)** para rotar nubes de puntos 3D en paralelo usando MPI. Cada proceso recibe un subconjunto de puntos, aplica la misma transformación matricial y devuelve los resultados al proceso raíz.

---

## Archivos

| Archivo | Descripción |
|---|---|
| `mpi_rotation.py` | Programa principal MPI: scatter → rotación → gather |
| `matrices.py` | Matrices de rotación Rx, Ry, Rz y eje arbitrario (Rodrigues) |
| `benchmark.py` | Mide T_total, T_cómputo y T_comm variando objetos, tamaños y procesos |
| `visualize.py` | Genera `visualization.png` con los objetos antes y después de rotar |

---

## Requisitos

```bash
pip install mpi4py numpy scikit-learn matplotlib
```

Para `mpi4py` también se necesita tener OpenMPI instalado en el sistema:

```bash
# Ubuntu / Debian
sudo apt-get install libopenmpi-dev openmpi-bin

# macOS
brew install open-mpi
```

---

## Cómo ejecutar

### Programa MPI principal

```bash
# 4 procesos — rota un cubo 45° alrededor de Z
mpiexec -n 4 python mpi_rotation.py

# Cambiar número de procesos
mpiexec -n 2 python mpi_rotation.py
mpiexec -n 8 python mpi_rotation.py
```

### Matrices (demo y verificación)

```bash
python matrices.py
```

Imprime las tres matrices fundamentales y verifica que son ortogonales con det = +1.

### Benchmark

```bash
python benchmark.py
```

Varía 3 objetos × 4 tamaños × 4 configuraciones de procesos y guarda `benchmark_results.csv`.

### Visualización

```bash
python visualize.py
```

Guarda `visualization.png` con los 4 objetos en su estado original y rotado.

---

## Cómo funciona

### Modelo SPMD

Todos los procesos ejecutan el **mismo archivo** (`mpi_rotation.py`). La diferencia de comportamiento se controla con el `rank`:

```
┌────────────────────────────────────────────┐
│  Rank 0         Rank 1   Rank 2   Rank k   │
│  genera puntos                             │
│       │                                    │
│    Scatter ──────►  chunk_1  chunk_2  ...  │
│       │                │        │          │
│   chunk_0          P'= P@R.T           ... │  ← todos rotan
│   P'= P@R.T            │        │          │
│       │                                    │
│    Gather ◄──────── result_1 result_2  ... │
│       │                                    │
│   reconstruye objeto completo              │
└────────────────────────────────────────────┘
```

### Distribución de puntos

Los N puntos se dividen entre p procesos de forma equitativa. Cuando N no es divisible por p, los primeros `N % p` procesos reciben un punto extra:

```
base      = N // p
counts[i] = base + 1   si i < N % p
counts[i] = base        en otro caso
```

---

## Matrices de Rotación

La transformación de cada punto $\mathbf{p} = (x, y, z)^\top$ se calcula como:

$$\mathbf{p}' = R \cdot \mathbf{p}$$

Para N puntos apilados en la matriz $P \in \mathbb{R}^{N \times 3}$, la operación vectorizada es:

$$P' = P \cdot R^\top$$

### Rotación alrededor del eje X

$$R_x(\theta) = \begin{pmatrix} 1 & 0 & 0 \\ 0 & \cos\theta & -\sin\theta \\ 0 & \sin\theta & \cos\theta \end{pmatrix}$$

El eje X permanece fijo; el plano YZ rota.

### Rotación alrededor del eje Y

$$R_y(\theta) = \begin{pmatrix} \cos\theta & 0 & \sin\theta \\ 0 & 1 & 0 \\ -\sin\theta & 0 & \cos\theta \end{pmatrix}$$

El eje Y permanece fijo; el plano XZ rota.

### Rotación alrededor del eje Z

$$R_z(\theta) = \begin{pmatrix} \cos\theta & -\sin\theta & 0 \\ \sin\theta & \cos\theta & 0 \\ 0 & 0 & 1 \end{pmatrix}$$

El eje Z permanece fijo; el plano XY rota.

### Rotación alrededor de eje arbitrario (Rodrigues)

Para rotar $\theta$ grados alrededor de un eje unitario $\hat{\mathbf{u}} = (u_x, u_y, u_z)^\top$, se usa la **fórmula de Rodrigues**:

$$R(\hat{\mathbf{u}}, \theta) = \cos\theta \; I + (1 - \cos\theta)\, \hat{\mathbf{u}}\hat{\mathbf{u}}^\top + \sin\theta \; [\hat{\mathbf{u}}]_\times$$

donde $[\hat{\mathbf{u}}]_\times$ es la matriz antisimétrica (*skew-symmetric*) asociada a $\hat{\mathbf{u}}$:

$$[\hat{\mathbf{u}}]_\times = \begin{pmatrix} 0 & -u_z & u_y \\ u_z & 0 & -u_x \\ -u_y & u_x & 0 \end{pmatrix}$$

Expandiendo término a término:

$$R(\hat{\mathbf{u}}, \theta) = \begin{pmatrix} \cos\theta + u_x^2(1-\cos\theta) & u_x u_y(1-\cos\theta) - u_z\sin\theta & u_x u_z(1-\cos\theta) + u_y\sin\theta \\ u_y u_x(1-\cos\theta) + u_z\sin\theta & \cos\theta + u_y^2(1-\cos\theta) & u_y u_z(1-\cos\theta) - u_x\sin\theta \\ u_z u_x(1-\cos\theta) - u_y\sin\theta & u_z u_y(1-\cos\theta) + u_x\sin\theta & \cos\theta + u_z^2(1-\cos\theta) \end{pmatrix}$$

### Propiedades verificadas

Toda matriz de rotación válida satisface:

$$R^\top R = I \qquad \det(R) = +1$$

La primera garantiza que preserva distancias (isometría); la segunda que es una rotación propia (no una reflexión). Ambas se verifican numéricamente en `matrices.py`.

---

## Métricas de desempeño

| Métrica | Fórmula |
|---|---|
| $T_\text{total}$ | tiempo de pared completo |
| $T_\text{cómputo}$ | solo la operación $P \cdot R^\top$ |
| $T_\text{comm}$ | scatter + gather |
| Speedup | $S = T_\text{serial} / T_\text{paralelo}$ |
| Eficiencia | $E = S / p$ |

> **Nota sobre el benchmark:** el script usa `ProcessPoolExecutor` como proxy de MPI (para ejecutar sin OpenMPI instalado). Cada llamada paga ~280 ms de arranque de proceso Python, lo que explica los speedups bajos. Con `mpiexec` real, los procesos ya están vivos antes de la rotación y el overhead de comunicación real es < 1 ms para N < 10 000 puntos.

---

## Objetos incluidos

- **Cubo** — 26 puntos: 8 vértices + 12 centros de arista + 6 centros de cara
- **Esfera** — espiral de Fibonacci para distribución uniforme sin concentración en los polos
- **Swiss Roll** — variedad 2D enrollada en 3D generada con `sklearn.datasets.make_swiss_roll`
