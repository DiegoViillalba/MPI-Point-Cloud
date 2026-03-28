# Rotación 3D Paralela con MPI — SPMD

Aplicación del modelo **Single Program, Multiple Data (SPMD)** para rotar nubes de puntos 3D en paralelo usando MPI. Esta versión utiliza `uv` para una gestión de dependencias ultra rápida y ejecución determinista.

---

## Archivos

| Archivo | Descripción |
|---|---|
| `mpi_rotation.py` | Programa principal MPI: rotación, benchmark (`--performance`) y visualización (`--plot`) |
| `matrices.py` | Lógica matemática: matrices $R_x$, $R_y$, $R_z$ y Rodrigues |
| `clouds.py` | Generadores vectorizados de nubes de puntos: Cubo, Esfera, Swiss Roll |
| `run_experiments.py` | Automatiza múltiples tests y genera gráficas de speedup |

---

## Instalación y requisitos

Este proyecto requiere `uv` y OpenMPI.

```bash
# 1. Instalar OpenMPI
brew install open-mpi           # macOS
sudo apt install openmpi-bin    # Ubuntu/Debian

# 2. Sincronizar entorno y dependencias
uv sync
```

> `uv sync` lee `pyproject.toml` e instala `mpi4py`, `numpy`, `scikit-learn` y `matplotlib` en un entorno virtual aislado de forma determinista.

---

## Cómo ejecutar

Es fundamental anteponer `uv run` para que MPI use el intérprete correcto del entorno virtual.

### 1. Ejecución básica

```bash
uv run mpiexec -n 4 python mpi_rotation.py --axis z --angle 45
```

### 2. Elegir objeto y tamaño

```bash
# Esfera con 1000 puntos, rotación 60° en Y
uv run mpiexec -n 4 python mpi_rotation.py --object sphere --n 1000 --axis y --angle 60

# Swiss Roll con 5000 puntos, eje arbitrario [1,1,0]
uv run mpiexec -n 4 python mpi_rotation.py --object sklearn --n 5000 --axis arbitrary --angle 135
```

### 3. Visualización 3D

```bash
uv run mpiexec -n 4 python mpi_rotation.py --object sphere --n 1000 --plot
```

Guarda `visualization.png` con el objeto antes y después de la transformación.

### 4. Test de performance

```bash
uv run mpiexec -n 8 python mpi_rotation.py --object sklearn --n 50000 --performance
```

Guarda los tiempos en `benchmark_results.csv`.

### 5. Batería de experimentos completa

```bash
uv run python run_experiments.py
```

Ejecuta todas las combinaciones de objetos × tamaños × procesos y genera `performance_charts.png`.

---

## Opciones de `mpi_rotation.py`

| Argumento | Valores | Default | Descripción |
|---|---|---|---|
| `--object` | `cube` `sphere` `sklearn` | `cube` | Nube de puntos a usar |
| `--axis` | `x` `y` `z` `arbitrary` | `z` | Eje de rotación |
| `--angle` | float | `45.0` | Ángulo en grados |
| `--n` | int | según objeto | Número de puntos |
| `--plot` | flag | — | Guardar visualización PNG |
| `--performance` | flag | — | Correr benchmark y guardar CSV |

---

## Cómo funciona

### Modelo SPMD

Todos los procesos ejecutan el mismo archivo. La diferencia de comportamiento se controla con el `rank`:

```
  Rank 0              Rank 1        Rank 2        Rank k
  genera nube
       │
   Scatter ─────────► chunk_1      chunk_2      chunk_k
       │                  │           │            │
   chunk_0           P'=P@R.T    P'=P@R.T     P'=P@R.T   ← paralelo
       │                  │           │            │
   Gather ◄────────────  res_1       res_2        res_k
       │
   reconstruye P' completo
```

**Distribución equitativa:** cuando $N$ no es divisible entre $p$, los primeros $N \bmod p$ procesos reciben un punto extra, garantizando una diferencia máxima de 1 punto entre procesos.

---

## Fundamentos matemáticos

### Transformación de un punto

La rotación de cada punto $\mathbf{p} = (x,\, y,\, z)^\top$ se calcula como:

$$\mathbf{p}' = R\,\mathbf{p}$$

Para $N$ puntos en la matriz $P \in \mathbb{R}^{N \times 3}$, la operación vectorizada (nivel BLAS-3) es:

$$P' = P \cdot R^\top$$

### Rotación alrededor del eje X

El eje $X$ permanece fijo; el plano $YZ$ rota.

$$R_x(\theta) = \begin{pmatrix} 1 & 0 & 0 \\ 0 & \cos\theta & -\sin\theta \\ 0 & \sin\theta & \phantom{-}\cos\theta \end{pmatrix}$$

### Rotación alrededor del eje Y

El eje $Y$ permanece fijo; el plano $XZ$ rota.

$$R_y(\theta) = \begin{pmatrix} \cos\theta & 0 & \sin\theta \\ 0 & 1 & 0 \\ -\sin\theta & 0 & \cos\theta \end{pmatrix}$$

### Rotación alrededor del eje Z

El eje $Z$ permanece fijo; el plano $XY$ rota.

$$R_z(\theta) = \begin{pmatrix} \cos\theta & -\sin\theta & 0 \\ \sin\theta & \phantom{-}\cos\theta & 0 \\ 0 & 0 & 1 \end{pmatrix}$$

### Fórmula de Rodrigues — eje arbitrario

Para rotar $\theta$ grados alrededor de un eje unitario $\hat{\mathbf{u}} = (u_x,\, u_y,\, u_z)^\top$:

$$R(\hat{\mathbf{u}},\,\theta) = \cos\theta\; I + (1-\cos\theta)\,\hat{\mathbf{u}}\hat{\mathbf{u}}^\top + \sin\theta\;[\hat{\mathbf{u}}]_\times$$

donde $[\hat{\mathbf{u}}]_\times$ es la matriz antisimétrica de $\hat{\mathbf{u}}$:

$$[\hat{\mathbf{u}}]_\times = \begin{pmatrix} 0 & -u_z & u_y \\ u_z & 0 & -u_x \\ -u_y & u_x & 0 \end{pmatrix}$$

### Propiedades verificadas

Toda matriz de rotación válida cumple:

$$R^\top R = I \qquad \det(R) = +1$$

La primera garantiza que se preservan distancias (isometría); la segunda que es una rotación propia (no una reflexión). Ambas se verifican automáticamente en `matrices.py`.

---

## Métricas de desempeño

| Métrica | Fórmula | Descripción |
|---|---|---|
| $T_\text{total}$ | wall clock completo | Desde scatter hasta gather |
| $T_\text{comp}$ | tiempo de $P \cdot R^\top$ | Trabajo útil de cómputo |
| $T_\text{comm}$ | $T_\text{total} - T_\text{comp}$ | Overhead de scatter + gather |
| Speedup | $S = T_1 / T_p$ | Aceleración respecto a 1 proceso |
| Eficiencia | $E = S\,/\,p$ | Uso efectivo de los $p$ procesos |

> **Nota:** para obtener speedups reales ($S > 1$) se recomienda usar al menos $N = 50\,000$ puntos. Con cargas pequeñas, la latencia de comunicación MPI domina sobre el cómputo de rotación.