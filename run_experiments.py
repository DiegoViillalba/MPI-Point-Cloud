import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import os

def run_benchmark():
    # --- Configuración del Experimento ---
    objects = ["cube", "sphere", "sklearn"]
    # n_points = 50000  # Usamos muchos puntos para que el paralelismo valga la pena
    n_points = 1_000_000
    process_counts = [1, 2, 4, 8]
    csv_file = "benchmark_results.csv"

    # Limpiar benchmark anterior si existe
    if os.path.exists(csv_file):
        os.remove(csv_file)

    print(f"=== Iniciando Batería de Tests (N={n_points}) ===")

    for obj in objects:
        for p in process_counts:
            print(f"Ejecutando: {obj} con {p} procesos...")
            # Construir el comando
            cmd = [
                "uv", "run", "mpiexec", "-n", str(p), 
                "python", "mpi_rotation.py", 
                "--object", obj, 
                "--n", str(n_points), 
                "--performance"
            ]
            # Ejecutar y esperar a que termine
            subprocess.run(cmd, check=True)

    print(f"\n[!] Tests completados. Datos guardados en {csv_file}")
    plot_results(csv_file)

def plot_results(csv_path):
    df = pd.read_csv(csv_path)
    
    # Crear figura con dos subgráficas (Tiempo y Speedup)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for obj in df['objeto'].unique():
        subset = df[df['objeto'] == obj]
        
        # Referencia serial (T para p=1)
        t_serial = subset[subset['n_procesos'] == 1]['total_ms'].values[0]
        
        # Calcular Speedup (T_serial / T_paralelo)
        subset = subset.copy()
        subset['speedup'] = t_serial / subset['total_ms']

        # Gráfica 1: Tiempo Total
        ax1.plot(subset['n_procesos'], subset['total_ms'], marker='o', label=f'{obj}')
        
        # Gráfica 2: Speedup
        ax2.plot(subset['n_procesos'], subset['speedup'], marker='s', label=f'{obj}')

    # Configurar Eje de Tiempos
    ax1.set_title("Tiempo de Ejecución vs Procesos")
    ax1.set_xlabel("Número de Procesos (p)")
    ax1.set_ylabel("Tiempo Total (ms)")
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    # Configurar Eje de Speedup
    ax2.set_line = ax2.plot(df['n_procesos'].unique(), df['n_procesos'].unique(), 
                            color='black', linestyle=':', label='Ideal')
    ax2.set_title("Speedup (Escalabilidad)")
    ax2.set_xlabel("Número de Procesos (p)")
    ax2.set_ylabel("Speedup (T1 / Tp)")
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()

    plt.tight_layout()
    plt.savefig("performance_charts.png")
    print("[!] Gráficas guardadas en 'performance_charts.png'")
    plt.show()

if __name__ == "__main__":
    run_benchmark()