import os
# IMPORTANTE: Limitar threads de BLAS ANTES de importar numpy
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
from mpi4py import MPI
from time import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Dimensión (N) - Debe ser lo suficientemente grande para amortizar overhead MPI
N = 4800  # Divisible por 1,2,3,4,6,8
if N % size != 0:
    if rank == 0: print("Error: N debe ser divisible por size.")
    exit()

N_loc = N // size

# --- 1. Inicialización de Datos (Distribuida) ---
# Cada proceso genera SOLO sus columnas de A para ahorrar memoria
# A_loc tendrá forma (N, N_loc)
# Fórmula: A_ij = (i+j) % N + 1.  (Ojo con los índices globales)

# El rango de columnas globales que maneja este proceso
j_start = rank * N_loc
j_end   = j_start + N_loc

# Generamos A localmente con numpy vectorizado (MUCHO más rápido)
i_indices = np.arange(N).reshape(-1, 1)  # columna de índices i
j_indices = np.arange(j_start, j_end).reshape(1, -1)  # fila de índices j globales
A_local = ((i_indices + j_indices) % N + 1).astype(np.float64)

# Vector u completo
u = np.arange(1, N+1, dtype=np.float64)

# Seleccionamos solo la parte de u que corresponde a mis columnas
u_local = u[j_start:j_end]

comm.Barrier()
t_start = MPI.Wtime()

# --- 2. Cálculo Local ---
# Producto punto: (N x N_loc) @ (N_loc) -> (N)
# Cada proceso obtiene un vector 'v' parcial de tamaño N
v_partial = np.dot(A_local, u_local)

t_compute = MPI.Wtime()

# --- 3. Comunicación (Allreduce) ---
# Sumamos todos los v_partial en v_final
v_final = np.zeros(N, dtype=np.float64)
comm.Allreduce(v_partial, v_final, op=MPI.SUM)

t_end = MPI.Wtime()

if rank == 0:
    print(f"np={size}: Cómputo={t_compute-t_start:.6f}s, Comm={t_end-t_compute:.6f}s, Total={t_end-t_start:.6f}s")