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

N = 4800  # Divisible por 1,2,3,4,6,8
N_loc = N // size

# --- 1. Inicialización ---
# Rango de filas globales
i_start = rank * N_loc
i_end   = i_start + N_loc

# A_local tendrá forma (N_loc, N)
# Generamos solo nuestras filas con numpy vectorizado (mucho más rápido)
i_indices = np.arange(i_start, i_end).reshape(-1, 1)  # columna de índices i globales
j_indices = np.arange(N).reshape(1, -1)  # fila de índices j
A_local = ((i_indices + j_indices) % N + 1).astype(np.float64)

# Vector u
u = np.arange(1, N+1, dtype=np.float64)

comm.Barrier()
t_start = MPI.Wtime()

# --- 2. Cálculo Local ---
# (N_loc x N) @ (N) -> (N_loc)
# Obtenemos un trozo del vector final
v_local_part = np.dot(A_local, u)

t_compute = MPI.Wtime()

# --- 3. Comunicación (Allgather) ---
# Juntamos todas las partes.
# Allgather espera un buffer de recepción del tamaño TOTAL
v_final = np.zeros(N, dtype=np.float64)

# Nota: Allgather en mpi4py con numpy requiere tipos cuidadosos
# v_local_part -> sendbuf, v_final -> recvbuf
comm.Allgather(v_local_part, v_final)

t_end = MPI.Wtime()

if rank == 0:
    print(f"np={size}: Cómputo={t_compute-t_start:.6f}s, Comm={t_end-t_compute:.6f}s, Total={t_end-t_start:.6f}s")