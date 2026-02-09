import numpy as np
from dataclasses import dataclass
from PIL import Image
from math import log
from time import time
import matplotlib.cm
from mpi4py import MPI

@dataclass
class MandelbrotSet:
    max_iterations: int
    escape_radius:  float = 2.0

    def __contains__(self, c: complex) -> bool:
        return self.stability(c) == 1

    def convergence(self, c: complex, smooth=False, clamp=True) -> float:
        value = self.count_iterations(c, smooth)/self.max_iterations
        return max(0.0, min(value, 1.0)) if clamp else value

    def count_iterations(self, c: complex,  smooth=False) -> int | float:
        z:    complex
        iter: int

        # Short-circuit points known to converge (bulbs and main cardioid)
        if c.real*c.real+c.imag*c.imag < 0.0625:
            return self.max_iterations
        if (c.real+1)*(c.real+1)+c.imag*c.imag < 0.0625:
            return self.max_iterations
        if (c.real > -0.75) and (c.real < 0.5):
            ct = c.real-0.25 + 1.j * c.imag
            ctnrm2 = abs(ct)
            if ctnrm2 < 0.5*(1-ct.real/max(ctnrm2, 1.E-14)):
                return self.max_iterations
        
        # Escape-time iteration
        z = 0
        for iter in range(self.max_iterations):
            z = z*z + c
            if abs(z) > self.escape_radius:
                if smooth:
                    return iter + 1 - log(log(abs(z)))/log(2)
                return iter
        return self.max_iterations

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

mandelbrot_set = MandelbrotSet(max_iterations=50, escape_radius=10)
width, height = 1024, 1024
scaleX = 3./width
scaleY = 2.25/height

# Cyclic row assignment: rank, rank + size, rank + 2*size, ...
# Example with 4 ranks: rank 0 computes rows 0, 4, 8, 12, ...

my_rows = list(range(rank, height, size))
num_local_rows = len(my_rows)

# Local buffer stores rows compactly; order matches my_rows
local_convergence = np.empty((width, num_local_rows), dtype=np.double)

comm.Barrier()
deb = MPI.Wtime()

print(f"Rank {rank}: Iniciando cálculo cíclico ({num_local_rows} filas)...")

for local_index, global_y in enumerate(my_rows):
    for x in range(width):
        c = complex(-2. + scaleX*x, -1.125 + scaleY * global_y)
        local_convergence[x, local_index] = mandelbrot_set.convergence(c, smooth=True)

# Gather row blocks to rank 0; each entry keeps the sender's cyclic order
gathered_data = comm.gather(local_convergence, root=0)

fin = MPI.Wtime()

if rank == 0:
    print(f"Tiempo total (Cíclico): {fin-deb:.4f} s con {size} procesos.")
    
    full_convergence = np.empty((width, height), dtype=np.double)
    
    # Reorder cyclic rows into their global positions
    for p in range(size):
        process_data = gathered_data[p] # shape (width, num_rows_p)
        rows_for_p = range(p, height, size)
        
        for i, global_y in enumerate(rows_for_p):
            full_convergence[:, global_y] = process_data[:, i]

    image = Image.fromarray(np.uint8(matplotlib.cm.plasma(full_convergence.T)*255))
    image.save("mandelbrot_cyclic.png")
    print("Imagen guardada.")
