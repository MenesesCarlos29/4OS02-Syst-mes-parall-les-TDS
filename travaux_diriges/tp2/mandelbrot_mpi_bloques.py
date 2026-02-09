# mandelbrot_mpi_bloques.py
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

        if c.real*c.real+c.imag*c.imag < 0.0625:
            return self.max_iterations
        if (c.real+1)*(c.real+1)+c.imag*c.imag < 0.0625:
            return self.max_iterations
        if (c.real > -0.75) and (c.real < 0.5):
            ct = c.real-0.25 + 1.j * c.imag
            ctnrm2 = abs(ct)
            if ctnrm2 < 0.5*(1-ct.real/max(ctnrm2, 1.E-14)):
                return self.max_iterations
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

# Block row decomposition: each rank owns a contiguous slice of rows
rows_per_process = height // size
start_row = rank * rows_per_process
end_row   = start_row + rows_per_process

if rank == size - 1:
    # Last rank absorbs any remainder rows
    end_row = height

# Local buffer stores this rank's rows in [x, local_y] layout
local_convergence = np.empty((width, end_row - start_row), dtype=np.double)

# Synchronize before timing to measure only the parallel section
comm.Barrier()
deb = MPI.Wtime()

print(f"Rank {rank}: Calculando filas {start_row} a {end_row}...")

for y in range(start_row, end_row):
    for x in range(width):
        c = complex(-2. + scaleX*x, -1.125 + scaleY * y)
        # Map global row y to local index y - start_row
        local_convergence[x, y - start_row] = mandelbrot_set.convergence(c, smooth=True)

# Gather all row blocks to rank 0
gathered_data = comm.gather(local_convergence, root=0)

fin = MPI.Wtime()

if rank == 0:
    print(f"Tiempo total de cálculo (paralelo): {fin-deb:.4f} segundos con {size} procesos.")
    
    # Rebuild the full image by concatenating row blocks along y
    full_convergence = np.concatenate(gathered_data, axis=1)
    
    image = Image.fromarray(np.uint8(matplotlib.cm.plasma(full_convergence.T)*255))
    # Save to file to avoid GUI dependencies
    image.save("mandelbrot_mpi.png")
    print("Imagen guardada como mandelbrot_mpi.png")
