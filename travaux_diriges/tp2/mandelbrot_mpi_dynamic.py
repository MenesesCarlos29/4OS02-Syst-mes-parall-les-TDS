import numpy as np
from dataclasses import dataclass
from PIL import Image
from math import log
from time import time
import matplotlib.cm
from mpi4py import MPI

# Tag values used for the master/worker protocol
TAG_WORK = 1
TAG_RESULT = 2
TAG_KILL = 3
CHUNK_SIZE = 32  # Trade-off between load balance and message overhead

@dataclass
class MandelbrotSet:
    max_iterations: int
    escape_radius:  float = 2.0

    def convergence(self, c: complex, smooth=False, clamp=True) -> float:
        value = self.count_iterations(c, smooth)/self.max_iterations
        return max(0.0, min(value, 1.0)) if clamp else value

    def count_iterations(self, c: complex,  smooth=False) -> int | float:
        # Short-circuit points known to converge (bulbs and main cardioid)
        if c.real*c.real + c.imag*c.imag < 0.0625:
            return self.max_iterations
        if (c.real+1)*(c.real+1) + c.imag*c.imag < 0.0625:
            return self.max_iterations
        if (c.real > -0.75) and (c.real < 0.5):
            ct = c.real - 0.25 + 1.j * c.imag
            ctnrm2 = abs(ct)
            if ctnrm2 < 0.5*(1 - ct.real/max(ctnrm2, 1.E-14)):
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

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
status = MPI.Status()

mandelbrot_set = MandelbrotSet(max_iterations=50, escape_radius=10)
width, height = 1024, 1024
scaleX = 3./width
scaleY = 2.25/height

# Master: distributes chunks and assembles the final image
if rank == 0:
    print(f"MAESTRO: Dinámico con {size-1} esclavos, chunks={CHUNK_SIZE}.")
    convergence = np.empty((width, height), dtype=np.double)
    
    # Queue of chunk start indices (each chunk spans CHUNK_SIZE rows)
    pending = list(range(0, height, CHUNK_SIZE))
    active_workers = 0
    
    # Receive buffer for chunk bounds: [y_start, y_end]
    info_buf = np.empty(2, dtype=np.int32)
    
    deb = MPI.Wtime()

    # 1) Seed workers with initial chunks
    for worker in range(1, size):
        if pending:
            y_start = pending.pop(0)
            y_end = min(y_start + CHUNK_SIZE, height)
            task_buf = np.array([y_start, y_end], dtype=np.int32)
            comm.Send(task_buf, dest=worker, tag=TAG_WORK)
            active_workers += 1
        else:
            kill_buf = np.array([-1, -1], dtype=np.int32)
            comm.Send(kill_buf, dest=worker, tag=TAG_KILL)

    # 2) Dynamic scheduling loop
    while active_workers > 0:
        # Receive completed chunk bounds
        comm.Recv(info_buf, source=MPI.ANY_SOURCE, tag=TAG_RESULT, status=status)
        source_rank = status.Get_source()
        y_start, y_end = info_buf[0], info_buf[1]
        chunk_height = y_end - y_start
        
        # Receive raw chunk data (direct buffer, no pickling)
        chunk_data = np.empty((width, chunk_height), dtype=np.double)
        comm.Recv(chunk_data, source=source_rank, tag=TAG_RESULT)
        
        # Place chunk into its global rows
        convergence[:, y_start:y_end] = chunk_data
        
        # Send more work or stop the worker
        if pending:
            next_y = pending.pop(0)
            next_end = min(next_y + CHUNK_SIZE, height)
            task_buf = np.array([next_y, next_end], dtype=np.int32)
            comm.Send(task_buf, dest=source_rank, tag=TAG_WORK)
        else:
            kill_buf = np.array([-1, -1], dtype=np.int32)
            comm.Send(kill_buf, dest=source_rank, tag=TAG_KILL)
            active_workers -= 1

    fin = MPI.Wtime()
    print(f"Tiempo total (Dinámico): {fin-deb:.4f} s.")

# Worker: receives chunks, computes rows, sends results back
else:
    task_buf = np.empty(2, dtype=np.int32)
    
    while True:
        # Receive chunk bounds
        comm.Recv(task_buf, source=0, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()
        y_start, y_end = task_buf[0], task_buf[1]

        if tag == TAG_KILL or y_start < 0:
            break

        if tag == TAG_WORK:
            chunk_height = y_end - y_start
            
            chunk_data = np.empty((width, chunk_height), dtype=np.double)
            
            for i, y in enumerate(range(y_start, y_end)):
                for x in range(width):
                    c = complex(-2. + scaleX*x, -1.125 + scaleY * y)
                    chunk_data[x, i] = mandelbrot_set.convergence(c, smooth=True)
            
            # Send bounds and data as raw buffers
            info_buf = np.array([y_start, y_end], dtype=np.int32)
            comm.Send(info_buf, dest=0, tag=TAG_RESULT)
            comm.Send(chunk_data, dest=0, tag=TAG_RESULT)
