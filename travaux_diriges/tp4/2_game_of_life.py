"""
Le jeu de la vie
################
... (tu docstring igual) ...
"""

from mpi4py import MPI
import pygame as pg
import numpy as np

# --- MPI tags ---
TAG_INIT = 10     # envío inicial de la grilla
TAG_GRID = 11     # (no usado aquí)
TAG_CTRL = 13     # handshake: continuar / parar (bool)
TAG_STATS = 14
TAG_BLOCK = 15
TAG_UP = 20
TAG_DOWN = 21

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class Grille:
    def __init__(self, dim, init_pattern=None, color_life=pg.Color("black"), color_dead=pg.Color("white")):
        self.dimensions = dim
        if init_pattern is not None:
            self.cells = np.zeros(self.dimensions, dtype=np.uint8)
            indices_i = [v[0] for v in init_pattern]
            indices_j = [v[1] for v in init_pattern]
            self.cells[indices_i, indices_j] = 1
        else:
            self.cells = np.random.randint(2, size=dim, dtype=np.uint8)
        self.col_life = color_life
        self.col_dead = color_dead

    def compute_next_iteration(self):
        ny = self.dimensions[0]
        nx = self.dimensions[1]
        next_cells = np.empty(self.dimensions, dtype=np.uint8)
        diff_cells = []

        for i in range(ny):
            i_above = (i + ny - 1) % ny
            i_below = (i + 1) % ny
            for j in range(nx):
                j_left = (j - 1 + nx) % nx
                j_right = (j + 1) % nx

                voisins_i = [i_above, i_above, i_above, i, i, i_below, i_below, i_below]
                voisins_j = [j_left,  j,       j_right, j_left, j_right, j_left,  j,       j_right]
                voisines = np.array(self.cells[voisins_i, voisins_j])
                nb_voisines_vivantes = np.sum(voisines)

                if self.cells[i, j] == 1:
                    if (nb_voisines_vivantes < 2) or (nb_voisines_vivantes > 3):
                        next_cells[i, j] = 0
                        diff_cells.append(i * nx + j)
                    else:
                        next_cells[i, j] = 1
                elif nb_voisines_vivantes == 3:
                    next_cells[i, j] = 1
                    diff_cells.append(i * nx + j)
                else:
                    next_cells[i, j] = 0

        self.cells = next_cells
        return diff_cells


class App:
    def __init__(self, geometry, grid):
        self.grid = grid
        self.size_x = geometry[1] // grid.dimensions[1]
        self.size_y = geometry[0] // grid.dimensions[0]
        self.draw_color = pg.Color('lightgrey') if (self.size_x > 4 and self.size_y > 4) else None

        self.width = grid.dimensions[1] * self.size_x
        self.height = grid.dimensions[0] * self.size_y
        self.screen = pg.display.set_mode((self.width, self.height))

    def compute_rectangle(self, i: int, j: int):
        return (self.size_x * j, self.height - self.size_y * (i + 1), self.size_x, self.size_y)

    def compute_color(self, i: int, j: int):
        return self.grid.col_dead if self.grid.cells[i, j] == 0 else self.grid.col_life

    def draw(self):
        [self.screen.fill(self.compute_color(i, j), self.compute_rectangle(i, j))
         for i in range(self.grid.dimensions[0]) for j in range(self.grid.dimensions[1])]

        if self.draw_color is not None:
            [pg.draw.line(self.screen, self.draw_color, (0, i * self.size_y), (self.width, i * self.size_y))
             for i in range(self.grid.dimensions[0])]
            [pg.draw.line(self.screen, self.draw_color, (j * self.size_x, 0), (j * self.size_x, self.height))
             for j in range(self.grid.dimensions[1])]

        pg.display.update()


def display_main_p2(geometry, dim, pattern):
    ny, nx = dim
    ncalc = size - 1
    counts, starts = split_rows(ny, ncalc)

    # global grid solo para display
    grid = Grille(dim, init_pattern=pattern)

    # enviar bloques iniciales a ranks 1..size-1
    for r in range(1, size):
        k = r - 1
        start = starts[k]
        cnt = counts[k]
        block0 = np.ascontiguousarray(grid.cells[start:start+cnt, :])
        comm.Send([block0, MPI.UNSIGNED_CHAR], dest=r, tag=TAG_INIT)

    app = App(geometry, grid)
    mustContinue = True

    # ---------------- METRICS (rank 0) ----------------
    warmup = 10
    it = 0
    n = 0
    sum_gather = 0.0   # recv blocks + copy into global
    sum_draw = 0.0     # app.draw()
    sum_frame = 0.0    # whole loop iteration
    bytes_total = grid.cells.nbytes
    # --------------------------------------------------

    while mustContinue:
        t_frame0 = MPI.Wtime()

        # recibir blocks y reconstruir grilla global
        t_g0 = MPI.Wtime()
        for r in range(1, size):
            k = r - 1
            start = starts[k]
            cnt = counts[k]
            recv_block = np.empty((cnt, nx), dtype=np.uint8)
            comm.Recv([recv_block, MPI.UNSIGNED_CHAR], source=r, tag=TAG_BLOCK)
            grid.cells[start:start+cnt, :] = recv_block
        t_g1 = MPI.Wtime()

        # eventos
        for event in pg.event.get():
            if event.type == pg.QUIT:
                mustContinue = False

        # draw
        t_d0 = MPI.Wtime()
        app.draw()
        t_d1 = MPI.Wtime()

        # mandar ctrl a todos los ranks de cálculo
        for r in range(1, size):
            comm.send(mustContinue, dest=r, tag=TAG_CTRL)

        t_frame1 = MPI.Wtime()

        if it >= warmup:
            sum_gather += (t_g1 - t_g0)
            sum_draw += (t_d1 - t_d0)
            sum_frame += (t_frame1 - t_frame0)
            n += 1
        it += 1

    # ---------------- METRICS: gather stats from compute ranks ----------------
    stats_list = []
    for r in range(1, size):
        stats_list.append(comm.recv(source=r, tag=TAG_STATS))

    # print metrics
    if n > 0:
        avg_gather_ms = (sum_gather / n) * 1e3
        avg_draw_ms = (sum_draw / n) * 1e3
        avg_frame_ms = (sum_frame / n) * 1e3
        fps = 1.0 / (sum_frame / n) if (sum_frame / n) > 0 else 0.0
    else:
        avg_gather_ms = avg_draw_ms = avg_frame_ms = fps = 0.0

    print("\n===== Punto 2: métricas simples =====")
    print(f"Procs: total={size} | calcul={size-1} | Grid={ny}x{nx} | msg_total={bytes_total/1024:.1f} KiB | samples={n}")
    print(f"Rank0 (affichage): avg gather={avg_gather_ms:.3f} ms | avg draw={avg_draw_ms:.3f} ms | avg frame={avg_frame_ms:.3f} ms | FPS≈{fps:.2f}")

    # compute-side summary
    if len(stats_list) > 0:
        def mean(key):
            vals = [s[key] for s in stats_list]
            return sum(vals) / len(vals)

        # simple mean over compute ranks
        avg_halo_ms = mean("avg_halo_ms")
        avg_compute_ms = mean("avg_compute_ms")
        avg_send_ms = mean("avg_send_ms")
        avg_ctrl_ms = mean("avg_ctrl_ms")

        print(f"Calc ranks (mean): avg halo={avg_halo_ms:.3f} ms | avg compute={avg_compute_ms:.3f} ms | avg send(block)={avg_send_ms:.3f} ms | avg ctrl(wait)={avg_ctrl_ms:.3f} ms")

        # optional: per-rank lines (compact)
        for s in stats_list:
            print(f"  rank {s['rank']:>2} | rows={s['local_ny']:<4} | halo={s['avg_halo_ms']:.3f} ms | compute={s['avg_compute_ms']:.3f} ms | send={s['avg_send_ms']:.3f} ms | ctrl={s['avg_ctrl_ms']:.3f} ms | samples={s['samples']}")


def compute_main_p2(dim):
    ny, nx = dim
    ncalc = size - 1
    counts, starts = split_rows(ny, ncalc)

    k = rank - 1                 # index en [0..ncalc-1]
    local_ny = counts[k]

    # recibir bloque inicial (sin ghosts)
    real0 = np.empty((local_ny, nx), dtype=np.uint8)
    comm.Recv([real0, MPI.UNSIGNED_CHAR], source=0, tag=TAG_INIT)

    # local con ghosts
    local = np.empty((local_ny + 2, nx), dtype=np.uint8)
    local[1:local_ny+1, :] = real0

    # vecinos en anillo de ranks de cálculo (1..size-1)
    if ncalc > 1:
        up_rank   = 1 + ((k - 1) % ncalc)
        down_rank = 1 + ((k + 1) % ncalc)
    else:
        up_rank = down_rank = rank  # no se usa (caso 1 solo compute)

    # ---------------- METRICS (compute ranks) ----------------
    warmup = 10
    it = 0
    n = 0
    sum_halo = 0.0
    sum_compute = 0.0
    sum_send = 0.0
    sum_ctrl = 0.0
    # ---------------------------------------------------------

    mustContinue = True
    while mustContinue:
        # --- halo exchange ---
        t0 = MPI.Wtime()
        if ncalc == 1:
            # wrap vertical dentro del mismo bloque
            local[0, :] = local[local_ny, :]
            local[local_ny+1, :] = local[1, :]
        else:
            comm.Sendrecv(
                sendbuf=local[1, :], dest=up_rank, sendtag=TAG_UP,
                recvbuf=local[local_ny + 1, :], source=down_rank, recvtag=TAG_UP
            )
            comm.Sendrecv(
                sendbuf=local[local_ny, :], dest=down_rank, sendtag=TAG_DOWN,
                recvbuf=local[0, :], source=up_rank, recvtag=TAG_DOWN
            )
        t1 = MPI.Wtime()

        # --- compute local ---
        next_real = step_local(local, local_ny, nx)
        local[1:local_ny+1, :] = next_real
        t2 = MPI.Wtime()

        # --- enviar bloque actualizado al rank0 ---
        comm.Send([next_real, MPI.UNSIGNED_CHAR], dest=0, tag=TAG_BLOCK)
        t3 = MPI.Wtime()

        # --- ctrl ---
        mustContinue = comm.recv(source=0, tag=TAG_CTRL)
        t4 = MPI.Wtime()

        if it >= warmup:
            sum_halo += (t1 - t0)
            sum_compute += (t2 - t1)
            sum_send += (t3 - t2)
            sum_ctrl += (t4 - t3)
            n += 1
        it += 1

    # send stats to rank0
    if n > 0:
        avg_halo_ms = (sum_halo / n) * 1e3
        avg_compute_ms = (sum_compute / n) * 1e3
        avg_send_ms = (sum_send / n) * 1e3
        avg_ctrl_ms = (sum_ctrl / n) * 1e3
    else:
        avg_halo_ms = avg_compute_ms = avg_send_ms = avg_ctrl_ms = 0.0

    comm.send(
        {
            "rank": rank,
            "local_ny": int(local_ny),
            "avg_halo_ms": float(avg_halo_ms),
            "avg_compute_ms": float(avg_compute_ms),
            "avg_send_ms": float(avg_send_ms),
            "avg_ctrl_ms": float(avg_ctrl_ms),
            "samples": int(n),
        },
        dest=0,
        tag=TAG_STATS
    )


def step_local(local: np.ndarray, local_ny: int, nx: int) -> np.ndarray:
    """
    local tiene shape (local_ny+2, nx)
    filas reales: 1..local_ny
    ghosts: 0 y local_ny+1
    devuelve next_real con shape (local_ny, nx)
    """
    next_real = np.empty((local_ny, nx), dtype=np.uint8)

    for i in range(1, local_ny + 1):
        for j in range(nx):
            # contar 8 vecinas usando ghosts + wrap horizontal
            s = 0
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    s += local[i + di, (j + dj) % nx]

            cell = local[i, j]
            if cell == 1:
                next_real[i - 1, j] = 0 if (s < 2 or s > 3) else 1
            else:
                next_real[i - 1, j] = 1 if s == 3 else 0

    return next_real


def split_rows(ny: int, ncalc: int):
    base = ny // ncalc
    rem  = ny % ncalc
    counts = [base + (1 if k < rem else 0) for k in range(ncalc)]
    starts = [0] * ncalc
    s = 0
    for k in range(ncalc):
        starts[k] = s
        s += counts[k]
    return counts, starts


if __name__ == "__main__":
    import sys

    if size < 2:
        if rank == 0:
            print("Punto 2 requiere al menos 2 procesos: mpirun -n <>=2 python game_of_life.py ...")
        comm.Abort(1)

    dico_patterns = {
        'blinker': ((5, 5), [(2, 1), (2, 2), (2, 3)]),
        'toad': ((6, 6), [(2, 2), (2, 3), (2, 4), (3, 3), (3, 4), (3, 5)]),
        "acorn": ((100, 100), [(51, 52), (52, 54), (53, 51), (53, 52), (53, 55), (53, 56), (53, 57)]),
        "beacon": ((6, 6), [(1, 3), (1, 4), (2, 3), (2, 4), (3, 1), (3, 2), (4, 1), (4, 2)]),
        "boat": ((5, 5), [(1, 1), (1, 2), (2, 1), (2, 3), (3, 2)]),
        "glider": ((100, 90), [(1, 1), (2, 2), (2, 3), (3, 1), (3, 2)]),
        "glider_gun": ((400, 400), [(51,76),(52,74),(52,76),(53,64),(53,65),(53,72),(53,73),(53,86),(53,87),
                                     (54,63),(54,67),(54,72),(54,73),(54,86),(54,87),(55,52),(55,53),(55,62),
                                     (55,68),(55,72),(55,73),(56,52),(56,53),(56,62),(56,66),(56,68),(56,69),
                                     (56,74),(56,76),(57,62),(57,68),(57,76),(58,63),(58,67),(59,64),(59,65)]),
        "space_ship": ((25, 25), [(11,13),(11,14),(12,11),(12,12),(12,14),(12,15),(13,11),(13,12),(13,13),(13,14),(14,12),(14,13)]),
        "die_hard": ((100, 100), [(51,57),(52,51),(52,52),(53,52),(53,56),(53,57),(53,58)]),
        "pulsar": ((17, 17), [(2,4),(2,5),(2,6),(7,4),(7,5),(7,6),(9,4),(9,5),(9,6),(14,4),(14,5),(14,6),
                              (2,10),(2,11),(2,12),(7,10),(7,11),(7,12),(9,10),(9,11),(9,12),(14,10),(14,11),(14,12),
                              (4,2),(5,2),(6,2),(4,7),(5,7),(6,7),(4,9),(5,9),(6,9),(4,14),(5,14),(6,14),
                              (10,2),(11,2),(12,2),(10,7),(11,7),(12,7),(10,9),(11,9),(12,9),(10,14),(11,14),(12,14)]),
        "floraison": ((40, 40), [(19,18),(19,19),(19,20),(20,17),(20,19),(20,21),(21,18),(21,19),(21,20)]),
        "block_switch_engine": ((400, 400), [(201,202),(201,203),(202,202),(202,203),(211,203),(212,204),(212,202),
                                             (214,204),(214,201),(215,201),(215,202),(216,201)]),
        "u": ((200, 200), [(101,101),(102,102),(103,102),(103,101),(104,103),(105,103),(105,102),(105,101),
                           (105,105),(103,105),(102,105),(101,105),(101,104)]),
        "flat": ((200, 400), [(80,200),(81,200),(82,200),(83,200),(84,200),(85,200),(86,200),(87,200),
                              (89,200),(90,200),(91,200),(92,200),(93,200),(97,200),(98,200),(99,200),
                              (106,200),(107,200),(108,200),(109,200),(110,200),(111,200),(112,200),(114,200),
                              (115,200),(116,200),(117,200),(118,200)]),
    }

    choice = "glider"
    if len(sys.argv) > 1:
        choice = sys.argv[1]

    resx, resy = 800, 800
    if len(sys.argv) > 3:
        resx = int(sys.argv[2])
        resy = int(sys.argv[3])

    if choice not in dico_patterns:
        if rank == 0:
            print("No such pattern. Available ones are:", dico_patterns.keys())
        comm.Abort(1)

    dim, pattern = dico_patterns[choice]
    geometry = (resx, resy)

    if rank == 0:
        pg.init()
        display_main_p2(geometry, dim, pattern)
        pg.quit()
    else:
        compute_main_p2(dim)
