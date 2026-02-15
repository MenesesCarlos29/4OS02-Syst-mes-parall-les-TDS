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
TAG_GRID = 11     # envío de la grilla a dibujar
TAG_CTRL = 13     # handshake: continuar / parar (bool)
TAG_STATS = 14



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


def display_main(geometry, dim, pattern):
    """
    Rank 0: UI.
    - Inicializa grilla y la manda al rank 1
    - Loop: Recv(grid), procesa eventos, dibuja, send(ctrl)
    """
    grid = Grille(dim, init_pattern=pattern)
    comm.Send([grid.cells, MPI.UNSIGNED_CHAR], dest=1, tag=TAG_INIT)

    app = App(geometry, grid)

    warmup = 10
    it = 0

    sum_recv = 0.0
    sum_draw = 0.0
    sum_frame = 0.0
    n = 0

    bytes_msg = grid.cells.nbytes

    mustContinue = True

    while mustContinue:
        # Recibir el siguiente estado (bloqueante, Punto 1)
        t0 = MPI.Wtime()
        comm.Recv([grid.cells, MPI.UNSIGNED_CHAR], source=1, tag=TAG_GRID)
        t1 = MPI.Wtime()

        # Eventos
        for event in pg.event.get():
            if event.type == pg.QUIT:
                mustContinue = False

        # Dibujar (redraw completo)
        app.draw()
        t2 = MPI.Wtime()

        # Handshake: orden al rank1 (seguir/parar)
        comm.send(mustContinue, dest=1, tag=TAG_CTRL)
        t3 = MPI.Wtime()
        if it >= warmup:
            sum_recv += (t1 - t0)
            sum_draw += (t2 - t1)
            sum_frame += (t3 - t0)
            n += 1

        it += 1
    
    # recibe resumen de rank1 y muestra todo
    stats1 = comm.recv(source=1, tag=TAG_STATS)

    if n > 0:
        avg_recv = sum_recv / n
        avg_draw = sum_draw / n
        avg_frame = sum_frame / n
        fps = 1.0 / avg_frame if avg_frame > 0 else 0.0
    else:
        avg_recv = avg_draw = avg_frame = fps = 0.0

    print("\n===== Punto 1: métricas simples =====")
    print(f"Grid: {dim[0]}x{dim[1]} | msg={bytes_msg/1024:.1f} KiB | samples={n}")
    print(f"Rank0 (affichage): avg recv={avg_recv*1e3:.3f} ms | avg draw={avg_draw*1e3:.3f} ms | avg frame={avg_frame*1e3:.3f} ms | FPS≈{fps:.2f}")
    print(f"Rank1 (calcul):    avg compute={stats1['avg_compute_ms']:.3f} ms | avg send={stats1['avg_send_ms']:.3f} ms")


def compute_main(dim):
    """
    Rank 1: cálculo.
    - Recibe grilla inicial
    - Loop: compute -> Send(grid) -> Recv(ctrl)
    """
    cells = np.empty(dim, dtype=np.uint8)
    comm.Recv([cells, MPI.UNSIGNED_CHAR], source=0, tag=TAG_INIT)

    grid = Grille(dim, init_pattern=None)
    grid.cells[:, :] = cells

    warmup = 10
    it = 0

    sum_compute = 0.0
    sum_send = 0.0
    n = 0

    mustContinue = True
    while mustContinue:
        t0 = MPI.Wtime()
        grid.compute_next_iteration()
        t1 = MPI.Wtime()

        comm.Send([grid.cells, MPI.UNSIGNED_CHAR], dest=0, tag=TAG_GRID)
        t2 = MPI.Wtime()

        mustContinue = comm.recv(source=0, tag=TAG_CTRL)

        if it >= warmup:
            sum_compute += (t1 - t0)
            sum_send += (t2 - t1)
            n += 1

        it += 1
    avg_compute_ms = (sum_compute / n) * 1e3 if n > 0 else 0.0
    avg_send_ms = (sum_send / n) * 1e3 if n > 0 else 0.0

    comm.send(
        {"avg_compute_ms": avg_compute_ms, "avg_send_ms": avg_send_ms},
        dest=0,
        tag=TAG_STATS
    )

if __name__ == "__main__":
    import sys

    # Punto 1: exactamente 2 procesos
    if size != 2:
        if rank == 0:
            print("Punto 1 requiere 2 procesos: mpirun -n 2 python game_of_life.py <pattern> <resx> <resy>")
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
        print(f"Pattern initial choisi : {choice}")
        print(f"resolution ecran : {resx, resy}")
        pg.init()
        display_main(geometry, dim, pattern)
        pg.quit()
    else:
        compute_main(dim)
