"""
basic_life_convo.py
Le jeu de la vie
################
(igual que tu docstring)
"""

from mpi4py import MPI
import pygame as pg
import numpy as np

# --- MPI tags ---
TAG_INIT  = 10   # envoi initial de la grille
TAG_GRID  = 11   # envoi des états de grille (calcul -> affichage)
TAG_STOP  = 12   # stop propre (affichage -> calcul)
TAG_STATS = 14   # stats (calcul -> affichage)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class Grille:
    """
    Grille torique décrivant l'automate cellulaire.
    (igual que tu base)
    """
    def __init__(self, dim, init_pattern=None, color_life=pg.Color("black"), color_dead=pg.Color("white")):
        import random
        self.dimensions = dim
        if init_pattern is not None:
            self.cells = np.zeros(self.dimensions, dtype=np.uint8)  # uint8
            indices_i = [v[0] for v in init_pattern]
            indices_j = [v[1] for v in init_pattern]
            self.cells[indices_i, indices_j] = 1
        else:
            self.cells = np.random.randint(2, size=dim, dtype=np.uint8)
        self.col_life = color_life
        self.col_dead = color_dead

    @staticmethod
    def h(x):
        x[x <= 1] = -1
        x[x >= 4] = -1
        x[x == 2] = 0
        x[x == 3] = 1

    def compute_next_iteration(self):
        """
        Calcule la prochaine génération de cellules en suivant les règles du jeu de la vie
        """
        ny = self.dimensions[0]
        nx = self.dimensions[1]
        next_cells = np.zeros(self.dimensions, dtype=np.uint8)
        diff_cells = []

        # Convolution de base 2.
        from scipy.signal import convolve2d
        C = np.ones((3, 3), dtype=np.int8)
        C[1, 1] = 0
        voisins = convolve2d(self.cells, C, mode='same', boundary='wrap')
        Grille.h(voisins)

        # Point de vue continue
        temp = self.cells + voisins
        # Keep uint8 to match MPI buffer type (MPI.UNSIGNED_CHAR).
        next_cells = np.clip(temp, 0, 1).astype(np.uint8, copy=False)

        self.cells = next_cells
        return diff_cells


class App:
    """
    Cette classe décrit la fenêtre affichant la grille à l'écran
    (igual que tu base)
    """
    def __init__(self, geometry, grid):
        self.grid = grid
        self.size_x = geometry[1] // grid.dimensions[1]
        self.size_y = geometry[0] // grid.dimensions[0]
        if self.size_x > 4 and self.size_y > 4:
            self.draw_color = pg.Color('lightgrey')
        else:
            self.draw_color = None

        self.width = grid.dimensions[1] * self.size_x
        self.height = grid.dimensions[0] * self.size_y
        self.screen = pg.display.set_mode((self.width, self.height))
        self.canvas_cells = []

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


def display_main_async(geometry, dim, pattern):
    """
    Rank 0 (affichage) ASYNC:
    - envoie init au rank1
    - boucle UI:
        - events
        - Iprobe pour recevoir toutes les grilles dispo (sans bloquer)
        - draw du dernier état reçu
    - stop propre: envoie TAG_STOP puis récupère stats
    """
    grid = Grille(dim, init_pattern=pattern)
    comm.Send([grid.cells, MPI.UNSIGNED_CHAR], dest=1, tag=TAG_INIT)

    app = App(geometry, grid)

    clock = pg.time.Clock()
    mustContinue = True

    warmup = 30
    it = 0

    sum_draw = 0.0
    sum_frame = 0.0
    sum_poll = 0.0
    frames = 0
    updates = 0

    while mustContinue:
        frame_t0 = MPI.Wtime()

        # Events
        for event in pg.event.get():
            if event.type == pg.QUIT:
                mustContinue = False

        # Poll + drain messages (NO BLOQUEA si no hay nada)
        poll_t0 = MPI.Wtime()
        got = 0
        while comm.Iprobe(source=1, tag=TAG_GRID):
            comm.Recv([grid.cells, MPI.UNSIGNED_CHAR], source=1, tag=TAG_GRID)
            got += 1
        poll_t1 = MPI.Wtime()

        # Draw
        draw_t0 = MPI.Wtime()
        app.draw()
        draw_t1 = MPI.Wtime()

        # throttle suave para no quemar CPU (puedes subir/bajar)
        clock.tick(60)

        frame_t1 = MPI.Wtime()

        if it >= warmup:
            sum_poll += (poll_t1 - poll_t0)
            sum_draw += (draw_t1 - draw_t0)
            sum_frame += (frame_t1 - frame_t0)
            frames += 1
            updates += got

        it += 1

    # Stop propre (un seul message)
    comm.send(True, dest=1, tag=TAG_STOP)

    # Drain pending TAG_GRID messages so rank1 can complete any in-flight Isend.
    # Without this, rank1 may block in req.Wait() and both ranks can deadlock on exit.
    while not comm.Iprobe(source=1, tag=TAG_STATS):
        while comm.Iprobe(source=1, tag=TAG_GRID):
            comm.Recv([grid.cells, MPI.UNSIGNED_CHAR], source=1, tag=TAG_GRID)
        pg.time.wait(1)

    # Recibir stats del rank1
    stats = comm.recv(source=1, tag=TAG_STATS)

    if frames > 0:
        avg_poll = sum_poll / frames
        avg_draw = sum_draw / frames
        avg_frame = sum_frame / frames
        fps = 1.0 / avg_frame if avg_frame > 0 else 0.0
        upd_per_frame = updates / frames
    else:
        avg_poll = avg_draw = avg_frame = fps = upd_per_frame = 0.0

    print("\n===== Punto 3 (async) : métricas =====")
    print(f"Grid: {dim[0]}x{dim[1]} | frames(samples)={frames} | updates reçues={updates}")
    print(f"Rank0 (UI): avg poll+recv={avg_poll*1e3:.3f} ms | avg draw={avg_draw*1e3:.3f} ms | avg frame={avg_frame*1e3:.3f} ms | FPS≈{fps:.2f} | updates/frame≈{upd_per_frame:.2f}")
    print(f"Rank1 (calcul): avg compute={stats['avg_compute_ms']:.3f} ms | sends_ok={stats['sends_ok']} | drops={stats['drops']} | send_ok_rate≈{stats['send_ok_rate']:.2f}")


def compute_main_async(dim):
    """
    Rank 1 (calcul) ASYNC:
    - reçoit init
    - boucle:
        - check stop via Iprobe (non bloquant)
        - compute
        - Isend seulement si le send précédent est fini (sinon drop)
    """
    cells = np.empty(dim, dtype=np.uint8)
    comm.Recv([cells, MPI.UNSIGNED_CHAR], source=0, tag=TAG_INIT)

    grid = Grille(dim, init_pattern=None)
    grid.cells[:, :] = cells

    sendbuf = np.empty_like(grid.cells)
    req = None

    warmup = 50
    it = 0
    sum_compute = 0.0
    n_compute = 0

    sends_ok = 0
    drops = 0

    while True:
        # Stop check (non bloquant)
        if comm.Iprobe(source=0, tag=TAG_STOP):
            _ = comm.recv(source=0, tag=TAG_STOP)
            break

        # Compute
        t0 = MPI.Wtime()
        grid.compute_next_iteration()
        t1 = MPI.Wtime()

        if it >= warmup:
            sum_compute += (t1 - t0)
            n_compute += 1

        # Send attempt (non bloquant)
        ready = True
        if req is not None:
            ready = req.Test()

        if ready:
            # copy to stable buffer then Isend
            np.copyto(sendbuf, grid.cells)
            req = comm.Isend([sendbuf, MPI.UNSIGNED_CHAR], dest=0, tag=TAG_GRID)
            if it >= warmup:
                sends_ok += 1
        else:
            # drop this frame to avoid blocking
            if it >= warmup:
                drops += 1

        it += 1

    # Clean: wait last send (optional but propre)
    if req is not None:
        req.Wait()

    avg_compute_ms = (sum_compute / n_compute) * 1e3 if n_compute > 0 else 0.0
    total = (sends_ok + drops)
    send_ok_rate = (sends_ok / total) if total > 0 else 0.0

    comm.send(
        {
            "avg_compute_ms": avg_compute_ms,
            "sends_ok": sends_ok,
            "drops": drops,
            "send_ok_rate": send_ok_rate,
        },
        dest=0,
        tag=TAG_STATS
    )


if __name__ == '__main__':
    import sys

    # Punto 3 (async) : 2 processus (rank0 affichage, rank1 calcul)
    if size != 2:
        if rank == 0:
            print("Punto 3 requiere 2 procesos: mpirun -n 2 python3 game_of_life_vec.py <pattern> <resx> <resy>")
        comm.Abort(1)

    dico_patterns = {  # Dimension et pattern dans un tuple
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

    choice = 'glider'
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
        display_main_async(geometry, dim, pattern)
        pg.quit()
    else:
        compute_main_async(dim)
