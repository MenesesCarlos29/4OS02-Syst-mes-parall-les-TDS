# Simulation d'une galaxie à n corps en utilisant une grille spatiale pour accélérer le calcul des forces gravitationnelles.
#     On crée une classe représentant le système de corps avec la méthode d'intégration basée sur une grille.
# On utilise numba pour accélérer les calculs.
import numpy as np
import visualizer3d
import sys
import time
from numba import njit, prange

try:
    from mpi4py import MPI
except ImportError as exc:
    raise SystemExit(
        "mpi4py est requis pour cette etape. Installez-le avec: python3 -m pip install mpi4py"
    ) from exc

# Unités:
# - Distance: année-lumière (ly)
# - Masse: masse solaire (M_sun)
# - Vitesse: année-lumière par an (ly/an)
# - Temps: année

# Constante gravitationnelle en unités [ly^3 / (M_sun * an^2)]
G = 1.560339e-13

def generate_star_color(mass : float) -> tuple[int, int, int]:
    """
    Génère une couleur pour une étoile en fonction de sa masse.
    Les étoiles massives sont bleues, les moyennes sont jaunes, les petites sont rouges.
    
    Parameters:
    -----------
    mass : float
        Masse de l'étoile en masses solaires
    
    Returns:
    --------
    color : tuple
        Couleur RGB (R, G, B) avec des valeurs entre 0 et 255
    """
    if mass > 5.0:
        # Étoiles massives: bleu-blanc
        return (150, 180, 255)
    elif mass > 2.0:
        # Étoiles moyennes-massives: blanc
        return (255, 255, 255)
    elif mass >= 1.0:
        # Étoiles comme le Soleil: jaune
        return (255, 255, 200)
    else:
        # Étoiles de faible masse: rouge-orange
        return (255, 150, 100)

@njit
def update_stars_in_grid( cell_start_indices : np.ndarray, body_indices : np.ndarray,
                          cell_masses : np.ndarray, cell_com_positions : np.ndarray,
                          masses: np.ndarray,
                          positions : np.ndarray, grid_min : np.ndarray, grid_max : np.ndarray,
                          cell_size : np.ndarray, n_cells : np.ndarray):
    n_bodies = positions.shape[0]
    # Réinitialise les compteurs de début des cellules
    cell_start_indices.fill(-1)
    # Compte le nombre de corps dans chaque cellule
    cell_counts = np.zeros(shape=(np.prod(n_cells),), dtype=np.int64)
    for ibody in range(n_bodies):
        cell_idx = np.floor((positions[ibody] - grid_min) / cell_size).astype(np.int64)
        # Gère le cas où un corps est exactement sur la borne max   
        for i in range(3):
            if cell_idx[i] >= n_cells[i]:
                cell_idx[i] = n_cells[i] - 1
            elif cell_idx[i] < 0:
                cell_idx[i] = 0
        morse_idx = cell_idx[0] + cell_idx[1]*n_cells[0] + cell_idx[2]*n_cells[0]*n_cells[1]
        cell_counts[morse_idx] += 1
    # Calcule les indices de début des cellules
    running_index = 0
    for i in range(len(cell_counts)):
        cell_start_indices[i] = running_index
        running_index += cell_counts[i]
    cell_start_indices[len(cell_counts)] = running_index # Fin du dernier corps
    # Remplit les indices des corps dans les cellules
    current_counts = np.zeros(shape=(np.prod(n_cells),), dtype=np.int64)
    for ibody in range(n_bodies):
        cell_idx = np.floor((positions[ibody] - grid_min) / cell_size).astype(np.int64)
        for i in range(3):
            if cell_idx[i] >= n_cells[i]:
                cell_idx[i] = n_cells[i] - 1
            elif cell_idx[i] < 0:
                cell_idx[i] = 0
        morse_idx = cell_idx[0] + cell_idx[1]*n_cells[0] + cell_idx[2]*n_cells[0]*n_cells[1]
        index_in_cell = cell_start_indices[morse_idx] + current_counts[morse_idx]
        body_indices[index_in_cell] = ibody
        current_counts[morse_idx] += 1
    # Maintenant, on peut calculer le centre de masse et la masse totale de chaque cellule
    for i in range(len(cell_counts)):
        cell_mass = 0.0
        com_position = np.zeros(3, dtype=np.float32)
        start_idx = cell_start_indices[i]
        end_idx = cell_start_indices[i+1]
        for j in range(start_idx, end_idx):
            ibody = body_indices[j]
            m = masses[ibody] 
            cell_mass += m
            com_position += positions[ibody] * m
        if cell_mass > 0.0:
            com_position /= cell_mass
        # Stocke les résultats dans des tableaux globaux
        cell_masses[i] = cell_mass
        cell_com_positions[i] = com_position

@njit(parallel=True)
def compute_acceleration( positions : np.ndarray, masses : np.ndarray,
                          cell_start_indices : np.ndarray, body_indices : np.ndarray,
                          cell_masses : np.ndarray, cell_com_positions : np.ndarray,
                          grid_min : np.ndarray, grid_max : np.ndarray,
                          cell_size : np.ndarray, n_cells : np.ndarray):
    n_bodies = positions.shape[0]
    a = np.zeros_like(positions)
    for ibody in prange(n_bodies):
        pos = positions[ibody]
        cell_idx = np.floor((pos - grid_min) / cell_size).astype(np.int64)
        for i in range(3):
            if cell_idx[i] >= n_cells[i]:
                cell_idx[i] = n_cells[i] - 1
            elif cell_idx[i] < 0:
                cell_idx[i] = 0
        # Parcourt toutes les cellules pour calculer la contribution gravitationnelle
        for ix in range(n_cells[0]):
            for iy in range(n_cells[1]):
                for iz in range(n_cells[2]):
                    morse_idx = ix + iy*n_cells[0] + iz*n_cells[0]*n_cells[1]
                    if (abs(ix-cell_idx[0]) > 2) or (abs(iy-cell_idx[1]) > 2) or (abs(iz-cell_idx[2]) > 2):
                        cell_com = cell_com_positions[morse_idx]    
                        cell_mass = cell_masses[morse_idx]
                        if cell_mass > 0.0:
                            direction = cell_com - pos
                            distance = np.sqrt(direction[0]**2 + direction[1]**2 + direction[2]**2)
                            if distance > 1.E-10:
                                inv_dist3 = 1.0 / (distance ** 3)
                                a[ibody,:] += G * direction[:] * inv_dist3 * cell_mass
                    else:
                        # Parcourt les corps dans cette cellule
                        start_idx = cell_start_indices[morse_idx]
                        end_idx = cell_start_indices[morse_idx+1]
                        for j in range(start_idx, end_idx):
                            jbody = body_indices[j]
                            if jbody != ibody:
                                direction = positions[jbody] - pos
                                distance = np.sqrt(direction[0]**2 + direction[1]**2 + direction[2]**2)
                                if distance > 1.E-10:
                                    inv_dist3 = 1.0 / (distance ** 3)
                                    a[ibody,:] += G * direction[:] * inv_dist3 * masses[jbody]
    return a

# On crée une grille cartésienne régulière pour diviser l'espace englobant la galaxie en cellules
class SpatialGrid:
    """_summary_
    """
    def __init__(self, positions : np.ndarray, nb_cells_per_dim : tuple[int, int, int]):
        self.min_bounds = np.min(positions, axis=0) - 1.E-6
        self.max_bounds = np.max(positions, axis=0) + 1.E-6
        self.n_cells = np.array(nb_cells_per_dim)
        self.cell_size = (self.max_bounds - self.min_bounds) / self.n_cells
        # On va stocker les indices des corps dans chaque cellule adéquate
        # Les cellules seront stockées sous une forme morse : indice de la cellule = ix + iy*n_cells_x + iz*n_cells_x*n_cells_y
        # et on gère deux tableaux : un pour le début des indices de chaque cellule, un autre pour les indices des corps
        self.cell_start_indices = np.full(np.prod(self.n_cells) + 1, -1, dtype=np.int64)
        self.body_indices = np.empty(shape=(positions.shape[0],), dtype=np.int64)
        # Stockage du centre de masse de chaque cellule et de la masse totale contenue dans chaque cellule
        self.cell_masses = np.zeros(shape=(np.prod(self.n_cells),), dtype=np.float32)
        self.cell_com_positions = np.zeros(shape=(np.prod(self.n_cells), 3), dtype=np.float32)
        
    def update_bounds(self, positions : np.ndarray):
        self.min_bounds = np.min(positions, axis=0) - 1.E-6
        self.max_bounds = np.max(positions, axis=0) + 1.E-6
        self.cell_size = (self.max_bounds - self.min_bounds) / self.n_cells
        
    def update(self, positions : np.ndarray, masses : np.ndarray):
        #self.update_bounds(positions)
        update_stars_in_grid( self.cell_start_indices, self.body_indices,                             
                              self.cell_masses, self.cell_com_positions,
                              masses,
                              positions, self.min_bounds, self.max_bounds,
                              self.cell_size, self.n_cells)

class NBodySystem:
    def __init__(self, filename, ncells_per_dir : tuple[int, int, int] = (10,10,10)):
        positions = []
        velocities = []
        masses    = []
        
        self.max_mass = 0.
        self.box = np.array([[-1.E-6,-1.E-6,-1.E-6],[1.E-6,1.E-6,1.E-6]], dtype=np.float64) # Contient les coins min et max du système
        with open(filename, "r") as fich:
            line = fich.readline() # Récupère la masse, la position et la vitesse sous forme de chaîne
            # Récupère les données numériques pour instancier un corps qu'on rajoute aux corps déjà présents :
            while line:
                data = line.split()
                masses.append(float(data[0]))
                positions.append([float(data[1]), float(data[2]), float(data[3])])
                velocities.append([float(data[4]), float(data[5]), float(data[6])])
                self.max_mass = max(self.max_mass, masses[-1])
                
                for i in range(3):
                    self.box[0][i] = min(self.box[0][i], positions[-1][i]-1.E-6)
                    self.box[1][i] = max(self.box[1][i], positions[-1][i]+1.E-6)
                    
                line = fich.readline()
        
        self.positions  = np.array(positions, dtype=np.float32)
        self.velocities = np.array(velocities, dtype=np.float32)
        self.masses     = np.array(masses, dtype=np.float32)
        self.colors = [generate_star_color(m) for m in masses]
        self.grid = SpatialGrid(self.positions, ncells_per_dir)
        self.grid.update(self.positions, self.masses)
        
    def update_positions(self, dt):
        """Applique la méthode de Verlet vectorisée pour mettre à jour les positions et vitesses des corps."""
        a = compute_acceleration( self.positions, self.masses,
                                  self.grid.cell_start_indices, self.grid.body_indices,
                                  self.grid.cell_masses, self.grid.cell_com_positions,
                                  self.grid.min_bounds, self.grid.max_bounds,
                                  self.grid.cell_size, self.grid.n_cells)
        self.positions += self.velocities * dt + 0.5 * a * dt * dt
        self.grid.update(self.positions, self.masses)
        a_new = compute_acceleration( self.positions, self.masses,
                                      self.grid.cell_start_indices, self.grid.body_indices,
                                      self.grid.cell_masses, self.grid.cell_com_positions,
                                      self.grid.min_bounds, self.grid.max_bounds,
                                      self.grid.cell_size, self.grid.n_cells)
        self.velocities += 0.5 * (a + a_new) * dt

system : NBodySystem

def update_positions(dt : float):
    global system
    system.update_positions(dt)
    return system.positions

def build_display_payload(system : NBodySystem):
    return {
        "positions": system.positions.copy(),
        "colors": np.array(system.colors, dtype=np.float32),
        "intensity": np.clip(system.masses / system.max_mass, 0.5, 1.0).astype(np.float32),
        "bounds": [
            [float(system.box[0][0]), float(system.box[1][0])],
            [float(system.box[0][1]), float(system.box[1][1])],
            [float(system.box[0][2]), float(system.box[1][2])],
        ],
    }


def warmup_numba(filename, ncells_per_dir : tuple[int, int, int], dt : float):
    warmup_system = NBodySystem(filename, ncells_per_dir=ncells_per_dir)
    warmup_system.update_positions(dt)


def worker_loop(comm, filename, ncells_per_dir : tuple[int, int, int], dt : float):
    global system
    warmup_numba(filename, ncells_per_dir, dt)
    system = NBodySystem(filename, ncells_per_dir=ncells_per_dir)
    comm.send(build_display_payload(system), dest=0)

    while True:
        command = comm.recv(source=0)
        if command == "step":
            system.update_positions(dt)
            comm.send(system.positions.copy(), dest=0)
        elif command == "stop":
            break


def run_visual_simulation(comm, payload, dt : float):
    visu = visualizer3d.Visualizer3D(
        payload["positions"],
        payload["colors"],
        payload["intensity"],
        payload["bounds"],
    )

    def remote_update(_dt):
        comm.send("step", dest=1)
        return comm.recv(source=1)

    try:
        visu.run(updater=remote_update, dt=dt)
    finally:
        comm.send("stop", dest=1)


def run_benchmark(comm, filename, ncells_per_dir : tuple[int, int, int], dt=0.001, n_steps=5):
    timings_ms = []
    for _ in range(n_steps):
        t0 = time.perf_counter()
        comm.send("step", dest=1)
        _ = comm.recv(source=1)
        t1 = time.perf_counter()
        timings_ms.append((t1 - t0) * 1000.0)

    avg_ms = sum(timings_ms) / len(timings_ms)
    print(f"Benchmark MPI display/calcul sur {filename}")
    print(f"dt = {dt}, grille = {ncells_per_dir}, iterations mesurees = {n_steps}")
    for i, timing_ms in enumerate(timings_ms, start=1):
        print(f"  Iteration {i}: {timing_ms:.3f} ms")
    print(f"Average step time (calcul + communication): {avg_ms:.3f} ms")
    comm.send("stop", dest=1)
    return avg_ms


def parse_arguments(argv):
    filename = "data/galaxy_1000"
    dt = 0.001
    n_cells_per_dir = (20, 20, 1)
    benchmark_steps = 0

    positionals = []
    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg == "--benchmark":
            benchmark_steps = 5
            if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                benchmark_steps = int(argv[i + 1])
                i += 1
        else:
            positionals.append(arg)
        i += 1

    if len(positionals) > 0:
        filename = positionals[0]
    if len(positionals) > 1:
        dt = float(positionals[1])
    if len(positionals) > 4:
        n_cells_per_dir = (int(positionals[2]), int(positionals[3]), int(positionals[4]))

    return filename, dt, n_cells_per_dir, benchmark_steps


def main():
    filename, dt, n_cells_per_dir, benchmark_steps = parse_arguments(sys.argv)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size != 2:
        if rank == 0:
            print("Cette etape doit etre executee avec exactement 2 processus MPI.")
            print("Exemple : mpirun -np 2 python3 -B 03_nbodies_grid_mpi_display_compute.py ...")
        return

    if rank == 0:
        payload = comm.recv(source=1)
        if benchmark_steps > 0:
            print(f"Benchmark MPI de {filename} avec dt = {dt} et grille {n_cells_per_dir}")
            run_benchmark(comm, filename, n_cells_per_dir, dt=dt, n_steps=benchmark_steps)
        else:
            print(f"Simulation MPI de {filename} avec dt = {dt} et grille {n_cells_per_dir}")
            run_visual_simulation(comm, payload, dt)
    else:
        worker_loop(comm, filename, n_cells_per_dir, dt)


if __name__ == "__main__":
    main()
    
