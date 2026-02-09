# Calcul de l'ensemble de Mandelbrot en python
import numpy as np
from dataclasses import dataclass
from PIL import Image
from math import log
from time import time
import matplotlib.cm


class MandelbrotSet:

    def __init__(self, max_iterations : int, escape_radius : float = 2. ):
        self.max_iterations = max_iterations
        self.escape_radius  = escape_radius

    def __contains__(self, c: complex) -> bool:
        return self.stability(c) == 1

    def convergence(self, c: np.ndarray, smooth=False, clamp=True) -> np.ndarray:
        value = self.count_iterations(c, smooth)/self.max_iterations
        return np.maximum(0.0, np.minimum(value, 1.0)) if clamp else value

    def count_iterations(self, c: np.ndarray,  smooth=False) -> np.ndarray:
        z:    np.ndarray
        iter: np.ndarray

        # On vérifie dans un premier temps si le complexe
        # n'appartient pas à une zone de convergence connue :
        #   1. Appartenance aux disques  C0{(0,0),1/4} et C1{(-1,0),1/4}
        iter = self.max_iterations * np.ones(c.shape, dtype=np.double)
        mask = (np.abs(c) >= 0.25) | (np.abs(c+1.) >= 0.25)
        #  2.  Appartenance à la cardioïde {(1/4,0),1/2(1-cos(theta))}
        #if (c.real > -0.75) and (c.real < 0.5):
        #    ct = c.real-0.25 + 1.j * c.imag
        #    ctnrm2 = abs(ct)
        #    if ctnrm2 < 0.5*(1-ct.real/max(ctnrm2, 1.E-14)):
        #        return self.max_iterations
        # Sinon on itère
        from scipy import linalg as slinalg
        z = np.zeros(c.shape, dtype=np.complex128)
        for it in range(self.max_iterations):
            z[mask] = z[mask]*z[mask] + c[mask]
            has_diverged = np.abs(z) > self.escape_radius
            if has_diverged.size > 0:
                iter[has_diverged] = np.minimum(iter[has_diverged], it)
                mask = mask & ~has_diverged
            if np.any(mask) == False : break
        has_diverged = np.abs(z) > 2
        if smooth:
            iter[has_diverged] += 1 - np.log(np.log(np.abs(z[has_diverged])))/log(2)
        return iter

# On peut changer les paramètres des deux prochaines lignes
mandelbrot_set = MandelbrotSet(max_iterations=200, escape_radius=2.)
width, height = 1024, 1024

scaleX = 3./width
scaleY = 2.25/height
convergence = np.empty((width, height), dtype=np.double)
# Calcul de l'ensemble de mandelbrot :
deb = time()
for y in range(height):
    #for x in range(width):
    c = np.array([complex(-2. + scaleX*x, -1.125 + scaleY * y) for x in range(width)])
    convergence[:, y] = mandelbrot_set.convergence(c, smooth=True)
fin = time()
print(f"Temps du calcul de l'ensemble de Mandelbrot : {fin-deb}")

# Constitution de l'image résultante :
deb = time()
image = Image.fromarray(np.uint8(matplotlib.cm.plasma(convergence.T)*255))
fin = time()
print(f"Temps de constitution de l'image : {fin-deb}")
image.show()
