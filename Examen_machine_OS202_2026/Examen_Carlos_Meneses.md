# Examen_Systemes_Paralleles
MENESES GAMBOA Carlos Adrian

## Etape 1 - Analyse initiale

### Question preliminaire

Il n'est pas pertinent de choisir une valeur de `N_k` strictement supérieure a 1, car la galaxie simulee est essentiellement plate dans le plan `Oxy` et tres peu etendue suivant `Oz`. Decouper davantage l'axe `z` produirait donc surtout des cellules presque vides, ce qui augmenterait le cout de gestion de la grille sans apporter de gain significatif en precision ou en performance.

### Mesure du temps initial

Commande utilisee :

```bash
python3 -B 01_nbodies_grid_numba_base.py data/galaxy_5000 0.0015 15 15 1
```

Mesures observees :

| Essai | Render time (ms) | Update time (ms) |
|---|---:|---:|
| 1 | 17 | 1152 |
| 2 | 16 | 1172 |
| 3 | 15 | 1170 |
| 4 | 17 | 1119 |
| 5 | 16 | 1202 |
| Moyenne | 16.2 | 1163.0 |

### Conclusion

La partie la plus interessante a paralleliser est clairement le calcul des trajectoires, c'est-a-dire la phase `Update`. En effet, le temps moyen de rendu est d'environ `16.2 ms`, alors que le temps moyen de mise a jour est d'environ `1163 ms`, soit un cout tres largement dominant pour le calcul. La suite du travail se concentrera donc sur l'acceleration de cette partie avec Numba puis MPI.

## Etape 2 - Parallelisation en Numba

### Modifications apportees

Pour cette etape, j'ai applique une parallelisation minimale et ciblee sur la partie dominante du calcul. Dans le fichier `02_nbodies_grid_numba_parallel.py`, la fonction `compute_acceleration` a ete modifiee en remplacant `@njit` par `@njit(parallel=True)` et en remplacant la boucle externe sur les corps par `prange`. Le reste de l'algorithme a ete conserve au maximum afin de limiter les modifications du code fourni.

### Methode de mesure

Afin de mesurer correctement l'effet de la parallelisation Numba, les mesures ont ete realisees sans affichage graphique, a l'aide du mode benchmark. Ce choix permet d'isoler le cout du calcul des trajectoires et d'eviter de biaiser les resultats par le rendu graphique.

Commande type utilisee :

```bash
NUMBA_NUM_THREADS=<p> python3 -B 02_nbodies_grid_numba_parallel.py data/galaxy_5000 0.0015 15 15 1 --benchmark 5
```

Le jeu de donnees utilise est `data/galaxy_5000`, avec `dt = 0.0015` et une grille `(15, 15, 1)`. Une iteration d'echauffement est effectuee avant la mesure afin d'exclure le cout de compilation JIT.

### Resultats

| Threads | Temps moyen update (ms) | Speedup | Efficacite |
|---|---:|---:|---:|
| 1 | 233.964 | 1.000 | 1.000 |
| 2 | 150.580 | 1.554 | 0.777 |
| 4 | 118.047 | 1.982 | 0.496 |
| 8 | 80.685 | 2.900 | 0.362 |

Avec :

- `Speedup = T1 / Tp`
- `Efficacite = Speedup / p`

### Analyse

On observe que la parallelisation avec Numba apporte bien une acceleration du calcul. Le temps moyen de mise a jour passe de `233.964 ms` avec un seul thread a `80.685 ms` avec `8` threads, soit un speedup d'environ `2.90`.

L'acceleration obtenue augmente donc avec le nombre de threads, mais elle n'est pas lineaire. Ce comportement est normal, car seule la partie `compute_acceleration` a ete parallelisee. D'autres portions du code, notamment la mise a jour de la grille et certaines operations de gestion des donnees, restent sequentielles. Il faut egalement tenir compte du cout de synchronisation entre threads et des limites liees aux acces memoire.

Un avertissement Numba concernant la couche `TBB` est apparu pendant les mesures. Cela n'empeche pas l'execution ni la parallelisation effective, mais indique simplement que la bibliotheque `TBB` disponible n'est pas dans la version recommandee. Les resultats restent coherents et exploitables, comme le montre la diminution nette des temps avec l'augmentation du nombre de threads.

### Conclusion

Sur ma machine, la meilleure configuration mesuree dans cette etape est `8` threads, avec un temps moyen de `80.685 ms`. Les mesures montrent donc clairement que la parallelisation avec Numba accelere la partie la plus couteuse du programme. La suite logique consiste maintenant a separer l'affichage et le calcul avec MPI afin d'evaluer l'impact de cette separation sur les performances globales.

## Etape 3 - Separation de l'affichage et du calcul avec MPI

### Principe retenu

Dans cette etape, le programme a ete modifie pour separer les deux roles principaux :

- le processus `rank 0` est charge de l'affichage,
- le processus `rank 1` est charge du calcul des trajectoires.

L'objectif ici est uniquement de decoupler l'affichage et le calcul. Le calcul du processus `rank 1` continue cependant a utiliser Numba avec un nombre variable de threads, afin de comparer cette version a l'etape precedente.

### Methode de mesure

Pour evaluer proprement cette etape, les mesures ont ete effectuees avec un mode benchmark sans affichage, de facon a mesurer le temps d'un pas complet comprenant :

- l'envoi de l'ordre de calcul du processus `0` vers le processus `1`,
- le calcul du nouveau pas de simulation dans le processus `1`,
- le retour des positions calculees vers le processus `0`.

Commande type utilisee :

```bash
NUMBA_NUM_THREADS=<p> mpirun -np 2 python3 -B 03_nbodies_grid_mpi_display_compute.py data/galaxy_5000 0.0015 15 15 1 --benchmark 5
```

Le jeu de donnees utilise est `data/galaxy_5000`, avec `dt = 0.0015` et une grille `(15, 15, 1)`.

### Resultats

| Threads | Temps moyen etape 3 (ms) | Speedup | Efficacite |
|---|---:|---:|---:|
| 1 | 288.337 | 1.000 | 1.000 |
| 2 | 179.233 | 1.609 | 0.804 |
| 4 | 174.975 | 1.648 | 0.412 |
| 8 | 160.606 | 1.795 | 0.224 |

Avec :

- `Speedup = T1 / Tp`
- `Efficacite = Speedup / p`

### Analyse

On constate que la separation entre affichage et calcul fonctionne correctement, mais que l'acceleration obtenue est moins bonne que dans l'etape 2. En effet, la meilleure mesure de cette etape est `160.606 ms` avec `8` threads, alors que l'etape 2 obtenait `80.685 ms` avec le meme nombre de threads.

Ce resultat est logique. Un seul processus reste responsable du calcul physique, tandis que l'autre s'occupe de l'affichage. A chaque iteration, il faut donc envoyer un ordre de calcul, attendre l'execution du pas de simulation, puis renvoyer les positions calculees. Cela introduit un surcout de communication et de synchronisation qui n'existait pas dans l'etape 2.

On observe egalement que l'augmentation du nombre de threads continue a reduire le temps d'execution, mais avec un gain plus limite. Le speedup passe seulement de `1.609` a `1.795` entre `2` et `8` threads. Cela montre que, dans cette configuration, le cout MPI vient amortir une partie importante des gains obtenus par la parallelisation Numba.

### Conclusion

Sur ma machine, la meilleure configuration mesuree dans cette etape est `8` threads, avec un temps moyen de `160.606 ms`. La separation de l'affichage et du calcul avec MPI clarifie bien l'architecture du programme, mais elle n'apporte pas une acceleration superieure a celle de l'etape 2, car elle ajoute des communications MPI sans encore distribuer le calcul entre plusieurs processus.

## Etape 4 - Parallelisation du calcul avec MPI

### Principe retenu

Dans cette etape, le calcul n'est plus effectue par un seul processus. Le programme a ete modifie pour distribuer le travail entre plusieurs processus MPI de calcul, en conservant la parallelisation Numba a l'interieur de chaque processus.

La version implemente utilise une distribution simple par blocs de corps :

- le processus `rank 0` joue le role de coordinateur,
- les processus `rank 1..P-1` calculent chacun un sous-ensemble de corps,
- chaque processus de calcul recoit les positions globales, calcule les mises a jour sur son bloc local, puis renvoie ses resultats au coordinateur.

Cette approche reste volontairement simple afin de limiter les modifications du code et de produire une version executable rapidement dans le cadre de l'examen.

La version implementee est une simplification par blocs de corps ; une vraie version a cellules fantomes sur la grille n'a pas ete mise en oeuvre ici.

### Methode de mesure

Les mesures ont ete effectuees en mode benchmark sans affichage, afin d'evaluer le cout global d'un pas distribue comprenant :

- la distribution des donnees aux processus de calcul,
- le calcul local dans chaque processus,
- le rassemblement des resultats par le processus coordinateur.

Commandes utilisees :

```bash
NUMBA_NUM_THREADS=1 mpirun -np 2 python3 -B 04_nbodies_grid_mpi_parallel.py data/galaxy_1000 0.001 20 20 1 --benchmark 5
NUMBA_NUM_THREADS=2 mpirun -np 2 python3 -B 04_nbodies_grid_mpi_parallel.py data/galaxy_1000 0.001 20 20 1 --benchmark 5
NUMBA_NUM_THREADS=4 mpirun -np 2 python3 -B 04_nbodies_grid_mpi_parallel.py data/galaxy_1000 0.001 20 20 1 --benchmark 5

NUMBA_NUM_THREADS=1 mpirun -np 3 python3 -B 04_nbodies_grid_mpi_parallel.py data/galaxy_1000 0.001 20 20 1 --benchmark 5
NUMBA_NUM_THREADS=2 mpirun -np 3 python3 -B 04_nbodies_grid_mpi_parallel.py data/galaxy_1000 0.001 20 20 1 --benchmark 5
NUMBA_NUM_THREADS=4 mpirun -np 3 python3 -B 04_nbodies_grid_mpi_parallel.py data/galaxy_1000 0.001 20 20 1 --benchmark 5
```

Le jeu de donnees utilise est `data/galaxy_1000`, avec `dt = 0.001` et une grille `(20, 20, 1)`.

### Resultats

| Processus MPI | Threads Numba | Temps moyen (ms) | Speedup |
|---|---:|---:|---:|
| 2 | 1 | 23.151 | 1.000 |
| 2 | 2 | 13.927 | 1.662 |
| 2 | 4 | 12.038 | 1.924 |
| 3 | 1 | 14.763 | 1.568 |
| 3 | 2 | 11.911 | 1.944 |
| 3 | 4 | 15.010 | 1.543 |

Ici, le speedup est calcule relativement a la configuration `2 processus / 1 thread`.

### Analyse

Cette etape montre une vraie parallelisation MPI du calcul, puisque plusieurs processus participent desormais au traitement des trajectoires. Les resultats montrent une amelioration nette par rapport a la configuration la plus simple de cette etape.

La meilleure configuration mesuree est `3 processus / 2 threads`, avec un temps moyen de `11.911 ms`, contre `23.151 ms` pour `2 processus / 1 thread`. Cela correspond a un speedup d'environ `1.94`.

On observe toutefois que l'augmentation simultanee du nombre de processus et du nombre de threads n'apporte pas toujours un gain supplementaire. Par exemple, la configuration `3 processus / 4 threads` est moins bonne que `3 processus / 2 threads`. Ce comportement est normal pour un probleme de taille moderee comme `galaxy_1000`, car le surcout de communication MPI, la synchronisation entre processus et le cout de coordination entre threads finissent par limiter les gains.

Autrement dit, cette etape confirme qu'une approche hybride `MPI + Numba` peut accelerer le calcul, mais qu'il existe un compromis entre :

- la quantite de travail reellement parallelisee,
- le cout des communications MPI,
- le cout de synchronisation entre threads.

Comme dans les etapes precedentes, un avertissement Numba concernant `TBB` est apparu pendant les mesures. Cet avertissement n'empeche pas l'execution et ne remet pas en cause les tendances observees.

### Questions theoriques

En observant que la densite d'etoiles diminue avec l'eloignement du trou noir, le principal probleme de performance attendu est un desequilibre de charge. En effet, les processus qui traitent une zone proche du centre de la galaxie auront davantage d'etoiles a gerer et donc davantage de calcul a effectuer que ceux traitant des zones peripheriques.

Une distribution plus intelligente consisterait a donner moins de cellules centrales a un meme processus et davantage de cellules peu denses en peripherie, de facon a equilibrer la charge totale entre processus.

Le probleme qui apparait alors est l'augmentation de la complexite de communication et de synchronisation. Une telle distribution rend plus delicate la gestion des frontieres, des echanges de donnees entre zones voisines et, dans une version plus avancee, des cellules fantomes.

### Remarque experimentale

Une tentative de mesure avec `5` processus MPI a echoue non pas a cause du code, mais a cause d'une limitation de l'environnement OpenMPI concernant le nombre de `slots` disponibles. Ces mesures n'ont donc pas ete retenues dans l'analyse.

### Conclusion

Sur ma machine, la meilleure configuration mesuree dans cette etape est `3 processus / 2 threads`, avec un temps moyen de `11.911 ms`. Les mesures montrent que la combinaison `MPI + threads Numba` peut apporter une acceleration supplementaire, mais que les gains ne sont pas lineaires en raison du cout des communications et des synchronisations. Cette observation est coherente avec le comportement attendu d'une parallelisation hybride sur un probleme de taille moderee.

## Barnes-Hut

### Distribution des boites et sous-boites

Dans le cas d'un quadtree Barnes-Hut, une strategie efficace consiste a partager les niveaux les plus hauts de l'arbre entre tous les processus, puis a repartir les sous-arbres plus profonds entre les differents processus.

L'idee est la suivante :

- la racine et les premiers niveaux du quadtree sont recopies sur tous les processus, car ils representent une vue globale compacte de la galaxie ;
- les sous-boites plus fines, situees aux niveaux inferieurs, sont distribuees entre les processus ;
- chaque processus devient responsable d'un ensemble de sous-arbres et des etoiles qui y sont associees.

Dans Cette solution les niveaux hauts contiennent peu de noeuds et peuvent donc etre partages a faible cout, tandis que le travail detaille est reparti entre processus.

### Proposition de parallelisation MPI

Une parallelisation possible avec MPI serait la suivante :

1. construire ou mettre a jour la structure globale du quadtree ;
2. diffuser a tous les processus les informations agrégées des niveaux hauts de l'arbre, en particulier les masses totales et centres de masse ;
3. repartir les sous-arbres profonds ou les groupes d'etoiles entre les processus ;
4. chaque processus calcule l'acceleration des etoiles dont il a la charge en utilisant :
   - les noeuds lointains du quadtree sous forme de masses aggregees,
   - et une descente plus fine dans les sous-arbres proches ;
5. rassembler ensuite les accelerations ou les nouvelles positions calculees.


### Conclusion

Une parallelisation MPI de Barnes-Hut peut donc etre construite en combinant une replication des niveaux hauts de l'arbre et une distribution des sous-arbres plus profonds entre les processus. Cette strategie permet de profiter de la structure hierarchique du quadtree tout en limitant la quantite de donnees a echanger.
