# 🧩 Labyrinthe Solver

Un projet Python pour **résoudre automatiquement des labyrinthes** à partir d'une image, en utilisant la reconnaissance d'image (YOLO), la classification couleur, et un algorithme de recherche de chemin.  
Ce script t'aide à prendre une photo de ton labyrinthe (webcam ou image), détecte la grille, identifie les cases (départ, sortie, murs, chemins), puis **trouve et trace la meilleure solution** sur l'image.  

---

## 🛠️ Fonctionnalités principales

- **Détection automatique de la grille et des cases** sur une photo grâce à YOLO (`grid.pt` et `colors.pt`)
- **Classification des couleurs** des cases pour identifier le départ, l’arrivée, les murs, les chemins...
- **Extraction de la grille** sous forme de tableau utilisable en Python
- **Recherche de chemin ("solveur")** pour trouver la sortie depuis le départ (plus court chemin)
- **Affichage et sauvegarde de la solution** directement tracée sur l’image de départ
- **Interface terminale** interactive (choix source image, correction dimensions grille...)

---

## 📷 Comment ça marche (workflow)

1. **Lancement du script**  
   `python solver.py`

2. **Choix de la source d'image**  
   - Webcam (appuie sur "espace" pour capturer)
   - Fichier image (chemin à fournir)

3. **Détection de la grille**  
   - Utilise un modèle YOLO (`grid.pt`) pour localiser la grille et la première case

4. **Estimation et confirmation de la taille de la grille**  
   - L'utilisateur peut corriger le nombre de lignes/colonnes détectées

5. **Découpage et classification des cases**  
   - Chaque case est sauvegardée (dans `cases/`) et classifiée par couleur (`colors.pt`)
   - Codes couleurs supportés :  
     - **vert** : départ (`g`)
     - **rouge** : sortie (`r`)
     - **bleu** : mur (`b`)
     - **blanc** : chemin (`w`)

6. **Affichage de la grille extraite** (tableau Python)

7. **Recherche et affichage de la ou des solutions**
   - Algorithme de backtracking pour le plus court chemin
   - Affiche le chemin retenu

8. **Tracé du chemin sur l’image originale**  
   - Résultat sauvegardé dans `labyrinthe_solution.png` et ouvert automatiquement

---

## 🧑‍💻 Utilisation rapide

```bash
pip install -r requirements.txt
python solver.py
```

- Tu peux utiliser ta webcam ou une image existante.
- Les modèles YOLO `grid.pt` et `colors.pt` doivent être présents dans le dossier.

---

## 📦 Dépendances

- Python 3.x
- [OpenCV](https://opencv.org/) (`cv2`)
- [Numpy](https://numpy.org/)
- [ultralytics YOLO](https://docs.ultralytics.com/)
- Modèles `grid.pt` et `colors.pt` (non fournis ici)

Installe tout avec :
```bash
pip install opencv-python numpy ultralytics
```

---

## 📁 Structure du repo

- `solver.py` : le coeur du projet (tout y est !)
- `cases/` : images de chaque case extraites automatiquement (créé au runtime)
- `labyrinthe_solution.png` : image de sortie tracée
- `grid.pt` / `colors.pt` : modèles YOLO pour la détection/classification

---

## ⚙️ Explications des fonctions clés

- **ask_image_source / get_image** : dialogue avec toi pour obtenir une image (webcam ou fichier)
- **detect_grid_and_top_left_cell** : détecte et localise la grille et la première case
- **deduce_cells** : découpe la grille en cases individuelles
- **classify_cells** : classe chaque case (couleur) via YOLO
- **solve** : cherche la solution (parcours en profondeur, backtracking)
- **draw_solution_on_image** : trace le chemin trouvé par le solveur

---

## 🎨 À personnaliser / améliorer

- Ajouter d'autres couleurs ou types de cases
- Supporter d'autres formats d'image
- Améliorer la robustesse des modèles YOLO utilisés
- Ajouter une interface graphique

---

## 👽 Exemples d'utilisation

1. Prendre un labyrinthe en photo avec ta webcam
2. Vérifier/corriger la taille détectée de la grille
3. Laisser le script "lire" les couleurs et résoudre automatiquement
4. Ouvrir l'image annotée avec le chemin de la sortie !

---

## 📝 Remarques

- Nécessite d’avoir les modèles YOLO `grid.pt` et `colors.pt` entraînés pour ton type de labyrinthe.
- Testé sous Linux (pour l'ouverture automatique de l'image, modifie la ligne 272 selon ton OS si besoin).
- Code 100% open-source, améliore-le et propose tes PR !

---

## 👤 Auteur

- parzizou

---

Bon courage pour hacker des labyrinthes !  
Pour toute question ou bug, ouvre une issue 😉
