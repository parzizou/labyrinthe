import cv2
import os
import numpy as np
from ultralytics import YOLO

# Modèle YOLO pour détecter la grille et les coins
yolo_model = YOLO('grid.pt')
# Modèle de classification pour la couleur des cases
color_model = YOLO('colors.pt')

# Codes couleurs pour le tableau court
COLOR_CODE = {
    "bleu": "b",
    "rouge": "r",
    "vert": "g",
    "blanc": "w"
}
# Ordre des classes de ton modèle
COLOR_CLASS_NAMES = ["blanc", "bleu", "rouge", "vert"]

def ask_image_source():
    print("\nTu veux utiliser la webcam (1) ou importer une photo (2) ?")
    while True:
        choix = input("Tape 1 (caméra) ou 2 (import photo): ").strip()
        if choix == "1":
            return True, None
        elif choix == "2":
            chemin = input("Chemin du fichier image : ").strip()
            if not os.path.exists(chemin):
                print("Fichier introuvable, recommence.")
            else:
                return False, chemin
        else:
            print("Réponse invalide, recommence.")

def get_image(from_cam=True, image_path=None):
    if from_cam:
        cap = cv2.VideoCapture(0)
        print("Appuie sur 'espace' pour capturer la photo.")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Erreur de lecture caméra.")
                continue
            cv2.imshow("Prends la photo", frame)
            if cv2.waitKey(1) & 0xFF == ord(' '):
                break
        cap.release()
        cv2.destroyAllWindows()
        return frame
    else:
        img = cv2.imread(image_path)
        if img is None:
            print("Impossible de lire l'image, vérifie le chemin.")
        return img

def detect_grid_and_top_left_cell(image):
    results = yolo_model(image)
    boxes = results[0].boxes

    grid_box = None
    grid_conf = -1
    cell_boxes = []

    for box in boxes:
        cls = int(box.cls.cpu().numpy().item())
        b = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf.cpu().numpy().item())
        if cls == 1 and conf > grid_conf:
            grid_box = b
            grid_conf = conf
        elif cls == 0:
            cell_boxes.append((b, conf))

    print("Grille (meilleure confiance) :", grid_box, f"confiance={grid_conf:.3f}")
    print("Nombre de coins détectés :", len(cell_boxes))
    for i, (b, conf) in enumerate(cell_boxes):
        print(f"  Coin {i+1}: box={b}, conf={conf:.2f}")

    cell_box = cell_boxes[0][0] if cell_boxes else None
    return grid_box, cell_box

def ask_int(msg, default):
    while True:
        val = input(f"{msg} (Entrée pour {default}) : ").strip()
        if val == "":
            return default
        try:
            n = int(val)
            if n > 0:
                return n
            else:
                print("Il faut une valeur positive.")
        except:
            print("Entre un nombre entier valide.")

def deduce_cells(grid_box, cell_box):
    if grid_box is None or cell_box is None:
        print("Erreur : Grille ou case haut gauche non détectée.")
        return [], 0, 0

    xg1, yg1, xg2, yg2 = grid_box
    xc1, yc1, xc2, yc2 = cell_box
    grid_w, grid_h = xg2 - xg1, yg2 - yg1
    cell_w, cell_h = xc2 - xc1, yc2 - yc1

    n_cols = int(round(grid_w / cell_w)) if cell_w > 0 else 0
    n_rows = int(round(grid_h / cell_h)) if cell_h > 0 else 0

    print(f"Taille grille : w={grid_w}, h={grid_h}")
    print(f"Taille case haut gauche : w={cell_w}, h={cell_h}")
    print(f"Estimation automatique : {n_rows} lignes x {n_cols} colonnes")

    print("Confirme ou corrige :")
    n_cols = ask_int("Combien de colonnes (cases par ligne) ?", n_cols)
    n_rows = ask_int("Combien de lignes (cases par colonne) ?", n_rows)

    cell_w = grid_w / n_cols
    cell_h = grid_h / n_rows

    boxes = []
    for r in range(n_rows):
        row = []
        for c in range(n_cols):
            x1 = int(round(xg1 + c * cell_w))
            y1 = int(round(yg1 + r * cell_h))
            x2 = int(round(xg1 + (c+1) * cell_w))
            y2 = int(round(yg1 + (r+1) * cell_h))
            row.append([max(x1,0), max(y1,0), max(x2,0), max(y2,0)])
        boxes.append(row)
    return boxes, n_rows, n_cols

def classify_cells(image, boxes, output_folder="cases"):
    os.makedirs(output_folder, exist_ok=True)
    grid_codes = []
    for r, row in enumerate(boxes):
        code_row = []
        for c, box in enumerate(row):
            x1, y1, x2, y2 = box
            if x2 > x1 and y2 > y1:
                cell_img = image[y1:y2, x1:x2]
                filename = os.path.join(output_folder, f"case_{r}_{c}.png")
                cv2.imwrite(filename, cell_img)

                # Classification de la case
                pred = color_model.predict(cell_img, imgsz=224, verbose=False)
                pred_class = int(pred[0].probs.top1)
                pred_name = COLOR_CLASS_NAMES[pred_class]
                code = COLOR_CODE.get(pred_name, "?")
                print(f"Case {r},{c} : {pred_name} -> {code}")
                code_row.append(code)
            else:
                print(f"Case {r},{c} ignorée : coordonnées invalides ({x1},{y1},{x2},{y2})")
                code_row.append("?")
        grid_codes.append(code_row)
    return grid_codes

def print_grid(grid_codes):
    print("\nGrille des couleurs :")
    for row in grid_codes:
        print(row)
    print("\nFormat tableau python utilisable :")
    print("[")
    for row in grid_codes:
        print(f"  {row},")
    print("]")

def find_start(grid):
    for x in range(len(grid)):
        for y in range(len(grid[0])):
            if grid[x][y] == "g":
                return (x,y)

def autours(case,grid):
    result=[]

    top = (case[0]-1,case[1])
    bot = (case[0]+1,case[1])
    left = (case[0],case[1]-1)
    right = (case[0],case[1]+1)
    
    for direct in [top,left,right,bot]:
        if direct[0] <0 or direct[0] > len(grid)-1 or direct[1]<0 or direct[1] > len(grid[0])-1:
            pass
        else:
            result.append(direct)
    return result

solutions = []

def recursive_path(grid,historique,case):
    global solutions
    tour = autours(case,grid)

    for direct in tour:
        if grid[direct[0]][direct[1]] == "r": # si il trouve la sortie
            solutions.append(historique+[direct])

        if grid[direct[0]][direct[1]] == "w" and not direct in historique: # si une case blanche est encore dispo autour
            recursive_path(grid,historique+[direct],direct)

def solve(grid):
    global solutions
    solutions = []  # reset entre deux runs
    dep_case = find_start(grid)
    recursive_path(grid,[dep_case],dep_case)
    print("nbr de solutions : ",len(solutions))
    min_len = len(solutions[0]) if solutions else 0
    best_sol = None
    for sol in solutions:
        if len(sol) <= min_len:
            min_len = len(sol)
            best_sol = sol
    print("meilleure solution : ",best_sol)
    return best_sol

# --------- AJOUT DU TRAÇAGE --------

def draw_solution_on_image(image, boxes, solution, color=(0,0,255), thickness=4):
    """
    Trace le chemin de solution sur l'image du labyrinthe.
    - image: ton image d'origine
    - boxes: liste 2D des boxes des cases (même indices que la grille)
    - solution: liste de tuples (r,c) représentant le chemin (cases à relier)
    - color: couleur du tracé (BGR)
    - thickness: épaisseur du tracé
    """
    if not solution or len(solution) < 2:
        print("Pas de solution à tracer.")
        return image

    # On trace un trait du centre d'une case au centre de la suivante, dans l'ordre
    points = []
    for (r, c) in solution:
        x1, y1, x2, y2 = boxes[r][c]
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        points.append((cx, cy))

    for i in range(len(points)-1):
        cv2.line(image, points[i], points[i+1], color, thickness)
    return image

# --------- MAIN PATCHÉ ---------

def main():
    from_cam, image_path = ask_image_source()
    image = get_image(from_cam=from_cam, image_path=image_path)
    if image is None:
        print("Erreur à la récupération de l'image.")
        return

    grid_box, cell_box = detect_grid_and_top_left_cell(image)
    boxes, n_rows, n_cols = deduce_cells(grid_box, cell_box)
    if n_rows == 0 or n_cols == 0:
        print("Erreur : aucune case extraite.")
        return

    grid_codes = classify_cells(image, boxes)
    print_grid(grid_codes)
    print(f"\nGrille de {n_rows} lignes x {n_cols} colonnes extraite et classifiée.")

    best_sol = solve(grid_codes)

    if best_sol:
        print("Je trace la solution sur l'image...")
        img_with_path = draw_solution_on_image(image.copy(), boxes, best_sol)
        out_file = "labyrinthe_solution.png"
        cv2.imwrite(out_file, img_with_path)
        print(f"Solution tracée et enregistrée dans {out_file}")
        import subprocess
        subprocess.run(["xdg-open", "labyrinthe_solution.png"])
    else:
        print("Aucune solution trouvée, rien à tracer.")

if __name__ == "__main__":
    main()