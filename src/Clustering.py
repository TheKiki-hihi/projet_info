# -*- coding: utf-8 -*-
"""
Segmentation K-Means (RGB) sur toutes les images de:
C-NMC_Leukemia/training_data/fold_0/{hem, all}

→ Sauvegarde les images segmentées dans un dossier "seg_kmeans_fold0_k<K>_RGB"
→ Crée un CSV contenant les statistiques (surface et couleur moyenne des clusters)
"""

import os
import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


# PARAMÈTRES À MODIFIER

BASE_DIR = "C-NMC_Leukemia/training_data/fold_0"  # chemin de base
CLASSES = ["hem", "all"]                          # les deux types d'images
EXTS_OK = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")

K = 3                                             # nombre de clusters
N_INIT = 20                                       # relances KMeans (plus = plus stable)
RANDOM_STATE = 42
OUT_ROOT = f"seg_kmeans_fold0_k{K}_RGB"           # dossier de sortie global
LIMIT_PER_CLASS = None                            # None = tout traiter, sinon ex: 100 pour test

# FONCTIONS 
def list_images(folder):
    files = [f for f in os.listdir(folder) if f.lower().endswith(EXTS_OK)]
    files.sort()
    if LIMIT_PER_CLASS is not None:
        files = files[:LIMIT_PER_CLASS]
    return files

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def segment_image_rgb(img_rgb, k=3, n_init=20, random_state=42):
    """
    Segmentation simple K-Means sur l'espace RGB uniquement.
    """
    h, w, _ = img_rgb.shape
    rgb = img_rgb.reshape(-1, 3).astype(np.float32) / 255.0  # normalisé
    km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
    labels = km.fit_predict(rgb)
    centers_rgb = km.cluster_centers_
    seg_rgb = (centers_rgb[labels].reshape(h, w, 3) * 255).astype(np.uint8)
    return seg_rgb, labels, centers_rgb

# ==========================
# MAIN
# ==========================
if __name__ == "__main__":
    all_rows = []

    for cls in CLASSES:
        in_dir = os.path.join(BASE_DIR, cls)
        out_dir = os.path.join(OUT_ROOT, cls)
        ensure_dir(out_dir)

        if not os.path.isdir(in_dir):
            print(f"[!] Dossier introuvable : {in_dir}")
            continue

        files = list_images(in_dir)
        print(f"\nClasse '{cls}': {len(files)} images à segmenter → sortie : {out_dir}")

        for i, fn in enumerate(files, start=1):
            fp = os.path.join(in_dir, fn)
            img_bgr = cv2.imread(fp, cv2.IMREAD_COLOR)
            if img_bgr is None:
                print(f"  [!] Lecture impossible : {fp}")
                continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h, w, _ = img_rgb.shape

            seg_rgb, labels, centers_rgb = segment_image_rgb(
                img_rgb, k=K, n_init=N_INIT, random_state=RANDOM_STATE
            )

            # Sauvegarde de l'image segmentée
            stem = os.path.splitext(fn)[0]
            out_path = os.path.join(out_dir, f"{stem}_seg_k{K}.png")
            cv2.imwrite(out_path, cv2.cvtColor(seg_rgb, cv2.COLOR_RGB2BGR))

            # Statistiques
            counts = np.bincount(labels, minlength=K)
            perc = counts / counts.sum() * 100.0
            row = {
                "class": cls,
                "file": fp,
                **{f"px_c{c}": int(counts[c]) for c in range(K)},
                **{f"pct_c{c}": float(perc[c]) for c in range(K)},
                **{f"center_c{c}_r": float(centers_rgb[c, 0]) for c in range(K)},
                **{f"center_c{c}_g": float(centers_rgb[c, 1]) for c in range(K)},
                **{f"center_c{c}_b": float(centers_rgb[c, 2]) for c in range(K)},
                "out_path": out_path,
            }
            all_rows.append(row)

            if i % 50 == 0 or i == len(files):
                print(f"  → {i}/{len(files)} images traitées")

    # Export CSV global
    if all_rows:
        ensure_dir(OUT_ROOT)
        df = pd.DataFrame(all_rows)
        csv_path = os.path.join(OUT_ROOT, f"stats_fold0_k{K}.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n Stats sauvegardées : {csv_path}")
        print(f"Images segmentées : {OUT_ROOT}/<hem|all>/")
    else:
        print("\nAucune image traitée.")

