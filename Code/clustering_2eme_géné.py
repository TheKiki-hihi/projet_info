# -*- coding: utf-8 -*-
"""
Clustering K-Means sur les images représentées par des matrices (0=noir à 1=blanc).
Regrouper les images en clusters (ex: saines vs malades) en comparant chaque image ensemble.
Sauvegarde les assignations de clusters dans un CSV.
"""

import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# ==========================
# PARAMÈTRES À MODIFIER
# ==========================
DATA_DIR = "."  # dossier contenant les fichiers .npy des matrices d'images
K = 2           # nombre de clusters (ex: 2 pour sain/malade)
N_INIT = 20     # relances KMeans
RANDOM_STATE = 42
OUT_ROOT = f"clustering_images_k{K}"  # dossier de sortie
LIMIT = None    # None = tout traiter, sinon ex: 10 pour test

# ==========================
# FONCTIONS UTILITAIRES
# ==========================
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

# ==========================
# MAIN
# ==========================
if __name__ == "__main__":
    print("Démarrage du script...")
    all_rows = []

    # Lister les fichiers .npy
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.npy') and 'matrice_image' in f]
    files.sort()
    if LIMIT is not None:
        files = files[:LIMIT]

    print(f"{len(files)} matrices d'images à traiter.")

    data = []
    filenames = []

    for f in files:
        fp = os.path.join(DATA_DIR, f)
        mat = np.load(fp)
        vec = mat.flatten()
        data.append(vec)
        filenames.append(f)
        print(f"  Chargé : {f} - shape: {mat.shape}")

    if not data:
        print("Aucune matrice chargée.")
        exit()

    data = np.array(data)

    # Clustering K-Means
    km = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=N_INIT)
    labels = km.fit_predict(data)
    centers = km.cluster_centers_

    # Statistiques
    for i, f in enumerate(filenames):
        row = {
            "file": f,
            "cluster": labels[i],
            "mean_value": float(np.mean(data[i])),
            "std_value": float(np.std(data[i])),
        }
        all_rows.append(row)

    # Export CSV
    ensure_dir(OUT_ROOT)
    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(OUT_ROOT, f"clustering_k{K}.csv")
    df.to_csv(csv_path, index=False)

    # Sauvegarde des centres
    centers_path = os.path.join(OUT_ROOT, "centers.npy")
    np.save(centers_path, centers)

    print(f"\nClustering terminé.")
    print(f"CSV sauvegardé : {csv_path}")
    print(f"Centres sauvegardés : {centers_path}")
    print(f"Dossier de sortie : {OUT_ROOT}")
