# -*- coding: utf-8 -*-
"""
Clustering K-Means sur les images représentées par des matrices (0=noir à 1=blanc).
Regrouper les images en clusters (ex: saines vs malades) en comparant chaque image ensemble.
Sauvegarde les assignations de clusters dans un CSV.
Ajout : visualisation graphique avec PCA, couleurs par vrais labels (hem/all).
"""

import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# ==========================
# PARAMÈTRES À MODIFIER
# ==========================
DATA_DIR = "../processed_data/training_data"  # dossier contenant les fichiers .npy des matrices d'images
K = 2           # nombre de clusters (ex: 2 pour sain/malade)
N_INIT = 20     # relances KMeans
RANDOM_STATE = 42
OUT_ROOT = f"clustering_images_k{K}"  # dossier de sortie
LIMIT_PER_CLASS = None  # None = toutes les images par classe

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

    # Lister les fichiers .npy récursivement et équilibrer les classes
    print("Listage des fichiers .npy...")
    hem_files = []
    all_files = []
    for root, dirs, filenames in os.walk(DATA_DIR):
        for f in filenames:
            if f.endswith('.npy'):
                rel_path = os.path.relpath(os.path.join(root, f), DATA_DIR)
                if 'hem' in rel_path:
                    hem_files.append(rel_path)
                elif 'all' in rel_path:
                    all_files.append(rel_path)
    hem_files.sort()
    all_files.sort()
    
    # Prendre LIMIT_PER_CLASS de chaque classe
    if LIMIT_PER_CLASS is not None:
        hem_files = hem_files[:LIMIT_PER_CLASS]
        all_files = all_files[:LIMIT_PER_CLASS]
    files = hem_files + all_files
    
    print(f"{len(hem_files)} images hem et {len(all_files)} images all à traiter.")

    print(f"{len(files)} matrices d'images à traiter.")

    data = []
    filenames = []

    print("Chargement des données...")
    for i, f in enumerate(files):
        fp = os.path.join(DATA_DIR, f)
        mat = np.load(fp)
        vec = mat.flatten()
        data.append(vec)
        filenames.append(f)
        if (i + 1) % 100 == 0 or i == len(files) - 1:
            print(f"  Chargé {i + 1}/{len(files)} : {f} - shape: {mat.shape}")

    if not data:
        print("Aucune matrice chargée.")
        exit()

    data = np.array(data)
    print("Données chargées et converties en array.")

    # Charger les vrais labels depuis les chemins (hem/all)
    print("Inférence des vrais labels...")
    labels_true = []
    for f in files:
        if 'hem' in f:
            labels_true.append('hem')
        elif 'all' in f:
            labels_true.append('all')
        else:
            labels_true.append('unknown')
    print(f"Labels inférés pour {len(labels_true)} images.")

    # Clustering K-Means
    print("Démarrage du clustering K-Means...")
    km = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=N_INIT)
    labels = km.fit_predict(data)
    centers = km.cluster_centers_
    print("Clustering terminé.")

    # Réduction de dimension pour visualisation (PCA à 2D)
    print("Réduction de dimension avec PCA...")
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(data)
    centers_2d = pca.transform(centers)  # Projeter les centroïdes
    print("PCA terminé.")

    # Statistiques
    print("Calcul des statistiques...")
    for i, f in enumerate(filenames):
        row = {
            "file": f,
            "cluster": labels[i],
            "true_label": labels_true[i],
            "mean_value": float(np.mean(data[i])),
            "std_value": float(np.std(data[i])),
            "pca_x": float(data_2d[i, 0]),
            "pca_y": float(data_2d[i, 1]),
        }
        all_rows.append(row)
    print("Statistiques calculées.")

    # Export CSV
    print("Sauvegarde du CSV...")
    ensure_dir(OUT_ROOT)
    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(OUT_ROOT, f"clustering_k{K}.csv")
    df.to_csv(csv_path, index=False)
    print(f"CSV sauvegardé : {csv_path}")

    # Sauvegarde des centres
    centers_path = os.path.join(OUT_ROOT, "centers.npy")
    np.save(centers_path, centers)
    print(f"Centres sauvegardés : {centers_path}")

    # Graphique de visualisation
    print("Génération du graphique...")
    # Couleurs : rouge pour cluster 0, bleu pour cluster 1
    color_map = {0: 'red', 1: 'blue'}
    color2_map = {'hem': 'yellow', 'all': 'green', 'unknown': 'gray'}
    colors = [color_map.get(cluster, 'gray') for cluster in labels]
    markers = [color2_map.get(label, 'o') for label in labels_true]

    plt.figure(figsize=(10, 8))
    for i in range(len(data_2d)):
        plt.scatter(data_2d[i, 0], data_2d[i, 1], c=colors[i], alpha=0.5, marker=markers[i])
    # Ajouter les centroïdes
    plt.scatter(centers_2d[:, 0], centers_2d[:, 1], c=['red', 'blue'], marker='*', s=300, edgecolors='black', linewidth=2, label='Centroïdes')
    plt.title('Visualisation du clustering (PCA 2D, couleurs par cluster)')
    plt.xlabel('Composante PCA 1')
    plt.ylabel('Composante PCA 2')
    plt.grid(True)
    # Légende pour clusters et centroïdes
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Cluster 0'),
               plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Cluster 1'),
               plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='red', markersize=15, markeredgecolor='black', label='Centroïde 0'),
               plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='blue', markersize=15, markeredgecolor='black', label='Centroïde 1')]
    plt.legend(handles=handles, title='Légende')
    plot_path = os.path.join(OUT_ROOT, 'clustering_visualization.png')
    plt.savefig(plot_path)
    plt.close()  # Fermer pour éviter affichage si en mode non-interactif
    print(f"Graphique sauvegardé : {plot_path}")

    print(f"\nClustering terminé.")
    print(f"Dossier de sortie : {OUT_ROOT}")
