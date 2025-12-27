"""
src/predict.py

Prédiction (inférence) sur le jeu de test.

Principe (comme votre version de base) :
- On charge le modèle entraîné (fichier .pth)
- On parcourt les images de test prétraitées (format .npy)
- On prédit "Sain" ou "Malade"
- On calcule une "confiance" (probabilité associée à la classe prédite)
- On sauvegarde les résultats dans un CSV (resultats_test.csv)
"""

import os
from glob import glob

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.model import charger_modele


# ============================================================
# PARAMÈTRES GLOBAUX (à garder simples et lisibles)
# ============================================================

# Dossier contenant les données prétraitées
PROCESSED_DIR = "processed_data"

# Dossier test prétraité (créé par preprocess.py)
# Dans notre preprocess, on a mis l'ensemble = "testing"
TEST_DIR = os.path.join(PROCESSED_DIR, "testing")

# Chemin du modèle entraîné (créé par train.py)
MODEL_PATH = "resnet_cellule.pth"

# Fichier de sortie CSV
CSV_OUT = "resultats_test.csv"

# Mapping des classes (doit être cohérent avec train.py)
# 0 = hem = sain ; 1 = all = malade
IDX_TO_LABEL = {
    0: "Sain",
    1: "Malade"
}


# ============================================================
# OUTILS : récupérer les fichiers .npy de test
# ============================================================

def _liste_fichiers_test_npy():
    """
    Récupère tous les fichiers .npy présents dans processed_data/testing/
    (en cherchant récursivement au cas où il y aurait des sous-dossiers).
    """
    # **/*.npy : recherche dans tous les sous-dossiers
    pattern = os.path.join(TEST_DIR, "**", "*.npy")
    fichiers = sorted(glob(pattern, recursive=True))
    return fichiers


# ============================================================
# FONCTION PRINCIPALE (appelée par main.py)
# ============================================================

def lancer_predictions():
    """
    Fonction appelée depuis main.py.

    Étapes :
    1) Vérifier qu'on a le modèle et les données de test
    2) Charger le modèle
    3) Parcourir les images .npy de test
    4) Prédire une classe et calculer la confiance
    5) Sauvegarder un CSV des résultats
    """

    print("======================================================================")
    print("   PROGRAMME DE TEST - CLASSIFICATION DES CELLULES AVEC RESNET")
    print("======================================================================")

    # ------------------------------------------------------------
    # 1) Vérifications de base
    # ------------------------------------------------------------
    if not os.path.isfile(MODEL_PATH):
        print(f"[ERREUR] Modèle introuvable : {MODEL_PATH}")
        print("Lance d'abord l'entraînement (entrainer_modele).")
        return

    if not os.path.isdir(TEST_DIR):
        print(f"[ERREUR] Dossier test prétraité introuvable : {TEST_DIR}")
        print("Lance d'abord le prétraitement (preparer_donnees).")
        return

    fichiers_test = _liste_fichiers_test_npy()

    if len(fichiers_test) == 0:
        print(f"[ERREUR] Aucun fichier .npy trouvé dans : {TEST_DIR}")
        return

    print(f"\n→ Nombre d'images trouvées: {len(fichiers_test)}")
    print("Images prêtes à être analysées")

    # ------------------------------------------------------------
    # 2) Chargement du modèle
    # ------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n→ Device utilisé: {device}")

    # On charge l'architecture + les poids (via src/modele.py)
    modele = charger_modele(chemin_poids=MODEL_PATH, nb_classes=2, device=device)
    print(" Modèle chargé et prêt pour les prédictions")

    # ------------------------------------------------------------
    # 3) Boucle de prédiction
    # ------------------------------------------------------------
    resultats = []

    print("\nAnalyse des images en cours...\n")

    with torch.no_grad():
        for i, path_npy in enumerate(fichiers_test, start=1):
            # ---- Charger l'image prétraitée (.npy) ----
            img = np.load(path_npy)  # shape attendue : (H, W, 3), valeurs entre 0 et 1

            # ---- Convertir en tenseur PyTorch ----
            # ResNet attend [B, C, H, W]
            x = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            x = x.to(device)

            # ---- Prédiction ----
            logits = modele(x)  # sorties brutes du réseau

            # Convertir en probabilités (softmax car 2 classes)
            probs = F.softmax(logits, dim=1)  # shape : [1, 2]

            # Classe prédite = argmax
            pred_idx = int(torch.argmax(probs, dim=1).item())
            pred_label = IDX_TO_LABEL[pred_idx]

            # Confiance = probabilité de la classe prédite
            confiance = float(probs[0, pred_idx].item() * 100)

            # ---- ID image (on reprend le nom du fichier) ----
            # Exemple : ".../IMG_123.npy" -> "IMG_123"
            image_id = os.path.splitext(os.path.basename(path_npy))[0]

            # Stockage des résultats
            resultats.append({
                "Image_ID": image_id,
                "Prediction": pred_idx,          # 0/1 (utile pour calculs)
                "Etat": pred_label,              # texte lisible (Sain/Malade)
                "Confiance (%)": round(confiance, 2)
            })

            # Affichage de progression (comme votre script)
            if i % 200 == 0 or i == len(fichiers_test):
                pct = (i / len(fichiers_test)) * 100
                print(f" Progression: {i}/{len(fichiers_test)} images ({pct:.1f}%)")

    print("\n Analyse terminée!")

    # ------------------------------------------------------------
    # 4) Sauvegarde du CSV
    # ------------------------------------------------------------
    df = pd.DataFrame(resultats)
    df.to_csv(CSV_OUT, index=False)

    print("\n----------------------------------------------------------------------")
    print(" Sauvegarde des résultats")
    print("----------------------------------------------------------------------")
    print(f"→ Résultats sauvegardés dans: {CSV_OUT}")

    # ------------------------------------------------------------
    # 5) Petit résumé (utile et lisible)
    # ------------------------------------------------------------
    nb_total = len(df)
    nb_sain = int((df["Etat"] == "Sain").sum())
    nb_malade = int((df["Etat"] == "Malade").sum())
    confiance_moy = float(df["Confiance (%)"].mean())

    print("\n======================================================================")
    print("  RÉSUMÉ DES RÉSULTATS")
    print("======================================================================")
    print(f"• Nombre total d'images analysées: {nb_total}")
    print(f"• Cellules prédites SAINES:        {nb_sain} ({(nb_sain/nb_total)*100:.1f}%)")
    print(f"• Cellules prédites MALADES:       {nb_malade} ({(nb_malade/nb_total)*100:.1f}%)")
    print(f"• Confiance moyenne des prédictions: {confiance_moy:.2f}%")

    print("\nAperçu des 10 premiers résultats :")
    print(df.head(10).to_string(index=False))

    print("\n======================================================================")
    print("   PROGRAMME TERMINÉ AVEC SUCCÈS")
    print("======================================================================")
