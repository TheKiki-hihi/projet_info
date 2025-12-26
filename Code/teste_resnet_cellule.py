"""
Programme de test du modèle ResNet pour la classification de cellules
=====================================================================
Ce programme charge le modèle ResNet pré-entraîné et prédit si les cellules
sont saines ou malades (leucémie) à partir des images .npy du dossier testing_data.

Auteur: Projet INFO
Date: Décembre 2025
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models
import glob
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================
print("=" * 70)
print("   PROGRAMME DE TEST - CLASSIFICATION DES CELLULES AVEC RESNET")
print("=" * 70)
print(f"\n📅 Date d'exécution: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")

# Chemins des fichiers
MODEL_PATH = '/Users/kilperic/Desktop/projet_info/IA/resnet_cellule.pth'
IMAGES_DIR = '/Users/kilperic/Desktop/projet_info/processed_data/testing_data'
OUTPUT_FILE = '/Users/kilperic/Desktop/projet_info/IA/resultats_test.csv'

# =============================================================================
# ÉTAPE 1 : Vérification des fichiers
# =============================================================================
print("\n" + "-" * 70)
print("📂 ÉTAPE 1/5 : Vérification des fichiers")
print("-" * 70)

# Vérification du modèle
print(f"\n  → Vérification du modèle: {MODEL_PATH}")
if os.path.exists(MODEL_PATH):
    print("    ✅ Modèle trouvé")
else:
    print("    ❌ ERREUR: Le modèle n'existe pas!")
    exit(1)

# Vérification du dossier d'images
print(f"\n  → Vérification du dossier d'images: {IMAGES_DIR}")
if os.path.exists(IMAGES_DIR):
    print("    ✅ Dossier trouvé")
else:
    print("    ❌ ERREUR: Le dossier d'images n'existe pas!")
    exit(1)

# =============================================================================
# ÉTAPE 2 : Chargement des images
# =============================================================================
print("\n" + "-" * 70)
print("🖼️  ÉTAPE 2/5 : Chargement de la liste des images")
print("-" * 70)

# Recherche de tous les fichiers .npy
image_paths = glob.glob(os.path.join(IMAGES_DIR, '*.npy'))
image_paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

nb_images = len(image_paths)
print(f"\n  → Nombre d'images trouvées: {nb_images}")

if nb_images == 0:
    print("    ❌ ERREUR: Aucune image .npy trouvée!")
    exit(1)
else:
    print(f"    ✅ {nb_images} images prêtes à être analysées")

# =============================================================================
# ÉTAPE 3 : Chargement du modèle ResNet
# =============================================================================
print("\n" + "-" * 70)
print("🧠 ÉTAPE 3/5 : Chargement du modèle ResNet")
print("-" * 70)

# Détection du device (GPU ou CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n  → Device utilisé: {device}")

# Création de l'architecture du modèle
print("  → Création de l'architecture ResNet18...")
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes: sain (0) ou malade (1)

# Chargement des poids entraînés
print("  → Chargement des poids du modèle...")
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()  # Mode évaluation (pas d'entraînement)

print("    ✅ Modèle chargé et prêt pour les prédictions")

# =============================================================================
# ÉTAPE 4 : Prédiction sur toutes les images
# =============================================================================
print("\n" + "-" * 70)
print("🔬 ÉTAPE 4/5 : Analyse des cellules")
print("-" * 70)

# Listes pour stocker les résultats
resultats = []
compteur_sain = 0
compteur_malade = 0

print(f"\n  Analyse de {nb_images} images en cours...\n")

# Boucle sur toutes les images
for i, image_path in enumerate(image_paths):
    # Affichage de la progression tous les 100 images
    if (i + 1) % 100 == 0 or (i + 1) == nb_images:
        pourcentage = ((i + 1) / nb_images) * 100
        print(f"    📊 Progression: {i + 1}/{nb_images} images ({pourcentage:.1f}%)")
    
    # Récupération du nom de l'image (sans extension)
    nom_image = os.path.splitext(os.path.basename(image_path))[0]
    
    # Chargement de l'image
    img = np.load(image_path)
    img = img.astype(np.float32)
    
    # Conversion en 3 canaux si nécessaire (pour ResNet)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    
    # Normalisation (0-1) si nécessaire
    if img.max() > 1.0:
        img = img / 255.0
    
    # Conversion en tensor PyTorch
    img = img.transpose(2, 0, 1).copy()  # Changement de format: HWC -> CHW
    img = torch.tensor(img, dtype=torch.float32)
    
    # Normalisation finale
    img = (img - 0.5) / 0.5
    
    # Ajout d'une dimension batch
    img = img.unsqueeze(0).to(device)
    
    # Prédiction
    with torch.no_grad():
        output = model(img)
        probabilities = torch.softmax(output, dim=1)
        prediction = torch.argmax(output, dim=1).item()
        confidence = probabilities[0][prediction].item() * 100
    
    # Interprétation du résultat
    if prediction == 0:
        etat = "Sain"
        compteur_sain += 1
    else:
        etat = "Malade"
        compteur_malade += 1
    
    # Stockage du résultat
    resultats.append({
        'Image_ID': nom_image,
        'Prediction': prediction,
        'Etat': etat,
        'Confiance (%)': round(confidence, 2)
    })

print("\n    ✅ Analyse terminée!")

# =============================================================================
# ÉTAPE 5 : Sauvegarde et affichage des résultats
# =============================================================================
print("\n" + "-" * 70)
print("💾 ÉTAPE 5/5 : Sauvegarde des résultats")
print("-" * 70)

# Création du DataFrame et sauvegarde en CSV
df_resultats = pd.DataFrame(resultats)
df_resultats.to_csv(OUTPUT_FILE, index=False)
print(f"\n  → Résultats sauvegardés dans: {OUTPUT_FILE}")

# =============================================================================
# RÉSUMÉ FINAL
# =============================================================================
print("\n" + "=" * 70)
print("   📋 RÉSUMÉ DES RÉSULTATS")
print("=" * 70)

print(f"\n  📊 Statistiques globales:")
print(f"     • Nombre total d'images analysées: {nb_images}")
print(f"     • Cellules prédites SAINES:        {compteur_sain} ({compteur_sain/nb_images*100:.1f}%)")
print(f"     • Cellules prédites MALADES:       {compteur_malade} ({compteur_malade/nb_images*100:.1f}%)")

# Calcul de la confiance moyenne
confiance_moyenne = df_resultats['Confiance (%)'].mean()
print(f"\n  🎯 Confiance moyenne des prédictions: {confiance_moyenne:.2f}%")

# Affichage des 10 premiers résultats
print(f"\n  📝 Aperçu des 10 premiers résultats:")
print("  " + "-" * 50)
print(f"  {'Image_ID':<15} {'État':<10} {'Confiance':<10}")
print("  " + "-" * 50)
for i in range(min(10, len(resultats))):
    r = resultats[i]
    print(f"  {r['Image_ID']:<15} {r['Etat']:<10} {r['Confiance (%)']:.2f}%")

print("\n" + "=" * 70)
print("   ✅ PROGRAMME TERMINÉ AVEC SUCCÈS")
print("=" * 70)
