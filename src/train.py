"""
src/train.py

Entraînement du modèle (ResNet18) pour classifier des images de cellules.

Principe  :
- On utilise les images prétraitées au format .npy (créées par preprocess.py)
- On crée un Dataset PyTorch qui charge ces .npy + les labels
- On entraîne un ResNet18 (défini dans src/modele.py)
- On sauvegarde les poids entraînés dans un fichier .pth
"""

import os
from glob import glob

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from src.model import creer_modele


##############################################################
#                 PARAMÈTRES GLOBAUX                         #
############################################################## 

# Dossier contenant les données prétraitées (.npy) créées par preprocess.py
PROCESSED_DIR = "processed_data"

# Chemins attendus :
# processed_data/train/hem/*.npy
# processed_data/train/all/*.npy
TRAIN_HEM_DIR = os.path.join(PROCESSED_DIR, "train", "hem") # cellules saines
TRAIN_ALL_DIR = os.path.join(PROCESSED_DIR, "train", "all") # cellules malades

# Sauvegarde du modèle entraîné
MODEL_PATH = "resnet_cellule.pth"

# Hyperparamètres 
BATCH_SIZE = 32 # nombre d'images traitées en même temps
LR = 1e-4 # learning rate (vitesse d'apprentissage)
EPOCHS = 20 # nombre de passages complets sur le dataset
TRAIN_SPLIT = 0.80  # 80% pour entraîner, 20% pour valider

# Mapping des labels
LABEL_HEM = 0   # "hem" = sain
LABEL_ALL = 1   # "all" = malade


############################################################## 
#               charge les .npy + label                      #
############################################################## 

class CellulesNPYDataset(Dataset):
    """
    Dataset PyTorch qui lit des images prétraitées .npy.

    Chaque item retourne :
    - un tenseur image de forme [3, H, W] (format attendu par ResNet)
    - un label (0 ou 1)
    """

    def __init__(self, paths, labels):
        # Liste des chemins vers les fichiers .npy
        # Liste des labels associés
        self.paths = paths
        self.labels = labels

    def __len__(self):
        # Nombre total d'images dans le dataset
        return len(self.paths)

    def __getitem__(self, idx):
        # Charger le fichier .npy (image normalisée entre 0 et 1)
        img = np.load(self.paths[idx])  # shape typique : (H, W, 3)

        # Convertir en tenseur PyTorch
        # ResNet attend [C, H, W], donc on transpose (H, W, C) -> (C, H, W)
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)

        # Récupérer le label
        y = torch.tensor(self.labels[idx], dtype=torch.long)

        return img, y


############################################################## 
   #    Créer la liste (paths, labels) depuis processed_data    # 
################################################################# 


def _charger_paths_et_labels():
    """
    Construit les listes :
    - paths : chemins vers tous les fichiers .npy
    - labels : labels associés (0 ou 1)

    Avec lee label dépendant uniquement du dossier d'origine (hem ou all).
    """
    hem_paths = sorted(glob(os.path.join(TRAIN_HEM_DIR, "*.npy")))
    all_paths = sorted(glob(os.path.join(TRAIN_ALL_DIR, "*.npy")))

    paths = hem_paths + all_paths
    labels = ([LABEL_HEM] * len(hem_paths)) + ([LABEL_ALL] * len(all_paths))

    return paths, labels


############################################################## 
# FONCTION PRINCIPALE                      
############################################################## 

def entrainer_modele():
    """
    Fonction appelée depuis main.py.

    Étapes :
    1) Charger les chemins .npy + labels
    2) Créer Dataset + split train/val
    3) Définir DataLoaders
    4) Créer le modèle ResNet18
    5) Entraîner sur EPOCHS époques
    6) Sauvegarder le modèle (.pth)
    """
    print("\n---DÉMARRAGE DU PROGRAMME D'ENTRAÎNEMENT RESNET---")
    
    ############################################################## 
    #       Vérifier qu'on a bien des données prétraitées        #
    ############################################################## 
    if not os.path.isdir(TRAIN_HEM_DIR) or not os.path.isdir(TRAIN_ALL_DIR):
        print("[ERREUR] Dossiers de données prétraitées introuvables.")
        print("Attendu :")
        print(f" - {TRAIN_HEM_DIR}")
        print(f" - {TRAIN_ALL_DIR}")
        print("Lance d'abord le prétraitement (preparer_donnees).")
        return

    paths, labels = _charger_paths_et_labels()

    if len(paths) == 0:
        print("[ERREUR] Aucun fichier .npy trouvé dans training/hem ou training/all.")
        return

    # infos utiles pour le suivi
    nb_hem = sum(1 for y in labels if y == LABEL_HEM)
    nb_all = sum(1 for y in labels if y == LABEL_ALL)
    print(f"Images trouvées : {len(paths)}")
    print(f"Distribution - hem (sain): {nb_hem} | all (malade): {nb_all}")

    ############################################################## 
    #                 Création du dataset                        #
    ############################################################## 
    dataset = CellulesNPYDataset(paths, labels)

    ############################################################## 
    #                  Split train/validation                    #
    ############################################################## 
    taille_train = int(TRAIN_SPLIT * len(dataset))
    taille_val = len(dataset) - taille_train

    dataset_train, dataset_val = random_split(dataset, [taille_train, taille_val])

    print(f"Dataset d'entraînement : {len(dataset_train)} images")
    print(f"Dataset de validation : {len(dataset_val)} images")

    ############################################################## 
    #                  DataLoaders (batches)                     #
    ############################################################## 
    train_loader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False)

    ############################################################## 
    #      Choix du device (CPU par défaut, GPU si dispo)        #
    ############################################################## 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device utilisé : {device}")

    ############################################################## 
    #         Création du modèle (via src/modele.py)             #
    ############################################################## 
    model = creer_modele(nb_classes=2, pretrained=True)
    model.to(device)

    ############################################################## 
    #                  Loss + optimiseur                         #
    ############################################################## 
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    ############################################################## 
    #                Boucle d'entraînement                       #
    ############################################################## 
    print(f"\nEntraînement sur {EPOCHS} époques :")
    print("------------------------------------------------------------")

    for epoch in range(EPOCHS):
        # ====== Phase TRAIN ======
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, y_true in train_loader:
            images = images.to(device)
            y_true = y_true.to(device)

            # 1) Forward
            logits = model(images)

            # 2) Calcul de la loss
            loss = criterion(logits, y_true)

            # 3) Backprop + update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 4) Stats
            train_loss += loss.item() * images.size(0)
            y_pred = torch.argmax(logits, dim=1)
            train_correct += (y_pred == y_true).sum().item()
            train_total += y_true.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # ====== Phase VALIDATION ======
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        # On désactive le calcul des gradients (plus rapide)
        with torch.no_grad():
            for images, y_true in val_loader:
                images = images.to(device)
                y_true = y_true.to(device)

                logits = model(images)
                loss = criterion(logits, y_true)

                val_loss += loss.item() * images.size(0)
                y_pred = torch.argmax(logits, dim=1)
                val_correct += (y_pred == y_true).sum().item()
                val_total += y_true.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        # Affichage clair 
        print(f"Époque {epoch+1}/{EPOCHS}")
        print(f"  Train -> Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val   -> Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

    print("------------------------------------------------------------")
    print("\nEntraînement terminé")

    ############################################################## 
    #                Sauvegarde du modèle                        #
    ############################################################## 
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"✓ Modèle sauvegardé sous '{MODEL_PATH}'")
    print("\n---PROGRAMME TERMINÉ AVEC SUCCÈS---")
   
