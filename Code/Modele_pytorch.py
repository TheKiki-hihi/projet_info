########################
# --- IMPORATTIONS --- #
########################
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
import os
import Pathlib as Path

##################################
# --- CHARGEMENT DES DONNEES --- #
##################################
# charger les donnees + les rendres accessibles pour pytorch
ROOT_NPY_DIR = 'processed_data' 
TRAIN_NPY_DIR = os.path.join(ROOT_NPY_DIR, 'train')
VAL_NPY_DIR = os.path.join(ROOT_NPY_DIR, 'validation')

BATCH_SIZE = 42 # Nb images par lot (je savais pas quoi mettre mdrr, à tester)

class DataSet_Cell(DataSet):
def __init__(self, chemin_principal):
        # Initialisation
        self.fichiers_listes = []
        self.map_classes = {'hem': 0, 'all': 1}
        
        # On va parcourir les deux sous-dossiers ('hem' et 'all').
        for nom_classe in ['hem', 'all']:
            chemin_classe = os.path.join(chemin_principal, nom_classe)
            
            # On vérifie si le dossier existe
            if os.path.isdir(chemin_classe):
                # On liste les fichiers .npy dans ce dossier 
                for nom_fichier in os.listdir(chemin_classe):
                    if nom_fichier.endswith('.npy'):
                        # On stocke le chemin complet de chaque fichier
                        chemin_complet = os.path.join(chemin_classe, nom_fichier)
                        self.fichiers_listes.append(chemin_complet)
def __getitem__(self, idx):
        chemin_complet = self.fichiers_listes[idx]
        # Lecture du fichier .npy
        img_array = np.load(chemin_complet) 
        
        # On extrait la classe à partir du chemin complet
        label_str = Path(chemin_complet).parent.name 
        label_int = self.map_classes.get(label_str) 
        
        # Convertit la matrice NumPy de l'image (H, L, C) en Tenseur PyTorch (C, H, L) 
        tenseur_img = torch.from_numpy(img_array).permute(2, 0, 1).float()
        tenseur_label = torch.tensor(label_int, dtype=torch.long)
        
        return tenseur_img, tenseur_label
                    

if __name__ == "__main__":
# Créer les instances de Dataset pour le train et la validation
    train_dataset = DataSet_Cell(TRAIN_NPY_DIR)
    val_dataset = DataSet_Cell(VAL_NPY_DIR)
    
    train_loader = DataLoader(
            root = train_dataset, 
            batch_size = BATCH_SIZE, 
            shuffle=True, 
        )
        
    val_loader = DataLoader(
            root = val_dataset, 
            batch_size = BATCH_SIZE, 
            shuffle=False, #shuffle=False parce qu'on s'en fou de  l'ordre 
        )

########################
# --- MODÈLE (CNN) --- #
########################
# Partie pour faire notre modele. je propose de partir sur un CNN 


########################
# --- ENTRAINEMENT --- #
########################
# Partie pour entrainer notre modele 

######################
# --- EVALUATION --- #
######################
# Partie pour evaluer notre modele 
