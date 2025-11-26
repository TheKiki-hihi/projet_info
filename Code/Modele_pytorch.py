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

if __name__ == "__main__":
# Créer les instances de Dataset pour le train et la validation
    train_dataset = CellDataset(TRAIN_NPY_DIR)
    val_dataset = CellDataset(VAL_NPY_DIR)
    
    train_loader = DataLoader(
            root = train_dataset, 
            batch_size = BATCH_SIZE, 
            shuffle=True, 
            num_workers=NUM_WORKERS
        )
        
        # Validation : shuffle=False parce qu'on s'en fou de  l'ordre 
    val_loader = DataLoader(
            root = val_dataset, 
            batch_size = BATCH_SIZE, 
            shuffle=False, 
            num_workers=NUM_WORKERS
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
