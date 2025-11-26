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

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
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
