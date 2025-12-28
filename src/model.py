"""
src/modele.py

Ce fichier contient uniquement la création du modèle.
L'objectif est d'avoir UNE SEULE source pour l'architecture
(ResNet18 + couche finale modifiée) pour éviter de dupliquer le code
dans train.py et predict.py.
"""

import torch
import torch.nn as nn
import torchvision.models as models


def creer_modele(nb_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """
    Crée et retourne un modèle ResNet18 adapté à notre problème.

    Paramètres
    ----------
    nb_classes : int
        Nombre de classes à prédire.
        Dans notre cas : 2 (ex : "Sain" vs "Malade").
    pretrained : bool
        Si True, on charge des poids pré-entraînés (ImageNet) pour aider
        l'apprentissage (transfer learning).

    Retour
    ------
    modele : torch.nn.Module
        Le modèle ResNet18 avec la dernière couche (fc) remplacée.
    """

    # ------------------------------------------------------------
    # 1) Création du backbone ResNet18
    # ------------------------------------------------------------
    # Selon la version de torchvision, l'argument "pretrained" peut être déprécié.
    # On gère les deux cas pour éviter que ça casse chez quelqu'un d'autre.
    try:
        # Nouvelle API (torchvision >= 0.13)
        if pretrained:
            weights = models.ResNet18_Weights.DEFAULT
        else:
            weights = None
        modele = models.resnet18(weights=weights)
    except Exception:
        # Ancienne API (torchvision plus ancienne)
        modele = models.resnet18(pretrained=pretrained)

    # ------------------------------------------------------------
    # 2) Remplacement de la couche finale
    # ------------------------------------------------------------
    # ResNet18 se termine par une couche fully-connected : modele.fc
    # On remplace cette couche pour obtenir nb_classes sorties.
    in_features = modele.fc.in_features
    modele.fc = nn.Linear(in_features, nb_classes)

    return modele


def charger_modele(chemin_poids: str, nb_classes: int = 2, device: str = "cpu") -> nn.Module:
    """
    Charge un modèle (architecture + poids) prêt pour l'inférence.

    Paramètres
    ----------
    chemin_poids : str
        Chemin vers le fichier .pth (poids sauvegardés).
    nb_classes : int
        Nombre de classes (doit correspondre au modèle entraîné).
    device : str
        "cpu" ou "cuda" si GPU disponible.

    Retour
    ------
    modele : torch.nn.Module
        Modèle en mode évaluation (eval) avec ses poids chargés.
    """

    # On recrée exactement la même architecture
    modele = creer_modele(nb_classes=nb_classes, pretrained=False)

    # On charge les poids sur le bon device
    state_dict = torch.load(chemin_poids, map_location=device)
    modele.load_state_dict(state_dict)

    # On envoie le modèle sur le device + mode évaluation (pas de dropout, etc.)
    modele.to(device)
    modele.eval()

    return modele
