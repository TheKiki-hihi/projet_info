import os  # Pour la gestion des fichiers et dossiers
import numpy as np  # Pour manipuler les tableaux d'images
import torch  # Framework principal pour le deep learning
import torch.nn as nn  # Pour les couches de réseaux de neurones
import torch.optim as optim  # Pour les optimisateurs
from torch.utils.data import Dataset, DataLoader  # Pour la gestion des données
from torchvision import models, transforms  # Pour les modèles pré-entraînés et les transformations
from sklearn.model_selection import train_test_split  # Pour séparer les données en train/val

# =========================
# 1. Définition du Dataset personnalisé
# =========================
class CellDataset(Dataset):
    """
    Dataset personnalisé pour charger les images de cellules et leurs labels.
    Les images sont chargées depuis des fichiers .npy et transformées pour le modèle.
    """
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths  # Liste des chemins vers les images
        self.labels = labels  # Liste des labels (0 = sain, 1 = malade)
        self.transform = transform  # Transformations à appliquer aux images

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Chargement de l'image au format numpy
        img = np.load(self.image_paths[idx])
        img = img.astype(np.float32)
        # Si l'image est en niveaux de gris (2D), on la convertit en 3 canaux pour ResNet
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)
        # Application des transformations (normalisation, conversion en tensor)
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

# =========================
# 2. Chargement des données et des labels
# =========================
image_dir = '../processed_data/testing_data/'  # Dossier contenant les images .npy
label_file = '../validation_data/C-NMC_test_prelim_phase_data_labels.csv'  # Fichier CSV des labels

# Récupère tous les chemins des fichiers .npy (images)
image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith('.npy')]

# Chargement des labels depuis le CSV
import pandas as pd
labels_df = pd.read_csv(label_file)  # Le CSV doit contenir les colonnes 'id' et 'label'
labels_dict = dict(zip(labels_df['id'], labels_df['label']))  # Dictionnaire id -> label

# Création de la liste des labels pour chaque image
labels = []
for path in image_paths:
    # Récupère l'id à partir du nom de fichier (ex: '123.npy' -> 123)
    img_id = os.path.splitext(os.path.basename(path))[0]
    labels.append(labels_dict.get(int(img_id), 0))  # Si l'id n'est pas trouvé, label=0 (sain)

# =========================
# 3. Préparation des DataLoader
# =========================
train_paths, val_paths, train_labels, val_labels = train_test_split(
    image_paths, labels, test_size=0.2, random_state=42)  # Séparation 80% train, 20% val

# Transformations pour les images (conversion en tensor et normalisation)
transform = transforms.Compose([
    transforms.ToTensor(),  # Conversion numpy -> tensor
    transforms.Normalize([0.5]*3, [0.5]*3)  # Normalisation des 3 canaux
])

# Création des datasets pour l'entraînement et la validation
train_dataset = CellDataset(train_paths, train_labels, transform)
val_dataset = CellDataset(val_paths, val_labels, transform)

# Création des DataLoaders pour charger les données par batch
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# =========================
# 4. Définition du modèle ResNet
# =========================
model = models.resnet18(pretrained=True)  # Charge un ResNet18 pré-entraîné sur ImageNet
# Remplace la dernière couche pour avoir 2 sorties (sain/malade)
model.fc = nn.Linear(model.fc.in_features, 2)

# =========================
# 5. Boucle d'entraînement
# =========================

# Détection du device (GPU si dispo, sinon CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Définition de la fonction de perte et de l'optimiseur
criterion = nn.CrossEntropyLoss()  # Pour la classification multi-classes
optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Optimiseur Adam

num_epochs = 20  # Nombre d'époques d'entraînement
for epoch in range(num_epochs):
    model.train()  # Mode entraînement
    running_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)  # Passage sur le device
        optimizer.zero_grad()  # Remise à zéro des gradients
        outputs = model(imgs)  # Prédiction du modèle
        loss = criterion(outputs, labels)  # Calcul de la perte
        loss.backward()  # Rétropropagation
        optimizer.step()  # Mise à jour des poids
        running_loss += loss.item() * imgs.size(0)  # Accumulation de la perte
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

    # Évaluation sur le jeu de validation
    model.eval()  # Mode évaluation
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)  # Classe prédite
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  # Nombre de bonnes prédictions
    val_acc = correct / total
    print(f'Validation Accuracy: {val_acc:.4f}')

# =========================
# 6. Sauvegarde du modèle
# =========================

# Sauvegarde des poids du modèle entraîné
torch.save(model.state_dict(), 'resnet_cellule.pth')
print('Modèle sauvegardé sous resnet_cellule.pth')

# =========================
# Ce script charge les images, prépare les labels, entraîne un ResNet18 et sauvegarde le modèle.
# Adapte les chemins et paramètres selon tes besoins.
#
# Structure du script :
# 1. Définition du Dataset personnalisé
# 2. Chargement des données et des labels
# 3. Préparation des DataLoader
# 4. Définition du modèle ResNet
# 5. Boucle d'entraînement et validation
# 6. Sauvegarde du modèle
#
# Pour toute question ou adaptation, demande-moi !
