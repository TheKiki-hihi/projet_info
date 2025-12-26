import os  # Pour la gestion des fichiers et dossiers
import numpy as np  # Pour manipuler les tableaux d'images
import pandas as pd  # Pour lire les fichiers CSV
import torch  # Framework principal pour le deep learning
import torch.nn as nn  # Pour les couches de réseaux de neurones
import torch.optim as optim  # Pour les optimisateurs
from torch.utils.data import Dataset, DataLoader  # Pour la gestion des données
from torchvision import models, transforms  # Pour les modèles pré-entraînés et les transformations
from sklearn.model_selection import train_test_split  # Pour séparer les données en train/val
import glob  # Pour rechercher des fichiers avec des patterns

print("=" * 60)
print("DÉMARRAGE DU PROGRAMME D'ENTRAÎNEMENT RESNET")
print("=" * 60)

# =========================
# 1. Définition du Dataset personnalisé
# =========================
print("\n[1/6] Définition du Dataset personnalisé...")
class CellDataset(Dataset):
    """
    Dataset personnalisé pour charger les images de cellules et leurs labels.
    Les images sont chargées depuis des fichiers .npy et transformées pour le modèle.
    """
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Chargement de l'image au format numpy
        img = np.load(self.image_paths[idx])
        img = img.astype(np.float32)
        
        # Si l'image est en niveaux de gris (2D), on la convertit en 3 canaux pour ResNet
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)
        
        # Normalisation manuelle (0-1) si l'image n'est pas déjà normalisée
        if img.max() > 1.0:
            img = img / 255.0
        
        # Conversion en tensor : copie explicite pour éviter les problèmes
        img = img.transpose(2, 0, 1).copy()
        img = torch.tensor(img, dtype=torch.float32)
        
        # Normalisation
        img = (img - 0.5) / 0.5
        
        label = self.labels[idx]
        return img, label

print("✓ Dataset personnalisé défini")

# =========================
# 2. Chargement des données et des labels
# =========================
print("\n[2/6] Chargement des données et des labels...")
image_dir = '/Users/kilperic/Desktop/projet_info/processed_data/training_data'
label_file = '/Users/kilperic/Desktop/projet_info/C-NMC_Leukemia/validation_data/C-NMC_test_prelim_phase_data_labels.csv'

print(f"  → Vérification du dossier: {image_dir}")
if not os.path.exists(image_dir):
    raise FileNotFoundError(f"Le dossier {image_dir} n'existe pas.")

print(f"  → Recherche des fichiers .npy...")
image_paths = glob.glob(os.path.join(image_dir, '**', '*.npy'), recursive=True)

if len(image_paths) == 0:
    raise ValueError(f"Aucun fichier .npy trouvé dans {image_dir}")

print(f"  ✓ Trouvé {len(image_paths)} images")

print(f"  → Chargement des labels depuis {label_file}")
labels_df = pd.read_csv(label_file)
labels_dict = dict(zip(labels_df['Patient_ID'], labels_df['labels']))
print(f"  ✓ {len(labels_dict)} labels chargés depuis le CSV")

print(f"  → Attribution des labels aux images...")
labels = []
for path in image_paths:
    img_id = os.path.splitext(os.path.basename(path))[0]
    
    if img_id in labels_dict:
        labels.append(labels_dict[img_id])
    else:
        parent_folder = os.path.basename(os.path.dirname(path))
        if parent_folder == 'hem':
            labels.append(1)
        elif parent_folder == 'all':
            labels.append(0)
        else:
            labels.append(0)

print(f"  ✓ Distribution des labels - Sains: {labels.count(0)}, Malades: {labels.count(1)}")

# =========================
# 3. Préparation des DataLoader
# =========================
print("\n[3/6] Préparation des DataLoaders...")
train_paths, val_paths, train_labels, val_labels = train_test_split(
    image_paths, labels, test_size=0.2, random_state=42)

print(f"  → Dataset d'entraînement: {len(train_paths)} images")
print(f"  → Dataset de validation: {len(val_paths)} images")

train_dataset = CellDataset(train_paths, train_labels, transform=None)
val_dataset = CellDataset(val_paths, val_labels, transform=None)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
print(f"  ✓ DataLoaders créés (batch_size=32)")

# =========================
# 4. Définition du modèle ResNet
# =========================
print("\n[4/6] Définition du modèle ResNet18...")
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)
print(f"  ✓ ResNet18 chargé et modifié (sortie: 2 classes)")

# =========================
# 5. Boucle d'entraînement
# =========================
print("\n[5/6] Démarrage de l'entraînement...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  → Utilisation du device: {device}")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
print(f"  → Fonction de perte: CrossEntropyLoss")
print(f"  → Optimiseur: Adam (lr=1e-4)")

num_epochs = 20
print(f"\n  Entraînement sur {num_epochs} époques:")
print("-" * 60)

for epoch in range(num_epochs):
    # Entraînement
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    print(f"\n  Époque {epoch+1}/{num_epochs}")
    print(f"    Phase d'entraînement...", end=" ")
    
    for batch_idx, (imgs, labels) in enumerate(train_loader):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        
        # Calcul de l'accuracy
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader.dataset)
    train_acc = correct_train / total_train
    print(f"Loss: {epoch_loss:.4f}, Acc: {train_acc:.4f}")

    # Validation
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    
    print(f"    Phase de validation...", end=" ")
    
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * imgs.size(0)
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_acc = correct / total
    val_loss = val_loss / len(val_loader.dataset)
    print(f"Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

print("\n" + "-" * 60)
print("✓ Entraînement terminé")

# =========================
# 6. Sauvegarde du modèle
# =========================
print("\n[6/6] Sauvegarde du modèle...")
torch.save(model.state_dict(), 'resnet_cellule.pth')
print("  ✓ Modèle sauvegardé sous 'resnet_cellule.pth'")

print("\n" + "=" * 60)
print("PROGRAMME TERMINÉ AVEC SUCCÈS")
print("=" * 60)
