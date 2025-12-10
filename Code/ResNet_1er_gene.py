# Programme d'entraînement ResNet50 pour classification d'images de cellules

# ============================================================================
# IMPORTS
# ============================================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ImageFolder
from torchvision import transforms, models
import os

# ============================================================================
# CONFIGURATION DU PROGRAMME
# ============================================================================
# Nombre d'images à traiter par lot
BATCH_SIZE = 32

# Nombre de passages complets sur l'ensemble des données d'entraînement
EPOCHS = 20

# Vitesse d'apprentissage du modèle (plus petit = apprentissage plus lent mais stable)
LEARNING_RATE = 0.001

# Choix du processeur (GPU si disponible, sinon CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# CHEMIN VERS LES DONNÉES
# ============================================================================
# Chemin du dossier contenant les images d'entraînement
# Les images doivent être organisées en sous-dossiers par classe
DATA_PATH = r"c:\Users\Etudiant\Desktop\Licence Science et Technologique\L3\S1\Projet Info\training"

# ============================================================================
# PRÉPARATION DES DONNÉES
# ============================================================================
# Définition des transformations à appliquer aux images
transform = transforms.Compose([
    # Redimensionner toutes les images à 224x224 (taille attendue par ResNet)
    transforms.Resize((224, 224)),
    
    # Convertir les images PIL en tenseurs PyTorch
    transforms.ToTensor(),
    
    # Normaliser avec les statistiques ImageNet (pour meilleure performance)
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Charger les images depuis le dossier
# ImageFolder organise automatiquement les données par dossiers (1 dossier = 1 classe)
train_dataset = ImageFolder(DATA_PATH, transform=transform)

# Créer un chargeur de données qui distribue les images par lots
# shuffle=True : mélanger les données pour un meilleur apprentissage
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ============================================================================
# CRÉATION DU MODÈLE
# ============================================================================
# Charger ResNet50 pré-entraîné sur ImageNet
# pretrained=True : utiliser les poids initiaux entraînés sur ImageNet
model = models.resnet50(pretrained=True)

# Déterminer le nombre de classes à partir des données
# = nombre de dossiers dans le répertoire training
num_classes = len(train_dataset.classes)

# Remplacer la dernière couche pour s'adapter à notre nombre de classes
# ResNet50 a par défaut 1000 classes (ImageNet)
# On la remplace pour notre nombre de classes spécifique
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Déplacer le modèle sur le GPU ou CPU
model = model.to(DEVICE)

# ============================================================================
# CONFIGURATION DE L'OPTIMISEUR ET DE LA PERTE
# ============================================================================
# CrossEntropyLoss : fonction de perte pour la classification multi-classe
criterion = nn.CrossEntropyLoss()

# Adam : optimiseur qui adapte le taux d'apprentissage automatiquement
# optimizer.parameters() : ajuste tous les poids du modèle
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ============================================================================
# AFFICHAGE DES INFORMATIONS
# ============================================================================
print(f"Nombre de classes détectées: {num_classes}")
print(f"Classes trouvées: {train_dataset.classes}")
print(f"Nombre d'images d'entraînement: {len(train_dataset)}")
print(f"Appareil utilisé: {DEVICE}")
print(f"Début de l'entraînement...\n")

# ============================================================================
# BOUCLE D'ENTRAÎNEMENT
# ============================================================================
# Parcourir le nombre d'epochs défini
for epoch in range(EPOCHS):
    # Mettre le modèle en mode entraînement
    # (certaines couches comme Dropout et BatchNorm se comportent différemment)
    model.train()
    
    # Variable pour accumuler la perte totale de l'epoch
    total_loss = 0
    
    # Parcourir tous les lots de données
    for batch_idx, (images, labels) in enumerate(train_loader):
        # Déplacer les images et labels sur le GPU ou CPU
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        
        # ====== FORWARD PASS ======
        # Passer les images dans le modèle pour obtenir les prédictions
        outputs = model(images)
        
        # Calculer la perte (erreur entre prédictions et labels réels)
        loss = criterion(outputs, labels)
        
        # ====== BACKWARD PASS ======
        # Remettre les gradients à zéro (éviter l'accumulation)
        optimizer.zero_grad()
        
        # Calculer les gradients de la perte par rapport aux poids
        loss.backward()
        
        # Mettre à jour les poids du modèle
        optimizer.step()
        
        # Ajouter la perte du lot à la perte totale
        total_loss += loss.item()
    
    # Calculer la perte moyenne pour cet epoch
    avg_loss = total_loss / len(train_loader)
    
    # Afficher la progression
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss moyenne: {avg_loss:.4f}")

# ============================================================================
# SAUVEGARDE DU MODÈLE
# ============================================================================
# Sauvegarder les poids du modèle entraîné pour une utilisation future
model_path = "resnet50_cellules.pth"
torch.save(model.state_dict(), model_path)
print(f"\n✓ Modèle sauvegardé avec succès dans: {model_path}")
