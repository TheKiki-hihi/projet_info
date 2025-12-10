import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ImageFolder
from torchvision import transforms, models
import os

# Configuration
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Chemin vers vos données
DATA_PATH = r"c:\Users\Etudiant\Desktop\Licence Science et Technologique\L3\S1\Projet Info\training"

# Augmentation et normalisation des images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Chargement du dataset
train_dataset = ImageFolder(DATA_PATH, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Modèle ResNet50 pré-entraîné
model = models.resnet50(pretrained=True)
num_classes = len(train_dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(DEVICE)

# Optimiseur et fonction de perte
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Entraînement
print(f"Nombre de classes détectées: {num_classes}")
print(f"Appareil utilisé: {DEVICE}")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")

# Sauvegarde du modèle
torch.save(model.state_dict(), "resnet50_cellules.pth")
print("Modèle sauvegardé avec succès!")
