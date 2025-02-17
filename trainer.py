import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader
from kaggle.api.kaggle_api_extended import KaggleApi

# Configuration
DATASET_NAME = "trainingdatapro/gender-detection-and-classification-image-dataset"
DATA_DIR = "data"
MODEL_PTH = "gender-detection-and-classification-image-dataset.pth"
BATCH_SIZE = 32 * 2
EPOCHS = 100
IMG_SIZE = 224  # Standard size for pretrained models

# Setup Kaggle API
api = KaggleApi()
api.authenticate()

# Create directories
os.makedirs(DATA_DIR, exist_ok=True)

# Download and extract dataset
print("Downloading dataset...")
api.dataset_download_files(DATASET_NAME, path=DATA_DIR, unzip=True)

# Dataset paths
train_dir = os.path.join(DATA_DIR, "train")
test_dir = os.path.join(DATA_DIR, "test")

# Data transformations and augmentation
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
test_dataset = datasets.ImageFolder(test_dir, transform=val_transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model setup (using pretrained ResNet-18)
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(train_dataset.classes))  # 6 classes for Intel dataset

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Training loop
best_accuracy = 0.0

print("Starting training...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    
    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_corrects.double() / len(train_dataset)
    
    # Validation
    model.eval()
    val_running_corrects = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            val_running_corrects += torch.sum(preds == labels.data)
    
    val_acc = val_running_corrects.double() / len(test_dataset)
    
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
    print(f"Val Acc: {val_acc:.4f}")
    
    # Save best model
    if val_acc > best_accuracy:
        best_accuracy = val_acc
        torch.save(model.state_dict(), MODEL_PTH)
        print(f"New best model saved with accuracy {val_acc:.4f}")
    
    scheduler.step()

print(f"Training complete. Best validation accuracy: {best_accuracy:.4f}")