# train.py

import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Config
DATA_DIR = "test/test"
BATCH_SIZE = 16
EPOCHS = 5
IMG_SIZE = 224
MODEL_PATH = "model.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Dataset
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
class_names = dataset.classes  # ['ai', 'human']

print("Veri sayısı:", len(dataset))
print("Sınıflar:", dataset.classes)
print("CUDA durumu:", torch.cuda.is_available())

# Model
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(DEVICE)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training
for epoch in range(EPOCHS):
    running_loss = 0.0
    model.train()
    print(f">>> Epoch {epoch+1} başladı")

    for images, labels in dataloader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss:.4f}")

# Save
torch.save(model.state_dict(), MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
