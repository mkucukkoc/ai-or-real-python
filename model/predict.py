import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from io import BytesIO  # ✅ Gerekli import
import cv2
import numpy as np
from PIL import Image

# Modeli yükle
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("model.pt", map_location=torch.device("cpu")))
model.eval()

# Sınıf isimleri (0 = AI, 1 = Human)
class_names = ["ai", "human"]

# Görseli dönüştürmek için transform işlemi
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def is_image_high_quality(image: Image.Image) -> bool:
    # PIL → NumPy → OpenCV formatına çevir
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    
    # Laplacian kullanarak bulanıklık ölçümü
    laplacian_var = cv2.Laplacian(image_cv, cv2.CV_64F).var()

    # Eşik değer (deneysel olarak 100 iyi sonuç verir)
    return laplacian_var > 100


def predict_image(image_bytes: bytes):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)

    verdict = class_names[pred.item()]
    confidence = confidence.item()

    # QUALITY ANALİZİ BURADA
    quality_flag = is_image_high_quality(image)

    return {
        "verdict": verdict,
        "ai": {
            "is_detected": verdict == "ai",
            "confidence": confidence if verdict == "ai" else 1 - confidence
        },
        "human": {
            "is_detected": verdict == "human",
            "confidence": confidence if verdict == "human" else 1 - confidence
        },
        "quality": {
            "is_detected": quality_flag
        }
    }

