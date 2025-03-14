import torch
import torchvision.models as models
import torch.nn as nn
from torchsummary import summary


# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50()
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 34)  

# Load the trained weights
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

# Print model summary
print("📜 Model Summary:")
summary(model, (3, 224, 224))

# Save the model architecture to a file
with open("model_summary.txt", "w") as f:
    f.write(str(model))

print("✅ Model summary saved to 'model_summary.txt'.")

import pandas as pd

# Create a dataframe with training results
report_data = {
    "Epoch": [1, 2],  # Update dynamically if more epochs
    "Train Loss": [0.4745, 0.1684],
    "Train Accuracy": [86.07, 94.72],
    "Validation Loss": [0.3556, 0.2814],
    "Validation Accuracy": [89.51, 91.68],
}

df = pd.DataFrame(report_data)
df.to_csv("training_report.csv", index=False)

print("📊 Training report saved as 'training_report.csv'.")

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets, models
from sklearn.metrics import classification_report
import numpy as np

# Define test data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load test dataset
test_dataset = datasets.ImageFolder(root=r"C:\Users\prasa\Documents\ec_it\processed_dataset\test", transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50()
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(test_dataset.classes))

model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

# Perform evaluation
y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# Generate classification report
report = classification_report(y_true, y_pred, target_names=test_dataset.classes)
print(report)

# Save to a file
with open("classification_report.txt", "w") as f:
    f.write(report)

print("✅ Classification report saved as 'classification_report.txt'.")
