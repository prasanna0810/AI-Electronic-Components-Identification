import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# Define paths
DATA_DIR = r"C:\Users\prasa\Documents\ec_it\processed_dataset"
TRAIN_DIR = f"{DATA_DIR}/train"
VALID_DIR = f"{DATA_DIR}/valid"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load Dataset
train_dataset = ImageFolder(root=TRAIN_DIR, transform=transform)
valid_dataset = ImageFolder(root=VALID_DIR, transform=transform)

print(f"🗂 Training samples: {len(train_dataset)}")
print(f"🗂 Validation samples: {len(valid_dataset)}")

# Training loop
def train_model(model, train_loader, valid_loader, epochs=10):
    print("🚀 Training started...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_acc = 0
    for epoch in range(epochs):
        print(f"\n🟢 Epoch {epoch+1}/{epochs} started...")
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total
        avg_train_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0

        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(valid_loader)

        # Print accuracy and loss for the epoch
        print(f"📊 Epoch {epoch+1}/{epochs}")
        print(f"   🔹 Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
        print(f"   🔹 Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")

        # Save the best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print("✅ Model saved!")

    print("\n🎉 Training Complete!")
    return model


# Run training inside main check
if __name__ == "__main__":
    print("Script started...")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)  # 🔹 Set num_workers=0 to avoid multiprocessing issues
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Load Pre-trained ResNet50 model
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)  # 🔹 Updated to avoid deprecated 'pretrained=True'
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(train_dataset.classes))
    model = model.to(device)

    # Train the model
    model = train_model(model, train_loader, valid_loader, epochs=20)
