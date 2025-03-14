# AI Model for Electronic Component Detection

## Overview
This AI model is designed to detect and classify electronic components from image inputs. It leverages deep learning and transfer learning techniques to recognize various components, such as capacitors, transistors, relays, and cables.

## Dataset
The dataset used for training the model is available on:
- **GitHub**: [Electronic Components Dataset](https://github.com/prasanna0810/Electronic-components-dataset)
- **Kaggle**: [Electronic Components Dataset](https://www.kaggle.com/datasets/prasannadevika/electronic-components)

The dataset is divided into:
- **Training Set (80%)**
- **Validation Set (10%)**
- **Test Set (10%)**

This split is performed using the `data_split.py` script.

## Preprocessing Steps
Key preprocessing steps applied to the dataset:
1. **Resize Images**: Standardized all images to **224x224** (compatible with CNNs like ResNet, VGG, etc.).
2. **Normalize Pixel Values**: Scaled pixel values to [0,1] (or -1 to 1 for ResNet models).
3. **Data Augmentation** (optional but recommended):
   - Random rotation
   - Horizontal and vertical flipping
   - Zooming
   - Brightness adjustments
   - Adding noise

### Preprocessing & Augmentation Script
The script `data_prepare.py` handles:
✔️ Resizing all images to **224x224**
✔️ Normalizing pixel values (0-255 → 0-1)
✔️ Applying augmentations to improve generalization
✔️ Saving the preprocessed images to a new directory

## Model Architecture: Transfer Learning with ResNet50
The model is implemented using **PyTorch** and **ResNet50** for transfer learning.

### Model Summary
The model architecture is based on **ResNet50**, with the final fully connected layer modified to classify **34 electronic components**. Here is a summary of the architecture:
```
ResNet(
  (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU(inplace=True)
  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  (layer1): Sequential(...)
  (layer2): Sequential(...)
  (layer3): Sequential(...)
  (layer4): Sequential(...)
  (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
  (fc): Linear(in_features=2048, out_features=34, bias=True)
)
```

### Training Process
The `train.py` script includes:
✔ Loading the preprocessed dataset
✔ Applying data augmentation
✔ Fine-tuning **ResNet50** for component classification
✔ Training and evaluating the model

## Model Performance
The model achieved an **overall accuracy of 91%** on the test set. Here are key performance metrics:
- **Precision (macro avg):** 0.92
- **Recall (macro avg):** 0.91
- **F1-score (macro avg):** 0.91

### Training Metrics
The training process was completed in **2 epochs**, with the following performance:
- **Epoch 1:**
  - Train Loss: **0.4745**
  - Train Accuracy: **86.07%**
  - Validation Loss: **0.3556**
  - Validation Accuracy: **89.51%**
  - ✅ Model saved!

- **Epoch 2:**
  - Train Loss: **0.1684**
  - Train Accuracy: **94.72%**
  - Validation Loss: **0.2814**
  - Validation Accuracy: **91.68%**

### Category-wise Performance
Some notable class-wise F1-scores:
- **Fiber Optic Cables:** 1.00
- **LED Character and Numeric:** 0.99
- **Mica and PTFE Capacitors:** 0.99
- **Batteries (Rechargeable):** 0.81
- **Motors (AC/DC):** 0.73

For a detailed breakdown of precision, recall, and F1-scores for each component category, see `classification_report.txt`.

## Model Inference
To classify an image, the `test.py` script:
✔ Loads the trained model (`best_model.pth`)
✔ Preprocesses the input image (resize, normalize, convert to tensor)
✔ Runs inference and predicts the electronic component
✔ Displays details about the identified component

## Installation & Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/prasanna0810/Electronic-Component-Detection.git
   cd Electronic-Component-Detection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Prepare the dataset:
   ```bash
   python data_prepare.py
   ```
4. Train the model:
   ```bash
   python train.py
   ```
5. Test the model on an image:
   ```bash
   python test.py --image path/to/image.jpg
   ```

## Acknowledgments
Thanks to Digi-Key for providing access to component data and images.
