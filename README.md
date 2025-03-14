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

### Training Process
The `train.py` script includes:
✔ Loading the preprocessed dataset
✔ Applying data augmentation
✔ Fine-tuning **ResNet50** for component classification
✔ Training and evaluating the model

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

## Future Enhancements
- Improve accuracy with additional data augmentation techniques
- Experiment with different architectures (EfficientNet, ViT, etc.)
- Deploy as a web app for real-time classification

## License
This project is for research and educational purposes. Please ensure compliance with dataset usage policies.

## Acknowledgments
Thanks to Digi-Key for providing access to component data and images.

