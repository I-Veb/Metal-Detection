## Complex tasks like segmentation and object detection require the data to have pixel map annotations and bounding box annotations respectively and it becomes very time consuming. Since, considering time limitation and the dataset does not include a single object in every image and most of them either contains lots of tiny object OR in the tiles format. Instead of annotating every image, I classified into two categories and made two approaches. one with simple CNN model(With 30 epochs) and other with data augmentation and transfer learning(With less epochs).



# Metal Detection using Deep Learning

This project uses deep learning techniques to classify and detect metal images into two categories: Copper/Brass and Steel/Other. It employs transfer learning with a VGG16 base model, data augmentation, and custom layers for fine-tuning.

## Project Overview

1. Data Collection
2. Data Preprocessing and Augmentation
3. Model Architecture
4. Model Training
5. Model Evaluation
6. Single Image Testing

## Detailed Steps

### 1. Data Collection

- The dataset consists of metal images categorized into Copper/Brass and Steel/Other.
- Images are stored in a directory structure where each category has its own folder.
- The main data directory is located at '/content/drive/MyDrive/Dataset_Metal/Dataset'.

### 2. Data Preprocessing and Augmentation

- Images are resized to 224x224 pixels to match VGG16's input size.
- Data augmentation is applied using Keras' ImageDataGenerator:
  - Random rotations (up to 20 degrees)
  - Width and height shifts (up to 20%)
  - Shear transformations
  - Zoom (up to 20%)
  - Horizontal flips
- Images are rescaled to have pixel values between 0 and 1.
- The dataset is split into training (80%) and validation (20%) sets.

### 3. Model Architecture

- Base Model: Pre-trained VGG16 (weights from ImageNet, without top layers)
- Custom Top Layers:
  - Flatten layer
  - Dense layer (512 units, ReLU activation)
  - Dense layer (256 units, ReLU activation)
  - Output Dense layer (1 unit, Sigmoid activation)

### 4. Model Training

- Optimizer: Adam with a learning rate of 0.0001
- Loss Function: Binary Crossentropy
- Metrics: Accuracy
- Number of Epochs: 30
- The base VGG16 layers are frozen during training (transfer learning).

### 5. Model Evaluation

- The model's performance is evaluated on a separate validation set.
- Evaluation metrics include loss and accuracy.
- Training and validation accuracy/loss are plotted over epochs to visualize the model's learning progress.

### 6. Single Image Testing

- The trained model can classify individual metal images.
- Test images are preprocessed (resized, normalized) before prediction.
- The model outputs a probability, which is then classified as Copper/Brass (>0.5) or Steel/Other (≤0.5).

## Requirements

- TensorFlow 2.x
- Keras
- NumPy
- OpenCV
- Matplotlib

## Improvements and Finetuning can be done if more time is provided.

- Fine-tune hyperparameters for better performance.
- Experiment with different pre-trained models.
- Implement cross-validation for more robust evaluation.
- Expand the dataset or use data synthesis techniques for better generalization.
