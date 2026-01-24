# Mask Detection Using MobileNetV2

## Project Overview
This project implements a **Mask Detection System** using **MobileNetV2** as the base model.  
The goal is to classify images as **Mask** or **No Mask** with high accuracy.

**Key Features:**
- Binary classification (Mask / No Mask)  
- Transfer learning with MobileNetV2  
- Data augmentation for robust training  
- Model evaluation using precision, recall, F1-score  
- Training and validation loss/accuracy visualization  
- Model serialization for deployment  

---

## Dataset
- The dataset contains **images of faces with and without masks**.  
- Each image is labeled as `Mask` or `No Mask`.  
- Images are **preprocessed and resized to 224x224 pixels** for MobileNetV2.

---

## Project Steps

| Step No. | Description |
|----------|-------------|
| 1 | Load and preprocess images and labels |
| 2 | Split dataset into training (80%) and testing (20%) sets |
| 3 | Apply data augmentation to training images to improve generalization |
| 4 | Load MobileNetV2 without the top layer (transfer learning) |
| 5 | Build the head model with AveragePooling, Flatten, Dense, Dropout, and final Dense layer |
| 6 | Freeze base MobileNetV2 layers to avoid updating pre-trained weights during initial training |
| 7 | Compile the model with binary crossentropy loss and Adam optimizer |
| 8 | Train the model on augmented training data |
| 9 | Make predictions on the testing set and convert probabilities to binary labels |
|10 | Evaluate model using classification report (precision, recall, F1-score) |
|11 | Plot training/validation loss and accuracy curves |
|12 | Save the trained model to disk (.h5) for future use |

---

## Model Architecture

### Data Splitting & Augmentation

| Layer              | Output Shape       | Purpose                      |
| ------------------ | ------------------ | ---------------------------- |
| MobileNetV2 (base) | (None, 7, 7, 1280) | Feature extraction           |
| AveragePooling2D   | (None, 1, 1, 1280) | Reduce spatial size          |
| Flatten            | (None, 1280)       | Flatten features             |
| Dense(128, ReLU)   | (None, 128)        | Learn dense features         |
| Dropout(0.5)       | (None, 128)        | Prevent overfitting          |
| Dense(1, Sigmoid)  | (None, 1)          | Binary classification output |

---

## Model Evaluation

-Training & Evaluation
-Loss Function: Binary Crossentropy
-Optimizer: Adam
-Metrics: Accuracy
-Epochs: EPOCHS (set in code)
-Batch Size: BS (set in code)
-Performance is evaluated using classification report (precision, recall, F1-score).

---

## Dependencies

-Python 3.x
-TensorFlow / Keras
-NumPy
-Matplotlib
-scikit-learn
-OpenCV

---

## References

-MobileNetV2 Paper: https://arxiv.org/abs/1801.04381
-Keras ImageDataGenerator: https://keras.io/api/preprocessing/image/
