# 🖼️ CIFAR-10 Image Classifier (Streamlit + TensorFlow)

This project is a simple **image classifier** built with:
- [TensorFlow/Keras](https://www.tensorflow.org/) for training a CNN on the **CIFAR-10 dataset**
- [Streamlit](https://streamlit.io/) for an interactive drag-and-drop web app
- [Flask](https://flask.palletsprojects.com/) (optional) for backend API deployment

---

## 🚀 Features
- Upload any image (e.g., `.jpg`, `.png`)
- Model resizes image to **32x32x3** and predicts CIFAR-10 classes:
  - Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck
- Displays prediction label + confidence score
- Option to show the full probability table

---

## 🧠 Model Architecture

The model is a **Convolutional Neural Network (CNN)** trained on the **CIFAR-10 dataset**.  
It achieved **~80% training and test accuracy** using **Batch Normalization** and **Image Data Augmentation**.

### 📊 Layer-by-Layer Summary

| Layer (Type)             | Output Shape   | Param #   | Description |
|---------------------------|----------------|-----------|-------------|
| **Input**                | (32, 32, 3)    | 0         | Input image (RGB, 32×32) |
| **Conv2D (32, 3×3)**     | (32, 32, 32)   | 896       | Feature extraction |
| **BatchNormalization**   | (32, 32, 32)   | 128       | Normalizes activations |
| **Conv2D (32, 3×3)**     | (32, 32, 32)   | 9,248     | Deeper features |
| **BatchNormalization**   | (32, 32, 32)   | 128       | Stabilization |
| **MaxPooling2D (2×2)**   | (16, 16, 32)   | 0         | Downsampling |
| **Dropout (0.25)**       | (16, 16, 32)   | 0         | Prevents overfitting |
| **Conv2D (64, 3×3)**     | (16, 16, 64)   | 18,496    | Deeper feature extraction |
| **BatchNormalization**   | (16, 16, 64)   | 256       | Normalization |
| **Conv2D (64, 3×3)**     | (16, 16, 64)   | 36,928    | Complex features |
| **BatchNormalization**   | (16, 16, 64)   | 256       | Stabilization |
| **MaxPooling2D (2×2)**   | (8, 8, 64)     | 0         | Downsampling |
| **Dropout (0.25)**       | (8, 8, 64)     | 0         | Regularization |
| **Flatten**              | (4096)         | 0         | Convert 2D → 1D |
| **Dense (128 units)**    | (128)          | 524,416   | Fully connected |
| **BatchNormalization**   | (128)          | 512       | Stabilization |
| **Dropout (0.5)**        | (128)          | 0         | Regularization |
| **Dense (10, Softmax)**  | (10)           | 1,290     | Output class probabilities |

---

### 🔹 Model Highlights
- **Batch Normalization** → speeds up training & stabilizes learning.  
- **Dropout Layers** → reduce overfitting.  
- **ImageDataGenerator** → improves generalization via data augmentation.  
- **Softmax Output** → predicts probabilities for 10 classes.  

---

### 📈 Performance
- **Training Accuracy**: ~80%  
- **Test Accuracy**: ~80%  

---

## 👨‍💻 Author  

- **Name**: Rishimithan 
- **Email**: rishimithan@gmail.com  



