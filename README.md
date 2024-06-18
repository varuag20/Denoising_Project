# Denoising_Project
This project focuses on developing a deep learning model to denoise images. The architecture used is a convolutional autoencoder implemented using TensorFlow and Keras. The primary goal is to improve the quality of noisy images by learning to reconstruct clean images from their noisy counterparts.

# Features
Load and preprocess images
Split data into training and testing sets
Create TensorFlow datasets
Define and train a convolutional autoencoder
Custom PSNR callback for monitoring during training
Evaluate and visualize results

# Dependencies
opencv-python
pillow
numpy
scikit-learn
tensorflow
torch
torchvision

# Installation
Install the required dependencies using the following command:
```pip install opencv-python pillow numpy scikit-learn tensorflow torch torchvision```

# Usage
Loading and Preprocessing Images
```
import os
import cv2
import numpy as np

def load_images_from_folder(folder, img_size=(128, 128)):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            img = cv2.resize(img, img_size)  # Resize image
            img = img / 255.0  # Normalize to [0, 1]
            images.append(img)
    return images

noisy_folder_path = 'C:/Users/Gaurav Sharma/Desktop/Train/Train/low' //Note :- change your path accoringly
clean_folder_path = 'C:/Users/Gaurav Sharma/Desktop/Train/Train/high' //Note :- change your path accoringly
img_size = (128, 128)  

noisy_images = load_images_from_folder(noisy_folder_path, img_size)
clean_images = load_images_from_folder(clean_folder_path, img_size)
```

# Future Improvements:-

Data Augmentation: Incorporate techniques like rotation, flipping, and cropping to increase the diversity of training data.
Advanced Architectures: Explore more complex architectures such as U-Net or GANs for better performance.
Hyperparameter Tuning: Fine-tune parameters such as learning rate, batch size, and the number of layers to optimize model performance.

