# SkinCancerDetection
Interpretable Deep Learning Framework for Skin Cancer Classification Using HAM10000 and Grad-CAM

* Project Overview:
Skin cancer detection using HAM10000 dataset (10,015 dermatoscopic images, 7 lesion types). DenseNet121 + Grad-CAM achieved **83.03% validation accuracy** with explainable heatmaps for clinical trust.

* Repository Structure:
SkinCancerDetection/
├── code/skincancerfinal.ipynb # Complete implementation
├── results/ # Grad-CAM heatmaps (6 images)
│ ├── WhatsApp Image 2025-12-02... # Visualization results
└── README.md # This file

* Methodology

### 1. Dataset Preparation
- **HAM10000**: 10,015 images (AKIEC, BCC, BKL, DF, MEL, NV, VASC)
- **Preprocessing**: Resize 224×224, normalize, one-hot encode
- **Augmentation**: Rotation, zoom, flip, shear, shifts [web:21]

### 2. Models Implemented
Basic CNN (81.85% val acc)
├── ResNet50 (80.56% val acc)
├── MobileNetV2 (70.89% val acc)
└── DenseNet121 (83.03% val acc) ← BEST

text

### 3. Training Process
Hyperparameters:
├── Optimizer: Adam (lr=0.001)
├── Batch size: 32
├── Loss: Categorical Cross-Entropy
├── Early stopping
└── Google Colab GPU

text

### 4. Explainability
**Grad-CAM**: Highlights lesion regions model focuses on (not background)

## 📊 Results Summary

| Model         | Train Acc | Val Acc  | Train Loss | Val Loss |
|---------------|-----------|----------|------------|----------|
| Basic CNN     | 81.85%    | 68.85%   | 0.4792     | 0.8857   |
| ResNet50      | 82.09%    | 80.56%   | 0.4699     | 0.8857   |
| MobileNetV2   | 70.25%    | 70.89%   | 0.5868     | 0.7382   |
| **DenseNet121** | **94.57%** | **83.03%** | **0.2509** | **0.4747** | [web:30]

## 🖼️ Grad-CAM Visualizations
![Result 1](./results/WhatsApp%20Image%202025-12-02%20at%2022.02.36_808dc3ce.jpg)
![Result 2](./results/WhatsApp%20Image%202025-12-02%20at%2022.02.36_82cb38a4.jpg)
![Result 3](./results/WhatsApp%20Image%202025-12-02%20at%2022.02.37_499bbc85.jpg)
![Result 4](./results/WhatsApp%20Image%202025-12-02%20at%2022.02.37_6c399076.jpg)
![Result 5](./results/WhatsApp%20Image%202025-12-02%20at%2022.02.37_75b3b759.jpg)
![Result 6](./results/WhatsApp%20Image%202025-12-02%20at%2022.08.05_58dd6754.jpg) 

## 🚀 Run Locally
Google Colab/Jupyter
Open code/skincancerfinal.ipynb

!pip install tensorflow keras grad-cam

Runtime → Run all
