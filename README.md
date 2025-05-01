# Detective Mel: Melanoma Detection Using Machine Learning and Causal Inference

This repository contains code and documentation for a final project developed for **CS 3891-05: Machine Learning/Natural Language Processing for Healthcare**. The project explores hybrid machine learning and statistical modeling techniques to improve the early detection of melanoma from dermatoscopic images.

## Project Objective

To evaluate whether combining deep learning-based feature extraction with traditional machine learning models enhances the accuracy and reliability of melanoma classification. Additionally, the project uses causal inference techniques to assess the impact of UV exposure-related features on melanoma detection.

## Dataset

- **Source**: ISIC 2018 Challenge (Task 3: Lesion Diagnosis)
- **Content**: 10,015 training images, 1,512 test images, 193 validation images
- **Labels**: 7 classes including Melanoma, Nevus, Basal Cell Carcinoma, etc.
- **Format**: JPEG images + classification metadata in CSV

## Approach

### 1. Supervised Learning Pipeline

- **Feature Extraction**: ResNet50 (pre-trained, with top layer removed)
- **Classifiers**:
  - Support Vector Machine (SVM)
  - Random Forest (RF)
- **Model Fusion**: Ensemble features from ResNet50, InceptionV3, and EfficientNetB0
- **Evaluation**:
  - Train/validation/test split
  - Accuracy and k-fold cross-validation
  - Monte Carlo Dropout for uncertainty estimation

### 2. Causal Inference Pipeline

- **Goal**: Estimate causal effect of image-based dermatological features (RGB Mean, Symmetry, Border Irregularity) on melanoma detection
- **Methods**:
  - Propensity Score Estimation using Logistic Regression
  - Nearest Neighbor Matching
  - Average Treatment Effect (ATE) Estimation

## Key Findings

- SVM and RF classifiers outperformed ResNet50 in constrained environments.
- Hybrid pipelines leveraging ResNet50 for feature extraction improved classification accuracy.
- ATE analysis showed a moderate effect (ATE ≈ 0.31) of UV exposure-related features on melanoma detection.
- Model performance was limited by dataset size and computational constraints.

## Technologies Used

- Python, TensorFlow/Keras, scikit-learn, NumPy, Pandas, Matplotlib
- Pretrained models from Keras Applications (ResNet50, InceptionV3, EfficientNetB0)
