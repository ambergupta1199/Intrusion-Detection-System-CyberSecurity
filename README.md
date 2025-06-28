# System Call Sequence Classification for Intrusion Detection

This project focuses on classifying system call sequences to detect intrusions using various machine learning and deep learning models. The dataset used is the **ADFA-LD dataset**, which contains system call sequences for normal and attack scenarios.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Models Implemented](#models-implemented)
- [Results](#results)
- [Dependencies](#dependencies)
- [Usage](#usage)

---

## Overview
The goal of this project is to classify system call sequences into normal or attack categories. The project explores multiple models, including traditional machine learning models like SVM and Random Forest, as well as deep learning models like ANN, LSTM, GRU, and Transformer-based encoders.

---

## Dataset
The **ADFA-LD dataset** is used in this project. It contains:
- **Normal sequences**: Representing benign system call sequences.
- **Attack sequences**: Representing various types of intrusions.

The dataset is downloaded and extracted programmatically within the notebook.

---

## Preprocessing
1. **Data Extraction**:
   - System call sequences are read from text files.
   - Attack types are mapped to unique labels.

2. **Data Augmentation**:
   - Attack sequences are augmented by appending/prepending harmless system calls to balance the dataset.

3. **Feature Engineering**:
   - **Word2Vec** embeddings are generated for system call sequences.
   - Bigram and trigram transformations are applied to capture contextual information.

4. **Train-Test Split**:
   - The dataset is split into training (80%) and testing (20%) sets.

---

## Models Implemented
### 1. **Support Vector Machine (SVM)**
- Kernel: RBF
- Hyperparameters: `C=1`, `gamma=5`
- Used Word2Vec embeddings as input features.

### 2. **Random Forest**
- Number of estimators: 100
- Used Word2Vec embeddings as input features.

### 3. **Artificial Neural Network (ANN)**
- Fully connected layers with ReLU activation.
- Output layer with softmax activation for multi-class classification.

### 4. **LSTM**
- Bidirectional LSTM layers with dropout and layer normalization.
- Used sequence embeddings as input.

### 5. **GRU**
- Bidirectional GRU layers with dropout and layer normalization.
- Used sequence embeddings as input.

### 6. **Transformer Encoder**
- Multi-head attention mechanism.
- Feed-forward layers with dropout and layer normalization.

---

## Results
### Metrics Evaluated:
- **Accuracy**
- **Classification Report** (Precision, Recall, F1-Score)
- **Confusion Matrix**

Each model's performance is evaluated on the test set and visualized using confusion matrices.

---

## Dependencies
The following libraries are required to run the notebook:
- Python 3.8+
- NumPy
- Pandas
- Gensim
- Scikit-learn
- TensorFlow
- Matplotlib
- Seaborn


## Acknowledgments
- **ADFA-LD Dataset**: A labeled version of the ADFA-LD dataset is used for this project.
- **Libraries**: Thanks to the developers of NumPy, Pandas, Gensim, Scikit-learn, and TensorFlow for their excellent tools.

---