# 🔐 Multi-Class Intrusion Detection System using System Call Sequences

A deep learning-based **multi-class Intrusion Detection System (IDS)** that classifies system call sequences into normal or one of multiple attack types using models like **LSTM**, **GRU**, and **Transformer Encoders**. Built using the **ADFA-LD** dataset, the project implements comprehensive preprocessing, feature extraction, and data augmentation techniques.

---

## 📁 Table of Contents

- [🔍 Overview](#-overview)
- [📊 Dataset](#-dataset)
- [🧹 Preprocessing](#-preprocessing)
- [🧠 Models Implemented](#-models-implemented)
- [📈 Results](#-results)
- [⚙️ Dependencies](#-dependencies)
- [▶️ Usage](#-usage)
- [🙌 Acknowledgments](#-acknowledgments)

---

## 🔍 Overview

This project builds an intrusion detection system using **system call sequences** to detect and classify **malicious behavior** in real-time application processes. The model is trained to recognize both **normal** and **multiple attack** patterns using deep learning.

📌 **Core Features:**
- Multi-class classification of system calls
- Handles **imbalanced data** with oversampling and augmentation
- Sequence-based modeling with **LSTM**, **GRU**, and **Transformer**
- Supports **word embeddings** and **n-gram** feature extraction

---

## 📊 Dataset

We use the **ADFA-LD** (Australian Defence Force Academy Linux Dataset), which includes system call traces for both **normal activities** and **six distinct types of attacks**.

📦 **Dataset Details:**
- 📁 File: `system_calls_with_labels.csv`
- 📌 Columns:
  - `System_Calls`: List of system calls (e.g., `[240, 311, 78, 240, ...]`)
  - `Label`: `0` = Normal, `1-6` = Attack types
  - `Attack Type`: Name of the specific attack

🛡️ **Attack Types:**
| Label | Attack Type       |
|-------|-------------------|
| 0     | Normal            |
| 1     | Adduser           |
| 2     | Hydra FTP         |
| 3     | Hydra SSH         |
| 4     | Java Meterpreter  |
| 5     | Meterpreter       |
| 6     | Web Shell         |

📎 **Download Source**: [ADFA-LD Dataset](https://research.unsw.edu.au/projects/adfa-ids-datasets)

---

## 🧹 Preprocessing

The dataset undergoes several transformation steps:

1. **Parsing**: System calls in string format are converted into numerical lists.
2. **Tokenization**: Sequences are encoded for model input.
3. **Padding**: Ensures all sequences are of uniform length.
4. **Augmentation**:
   - Oversampling attack instances
   - Adding benign sequences to augment minority classes
5. **Splitting**:
   - 80% Training
   - 20% Testing
   - A portion of attacks is set aside for validation

---

## 🧠 Models Implemented

| Model               | Description |
|--------------------|-------------|
| 🤖 **Support Vector Machine (SVM)** | RBF kernel, `C=1`, `gamma=5`, with Word2Vec features |
| 🌲 **Random Forest**               | 100 estimators, uses Word2Vec embeddings |
| 🔗 **Artificial Neural Network (ANN)** | Fully connected layers, Softmax output |
| 🔄 **LSTM (Bi-directional)**        | LayerNorm + Dropout, ideal for long sequences |
| 🔁 **GRU (Bi-directional)**         | Similar to LSTM, faster training |
| ✨ **Transformer Encoder**          | Multi-head attention, sequence embeddings |

📌 **Feature Engineering:**
- **Word2Vec** embeddings for system calls
- **n-gram** (bigram, trigram) generation for contextual modeling

---

## 📈 Results

📊 **Evaluation Metrics:**
- Accuracy
- Precision / Recall / F1-Score
- Confusion Matrix (visualized using heatmaps)

Each model is assessed on:
- Multi-class classification capability
- Sensitivity to rare attack classes
- Generalization on unseen sequences

📉 Results are summarized in the Jupyter Notebook:  
➡️ `PGSL_project_LSTM_based_IDS.ipynb`

---

## ⚙️ Dependencies

Ensure the following packages are installed:

```bash
pip install numpy pandas scikit-learn gensim tensorflow matplotlib seaborn
