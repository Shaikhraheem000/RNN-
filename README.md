# 📰 Fake News Detection using NLP & Deep Learning

This project focuses on detecting fake news articles using Natural Language Processing (NLP) techniques and a deep learning model built with PyTorch.

---

## 📌 Overview

Fake news has become a major issue in the digital world. This project builds a machine learning pipeline that:

- Preprocesses news text
- Extracts features using TF-IDF
- Trains a neural network model
- Predicts whether a news article is **real or fake**

---

## 📂 Dataset

The model is trained on two datasets:

- `Fake.csv` → Fake news articles
- `True.csv` → Real news articles

Each dataset contains:
- Title
- Text
- Subject
- Date

---

## ⚙️ Project Pipeline

### 1. Data Loading
- Load datasets using Pandas
- Assign labels:
  - `0` → Fake news  
  - `1` → Real news
- Merge and shuffle data

---

### 2. Data Preprocessing

- Convert text to lowercase
- Remove URLs and punctuation
- Combine title + text into a single feature (`Content`)

---

### 3. Text Cleaning

- Tokenization (NLTK)
- Stopword removal
- Stemming using Porter Stemmer

---

### 4. Feature Extraction

- TF-IDF Vectorization
- Max features: `5000`

---

### 5. Train-Test Split

- 80% Training
- 20% Testing

---

### 6. Model

A custom neural network (RNN-based) implemented using PyTorch.

#### Key Components:
- Input size = TF-IDF feature size
- Loss Function: Binary Cross Entropy (`BCELoss`)
- Optimizer: Adam
- Epochs: 10

---

### 7. Training

- Data loaded using PyTorch `DataLoader`
- Batch size: 64
- Model trained over multiple epochs

---

### 8. Evaluation

- Model evaluated on test dataset
- Accuracy calculated using predictions vs actual labels

---

### 9. Prediction Function

Custom function to classify new news text:

```python
predict_news("Your news text here")