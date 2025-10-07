# 🌟 Month 02 – Developer Hub Corporation Internship Projects

This repository hosts the **Machine Learning & NLP projects** I completed during my internship at Developer Hub Corporation (DHC).  
Each folder corresponds to one of the tasks assigned during Month 02.

## 📂 Repository Structure

```

Month_02-Developer-Hub-Corporation-Internship/
│
├── TASK 01-News Topic Classifier/
├── TASK 02-Telco Churn Dataset/
├── TASK 03-House Price Pred (images + tabular data)/
└── README.md

```

---

## 🧾 Task 01 – News Topic Classifier (AG News + BERT)

### 🎯 Objective of the Task  
To build a transformer-based model (BERT) that classifies news headlines into four categories: World, Sports, Business, and Sci/Tech.

### 🧠 Methodology / Approach  
- Load **AG News** dataset using the Hugging Face datasets library.  
- Tokenize headlines using `BertTokenizer` with dynamic padding.  
- Fine-tune `BertForSequenceClassification` via the Hugging Face `Trainer` API.  
- Use AdamW optimizer and monitor validation loss & metrics during training.

### 📈 Key Results / Observations  
- Achieved **≈ 95% accuracy** on the test set.  
- Weighted **F1-score ~ 95%**.  
- Most misclassifications occurred between *Business* and *Sci/Tech*.  
- The model’s confidence distribution showed it was often very confident (> 0.8) about its predictions.

---

## 🧾 Task 02 – Telco Churn Dataset

### 🎯 Objective of the Task  
To predict whether a telecom customer will churn (i.e. discontinue service) using structured customer data (demographics, usage, account details).

### 🧠 Methodology / Approach  
- Load the Telco Customer Churn dataset via KaggleHub.  
- Clean data (e.g. convert `TotalCharges` to numeric, fill missing values).  
- Define features and binary target (`Churn`: Yes → 1, No → 0).  
- Split data into train and test sets (80 / 20).  
- Preprocess numeric and categorical features with pipelines (imputation + scaling / one-hot encoding).  
- Train two model pipelines: **Logistic Regression** and **Random Forest**.  
- Tune hyperparameters using `GridSearchCV`.  
- Evaluate models using accuracy, confusion matrix, and classification reports.

### 📈 Key Results / Observations  
- Random Forest outperformed Logistic Regression in accuracy on test set.  
- Logistic Regression provided more interpretability (feature coefficients).  
- Preprocessing pipelines ensured consistent and clean model input.  
- Model performance could be further improved by handling class imbalance or using stronger models (XGBoost, etc.).

---

## 🧾 Task 03 – House Price Prediction (Images + Tabular Data)

### 🎯 Objective of the Task  
To predict house prices in Austin by combining both **image data** (photos of houses) and **tabular data** (house features) in a multimodal deep learning model.

### 🧠 Methodology / Approach  
- Load housing data & associated image paths from a Kaggle dataset using KaggleHub.  
- Drop irrelevant columns and clean missing values.  
- Split into train / validation / test sets.  
- Tabular preprocessing: median imputation + scaling for numeric data, and one-hot encoding for categorical data.  
- Image preprocessing: load, resize (128×128), convert to arrays, preprocess using ResNet50 scheme.  
- Use a pre-trained **MobileNetV2** as feature extractor (frozen).  
- Build a multimodal neural network combining image features + processed tabular features.  
- Train model (Adam optimizer, MSE loss, MAE metric) for ~10 epochs.  

### 📈 Key Results / Observations  
- The multimodal model effectively fuses visual and structured data to produce price predictions.  
- MAE and RMSE on test set provided quantitative error estimates.  
- Visual features improved performance beyond using only tabular features.  
- This approach is promising for real-estate valuation tasks.

---

## 🧮 Summary & Highlights

- This repository showcases **three distinct ML/NLP tasks**, each using different data modalities (text, tabular, image + tabular).  
- I gained hands-on experience with **transformers**, **scikit-learn pipelines**, and **multimodal deep learning architectures**.  
- The structured layout and consistent README make it easy for others to navigate and understand the work.

---

