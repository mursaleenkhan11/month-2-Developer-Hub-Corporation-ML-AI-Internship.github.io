# 📂 Repository Name: **Telco-Customer-Churn-Prediction**

## 🏢 Organization

**Project Title:** Telco Customer Churn Prediction
**Developed by:** Huzaima Aneeq
**Organization/Institution:** Department of Computer Science
**Course/Task:** Machine Learning Task 02

---

## 🎯 Objective of the Task

The objective of this project is to **predict customer churn** — whether a telecom customer will discontinue their service — based on demographic, account, and service usage data.

This task aims to:

* Identify key factors influencing customer churn.
* Develop machine learning models to predict churn probability.
* Compare performance between **Logistic Regression** and **Random Forest** models.
* Provide actionable insights to help telecom companies improve **customer retention** and **reduce churn rates**.

---

## 🧠 Methodology / Approach

### **1. Data Loading & Preprocessing**

* The dataset was sourced from **Kaggle (Telco Customer Churn Dataset)**.
* Missing values were imputed using mean or mode strategies.
* Categorical features were encoded using **One-Hot Encoding**, and numerical features were **standardized** using `StandardScaler`.
* The data was split into **training (80%)** and **testing (20%)** subsets.

### **2. Model Development**

* Built two pipelines integrating preprocessing and model training:

  * **Logistic Regression:** For a simple, interpretable baseline.
  * **Random Forest Classifier:** For higher accuracy through ensemble learning.
* Each pipeline automatically handled data transformation and training.
* **Hyperparameter tuning** was performed using `GridSearchCV`.

### **3. Evaluation**

* Models were evaluated on:

  * **Accuracy**
  * **Precision, Recall, F1-Score**
  * **Confusion Matrix**
* Performance comparison visualized via bar charts and heatmaps.

### **4. Tools & Libraries**

* **Python 3**, **scikit-learn**, **pandas**, **NumPy**, **matplotlib**, **seaborn**
* **KaggleHub** for dataset access and integration.

---

## 📈 Key Results or Observations

| Model                   | Train Accuracy | Test Accuracy | Notes                                      |
| :---------------------- | :------------: | :-----------: | :----------------------------------------- |
| **Logistic Regression** |      80.2%     |     78.6%     | Simple, interpretable model.               |
| **Random Forest**       |      99.5%     |     81.3%     | Better generalization, slight overfitting. |

### **Insights**

* The **Random Forest model** achieved the highest accuracy and recall, making it more reliable for churn detection.
* **Logistic Regression** helps understand key drivers of churn (e.g., tenure, contract type, internet service).
* Preprocessing pipelines ensured consistent handling of numeric and categorical variables.
* Visualizations like confusion matrices helped understand model prediction patterns.

### **Future Improvements**

* Handle class imbalance with **SMOTE** or **class weighting**.
* Use more advanced models (e.g., **XGBoost**, **LightGBM**).
* Deploy the churn prediction model as an interactive web service.

---

## 🧾 Repository Structure

```
Telco-Customer-Churn-Prediction/
│
├── TASK_02_Telco_Churn_Dataset.ipynb     # Main Jupyter notebook
├── TASK_02_Telco_Churn_Dataset.py         # Converted Python script
├── README.md                              # Project documentation
├── requirements.txt                       # Dependencies list (optional)
└── /data                                  # Dataset directory (if applicable)
