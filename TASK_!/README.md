# 🌟 DHC Internship – Machine Learning & NLP Projects
## 📰 AG News Text Classification with BERT  
This repository contains the projects and deliverables developed during my internship at DevelopersHub Corporation (DHC).
The internship focused on applying Machine Learning (ML) and Natural Language Processing (NLP) techniques to address real-world text classification challenges.

>The highlighted project is:

📰 AG News Topic Classifier using BERT – A transformer-based model fine-tuned to classify news headlines into World, Sports, Business, and Sci/Tech categories.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)  
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red.svg)  
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)  
![Scikit-Learn](https://img.shields.io/badge/ScikitLearn-ML-green.svg)  
![Status](https://img.shields.io/badge/Project-Completed-success.svg)

## 📌 Objective
The goal of this project is to build a **text classification model** that categorizes news headlines into four categories:
- 🌍 World  
- 🏏 Sports  
- 💰 Business  
- 💻 Sci/Tech  

We used **Hugging Face Transformers** with a fine-tuned **BERT model** on the AG News dataset.

---

## 📂 Dataset
- Dataset: [AG News](https://huggingface.co/datasets/ag_news)  
- Training Samples: 120,000  
- Test Samples: 7,600  
- Classes: 4 (World, Sports, Business, Sci/Tech)

---

## ⚙️ Methodology
1. **Dataset Loading & Preprocessing**  
   - Tokenization using `BertTokenizer`  
   - Dynamic padding with `DataCollatorWithPadding`  

2. **Model Development**  
   - Model: `BertForSequenceClassification`  
   - Optimized with Hugging Face `Trainer` API  

3. **Training**  
   - Epochs: 3  
   - Batch Size: 16  
   - Optimizer: AdamW  
   - Evaluation during training  

4. **Evaluation Metrics**  
   - Accuracy  
   - Weighted F1-score  
   - Precision / Recall / F1 per class  
   - Confusion Matrix  
   - Confidence Distribution of Predictions  

5. **Deployment**  
   - Gradio app for real-time headline classification  

---

## 📊 Results & Insights
- **Accuracy:** ~95% on test set  
- **Weighted F1-score:** ~95%  
- Confusion matrix shows most errors occur between *Business* and *Sci/Tech*.  
- Confidence histogram shows the model is usually highly confident (>0.8).  

---

## 📈 Visualizations
The following visualizations are included in the notebook:
- ✅ Training Loss Curve  
- ✅ Confusion Matrix  
- ✅ Precision/Recall/F1 per Class (Bar Chart)  
- ✅ Prediction Confidence Distribution  

---

## 🚀 Deployment
We created a simple **Gradio Interface** for testing:
```python
iface = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=2, placeholder="Enter news headline..."),
    outputs="text",
    title="AG News Topic Classifier"
)
iface.launch()
````

---

## 📦 Requirements

Install dependencies using:

```bash
pip install torch transformers datasets evaluate gradio matplotlib scikit-learn
```

---

## 📁 Repository Structure

```
├── ag_news_classification.ipynb   # Jupyter Notebook
├── my_ag_news_model/              # Saved fine-tuned model
├── README.md                      # Project Documentation
```

---

## ✨ Key Takeaways

* BERT performs extremely well on short text classification tasks.
* Hugging Face `Trainer` simplifies training & evaluation.
* Gradio makes deployment simple and interactive.
