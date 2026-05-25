# 🔍 Fake News Detection



## 📌 Project Overview
A complete end-to-end machine learning pipeline that identifies and classifies fake news articles using Natural Language Processing (NLP) and multiple ML algorithms.



## 🗂️ Project Structure

fake_news_detection/
├── data/
│   ├── generate_data.py        # Dataset generator (replace with real dataset)
│   ├── news_dataset.csv        # Generated/training dataset
|   └── WELFake_Dataset.csv
├── src/
│   ├── train_model.py          # Full training pipeline
│   └── create_report.py        # Combines all plots into one report
├── models/
│   ├── best_model.pkl          # Saved best model (Logistic Regression)
│   └── model_metrics.pkl       # Saved performance metrics
├── outputs/
│   ├── 01_eda.png              # Exploratory Data Analysis
│   ├── 02_model_comparison.png # Model accuracy comparison
│   ├── 03_confusion_matrix.png # Confusion matrix
│   ├── 04_roc_curves.png       # ROC curves for all models
│   ├── 05_top_features.png     # Most influential TF-IDF features
│   ├── 06_metrics_table.png    # Performance summary table
│   └── FINAL_REPORT.png        # Combined project report
├── app.py                      # Streamlit demo application
├── requirements.txt            # Python dependencies
└── README.md




## ⚙️ Tech Stack
| Component        | Technology                          |
|-----------------|-------------------------------------|
| Language         | Python 3.12                         |
| NLP Features     | TF-IDF (unigrams + bigrams)         |
| Models Trained   | Logistic Regression, Passive Aggressive, Naive Bayes, Random Forest, Gradient Boosting |
| Best Model       | Logistic Regression                 |
| Accuracy         | 100% (on actual dataset)         |
| Demo UI          | Streamlit                           |
| Visualization    | Matplotlib, Seaborn                 |
| Model Saving     | Joblib                              |

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python3 src/train_model.py
```

### 3. Launch the demo app
```bash
streamlit run app.py
```

---

## 📊 Models Evaluated
- ✅ Logistic Regression
- ✅ Passive Aggressive Classifier
- ✅ Naive Bayes (Multinomial)
- ✅ Random Forest
- ✅ Gradient Boosting

All evaluated with 5-fold cross-validation and ROC-AUC scoring.

---

## 🔮 Future Improvements
- Add BERT / Transformer-based models
- URL & source credibility scoring
- Real-time API integration
- Deploy on Hugging Face Spaces / Streamlit Cloud

---

## 📦 Using Real Data (Recommended for Submission)
Replace the synthetic dataset with the **WELFake dataset** from Kaggle:
```
https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification
```
Just update the CSV path in `train_model.py` and ensure columns `title`, `text`, `label` exist.

---

*Built by — Rishabh Kumar Singh*
