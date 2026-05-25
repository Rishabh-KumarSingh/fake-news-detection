# 🔍 Fake News Detection

## 📌 Project Overview
A complete end-to-end machine learning pipeline that identifies and classifies fake news articles using Natural Language Processing (NLP) and multiple ML algorithms, trained on 72,134 real news articles.

---

## 🗂️ Project Structure

fake_news_detection/
├── data/
│   └── WELFake_Dataset.csv     # Real dataset (72,134 articles)
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




---

## ⚙️ Tech Stack
| Component      | Technology                                                                 |
|----------------|----------------------------------------------------------------------------|
| Language       | Python 3.12                                                                |
| Dataset        | WELFake Dataset — 72,134 real news articles                               |
| NLP Features   | TF-IDF (unigrams + bigrams)                                               |
| Models Trained | Logistic Regression, Passive Aggressive, Naive Bayes, Random Forest, Gradient Boosting |
| Best Model     | Logistic Regression                                                        |
| Accuracy       | 96.2% on real data                                                        |
| Demo UI        | Streamlit                                                                  |
| Visualization  | Matplotlib, Seaborn                                                        |
| Model Saving   | Joblib                                                                     |

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download the dataset
Download the WELFake dataset from Kaggle and place it in the `data/` folder: https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification

### 3. Train the model
```bash
python src/train_model.py
```

### 4. Launch the demo app
```bash
streamlit run app.py
```

---

## 📊 Models Evaluated
- ✅ Logistic Regression — **96.20%**
- ✅ Passive Aggressive Classifier — 96.19%
- ✅ Naive Bayes (Multinomial) — 86.16%
- ✅ Random Forest — 96.02%
- ✅ Gradient Boosting — 94.70%

All evaluated with 5-fold cross-validation and ROC-AUC scoring.

---

## 🔮 Future Improvements
- Add BERT / Transformer-based models
- URL & source credibility scoring
- Real-time news API integration
- Deploy on Hugging Face Spaces / Streamlit Cloud

---

*Built by Rishabh Kumar Singh*