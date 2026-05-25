import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import re
import string
import warnings
warnings.filterwarnings('ignore')

STOP_WORDS = set([
    'i','me','my','myself','we','our','ours','ourselves','you','your','yours',
    'yourself','he','him','his','himself','she','her','hers','herself','it','its',
    'itself','they','them','their','theirs','themselves','what','which','who','whom',
    'this','that','these','those','am','is','are','was','were','be','been','being',
    'have','has','had','having','do','does','did','doing','a','an','the','and','but',
    'if','or','because','as','until','while','of','at','by','for','with','about',
    'against','between','into','through','during','before','after','above','below',
    'to','from','up','down','in','out','on','off','over','under','again','further',
    'then','once','here','there','when','where','why','how','all','both','each',
    'few','more','most','other','some','such','no','nor','not','only','own','same',
    'so','than','too','very','s','t','can','will','just','don','should','now','d',
    'll','m','o','re','ve','y','ain','aren','couldn','didn','doesn','hadn','hasn',
    'haven','isn','ma','mightn','mustn','needn','shan','shouldn','wasn','weren',
    'won','wouldn'
])

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_curve, auc, roc_auc_score)
from sklearn.pipeline import Pipeline
import joblib
import os

OUTPUT_DIR = r"C:\\Users\\91830\\Downloads\\fake_news_detection_project\\fake_news_detection\\outputs"
MODEL_DIR  = r"C:\\Users\\91830\\Downloads\\fake_news_detection_project\\fake_news_detection\\models"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)

# ── 1. Load Data ──────────────────────────────────────────────────────────────
print("=" * 60)
print("FAKE NEWS DETECTION — END-TO-END PIPELINE")
print("=" * 60)

df = pd.read_csv("C:\\Users\\91830\\Downloads\\fake_news_detection_project\\fake_news_detection\\data\\WELFake_Dataset.csv")
df = df.drop(columns=['Unnamed: 0'])          # drop useless index column
df['title'] = df['title'].fillna('')          # handle NaN titles
df['text']  = df['text'].fillna('')           # handle NaN body text
df['combined_text'] = df['title'] + " " + df['text']
df['label_num'] = df['label']                 # already 0=REAL, 1=FAKE
df['label'] = df['label'].map({0: 'REAL', 1: 'FAKE'})  # for plots

print(f"\n✅ Dataset loaded: {len(df)} samples")
print(df['label'].value_counts().to_string())

# ── 2. Text Preprocessing ────────────────────────────────────────────────────
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = [w for w in text.split() if w not in STOP_WORDS and len(w) > 2]
    return " ".join(tokens)

print("\n⚙️  Preprocessing text …")
df['clean_text'] = df['combined_text'].apply(preprocess_text)
df['text_length'] = df['combined_text'].apply(len)
df['word_count']  = df['combined_text'].apply(lambda x: len(str(x).split()))
print("✅ Preprocessing complete")

# ── 3. EDA Plots ─────────────────────────────────────────────────────────────
print("\n📊 Generating EDA visualizations …")

palette = {'REAL': '#2ecc71', 'FAKE': '#e74c3c'}
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.patch.set_facecolor('#0f1117')
for ax in axes.flat:
    ax.set_facecolor('#1a1d27')
    ax.tick_params(colors='#cccccc')
    ax.xaxis.label.set_color('#cccccc')
    ax.yaxis.label.set_color('#cccccc')
    ax.title.set_color('#ffffff')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333344')

# 3a. Label distribution
counts = df['label'].value_counts()
bars = axes[0,0].bar(counts.index, counts.values,
                     color=[palette[l] for l in counts.index],
                     width=0.5, edgecolor='none')
axes[0,0].set_title("Label Distribution", fontsize=14, fontweight='bold')
axes[0,0].set_ylabel("Count")
for b in bars:
    axes[0,0].text(b.get_x()+b.get_width()/2, b.get_height()+5,
                   str(b.get_height()), ha='center', color='white', fontsize=12)

# 3b. Text length distribution
for label, color in palette.items():
    subset = df[df['label'] == label]['text_length']
    axes[0,1].hist(subset, bins=30, alpha=0.7, color=color, label=label, edgecolor='none')
axes[0,1].set_title("Text Length Distribution", fontsize=14, fontweight='bold')
axes[0,1].set_xlabel("Characters")
axes[0,1].set_ylabel("Frequency")
axes[0,1].legend(facecolor='#1a1d27', labelcolor='white')

# 3c. Word count box plot
real_wc = df[df['label']=='REAL']['word_count']
fake_wc = df[df['label']=='FAKE']['word_count']
bp = axes[0,2].boxplot([real_wc, fake_wc], patch_artist=True,
                        medianprops=dict(color='white', linewidth=2))
bp['boxes'][0].set_facecolor('#2ecc71')
bp['boxes'][1].set_facecolor('#e74c3c')
axes[0,2].set_xticklabels(['REAL', 'FAKE'], color='white')
axes[0,2].set_title("Word Count by Label", fontsize=14, fontweight='bold')
axes[0,2].set_ylabel("Word Count")

# 3d. Average word count bar
avg_wc = df.groupby('label')['word_count'].mean()
axes[1,0].barh(avg_wc.index, avg_wc.values,
               color=[palette[l] for l in avg_wc.index], edgecolor='none')
axes[1,0].set_title("Avg Word Count per Label", fontsize=14, fontweight='bold')
axes[1,0].set_xlabel("Avg Words")
for i, v in enumerate(avg_wc.values):
    axes[1,0].text(v+1, i, f"{v:.1f}", va='center', color='white')

# 3e. Unique word ratio
df['unique_ratio'] = df['clean_text'].apply(
    lambda x: len(set(x.split()))/max(len(x.split()),1))
for label, color in palette.items():
    subset = df[df['label']==label]['unique_ratio']
    axes[1,1].hist(subset, bins=25, alpha=0.75, color=color, label=label, edgecolor='none')
axes[1,1].set_title("Unique Word Ratio", fontsize=14, fontweight='bold')
axes[1,1].set_xlabel("Ratio")
axes[1,1].legend(facecolor='#1a1d27', labelcolor='white')

# 3f. Exclamation / caps usage (fake news signature)
df['exclamation'] = df['combined_text'].apply(lambda x: str(x).count('!'))
df['caps_words']  = df['combined_text'].apply(
    lambda x: sum(1 for w in str(x).split() if w.isupper() and len(w)>2))
avg_exc = df.groupby('label')['exclamation'].mean()
avg_cap = df.groupby('label')['caps_words'].mean()
x = np.arange(2)
w = 0.35
b1 = axes[1,2].bar(x-w/2, avg_exc.values, w, label='Exclamation Marks',
                    color='#f39c12', edgecolor='none')
b2 = axes[1,2].bar(x+w/2, avg_cap.values, w, label='ALL CAPS Words',
                    color='#9b59b6', edgecolor='none')
axes[1,2].set_xticks(x)
axes[1,2].set_xticklabels(avg_exc.index, color='white')
axes[1,2].set_title("Sensationalism Signals", fontsize=14, fontweight='bold')
axes[1,2].legend(facecolor='#1a1d27', labelcolor='white', fontsize=9)

plt.suptitle("Fake News Detection — Exploratory Data Analysis",
             fontsize=17, fontweight='bold', color='white', y=1.01)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_eda.png", dpi=150, bbox_inches='tight',
            facecolor='#0f1117')
plt.close()
print("✅  EDA saved → 01_eda.png")

# ── 4. Train / Test Split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['label_num'],
    test_size=0.2, random_state=42, stratify=df['label_num'])
print(f"\n✅ Train: {len(X_train)}  |  Test: {len(X_test)}")

# ── 5. Build & Evaluate Multiple Models ──────────────────────────────────────
print("\n🤖 Training models …")

models = {
    "Logistic Regression":        LogisticRegression(max_iter=1000, C=1.0),
    "Passive Aggressive":         PassiveAggressiveClassifier(max_iter=1000),
    "Naive Bayes":                MultinomialNB(alpha=0.1),
    "Random Forest":              RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting":          GradientBoostingClassifier(n_estimators=100, random_state=42),
}

tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1,2), sublinear_tf=True)

results = {}
best_acc   = 0
best_name  = ""
best_pipeline = None

for name, clf in models.items():
    pipe = Pipeline([('tfidf', TfidfVectorizer(max_features=10000,
                                               ngram_range=(1,2),
                                               sublinear_tf=True)),
                     ('clf', clf)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    cv     = cross_val_score(pipe, df['clean_text'], df['label_num'],
                             cv=5, scoring='accuracy')
    results[name] = {
        'accuracy':  acc,
        'cv_mean':   cv.mean(),
        'cv_std':    cv.std(),
        'y_pred':    y_pred,
        'pipeline':  pipe,
    }
    if acc > best_acc:
        best_acc  = acc
        best_name = name
        best_pipeline = pipe
    print(f"   {name:26s}  Acc={acc:.4f}  CV={cv.mean():.4f}±{cv.std():.4f}")

print(f"\n🏆 Best Model: {best_name}  (Accuracy = {best_acc:.4f})")

# ── 6. Model Comparison Plot ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor('#0f1117')
for ax in axes:
    ax.set_facecolor('#1a1d27')
    ax.tick_params(colors='#cccccc')
    ax.xaxis.label.set_color('#cccccc')
    ax.yaxis.label.set_color('#cccccc')
    ax.title.set_color('#ffffff')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333344')

names  = list(results.keys())
accs   = [results[n]['accuracy'] for n in names]
cv_m   = [results[n]['cv_mean']  for n in names]
cv_s   = [results[n]['cv_std']   for n in names]
colors = ['#2ecc71' if n == best_name else '#3498db' for n in names]
short  = ["LR", "PAC", "NB", "RF", "GB"]

bars = axes[0].bar(short, accs, color=colors, edgecolor='none', width=0.5)
axes[0].set_ylim(0.7, 1.02)
axes[0].set_title("Test Accuracy by Model", fontsize=13, fontweight='bold')
axes[0].set_ylabel("Accuracy")
for b, a in zip(bars, accs):
    axes[0].text(b.get_x()+b.get_width()/2, a+0.002, f"{a:.3f}",
                 ha='center', color='white', fontsize=10)

axes[1].bar(short, cv_m, color=colors, edgecolor='none', width=0.5, yerr=cv_s,
            error_kw=dict(ecolor='white', capsize=5))
axes[1].set_ylim(0.7, 1.02)
axes[1].set_title("5-Fold CV Accuracy", fontsize=13, fontweight='bold')
axes[1].set_ylabel("CV Accuracy")

green_patch = mpatches.Patch(color='#2ecc71', label='Best Model')
blue_patch  = mpatches.Patch(color='#3498db', label='Other Models')
axes[1].legend(handles=[green_patch, blue_patch],
               facecolor='#1a1d27', labelcolor='white')

plt.suptitle("Model Comparison", fontsize=15, fontweight='bold', color='white')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_model_comparison.png", dpi=150,
            bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("✅  Model comparison saved → 02_model_comparison.png")

# ── 7. Detailed Report for Best Model ────────────────────────────────────────
y_pred_best = results[best_name]['y_pred']
report = classification_report(y_test, y_pred_best,
                                target_names=['REAL', 'FAKE'], output_dict=True)
print(f"\n📋 Classification Report ({best_name}):\n")
print(classification_report(y_test, y_pred_best, target_names=['REAL','FAKE']))

# ── 8. Confusion Matrix ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
fig.patch.set_facecolor('#0f1117')
ax.set_facecolor('#1a1d27')

cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn',
            xticklabels=['REAL','FAKE'], yticklabels=['REAL','FAKE'],
            ax=ax, linewidths=0.5, linecolor='#0f1117',
            cbar_kws={'shrink': 0.8})
ax.set_xlabel("Predicted", color='white', fontsize=12)
ax.set_ylabel("Actual",    color='white', fontsize=12)
ax.tick_params(colors='white')
ax.set_title(f"Confusion Matrix — {best_name}", color='white',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_confusion_matrix.png", dpi=150,
            bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("✅  Confusion matrix saved → 03_confusion_matrix.png")

# ── 9. ROC Curves ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor('#0f1117')
ax.set_facecolor('#1a1d27')
ax.tick_params(colors='#cccccc')
ax.xaxis.label.set_color('#cccccc')
ax.yaxis.label.set_color('#cccccc')
ax.title.set_color('#ffffff')
for spine in ax.spines.values():
    spine.set_edgecolor('#333344')

roc_colors = ['#2ecc71','#3498db','#e74c3c','#f39c12','#9b59b6']
for (name, res), col in zip(results.items(), roc_colors):
    pipe = res['pipeline']
    if hasattr(pipe, 'predict_proba'):
        proba = pipe.predict_proba(X_test)[:,1]
    else:
        proba = pipe.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, proba)
    roc_auc_val = auc(fpr, tpr)
    short_n = name.replace(" ", "\n")
    ax.plot(fpr, tpr, color=col, lw=2,
            label=f"{name} (AUC={roc_auc_val:.3f})")

ax.plot([0,1],[0,1],'--', color='#555566', lw=1.5)
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate",  fontsize=12)
ax.set_title("ROC Curves — All Models", fontsize=13, fontweight='bold', color='white')
ax.legend(facecolor='#1a1d27', labelcolor='white', fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_roc_curves.png", dpi=150,
            bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("✅  ROC curves saved → 04_roc_curves.png")

# ── 10. Top TF-IDF Features ───────────────────────────────────────────────────
tfidf_fitted = best_pipeline.named_steps['tfidf']
clf_step     = best_pipeline.named_steps['clf']

if hasattr(clf_step, 'coef_'):
    coefs      = clf_step.coef_.ravel() if clf_step.coef_.ndim > 1 else clf_step.coef_
    feat_names = tfidf_fitted.get_feature_names_out()
    top_n      = 15

    top_fake_idx = np.argsort(coefs)[-top_n:][::-1]
    top_real_idx = np.argsort(coefs)[:top_n]

    top_fake_words  = feat_names[top_fake_idx]
    top_fake_scores = coefs[top_fake_idx]
    top_real_words  = feat_names[top_real_idx]
    top_real_scores = np.abs(coefs[top_real_idx])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor('#0f1117')
    for ax in [ax1, ax2]:
        ax.set_facecolor('#1a1d27')
        ax.tick_params(colors='#cccccc')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333344')

    ax1.barh(range(top_n), top_fake_scores, color='#e74c3c', edgecolor='none')
    ax1.set_yticks(range(top_n))
    ax1.set_yticklabels(top_fake_words, color='white', fontsize=10)
    ax1.set_title("Top 15 FAKE News Keywords", color='white',
                  fontsize=13, fontweight='bold')
    ax1.set_xlabel("Coefficient Weight", color='#cccccc')

    ax2.barh(range(top_n), top_real_scores, color='#2ecc71', edgecolor='none')
    ax2.set_yticks(range(top_n))
    ax2.set_yticklabels(top_real_words, color='white', fontsize=10)
    ax2.set_title("Top 15 REAL News Keywords", color='white',
                  fontsize=13, fontweight='bold')
    ax2.set_xlabel("|Coefficient Weight|", color='#cccccc')

    plt.suptitle(f"Most Influential Features — {best_name}",
                 fontsize=14, fontweight='bold', color='white')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/05_top_features.png", dpi=150,
                bbox_inches='tight', facecolor='#0f1117')
    plt.close()
    print("✅  Feature importance saved → 05_top_features.png")

# ── 11. Save Best Model ───────────────────────────────────────────────────────
joblib.dump(best_pipeline, f"{MODEL_DIR}/best_model.pkl")
joblib.dump({'best_model': best_name, 'accuracy': best_acc,
             'results': {k: {'accuracy': v['accuracy'],
                             'cv_mean':  v['cv_mean'],
                             'cv_std':   v['cv_std']} for k, v in results.items()}
             }, f"{MODEL_DIR}/model_metrics.pkl")
print(f"\n✅ Best model saved → models/best_model.pkl")

# ── 12. Metrics Summary Table ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
fig.patch.set_facecolor('#0f1117')
ax.set_facecolor('#0f1117')
ax.axis('off')

table_data = [[n,
               f"{results[n]['accuracy']:.4f}",
               f"{results[n]['cv_mean']:.4f}",
               f"±{results[n]['cv_std']:.4f}",
               "🏆" if n == best_name else ""]
              for n in names]

table = ax.table(cellText=table_data,
                 colLabels=["Model", "Test Acc", "CV Mean", "CV Std", ""],
                 cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.3, 2.0)

for (row, col), cell in table.get_celld().items():
    cell.set_facecolor('#1a1d27' if row % 2 == 0 else '#22263a')
    cell.set_edgecolor('#333344')
    cell.set_text_props(color='white')
    if row == 0:
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='#2ecc71', fontweight='bold')

plt.title("Model Performance Summary", color='white',
          fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_metrics_table.png", dpi=150,
            bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("✅  Metrics table saved → 06_metrics_table.png")

print("\n" + "="*60)
print("✅ TRAINING PIPELINE COMPLETE")
print(f"   Best Model : {best_name}")
print(f"   Accuracy   : {best_acc:.4f}")
print(f"   All outputs: /fake_news_detection/outputs/")
print("="*60)
