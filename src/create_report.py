import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

OUTPUT_DIR = "/home/claude/fake_news_detection/outputs"
imgs = ["01_eda.png","02_model_comparison.png",
        "03_confusion_matrix.png","04_roc_curves.png",
        "05_top_features.png","06_metrics_table.png"]

fig = plt.figure(figsize=(22, 30))
fig.patch.set_facecolor('#0a0c12')

# Title
fig.text(0.5, 0.985, "FAKE NEWS DETECTION — COMPLETE PROJECT REPORT",
         ha='center', va='top', fontsize=22, fontweight='bold', color='white',
         fontfamily='monospace')
fig.text(0.5, 0.974, "Pinnacle Labs Data Science Internship 2026",
         ha='center', va='top', fontsize=13, color='#2ecc71', fontfamily='monospace')

positions = [
    [0.02, 0.66, 0.96, 0.30],   # EDA  (full width)
    [0.02, 0.46, 0.96, 0.18],   # Model comparison
    [0.02, 0.26, 0.46, 0.18],   # Confusion matrix
    [0.52, 0.26, 0.46, 0.18],   # ROC curves
    [0.02, 0.06, 0.96, 0.18],   # Feature importance
    [0.02, 0.00, 0.96, 0.05],   # Metrics table
]

for img_name, pos in zip(imgs, positions):
    path = os.path.join(OUTPUT_DIR, img_name)
    if os.path.exists(path):
        ax = fig.add_axes(pos)
        img = mpimg.imread(path)
        ax.imshow(img)
        ax.axis('off')

plt.savefig(f"{OUTPUT_DIR}/FINAL_REPORT.png", dpi=130,
            bbox_inches='tight', facecolor='#0a0c12')
plt.close()
print("✅  FINAL_REPORT.png saved")
