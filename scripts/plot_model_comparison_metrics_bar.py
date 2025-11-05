import pandas as pd
import matplotlib.pyplot as plt

# Load metrics
csv_path = 'model_comparison_metrics.csv'
df = pd.read_csv(csv_path)
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']
models = df['Model']
bar_width = 0.2
x = range(len(models))

fig, ax = plt.subplots(figsize=(10, 6))
for i, metric in enumerate(metrics):
    ax.bar([p + i*bar_width for p in x], df[metric], bar_width, label=metric)

ax.set_xticks([p + bar_width*1.5 for p in x])
ax.set_xticklabels(models, fontsize=12)
ax.set_ylabel('Score')
ax.set_ylim(0, 1.05)
ax.set_title('Model Comparison: Accuracy, Precision, Recall, F1-score')
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig('model_comparison_bar.png', dpi=200)
plt.close()
print('Wrote model comparison bar chart to model_comparison_bar.png')
