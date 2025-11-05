import pandas as pd
import matplotlib.pyplot as plt

# Data for laboratory/test set
models = [
    'MobileNetV2 96x96',
    'MobileNetV2 EfficientNet-style 96x96',
    'SqueezeNet 96x96'
]
accuracy = [95.83, 96.92, 97.93]
precision = [96.0, 97.05, 98.0]
recall = [96.0, 96.92, 98.0]
f1 = [96.0, 96.89, 98.0]

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']
values = [accuracy, precision, recall, f1]
bar_width = 0.2
x = range(len(models))

fig, ax = plt.subplots(figsize=(10, 6))
for i, metric in enumerate(metrics):
    ax.bar([p + i*bar_width for p in x], values[i], bar_width, label=metric)

ax.set_xticks([p + bar_width*1.5 for p in x])
ax.set_xticklabels(models, fontsize=12)
ax.set_ylabel('Score (%)')
ax.set_ylim(0, 105)
ax.set_title('Model Comparison: Laboratory/Test Set Metrics')
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig('lab_metrics_bar.png', dpi=200)
plt.close()
print('Wrote laboratory metrics bar chart to lab_metrics_bar.png')
