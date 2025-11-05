import os
import pandas as pd
import matplotlib.pyplot as plt

csv_path = os.path.join('models', 'evaluation_metrics.csv')
out_path = os.path.join('models', 'model_comparison_bar.png')

# Model display names and folder keys
model_map = {
    'squeezenet_96x96_full_epochs_with_unknown': 'SqueezeNet-inspired',
    'mobilenetv2_96x96_full_epochs_with_unknown': 'MobileNetV2-baseline',
    'mobilenetv2_efficientnet_style_96x96': 'MobileNetV2-Eff-style',
}


# Use available metrics from the CSV
metrics = ['keras_accuracy', 'standard_tflite_accuracy', 'quantized_tflite_accuracy']
metric_labels = ['Keras Accuracy', 'Standard TFLite Accuracy', 'Quantized TFLite Accuracy']

df = pd.read_csv(csv_path)
df = df[df['model'].isin(model_map.keys())].copy()
df['display_name'] = df['model'].map(model_map)

bar_data = []
for m in metrics:
    bar_data.append(df[m].values)
bar_data = list(zip(*bar_data))  # shape: (num_models, num_metrics)

fig, ax = plt.subplots(figsize=(10, 6))
n_models = len(df)
n_metrics = len(metrics)
x = range(n_models)
width = 0.15

for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
    ax.bar([p + i*width for p in x], df[metric], width, label=label)

ax.set_xticks([p + width*(n_metrics-1)/2 for p in x])
ax.set_xticklabels(df['display_name'], rotation=0, fontsize=12)
ax.set_ylabel('Score')
ax.set_ylim(0, 1.05)
ax.set_title('Model Comparison: Accuracy, Precision, Recall, F1, Quantized Accuracy')
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig(out_path, dpi=200)
plt.close()
print('Wrote model comparison bar chart to', out_path)