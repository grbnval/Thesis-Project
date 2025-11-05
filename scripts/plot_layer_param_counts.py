import os
import csv
import math
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

models_to_check = [
    ("SqueezeNet-inspired", os.path.join(ROOT, 'models', 'squeezenet_96x96_full_epochs_with_unknown', 'model.keras')),
    ("MobileNetV2-baseline", os.path.join(ROOT, 'models', 'mobilenetv2_96x96_full_epochs_with_unknown', 'model.keras')),
    ("MobileNetV2-Eff-style", os.path.join(ROOT, 'models', 'mobilenetv2_efficientnet_style_96x96', 'model.keras')),
]

out_dir = os.path.join(ROOT, 'models')
os.makedirs(out_dir, exist_ok=True)

summary_rows = []

for display_name, model_path in models_to_check:
    print('Processing', display_name)
    if not os.path.exists(model_path):
        print('Model not found:', model_path)
        continue

    model = tf.keras.models.load_model(model_path)

    layer_info = []
    for layer in model.layers:
        try:
            params = layer.count_params()
        except Exception:
            params = 0
        layer_info.append((layer.name, params, layer.__class__.__name__))

    # Save per-model CSV
    safe_name = display_name.replace(' ', '_').replace('/', '_')
    csv_path = os.path.join(out_dir, f'{safe_name}_layer_params.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(['layer_name', 'param_count', 'layer_type'])
        for name, params, ltype in layer_info:
            writer.writerow([name, params, ltype])

    # Add to combined summary
    total_params = model.count_params()
    trainable = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    summary_rows.append({'model': display_name, 'total_params': total_params,
                         'trainable': trainable, 'non_trainable': non_trainable,
                         'csv': csv_path})

    # Plot top layers by params (descending)
    df = pd.DataFrame(layer_info, columns=['layer_name', 'param_count', 'layer_type'])
    df_sorted = df.sort_values('param_count', ascending=False)
    # Keep top 30 for readability
    topn = df_sorted.head(30).iloc[::-1]

    plt.figure(figsize=(10, max(4, len(topn) * 0.3)))
    bars = plt.barh(topn['layer_name'], topn['param_count'], color='C0')
    plt.xlabel('Parameter count')
    plt.xscale('log')
    plt.title(f'Per-layer parameter counts (top {len(topn)})\n{display_name}')
    plt.tight_layout()
    img_path = os.path.join(out_dir, f'{safe_name}_layer_param_counts.png')
    plt.savefig(img_path, dpi=150)
    plt.close()
    print('Wrote', csv_path, 'and', img_path)

# Combined totals CSV and plot
totals_csv = os.path.join(out_dir, 'model_param_totals.csv')
with open(totals_csv, 'w', newline='', encoding='utf-8') as fh:
    writer = csv.writer(fh)
    writer.writerow(['model', 'total_params', 'trainable', 'non_trainable', 'layer_params_csv'])
    for r in summary_rows:
        writer.writerow([r['model'], r['total_params'], r['trainable'], r['non_trainable'], r['csv']])

# Plot total params comparison
if summary_rows:
    df_tot = pd.DataFrame(summary_rows)
    df_tot_sorted = df_tot.sort_values('total_params', ascending=True)
    plt.figure(figsize=(6, 3 + 0.6 * len(df_tot_sorted)))
    plt.barh(df_tot_sorted['model'], df_tot_sorted['total_params'], color=['C1','C2','C3'])
    plt.xscale('log')
    plt.xlabel('Total parameter count (log scale)')
    plt.title('Model total parameter comparison')
    plt.tight_layout()
    totals_img = os.path.join(out_dir, 'model_param_totals.png')
    plt.savefig(totals_img, dpi=150)
    plt.close()
    print('Wrote totals CSV and plot:', totals_csv, totals_img)
else:
    print('No model summaries collected, skipping totals plot')

print('Done')
