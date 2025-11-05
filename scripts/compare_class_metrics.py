import os
import re
import pandas as pd
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, 'models')
OUT_CSV = os.path.join(MODELS_DIR, 'class_metrics_comparison.csv')
OUT_DIR = MODELS_DIR

MODELS = [
    'squeezenet_96x96_full_epochs_with_unknown',
    'mobilenetv2_96x96_full_epochs_with_unknown',
    'mobilenetv2_efficientnet_style_96x96',
]

CLASS_ORDER = ['early_blight_leaf', 'healthy_leaf', 'late_blight_leaf', 'septoria_leaf', 'unknown']

def read_text(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception:
        return ''

def parse_classification_report(text):
    # Parse sklearn-like classification report into dict[class] = {precision, recall, f1, support}
    res = {cls: {'precision': None, 'recall': None, 'f1_score': None, 'support': None} for cls in CLASS_ORDER}
    for line in text.splitlines():
        line_stripped = line.strip()
        if not line_stripped:
            continue
        parts = re.split(r"\s+", line_stripped)
        # Expect lines like: class_name  precision recall f1-score support
        if parts[0] in CLASS_ORDER and len(parts) >= 5:
            try:
                precision = float(parts[1])
                recall = float(parts[2])
                f1_score = float(parts[3])
                support = int(parts[4])
                res[parts[0]] = {
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1_score,
                    'support': support,
                }
            except Exception:
                # leave None on parse failure
                pass
    return res

def collect():
    rows = []
    for model in MODELS:
        eval_dir = os.path.join(MODELS_DIR, model, 'evaluation')
        # Keras/classification report
        kr = read_text(os.path.join(eval_dir, 'classification_report.txt'))
        # TFLite standard
        tr_std = read_text(os.path.join(eval_dir, 'standard_tflite_report.txt'))
        tr_q = read_text(os.path.join(eval_dir, 'quantized_tflite_report.txt'))
        # sometimes files have different names (check alternatives)
        if not tr_std:
            tr_std = read_text(os.path.join(eval_dir, 'standard_tflite_results.txt'))
        if not tr_q:
            tr_q = read_text(os.path.join(eval_dir, 'quantized_tflite_results.txt'))

        metrics_keras = parse_classification_report(kr)
        metrics_std = parse_classification_report(tr_std)
        metrics_q = parse_classification_report(tr_q)

        for cls in CLASS_ORDER:
            rows.append({
                'model': model,
                'runtime': 'keras',
                'class': cls,
                'precision': metrics_keras[cls]['precision'],
                'recall': metrics_keras[cls]['recall'],
                'f1_score': metrics_keras[cls]['f1_score'],
                'support': metrics_keras[cls]['support'],
            })
            rows.append({
                'model': model,
                'runtime': 'tflite_standard',
                'class': cls,
                'precision': metrics_std[cls]['precision'],
                'recall': metrics_std[cls]['recall'],
                'f1_score': metrics_std[cls]['f1_score'],
                'support': metrics_std[cls]['support'],
            })
            rows.append({
                'model': model,
                'runtime': 'tflite_quantized',
                'class': cls,
                'precision': metrics_q[cls]['precision'],
                'recall': metrics_q[cls]['recall'],
                'f1_score': metrics_q[cls]['f1_score'],
                'support': metrics_q[cls]['support'],
            })
    return rows

def write_and_plot(rows):
    df = pd.DataFrame(rows)
    # Coerce metric columns to numeric (some reports may have missing or non-numeric entries)
    for c in ['precision', 'recall', 'f1_score', 'support']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote class metrics CSV: {OUT_CSV}")

    # Plot precision, recall, f1 for each class (grouped by model/runtime)
    metrics = ['precision', 'recall', 'f1_score']
    for metric in metrics:
        plt.figure(figsize=(10,6))
        for model in MODELS:
            sub = df[df['model'] == model]
            # create labels combining runtime
            x = []
            y = []
            labels = []
            for cls in CLASS_ORDER:
                for runtime in ['keras', 'tflite_standard', 'tflite_quantized']:
                    val = sub[(sub['class'] == cls) & (sub['runtime'] == runtime)][metric].values
                    if len(val) == 0:
                        y.append(float('nan'))
                    else:
                        y.append(float(val[0]))
                    labels.append(f"{cls}\n({runtime.split('_')[-1]})")
            plt.plot(labels, y, marker='o', label=model)
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0,1)
        plt.ylabel(metric.capitalize())
        plt.title(f'Per-class {metric} comparison (Keras / TFLite std / TFLite quant)')
        plt.legend()
        out_png = os.path.join(OUT_DIR, f'class_{metric}_comparison.png')
        plt.tight_layout()
        plt.savefig(out_png)
        plt.close()
        print(f"Wrote plot: {out_png}")

    # --- Grouped bar charts per class (models on x-axis, runtimes as bars) ---
    import numpy as np
    runtimes = ['keras', 'tflite_standard', 'tflite_quantized']
    for metric in metrics:
        fig, axes = plt.subplots(1, len(CLASS_ORDER), figsize=(4 * len(CLASS_ORDER), 5), sharey=True)
        if len(CLASS_ORDER) == 1:
            axes = [axes]
        for i, cls in enumerate(CLASS_ORDER):
            ax = axes[i]
            x = np.arange(len(MODELS))
            width = 0.2
            for j, runtime in enumerate(runtimes):
                vals = []
                for model in MODELS:
                    v = df[(df['model'] == model) & (df['class'] == cls) & (df['runtime'] == runtime)][metric]
                    vals.append(float(v.values[0]) if not v.empty and pd.notna(v.values[0]) else np.nan)
                ax.bar(x + (j - 1) * width, vals, width, label=runtime.split('_')[-1])
            ax.set_xticks(x)
            ax.set_xticklabels([m.replace('_', '\n') for m in MODELS], rotation=0)
            ax.set_title(cls)
            if i == 0:
                ax.set_ylabel(metric.capitalize())
            ax.set_ylim(0, 1)
            ax.legend()
        plt.tight_layout()
        out_png = os.path.join(OUT_DIR, f'class_{metric}_bar_comparison.png')
        plt.savefig(out_png, dpi=200)
        plt.close()
        print(f"Wrote bar plot: {out_png}")

def main():
    rows = collect()
    write_and_plot(rows)

if __name__ == '__main__':
    main()
