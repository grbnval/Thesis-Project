import os
import pandas as pd
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, 'models')
IN_CSV = os.path.join(MODELS_DIR, 'evaluation_metrics.csv')
OUT_CSV = os.path.join(MODELS_DIR, 'comparison_three_models.csv')
OUT_PNG = os.path.join(MODELS_DIR, 'comparison_plot.png')

# Models to compare (exact folder names)
TARGET_MODELS = [
    'squeezenet_96x96_full_epochs_with_unknown',
    'mobilenetv2_96x96_full_epochs_with_unknown',
    'mobilenetv2_efficientnet_style_96x96',
]

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return float('nan')

def main():
    if not os.path.exists(IN_CSV):
        print(f"Input CSV not found: {IN_CSV}")
        return

    df = pd.read_csv(IN_CSV)
    df_filtered = df[df['model'].isin(TARGET_MODELS)].copy()

    # Convert numeric columns
    num_cols = [
        'keras_accuracy',
        'standard_tflite_accuracy',
        'standard_inference_ms',
        'standard_model_size_mb',
        'quantized_tflite_accuracy',
        'quantized_inference_ms',
        'quantized_model_size_mb',
    ]
    for c in num_cols:
        if c in df_filtered.columns:
            df_filtered[c] = df_filtered[c].apply(safe_float)

    # Save filtered CSV
    df_filtered.to_csv(OUT_CSV, index=False)
    print(f"Wrote filtered CSV: {OUT_CSV}")

    # Plotting
    # Use a built-in matplotlib style (avoid optional seaborn dependency)
    try:
        plt.style.use('seaborn-darkgrid')
    except Exception:
        plt.style.use('ggplot')
    fig, axes = plt.subplots(1, 2, figsize=(14,5))

    # Accuracy plot
    ax = axes[0]
    x = df_filtered['model']
    ax.plot(x, df_filtered['keras_accuracy'], marker='o', label='Keras')
    ax.plot(x, df_filtered['standard_tflite_accuracy'], marker='o', label='TFLite (standard)')
    ax.plot(x, df_filtered['quantized_tflite_accuracy'], marker='o', label='TFLite (quantized)')
    ax.set_ylim(0.0, 1.0)
    ax.set_title('Model accuracy comparison')
    ax.set_ylabel('Accuracy')
    ax.set_xticklabels(x, rotation=25, ha='right')
    ax.legend()

    # Inference time plot (ms)
    ax2 = axes[1]
    ax2.bar(x, df_filtered['standard_inference_ms'], alpha=0.7, label='TFLite standard (ms)')
    ax2.bar(x, df_filtered['quantized_inference_ms'], alpha=0.7, label='TFLite quantized (ms)')
    ax2.set_title('Average inference time (ms)')
    ax2.set_ylabel('Milliseconds per image')
    ax2.set_xticklabels(x, rotation=25, ha='right')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(OUT_PNG)
    plt.close()
    print(f"Wrote comparison plot: {OUT_PNG}")

if __name__ == '__main__':
    main()
