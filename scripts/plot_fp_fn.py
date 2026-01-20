import os
import re
import csv
import pandas as pd
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = [
    'squeezenet_96x96_full_epochs_with_unknown',
    'mobilenetv2_96x96_full_epochs_with_unknown',
    'mobilenetv2_efficientnet_style_96x96',
]
MODELS_DIR = os.path.join(ROOT, 'models')
OUT_CSV = os.path.join(MODELS_DIR, 'fp_fn_comparison.csv')
OUT_FN_PNG = os.path.join(MODELS_DIR, 'fn_comparison.png')
OUT_FP_PNG = os.path.join(MODELS_DIR, 'fp_comparison.png')

CLASS_ORDER = ['early_blight_leaf', 'healthy_leaf', 'late_blight_leaf', 'septoria_leaf', 'unknown']

def read_file(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception:
        return ''

def parse_fn_report(text):
    # returns dict class -> (fn_count, support, fn_rate)
    res = {}
    for cls in CLASS_ORDER:
        # match lines like: "early_blight_leaf: 38/596 (6.4% FN rate)" or the detailed format
        m = re.search(rf"{cls}:\s*(\d+)/(\d+)\s*\(([^)]*)FN rate\)?", text)
        if m:
            fn = int(m.group(1))
            sup = int(m.group(2))
            # rate parsing fallback
            res[cls] = {'fn': fn, 'support': sup, 'fn_rate': fn / sup}
            continue

        # try alternative verbose block
        m2 = re.search(rf"{cls}:[\s\S]*?False negatives:\s*(\d+)/(\d+)", text)
        if m2:
            fn = int(m2.group(1))
            sup = int(m2.group(2))
            res[cls] = {'fn': fn, 'support': sup, 'fn_rate': fn / sup}
            continue

        res[cls] = {'fn': None, 'support': None, 'fn_rate': None}
    return res

def parse_fp_report(text):
    # returns dict class -> (fp_count, total_predictions, fp_rate)
    res = {}
    for cls in CLASS_ORDER:
        # match like: "early_blight_leaf: 4/562 (0.7% FP rate, 99.3% precision)" or lines with "Predicted Class: early_blight_leaf\nFalse Positives: 7/529 (0.013)"
        m = re.search(rf"{cls}:\s*(\d+)/(\d+)\s*\(([^)]*?)FP rate", text)
        if m:
            fp = int(m.group(1))
            total = int(m.group(2))
            res[cls] = {'fp': fp, 'total_predictions': total, 'fp_rate': fp / total}
            continue

        m2 = re.search(rf"Predicted Class:\s*{cls}[\s\S]*?False Positives:\s*(\d+)/(\d+)", text)
        if m2:
            fp = int(m2.group(1))
            total = int(m2.group(2))
            res[cls] = {'fp': fp, 'total_predictions': total, 'fp_rate': fp / total}
            continue

        res[cls] = {'fp': None, 'total_predictions': None, 'fp_rate': None}
    return res

def collect_all():
    rows = []
    for m in MODELS:
        eval_dir = os.path.join(MODELS_DIR, m, 'evaluation')
        fn_text = read_file(os.path.join(eval_dir, 'false_negative_report.txt'))
        fp_text = read_file(os.path.join(eval_dir, 'false_positive_report.txt'))
        fn_parsed = parse_fn_report(fn_text)
        fp_parsed = parse_fp_report(fp_text)
        for cls in CLASS_ORDER:
            rows.append({
                'model': m,
                'class': cls,
                'fn': fn_parsed[cls]['fn'],
                'fn_rate': fn_parsed[cls]['fn_rate'],
                'support': fn_parsed[cls]['support'],
                'fp': fp_parsed[cls]['fp'],
                'fp_rate': fp_parsed[cls]['fp_rate'],
                'total_predictions': fp_parsed[cls]['total_predictions'],
            })
    return rows

def write_csv(rows):
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote FP/FN CSV: {OUT_CSV}")
    return df

def plot(df):
    # FN plot
    fig, axes = plt.subplots(1, 2, figsize=(14,5))
    ax1, ax2 = axes

    for m in MODELS:
        sub = df[df['model'] == m]
        ax1.plot(sub['class'], sub['fn_rate'], marker='o', label=m)
    ax1.set_title('False Negative Rate by Class')
    ax1.set_ylabel('FN rate')
    finite_fn = df['fn_rate'].dropna()
    if not finite_fn.empty:
        top1 = float(finite_fn.max() * 1.2)
    else:
        top1 = 0.05
    ax1.set_ylim(0, max(top1, 0.05))
    ax1.legend()

    for m in MODELS:
        sub = df[df['model'] == m]
        ax2.plot(sub['class'], sub['fp_rate'], marker='o', label=m)
    ax2.set_title('False Positive Rate by Class')
    ax2.set_ylabel('FP rate')
    finite_fp = df['fp_rate'].dropna()
    if not finite_fp.empty:
        top2 = float(finite_fp.max() * 1.2)
    else:
        top2 = 0.05
    ax2.set_ylim(0, max(top2, 0.05))
    ax2.legend()

    plt.tight_layout()
    plt.savefig(OUT_FN_PNG)
    # Save separate FP plot as well
    plt.savefig(OUT_FP_PNG)
    plt.close()
    print(f"Wrote FN/FP plots: {OUT_FN_PNG}, {OUT_FP_PNG}")

def main():
    rows = collect_all()
    df = write_csv(rows)
    plot(df)

if __name__ == '__main__':
    main()
