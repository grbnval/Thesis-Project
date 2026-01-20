import os
import re
import sys
import numpy as np
import matplotlib.pyplot as plt

def parse_classification_report(path):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Model name (optional)
    model_name = None
    m = re.search(r'^Model:\s*(.*)$', text, re.MULTILINE)
    if m:
        model_name = m.group(1).strip()

    # Extract the classification report block
    report_block = None
    m = re.search(r'Classification Report:\s*\n([\s\S]*?)\n\n=', text)
    if m:
        report_block = m.group(1)
    else:
        # fallback: take block between header and empty line before FN summary
        m2 = re.search(r'Classification Report:\s*\n([\s\S]*?)\n\s*\n', text)
        if m2:
            report_block = m2.group(1)

    classes = []
    precisions = []
    recalls = []
    f1s = []
    supports = []

    if report_block:
        for line in report_block.splitlines():
            # match lines like: name  0.98  0.86  0.92  596
            m = re.match(r"^\s*([A-Za-z0-9_ ]+?)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)\s*$", line)
            if m:
                name = m.group(1).strip()
                p = float(m.group(2))
                r = float(m.group(3))
                f = float(m.group(4))
                s = int(m.group(5))
                classes.append(name)
                precisions.append(p)
                recalls.append(r)
                f1s.append(f)
                supports.append(s)

    # Parse FN summary
    fn_rates = {c: 0.0 for c in classes}
    fp_rates = {c: 0.0 for c in classes}

    mfn = re.search(r'FALSE NEGATIVE SUMMARY:\s*\n([\s\S]*?)\n\s*\n=', text)
    if mfn:
        fn_block = mfn.group(1)
    else:
        mfn2 = re.search(r'FALSE NEGATIVE SUMMARY:\s*\n([\s\S]*?)\n\s*\n', text)
        fn_block = mfn2.group(1) if mfn2 else ''

    for line in fn_block.splitlines():
        m = re.match(r"^\s*([A-Za-z0-9_ ]+?):\s*(\d+)/(\d+)\s*\(([-0-9.]+)%\s*FN rate\)", line)
        if m:
            name = m.group(1).strip()
            num = int(m.group(2))
            denom = int(m.group(3))
            pct = float(m.group(4))
            if name in fn_rates:
                fn_rates[name] = pct

    # Parse FP summary
    mfp = re.search(r'FALSE POSITIVE SUMMARY:\s*\n([\s\S]*?)\n\s*$', text)
    if mfp:
        fp_block = mfp.group(1)
    else:
        mfp2 = re.search(r'FALSE POSITIVE SUMMARY:\s*\n([\s\S]*?)\n\s*\Z', text)
        fp_block = mfp2.group(1) if mfp2 else ''

    for line in fp_block.splitlines():
        m = re.match(r"^\s*([A-Za-z0-9_ ]+?):\s*(\d+)/(\d+)\s*\(([-0-9.]+)%\s*FP rate", line)
        if m:
            name = m.group(1).strip()
            pct = float(m.group(4))
            if name in fp_rates:
                fp_rates[name] = pct

    fn_list = [fn_rates.get(c, 0.0) for c in classes]
    fp_list = [fp_rates.get(c, 0.0) for c in classes]

    return {
        'model_name': model_name,
        'classes': classes,
        'precision': precisions,
        'recall': recalls,
        'f1': f1s,
        'support': supports,
        'fn_percent': fn_list,
        'fp_percent': fp_list,
    }


def plot_report(data, out_path):
    classes = data['classes']
    p = np.array(data['precision'])
    r = np.array(data['recall'])
    f = np.array(data['f1'])
    fn = np.array(data['fn_percent'])
    fp = np.array(data['fp_percent'])
    n = len(classes)

    x = np.arange(n)
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(8, n * 0.8), 6))
    ax.bar(x - width, p, width, label='Precision', color='C0')
    ax.bar(x, r, width, label='Recall', color='C1')
    ax.bar(x + width, f, width, label='F1-score', color='C2')

    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_ylabel('Score (0-1)')
    ax.set_title((data['model_name'] or 'Classification Report') + ' — precision / recall / f1 by class')
    ax.legend(loc='upper left')

    # Secondary axis for FN/FP percent
    ax2 = ax.twinx()
    ax2.plot(x, fn, marker='x', linestyle='--', color='red', label='FN %')
    ax2.plot(x, fp, marker='o', linestyle=':', color='black', label='FP %')
    ax2.set_ylim(0, max(10, max(fn.max() if len(fn) else 0, fp.max() if len(fp) else 0) * 1.2))
    ax2.set_ylabel('False rate (%)')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    if len(sys.argv) > 1:
        report_path = sys.argv[1]
    else:
        report_path = os.path.join('models', 'mobilenetv2_96x96_full_epochs_100', 'evaluation', 'classification_report.txt')

    if not os.path.exists(report_path):
        print('Report file not found:', report_path)
        sys.exit(1)

    data = parse_classification_report(report_path)
    out_png = os.path.join(os.path.dirname(report_path), 'classification_report_plot.png')
    plot_report(data, out_png)
    print('Wrote plot to', out_png)


if __name__ == '__main__':
    main()
