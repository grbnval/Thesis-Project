import os
import sys
import random
import numpy as np

# Ensure project root is on sys.path so we can import preprocess_dataset_96x96
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PROJECT_ROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from preprocess_dataset_96x96 import preprocess_image

PROCESSED_TEST_DIR = os.path.join(ROOT, 'processed_test_96x96')
RAW_DIR = os.path.join(ROOT, 'raw_dataset')

OUT_REPORT = os.path.join(ROOT, 'models', 'preprocess_parity_report.txt')

def sample_pairs(num_samples=50):
    pairs = []
    classes = [d for d in os.listdir(PROCESSED_TEST_DIR) if os.path.isdir(os.path.join(PROCESSED_TEST_DIR, d))]
    for cls in classes:
        cls_processed = os.path.join(PROCESSED_TEST_DIR, cls)
        files = [f for f in os.listdir(cls_processed) if f.endswith('.npy')]
        random.shuffle(files)
        for f in files[:num_samples]:
            # find corresponding raw image by searching raw_dataset class dir for basename
            basename = os.path.splitext(f)[0]
            raw_candidates = [x for x in os.listdir(os.path.join(RAW_DIR, cls)) if os.path.splitext(x)[0] == basename]
            if not raw_candidates:
                continue
            pairs.append((os.path.join(RAW_DIR, cls, raw_candidates[0]), os.path.join(cls_processed, f)))
    return pairs

def compare_pair(raw_path, npy_path):
    processed_saved = np.load(npy_path)
    processed_onfly = preprocess_image(raw_path, target_size=(96,96), enhance_contrast=True)
    if processed_onfly is None:
        return None
    # compute absolute differences
    diff = np.abs(processed_saved.astype('float32') - processed_onfly.astype('float32'))
    return {
        'npy_path': npy_path,
        'raw_path': raw_path,
        'max_abs_diff': float(np.max(diff)),
        'mean_abs_diff': float(np.mean(diff)),
    }

def main():
    if not os.path.isdir(PROCESSED_TEST_DIR):
        print(f"Processed test dir not found: {PROCESSED_TEST_DIR}")
        return
    pairs = sample_pairs(num_samples=5)  # small sample per class
    results = []
    for raw, npy in pairs:
        res = compare_pair(raw, npy)
        if res is not None:
            results.append(res)

    # Save report
    with open(OUT_REPORT, 'w', encoding='utf-8') as f:
        f.write('Preprocess parity test report\n')
        f.write('Pairs tested: {}\n\n'.format(len(results)))
        for r in results:
            f.write(f"RAW: {r['raw_path']}\n")
            f.write(f"NPY: {r['npy_path']}\n")
            f.write(f"Max abs diff: {r['max_abs_diff']:.6f}\n")
            f.write(f"Mean abs diff: {r['mean_abs_diff']:.6f}\n")
            f.write('\n')

    print(f"Wrote parity report to: {OUT_REPORT}")

if __name__ == '__main__':
    main()
