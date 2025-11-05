import os
import re
import csv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
OUT_CSV = os.path.join(MODELS_DIR, "evaluation_metrics.csv")

def read_text(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception:
        return ''

def parse_metrics(text):
    metrics = {}
    # Keras/Test accuracy patterns
    m = re.search(r"Test accuracy:\s*([0-9]*\.?[0-9]+)", text)
    if not m:
        m = re.search(r"OVERALL MODEL SCORE:\s*([0-9]*\.?[0-9]+)%", text)
        if m:
            # convert percent to 0-1
            metrics['keras_accuracy'] = float(m.group(1)) / 100.0
    else:
        metrics['keras_accuracy'] = float(m.group(1))

    # Generic accuracy (used in tflite reports)
    m = re.search(r"Accuracy:\s*([0-9]*\.?[0-9]+)", text)
    if m:
        metrics.setdefault('any_accuracy', float(m.group(1)))

    # Average inference time (ms)
    m = re.search(r"Average inference time:\s*([0-9]*\.?[0-9]+)\s*ms", text)
    if m:
        metrics['inference_ms'] = float(m.group(1))

    # Model size (MB)
    m = re.search(r"Model size:\s*([0-9]*\.?[0-9]+)\s*MB", text)
    if m:
        metrics['model_size_mb'] = float(m.group(1))

    return metrics

def collect_for_model(model_path):
    eval_dir = os.path.join(model_path, 'evaluation')
    result = {
        'model': os.path.basename(model_path),
        'keras_accuracy': '',
        'standard_tflite_accuracy': '',
        'standard_inference_ms': '',
        'standard_model_size_mb': '',
        'quantized_tflite_accuracy': '',
        'quantized_inference_ms': '',
        'quantized_model_size_mb': '',
    }

    if not os.path.isdir(eval_dir):
        return result

    # Files we might find
    candidates = os.listdir(eval_dir)
    # Read evaluation_summary or classification_report for keras accuracy
    for fname in ['evaluation_summary.txt', 'classification_report.txt']:
        fpath = os.path.join(eval_dir, fname)
        if os.path.exists(fpath):
            txt = read_text(fpath)
            parsed = parse_metrics(txt)
            if 'keras_accuracy' in parsed:
                result['keras_accuracy'] = parsed['keras_accuracy']
                break

    # Standard TFLite
    for fname in ['standard_tflite_results.txt', 'standard_tflite_report.txt', 'standard_tflite.txt']:
        fpath = os.path.join(eval_dir, fname)
        if os.path.exists(fpath):
            txt = read_text(fpath)
            parsed = parse_metrics(txt)
            if 'any_accuracy' in parsed:
                result['standard_tflite_accuracy'] = parsed['any_accuracy']
            if 'inference_ms' in parsed:
                result['standard_inference_ms'] = parsed['inference_ms']
            if 'model_size_mb' in parsed:
                result['standard_model_size_mb'] = parsed['model_size_mb']
            break

    # Quantized TFLite
    for fname in ['quantized_tflite_results.txt', 'quantized_tflite_report.txt', 'quantized_tflite.txt']:
        fpath = os.path.join(eval_dir, fname)
        if os.path.exists(fpath):
            txt = read_text(fpath)
            parsed = parse_metrics(txt)
            if 'any_accuracy' in parsed:
                result['quantized_tflite_accuracy'] = parsed['any_accuracy']
            if 'inference_ms' in parsed:
                result['quantized_inference_ms'] = parsed['inference_ms']
            if 'model_size_mb' in parsed:
                result['quantized_model_size_mb'] = parsed['model_size_mb']
            break

    return result

def main():
    models = [os.path.join(MODELS_DIR, d) for d in os.listdir(MODELS_DIR) if os.path.isdir(os.path.join(MODELS_DIR, d))]
    rows = []
    for m in models:
        rows.append(collect_for_model(m))

    # Write CSV
    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'model',
            'keras_accuracy',
            'standard_tflite_accuracy',
            'standard_inference_ms',
            'standard_model_size_mb',
            'quantized_tflite_accuracy',
            'quantized_inference_ms',
            'quantized_model_size_mb',
        ])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Wrote aggregated metrics to: {OUT_CSV}")

if __name__ == '__main__':
    main()
