import os
import io
import tensorflow as tf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

models_to_check = [
    ("SqueezeNet-inspired", os.path.join(ROOT, 'models', 'squeezenet_96x96_full_epochs_with_unknown', 'model.keras')),
    ("MobileNetV2-baseline", os.path.join(ROOT, 'models', 'mobilenetv2_96x96_full_epochs_with_unknown', 'model.keras')),
    ("MobileNetV2-Eff-style", os.path.join(ROOT, 'models', 'mobilenetv2_efficientnet_style_96x96', 'model.keras')),
]

output_path = os.path.join(ROOT, 'docs', 'model_summaries.md')

with open(output_path, 'w', encoding='utf-8') as out:
    out.write('# Model summaries and parameter counts\n\n')

    for display_name, model_path in models_to_check:
        out.write(f'## {display_name}\n')
        out.write(f'Model file: `{model_path}`\n\n')

        if not os.path.exists(model_path):
            out.write('**Model file not found**\n\n')
            print(f'Model file not found: {model_path}')
            continue

        try:
            model = tf.keras.models.load_model(model_path)
        except Exception as e:
            out.write(f'**Error loading model:** {e}\n\n')
            print(f'Error loading {model_path}: {e}')
            continue

        # Capture model.summary()
        stream = io.StringIO()
        model.summary(print_fn=lambda s: stream.write(s + '\n'))
        summary_text = stream.getvalue()
        out.write("```\n")
        out.write(summary_text)
        out.write("```\n\n")

        # Parameter counts
        total_params = model.count_params()
        trainable = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        non_trainable = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
        out.write(f'- Total params: {total_params}\n')
        out.write(f'- Trainable params: {trainable}\n')
        out.write(f'- Non-trainable params: {non_trainable}\n\n')

        print(f'Wrote summary for {display_name} -> {output_path}')

print('Done. Summaries written to', output_path)
