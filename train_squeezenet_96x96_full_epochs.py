import tensorflow as tf
import numpy as np
import os
import json
import datetime
import time
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Constants
IMAGE_SIZE = 96
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
PROCESSED_DATA_DIR = "processed_dataset_96x96"
TEST_SPLIT = 0.2

def load_preprocessed_data(data_dir):
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory not found: {data_dir}")
    class_names = sorted(os.listdir(data_dir))
    if not class_names:
        raise ValueError(f"No class folders found in {data_dir}")
    images = []
    labels = []
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        files = [f for f in os.listdir(class_dir) if f.endswith('.npy')]
        if not files:
            print(f"Warning: No .npy files found in {class_name}/")
            continue
        print(f"Loading {len(files)} images from {class_name}/")
        for img_file in tqdm(files, desc=f"Loading {class_name}"):
            img_path = os.path.join(class_dir, img_file)
            img = np.load(img_path)
            images.append(img)
            labels.append(class_idx)
    if not images:
        raise ValueError("No images found in any class folder!")
    return np.array(images), np.array(labels), class_names

def create_squeezenet_model(input_shape, num_classes):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    for layer in base_model.layers:
        layer.trainable = False
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model, base_model

def create_data_generators(X_train, y_train, X_val, y_val):
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator()
    train_generator = train_datagen.flow(
        X_train, y_train,
        batch_size=BATCH_SIZE
    )
    val_generator = val_datagen.flow(
        X_val, y_val,
        batch_size=BATCH_SIZE
    )
    return train_generator, val_generator

def train_model_full_epochs(model, base_model, train_generator, val_generator, callbacks):
    # Two-phase training: Phase 1 (top layers), Phase 2 (fine-tuning)
    print("\n=== Phase 1: Training top layers only ===")
    history1 = model.fit(
        train_generator,
        epochs=20,
        validation_data=val_generator,
        callbacks=callbacks
    )

    print("\n=== Phase 2: Fine-tuning last layers ===")
    for layer in base_model.layers[-30:]:
        layer.trainable = True
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE/10),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    history2 = model.fit(
        train_generator,
        epochs=EPOCHS,
        initial_epoch=20,
        validation_data=val_generator,
        callbacks=callbacks
    )
    # Combine histories
    combined_history = {}
    for key in history1.history:
        combined_history[key] = history1.history[key] + history2.history[key]
    return combined_history

def plot_training_history(history, output_dir):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plots", "training_history.png"))
    plt.close()

def convert_to_tflite(model, output_path):
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        print(f"TFLite model saved to: {output_path}")
        return True
    except Exception as e:
        print(f"Error converting to TFLite: {str(e)}")
        print("Try using the model in .keras format instead.")
        return False

if __name__ == "__main__":
    start_time = time.time()
    try:
        print("=== TRAINING SQUEEZENET 96x96 MODEL (FULL 100 EPOCHS) ===")
        PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
        PROCESSED_DIR = os.path.join(PROJECT_ROOT, PROCESSED_DATA_DIR)
        MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "squeezenet_96x96_full_epochs_100")
        os.makedirs(MODEL_DIR, exist_ok=True)
        print(f"Looking for processed data in: {PROCESSED_DIR}")
        if not os.path.exists(PROCESSED_DIR):
            print(f"\nError: Preprocessed data directory not found: {PROCESSED_DIR}")
            print("\nPlease run preprocess_mobilenetv2_96x96.py first to prepare the dataset.")
            exit(1)
        X, y, classes = load_preprocessed_data(PROCESSED_DIR)
        print(f"\nLoaded {len(X)} images from {len(classes)} classes")
        num_classes = len(classes)
        y_cat = tf.keras.utils.to_categorical(y, num_classes)
        X_train, X_temp, y_train_cat, y_temp_cat = train_test_split(
            X, y_cat, test_size=TEST_SPLIT, random_state=42, stratify=y
        )
        X_val, X_test, y_val_cat, y_test_cat = train_test_split(
            X_temp, y_temp_cat, test_size=0.5, random_state=42
        )
        print(f"\nTraining data shape: {X_train.shape}")
        print(f"Validation data shape: {X_val.shape}")
        print(f"Test data shape: {X_test.shape}")
        input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)
        model, base_model = create_squeezenet_model(input_shape, num_classes)
        model.summary()
        train_generator, val_generator = create_data_generators(
            X_train, y_train_cat, X_val, y_val_cat
        )
        callbacks = [
            ModelCheckpoint(
                os.path.join(MODEL_DIR, "best_model.h5"),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        history = train_model_full_epochs(
            model, base_model, train_generator, val_generator, callbacks
        )
        plot_training_history(history, MODEL_DIR)
        test_datagen = ImageDataGenerator()
        test_generator = test_datagen.flow(
            X_test, y_test_cat, batch_size=BATCH_SIZE
        )
        test_loss, test_accuracy = model.evaluate(test_generator)
        print(f"\nTest accuracy: {test_accuracy:.4f}")
        print(f"Test loss: {test_loss:.4f}")
        print("\nSaving final model (after all epochs)...")
        final_model_path = os.path.join(MODEL_DIR, "final_model.keras")
        model.save(final_model_path)
        print(f"- Final model saved to: {final_model_path}")
        best_model_path = os.path.join(MODEL_DIR, "best_model.h5")
        print(f"- Best model saved to: {best_model_path} (by ModelCheckpoint callback)")
        print("\nSaving models in different formats...")
        keras_path = os.path.join(MODEL_DIR, "model.keras")
        model.save(keras_path)
        print(f"- Keras model saved to: {keras_path}")
        tflite_path = os.path.join(MODEL_DIR, "model.tflite")
        convert_to_tflite(model, tflite_path)
        try:
            print("Creating quantized TFLite model...")
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
            def representative_dataset():
                for i in range(min(100, len(X_train))):
                    img = X_train[i:i+1].copy()
                    yield [img.astype(np.float32)]
            converter.representative_dataset = representative_dataset
            tflite_quant_model = converter.convert()
            tflite_quant_path = os.path.join(MODEL_DIR, "model_quantized.tflite")
            with open(tflite_quant_path, 'wb') as f:
                f.write(tflite_quant_model)
            print(f"- Quantized TFLite model saved to: {tflite_quant_path}")
            print(f"  Size: {os.path.getsize(tflite_quant_path) / (1024 * 1024):.2f} MB")
        except Exception as e:
            print(f"Error creating quantized model: {str(e)}")
            print("Full error details:", e)
            print("Proceeding without quantized model")
        class_info = {
            "classes": list(classes),
            "input_shape": [IMAGE_SIZE, IMAGE_SIZE, 3],
            "version": "1.0",
            "model_type": "SqueezeNet-inspired (MobileNetV2 base with SqueezeNet-like top layers)",
            "resolution": "96x96",
            "training_type": "full_100_epochs",
            "date_trained": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        class_info_path = os.path.join(MODEL_DIR, "class_info.json")
        with open(class_info_path, 'w') as f:
            json.dump(class_info, f, indent=2)
        print(f"- Class info saved to: {class_info_path}")
        end_time = time.time()
        training_time = end_time - start_time
        hours, remainder = divmod(training_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"\nTraining completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
        print("=== TRAINING COMPLETED SUCCESSFULLY ===")
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nPlease ensure:")
        print(f"1. You have run preprocess_mobilenetv2_96x96.py first")
        print(f"2. The processed dataset folder exists: {PROCESSED_DATA_DIR}")
        print("3. Each class folder contains .npy files")
        raise
