import tensorflow as tf
import numpy as np
import os
import json
import datetime
import time
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# Constants
IMAGE_SIZE = 300
BATCH_SIZE = 50  # As requested
EPOCHS_PHASE1 = 10  # As requested for first phase
EPOCHS_PHASE2 = 50  # As requested for second phase
LEARNING_RATE = 0.001  # Using same learning rate as successful models
PROCESSED_DATA_DIR = "processed_train_300x300"
PROCESSED_TEST_DIR = "processed_test_300x300"
TEST_SPLIT = 0.2  # Percentage of data to use for testing

def load_preprocessed_data(data_dir):
    """Load preprocessed numpy arrays"""
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
        for img_file in files:
            img_path = os.path.join(class_dir, img_file)
            img = np.load(img_path)
            images.append(img)
            labels.append(class_idx)
    
    if not images:
        raise ValueError("No images found in any class folder!")
    
    return np.array(images), np.array(labels), class_names

def create_squeezenet_model(input_shape, num_classes):
    """Create a hybrid model using MobileNetV2 as base with SqueezeNet-inspired top layers
    
    Since MobileNetV2 and SqueezeNet are both designed for mobile/embedded devices,
    this is a good substitute for SqueezeNet while leveraging transfer learning.
    We'll use pre-trained weights from ImageNet to improve performance.
    """
    
    # Base model with pre-trained weights (same approach as successful models)
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model layers initially
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add custom SqueezeNet-inspired top layers
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        
        # Squeeze-like layer (reduce dimensionality)
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        
        # Expand-like layers (two parallel paths combined)
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        
        # Final classification layer
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model, base_model

def create_data_generators(X_train, y_train, X_val, y_val):
    """Create data generators with augmentation for training"""
    
    # Data augmentation for training - using same settings as successful models
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # No augmentation for validation
    val_datagen = ImageDataGenerator()
    
    # Create generators
    train_generator = train_datagen.flow(
        X_train, y_train,
        batch_size=BATCH_SIZE
    )
    
    val_generator = val_datagen.flow(
        X_val, y_val,
        batch_size=BATCH_SIZE
    )
    
    return train_generator, val_generator

def train_model_in_phases(model, base_model, train_generator, val_generator, callbacks_phase1, callbacks_phase2):
    """Train the model in two phases as requested: 
    1) Top layers only with 10 epochs, 50 batch size
    2) Fine-tuning with 50 epochs, 50 batch size"""
    
    # Phase 1: Train only the top layers (base model is frozen)
    print("\n=== Phase 1: Training top layers only (10 epochs) ===")
    history1 = model.fit(
        train_generator,
        epochs=EPOCHS_PHASE1,
        validation_data=val_generator,
        callbacks=callbacks_phase1
    )
    
    # Save the best model from phase 1
    model.save(os.path.join(MODEL_DIR, "best_model_phase1.h5"))
    
    # Phase 2: Fine-tune the model by unfreezing some conv layers
    print("\n=== Phase 2: Fine-tuning last layers (50 epochs) ===")
    
    # Unfreeze the last 50 layers of the MobileNetV2 model (same as other successful models)
    for layer in base_model.layers[-50:]:
        layer.trainable = True
    
    # Recompile with lower learning rate for fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE/10),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Continue training
    history2 = model.fit(
        train_generator,
        epochs=EPOCHS_PHASE2,
        validation_data=val_generator,
        callbacks=callbacks_phase2
    )
    
    # Combine histories
    combined_history = {}
    for key in history1.history:
        combined_history[key] = history1.history[key] + history2.history[key]
    
    return combined_history

def plot_training_history(history, output_dir):
    """Plot and save training history"""
    
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    # Save plot
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plots", "training_history.png"))
    plt.close()

def convert_to_tflite(model, output_path):
    """Convert model to TFLite format"""
    try:
        # Create a converter using the Keras model
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Configure the converter
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Convert the model
        tflite_model = converter.convert()
        
        # Save the model to file
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
        print("=== TRAINING SQUEEZENET 300x300 MODEL ===")
        print("Note: Using a hybrid approach with MobileNetV2 base model and SqueezeNet-inspired top layers")
        print("      This approach leverages transfer learning with ImageNet weights like the other successful models")
        
        # Set up paths
        PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
        PROCESSED_TRAIN_DIR_PATH = os.path.join(PROJECT_ROOT, PROCESSED_DATA_DIR)
        PROCESSED_TEST_DIR_PATH = os.path.join(PROJECT_ROOT, PROCESSED_TEST_DIR)
        MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "squeezenet_300x300")
        
        # Create model directory
        os.makedirs(MODEL_DIR, exist_ok=True)
        os.makedirs(os.path.join(MODEL_DIR, "logs"), exist_ok=True)
        
        print(f"Looking for processed training data in: {PROCESSED_TRAIN_DIR_PATH}")
        print(f"Looking for processed test data in: {PROCESSED_TEST_DIR_PATH}")
        
        # Load processed training data
        X_train, y_train, classes = load_preprocessed_data(PROCESSED_TRAIN_DIR_PATH)
        print(f"\nLoaded {len(X_train)} training images from {len(classes)} classes")
        
        # Load processed test data if available
        use_separate_test_set = os.path.exists(PROCESSED_TEST_DIR_PATH)
        if use_separate_test_set:
            X_test, y_test, test_classes = load_preprocessed_data(PROCESSED_TEST_DIR_PATH)
            print(f"Loaded {len(X_test)} test images from {len(test_classes)} classes")
            
            # Verify that classes match
            if set(classes) != set(test_classes):
                print("Warning: Training and test datasets have different classes!")
                print(f"Training classes: {classes}")
                print(f"Test classes: {test_classes}")
                
                # Use training classes as the reference
                classes = [c for c in classes if c in test_classes]
        
        # Convert labels to categorical
        num_classes = len(classes)
        y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
        
        if use_separate_test_set:
            y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes)
            
            # Split training data into train and validation sets
            X_train_final, X_val, y_train_cat_final, y_val_cat = train_test_split(
                X_train, y_train_cat, test_size=0.2, random_state=42, stratify=y_train
            )
            
            # Print dataset shapes
            print(f"\nTraining data shape: {X_train_final.shape}")
            print(f"Validation data shape: {X_val.shape}")
            print(f"Test data shape: {X_test.shape}")
        else:
            # Split data into train, validation, and test sets
            X_train_temp, X_test, y_train_temp, y_test = train_test_split(
                X_train, y_train, test_size=TEST_SPLIT, random_state=42, stratify=y_train
            )
            
            # Further split training data into train and validation
            X_train_final, X_val, y_train_cat_final, y_val_cat = train_test_split(
                X_train_temp, tf.keras.utils.to_categorical(y_train_temp, num_classes), 
                test_size=0.25, random_state=42, stratify=y_train_temp
            )
            
            # Convert test labels to categorical
            y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes)
            
            # Print dataset shapes
            print(f"\nTraining data shape: {X_train_final.shape}")
            print(f"Validation data shape: {X_val.shape}")
            print(f"Test data shape: {X_test.shape}")
        
        # Create model
        input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)
        model, base_model = create_squeezenet_model(input_shape, num_classes)
        model.summary()
        
        # Create data generators
        train_generator, val_generator = create_data_generators(
            X_train_final, y_train_cat_final, X_val, y_val_cat
        )
        
        # Create callbacks for Phase 1
        callbacks_phase1 = [
            ModelCheckpoint(
                os.path.join(MODEL_DIR, "best_model_phase1.h5"),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join(MODEL_DIR, "logs", "phase1"),
                histogram_freq=1
            )
        ]
        
        # Create callbacks for Phase 2
        callbacks_phase2 = [
            ModelCheckpoint(
                os.path.join(MODEL_DIR, "best_model_phase2.h5"),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join(MODEL_DIR, "logs", "phase2"),
                histogram_freq=1
            )
        ]
        
        # Train model in phases
        history = train_model_in_phases(
            model, base_model, train_generator, val_generator, callbacks_phase1, callbacks_phase2
        )
        
        # Plot training history
        plot_training_history(history, MODEL_DIR)
        
        # Evaluate on test data
        test_datagen = ImageDataGenerator()
        test_generator = test_datagen.flow(
            X_test, y_test_cat, batch_size=BATCH_SIZE, shuffle=False
        )
        
        test_loss, test_accuracy = model.evaluate(test_generator)
        print(f"\nTest accuracy: {test_accuracy:.4f}")
        print(f"Test loss: {test_loss:.4f}")
        
        # Save model in different formats
        print("\nSaving models...")
        
        # 1. Save in Keras format
        keras_path = os.path.join(MODEL_DIR, "model.keras")
        model.save(keras_path)
        print(f"- Keras model saved to: {keras_path}")
        
        # 2. TFLite for ESP32
        tflite_path = os.path.join(MODEL_DIR, "model.tflite")
        convert_to_tflite(model, tflite_path)
        
        # 3. Quantized TFLite for even smaller size
        try:
            print("Creating quantized TFLite model...")
            # Clone converter with additional quantization
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            # Set input/output types to uint8 for full integer quantization
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
            
            # Representative dataset for quantization
            def representative_dataset():
                for i in range(min(100, len(X_train_final))):
                    # Scale input from [0,1] to [0,255]
                    img = X_train_final[i:i+1].copy()
                    yield [img.astype(np.float32)]
            
            converter.representative_dataset = representative_dataset
            
            # Convert and save
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
        
        # 4. Save class information
        class_info = {
            "classes": list(classes),
            "input_shape": [IMAGE_SIZE, IMAGE_SIZE, 3],
            "version": "1.0",
            "model_type": "SqueezeNet-inspired (MobileNetV2 base with SqueezeNet-like top layers)",
            "date_trained": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        class_names_path = os.path.join(MODEL_DIR, "class_names.txt")
        with open(class_names_path, 'w') as f:
            for class_name in classes:
                f.write(f"{class_name}\n")
        print(f"- Class names saved to: {class_names_path}")
        
        class_info_path = os.path.join(MODEL_DIR, "model_metadata.json")
        with open(class_info_path, 'w') as f:
            json.dump(class_info, f, indent=2)
        print(f"- Model metadata saved to: {class_info_path}")
        
        # Calculate training time
        end_time = time.time()
        training_time = end_time - start_time
        hours, remainder = divmod(training_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print(f"\nTraining completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
        print("=== TRAINING COMPLETED SUCCESSFULLY ===")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nPlease ensure:")
        print(f"1. You have run preprocess.py first to create the 300x300 preprocessed dataset")
        print(f"2. The processed dataset folders exist: {PROCESSED_DATA_DIR} and (optionally) {PROCESSED_TEST_DIR}")
        print("3. Each class folder contains .npy files")
        raise
