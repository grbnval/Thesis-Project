import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import time
import json
from datetime import datetime

# Constants
IMAGE_SIZE = 300
BATCH_SIZE = 32
EPOCHS = 50
INITIAL_LEARNING_RATE = 0.001
PROCESSED_TRAIN_DIR = "processed_train_300x300"
PROCESSED_TEST_DIR = "processed_test_300x300"
MODEL_DIR = "models/mobilenetv2_300x300_separate"

def load_data(data_dir):
    """Load preprocessed data from directory"""
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory not found: {data_dir}")
    
    # Get class names
    class_names = sorted([d for d in os.listdir(data_dir) 
                          if os.path.isdir(os.path.join(data_dir, d))])
    
    if not class_names:
        raise ValueError(f"No class directories found in {data_dir}")
    
    # Create label mapping
    label_map = {class_name: idx for idx, class_name in enumerate(class_names)}
    
    # Initialize arrays
    images = []
    labels = []
    
    # Load data for each class
    for class_name in class_names:
        print(f"Loading class: {class_name}")
        
        # Get numpy files
        class_dir = os.path.join(data_dir, class_name)
        files = [f for f in os.listdir(class_dir) if f.endswith('.npy')]
        
        if not files:
            print(f"Warning: No .npy files found in {class_name}/")
            continue
        
        print(f"Found {len(files)} images")
        
        # Load each file
        for file in files:
            file_path = os.path.join(class_dir, file)
            img = np.load(file_path)
            images.append(img)
            labels.append(label_map[class_name])
    
    # Convert to numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    
    # Convert labels to categorical
    labels_categorical = tf.keras.utils.to_categorical(labels, len(class_names))
    
    return images, labels, labels_categorical, class_names

def create_model(input_shape, num_classes):
    """Create MobileNetV2 model with custom top layers"""
    # Base model
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Create full model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=INITIAL_LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model, base_model

def train_model(model, base_model, train_images, train_labels, test_images, test_labels):
    """Train the model in two phases"""
    # Create model directory
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    # Create callback directory
    log_dir = os.path.join(MODEL_DIR, 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create phase 1 callbacks
    phase1_callbacks = [
        ModelCheckpoint(
            os.path.join(MODEL_DIR, 'best_model_phase1.h5'),
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
        )
    ]
    
    # Create phase 2 callbacks
    phase2_callbacks = [
        ModelCheckpoint(
            os.path.join(MODEL_DIR, 'best_model_phase2.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        ModelCheckpoint(
            os.path.join(MODEL_DIR, 'best_model.h5'),  # Keep the original best_model.h5 for compatibility
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
        )
    ]
    
    # Data augmentation for training
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
    test_datagen = ImageDataGenerator()
    
    # Create generators
    train_generator = train_datagen.flow(
        train_images, train_labels,
        batch_size=BATCH_SIZE
    )
    
    test_generator = test_datagen.flow(
        test_images, test_labels,
        batch_size=BATCH_SIZE
    )
    
    # Phase 1: Train only the top layers
    print("\n=== PHASE 1: TRAINING TOP LAYERS ===")
    phase1_history = model.fit(
        train_generator,
        epochs=10,
        validation_data=test_generator,
        callbacks=phase1_callbacks
    )
    
    # Save the best weights from Phase 1
    best_phase1_model_path = os.path.join(MODEL_DIR, 'best_model_phase1.h5')
    if os.path.exists(best_phase1_model_path):
        print("Loading best weights from Phase 1")
        model.load_weights(best_phase1_model_path)
    
    # Phase 2: Unfreeze some layers and fine-tune
    print("\n=== PHASE 2: FINE-TUNING ===")
    
    # Unfreeze the last 50 layers of the base model
    for layer in base_model.layers[-50:]:
        layer.trainable = True
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=INITIAL_LEARNING_RATE / 10),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Continue training
    phase2_history = model.fit(
        train_generator,
        epochs=EPOCHS,
        initial_epoch=10,
        validation_data=test_generator,
        callbacks=phase2_callbacks
    )
    
    # Combine histories
    combined_history = {}
    for key in phase1_history.history:
        combined_history[key] = phase1_history.history[key] + phase2_history.history[key]
    
    return combined_history

def plot_history(history):
    """Plot training history"""
    # Create plots directory
    plots_dir = os.path.join(MODEL_DIR, 'plots')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    # Plot accuracy
    plt.figure(figsize=(12, 4))
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
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'training_history.png'))
    plt.close()

def save_model_artifacts(model, class_names):
    """Save model in different formats with metadata"""
    # Create necessary directories
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    # 1. Save class names
    with open(os.path.join(MODEL_DIR, 'class_names.txt'), 'w') as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")
    
    # 2. Save model in Keras format
    keras_path = os.path.join(MODEL_DIR, 'model.keras')
    model.save(keras_path)
    print(f"Keras model saved to: {keras_path}")
    
    # 3. Convert to TFLite
    try:
        # Standard TFLite
        tflite_path = os.path.join(MODEL_DIR, 'model.tflite')
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"TFLite model saved to: {tflite_path}")
        print(f"TFLite model size: {os.path.getsize(tflite_path) / (1024 * 1024):.2f} MB")
        
        # Quantized TFLite
        tflite_quant_path = os.path.join(MODEL_DIR, 'model_quantized.tflite')
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_quant = converter.convert()
        
        with open(tflite_quant_path, 'wb') as f:
            f.write(tflite_quant)
        
        print(f"Quantized TFLite model saved to: {tflite_quant_path}")
        print(f"Quantized TFLite model size: {os.path.getsize(tflite_quant_path) / (1024 * 1024):.2f} MB")
    
    except Exception as e:
        print(f"Error creating TFLite models: {str(e)}")
    
    # 4. Save model metadata
    metadata = {
        "model_type": "MobileNetV2",
        "input_size": [IMAGE_SIZE, IMAGE_SIZE, 3],
        "classes": class_names,
        "date_trained": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "keras_version": tf.keras.__version__,
        "tensorflow_version": tf.__version__,
        "phases": {
            "phase1_best_model": "best_model_phase1.h5",
            "phase2_best_model": "best_model_phase2.h5",
            "combined_best_model": "best_model.h5"
        }
    }
    
    with open(os.path.join(MODEL_DIR, 'model_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"Model metadata saved to: {os.path.join(MODEL_DIR, 'model_metadata.json')}")

def main():
    start_time = time.time()
    
    try:
        print("=== TRAINING MOBILENETV2 300x300 WITH SEPARATE DATASETS ===")
        
        # Load training data
        print(f"\nLoading training data from: {PROCESSED_TRAIN_DIR}")
        train_images, train_labels_raw, train_labels, class_names = load_data(PROCESSED_TRAIN_DIR)
        print(f"Loaded {len(train_images)} training images from {len(class_names)} classes")
        
        # Load testing data
        print(f"\nLoading testing data from: {PROCESSED_TEST_DIR}")
        test_images, test_labels_raw, test_labels, test_class_names = load_data(PROCESSED_TEST_DIR)
        print(f"Loaded {len(test_images)} testing images from {len(test_class_names)} classes")
        
        # Check if classes match
        if set(class_names) != set(test_class_names):
            print("\nWarning: Training and testing datasets have different classes!")
            print(f"Training classes: {class_names}")
            print(f"Testing classes: {test_class_names}")
        
        # Create model
        input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)
        model, base_model = create_model(input_shape, len(class_names))
        
        # Print model summary
        model.summary()
        
        # Train model
        history = train_model(model, base_model, train_images, train_labels, test_images, test_labels)
        
        # Plot training history
        plot_history(history)
        
        # Compare phase 1 and phase 2 models
        print("\n=== COMPARING MODELS FROM BOTH PHASES ===")
        
        # Evaluate phase 1 model
        phase1_model_path = os.path.join(MODEL_DIR, 'best_model_phase1.h5')
        if os.path.exists(phase1_model_path):
            model.load_weights(phase1_model_path)
            phase1_loss, phase1_accuracy = model.evaluate(test_images, test_labels, batch_size=BATCH_SIZE)
            print(f"Phase 1 model - Test accuracy: {phase1_accuracy:.4f}, Test loss: {phase1_loss:.4f}")
        else:
            print("Phase 1 model not found!")
            phase1_accuracy = 0
            
        # Evaluate phase 2 model
        phase2_model_path = os.path.join(MODEL_DIR, 'best_model_phase2.h5')
        if os.path.exists(phase2_model_path):
            model.load_weights(phase2_model_path)
            phase2_loss, phase2_accuracy = model.evaluate(test_images, test_labels, batch_size=BATCH_SIZE)
            print(f"Phase 2 model - Test accuracy: {phase2_accuracy:.4f}, Test loss: {phase2_loss:.4f}")
        else:
            print("Phase 2 model not found!")
            phase2_accuracy = 0
            
        # Use the better model for final evaluation and saving
        if phase1_accuracy >= phase2_accuracy:
            print("\nPhase 1 model has better accuracy. Using it as the final model.")
            model.load_weights(phase1_model_path)
            best_phase = "Phase 1"
            # Copy phase 1 model to best_model.h5 for compatibility
            import shutil
            shutil.copy(phase1_model_path, os.path.join(MODEL_DIR, 'best_model.h5'))
        else:
            print("\nPhase 2 model has better accuracy. Using it as the final model.")
            model.load_weights(phase2_model_path)
            best_phase = "Phase 2"
            
        # Evaluate final model
        print("\nEvaluating final model...")
        test_loss, test_accuracy = model.evaluate(test_images, test_labels, batch_size=BATCH_SIZE)
        print(f"Final model ({best_phase}) - Test accuracy: {test_accuracy:.4f}, Test loss: {test_loss:.4f}")
        
        # Save model artifacts
        save_model_artifacts(model, class_names)
        
        # Calculate training time
        total_time = time.time() - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print(f"\nTotal training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        print("\n=== TRAINING COMPLETED ===")
        print(f"Model saved to: {MODEL_DIR}")
        print(f"Best model from: {best_phase}")
        print("You can evaluate the model using: python evaluate_mobilenetv2_separate.py")
    
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nPlease ensure:")
        print(f"1. You have run preprocess_mobilenetv2_separate.py first")
        print(f"2. Both {PROCESSED_TRAIN_DIR} and {PROCESSED_TEST_DIR} directories exist")
        print("3. Each class folder contains .npy files")
        raise

if __name__ == "__main__":
    main()
