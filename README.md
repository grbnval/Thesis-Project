The ESP32-CAM-based board is used for capturing tomato leaves and combined with a lightweight CNN model tuned for cloud based processing to identify diseases in tomato plants, namely Healthy Leaf, Early Blight, Late Blight, and Septoria Leaf Spot. The model is built using MobileNetV2 architecture (α=0.25), EfficientNetLite, and SqueezeNet which will be evaluated later on to determine the best performing model, and trained from scratch on a dataset of 500 images per class. Training the model from scratch rather than using transfer learning was a deliberate decision to optimize for smaller model size while maintaining acceptable accuracy with our substantial domain-specific dataset. The training process employs the Adam optimizer with an initial learning rate of 0.001, followed by a refinement phase using a reduced learning rate (1e-4) and early stopping to prevent overfitting. This approach allows the model to develop specialized feature extractors specifically tailored to tomato disease patterns. The network architecture includes a lightweight classification head consisting of GlobalAveragePooling2D, a 16-neuron dense layer with ReLU activation, and dropout for regularization. The final model is optimized through float32 and converted to TensorFlow Lite format, reducing its size for faster inference time for the cloud server which is Google Functions.

## Model Training Updates

### MobileNetV2 96x96 Training

Multiple models have been trained with different configurations for evaluation:

1. **MobileNetV2 (96x96)** - Base model with early stopping
   - Input size: 96x96
   - Two-phase training: initial phase for top layers, fine-tuning phase for convolutional layers
   - Early stopping to prevent overfitting

2. **MobileNetV2 (96x96) Full Epochs** - Training for full 100 epochs
   - Input size: 96x96
   - Two-phase training: 20 epochs for initial phase, 80 epochs for fine-tuning
   - No early stopping - completes all epochs for better feature learning

3. **MobileNetV2 EfficientNet-Style (96x96)** - Modified architecture with 100 epochs
   - Input size: 96x96
   - Architecture: MobileNetV2 base with EfficientNet-style top layers
   - Deeper classification head with 512 and 256 neuron dense layers with dropout
   - Full 100 epochs training for maximum feature learning
   - Two-phase approach for better feature adaptation

All models utilize:
- CLAHE preprocessing for contrast enhancement
- Data augmentation (rotation, shifts, shear, zoom, flips)
- Quantization for smaller model size
- Conversion to TFLite for deployment
