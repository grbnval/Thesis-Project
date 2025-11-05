import cv2
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

def compare_preprocessing_methods(image_path):
    """
    Compare your original Colab preprocessing vs correct training preprocessing
    """
    
    print("=== PREPROCESSING COMPARISON ===")
    
    # Method 1: Your original Colab method (WRONG)
    print("1. Original Colab Method (causing issues):")
    try:
        img1 = Image.open(image_path).resize((96, 96))
        img1_array = np.array(img1) / 255.0
        print("   ✓ Uses PIL")
        print("   ✓ Simple resize")
        print("   ❌ NO CLAHE enhancement")
        print("   ❌ May have different color handling")
        print(f"   Shape: {img1_array.shape}, Range: [{img1_array.min():.3f}, {img1_array.max():.3f}]")
    except Exception as e:
        print(f"   Error: {e}")
        img1_array = None
    
    # Method 2: Correct training method (RIGHT)
    print("\n2. Correct Training Method:")
    try:
        img2 = cv2.imread(image_path)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        img2 = cv2.resize(img2, (96, 96))
        
        # Apply CLAHE enhancement
        lab = cv2.cvtColor(img2, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced = cv2.merge((cl,a,b))
        img2 = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        img2_array = img2.astype('float32') / 255.0
        print("   ✓ Uses OpenCV")
        print("   ✓ BGR→RGB conversion")
        print("   ✓ CLAHE enhancement")
        print("   ✓ Proper normalization")
        print(f"   Shape: {img2_array.shape}, Range: [{img2_array.min():.3f}, {img2_array.max():.3f}]")
    except Exception as e:
        print(f"   Error: {e}")
        img2_array = None
    
    # Visual comparison
    if img1_array is not None and img2_array is not None:
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(img1_array)
        plt.title("Original Colab Method\n(Wrong - No CLAHE)")
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(img2_array)
        plt.title("Correct Training Method\n(Right - With CLAHE)")
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        diff = np.abs(img2_array - img1_array)
        plt.imshow(diff)
        plt.title("Difference\n(Higher = More Different)")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig("preprocessing_comparison.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        # Calculate difference metrics
        mse = np.mean((img1_array - img2_array) ** 2)
        print(f"\n📊 Difference Metrics:")
        print(f"   Mean Squared Error: {mse:.6f}")
        print(f"   Max difference: {diff.max():.3f}")
        print(f"   Mean difference: {diff.mean():.3f}")
        
        if mse > 0.01:
            print("   🚨 SIGNIFICANT DIFFERENCE detected!")
            print("   This explains why your model fails on real images.")
        else:
            print("   ✅ Preprocessing methods are similar")
    
    return img1_array, img2_array

# Test with a sample image
if __name__ == "__main__":
    # Test with an image from your test dataset
    test_image = "test_dataset/healthy_leaf/H1.jpg"
    
    if os.path.exists(test_image):
        compare_preprocessing_methods(test_image)
    else:
        print("Please run this script from your project directory")
        print("Or modify the test_image path to point to a valid image")