import cv2
import numpy as np
import os
import time
from PIL import Image
import matplotlib.pyplot as plt

def diagnose_image_issues(image_path):
    """
    Comprehensive diagnosis of image processing issues
    """
    print(f"🔍 DIAGNOSING IMAGE: {os.path.basename(image_path)}")
    print("=" * 60)
    
    if not os.path.exists(image_path):
        print(f"❌ ERROR: File does not exist!")
        return
    
    # 1. File size check
    file_size = os.path.getsize(image_path)
    file_size_mb = file_size / (1024 * 1024)
    print(f"📁 File size: {file_size:,} bytes ({file_size_mb:.2f} MB)")
    
    if file_size_mb > 10:
        print("⚠️  WARNING: Very large file! This will slow down processing.")
    elif file_size_mb > 5:
        print("⚠️  WARNING: Large file detected.")
    
    # 2. Try reading with OpenCV
    print(f"\n📖 Testing OpenCV imread...")
    start_time = time.time()
    try:
        img_cv = cv2.imread(image_path)
        cv_time = time.time() - start_time
        
        if img_cv is None:
            print("❌ ERROR: OpenCV cannot read this image!")
            print("   Possible causes:")
            print("   - Corrupted file")
            print("   - Unsupported format")
            print("   - File path contains special characters")
            return
        else:
            height, width, channels = img_cv.shape
            print(f"✅ OpenCV read successful ({cv_time:.3f}s)")
            print(f"   Dimensions: {width}x{height}x{channels}")
            print(f"   Data type: {img_cv.dtype}")
            
            # Check for unusual dimensions
            total_pixels = width * height
            if total_pixels > 10_000_000:  # > 10MP
                print(f"⚠️  WARNING: Very high resolution ({total_pixels:,} pixels)")
                print("   This will significantly slow down processing!")
            elif total_pixels > 5_000_000:  # > 5MP
                print(f"⚠️  WARNING: High resolution ({total_pixels:,} pixels)")
    
    except Exception as e:
        print(f"❌ OpenCV ERROR: {e}")
        return
    
    # 3. Try reading with PIL
    print(f"\n🖼️  Testing PIL/Pillow...")
    start_time = time.time()
    try:
        img_pil = Image.open(image_path)
        pil_time = time.time() - start_time
        print(f"✅ PIL read successful ({pil_time:.3f}s)")
        print(f"   Format: {img_pil.format}")
        print(f"   Mode: {img_pil.mode}")
        print(f"   Size: {img_pil.size}")
        
        # Check for unusual formats
        if img_pil.mode not in ['RGB', 'L', 'RGBA']:
            print(f"⚠️  WARNING: Unusual color mode '{img_pil.mode}'")
        
        if img_pil.format not in ['JPEG', 'PNG', 'BMP']:
            print(f"⚠️  WARNING: Unusual format '{img_pil.format}'")
            
    except Exception as e:
        print(f"❌ PIL ERROR: {e}")
    
    # 4. Test preprocessing steps individually
    print(f"\n⚙️  Testing preprocessing steps...")
    
    try:
        # Step 1: Color conversion
        start_time = time.time()
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        conv_time = time.time() - start_time
        print(f"   BGR→RGB conversion: {conv_time:.3f}s")
        
        # Step 2: Resize
        start_time = time.time()
        img_resized = cv2.resize(img_rgb, (96, 96), interpolation=cv2.INTER_AREA)
        resize_time = time.time() - start_time
        print(f"   Resize to 96x96: {resize_time:.3f}s")
        
        if resize_time > 0.1:
            print("   ⚠️  Resize is unusually slow!")
        
        # Step 3: CLAHE preprocessing
        start_time = time.time()
        lab = cv2.cvtColor(img_resized, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced = cv2.merge((cl, a, b))
        img_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        clahe_time = time.time() - start_time
        print(f"   CLAHE enhancement: {clahe_time:.3f}s")
        
        # Step 4: Normalization
        start_time = time.time()
        img_normalized = img_enhanced.astype('float32') / 255.0
        norm_time = time.time() - start_time
        print(f"   Normalization: {norm_time:.3f}s")
        
        total_preprocess_time = conv_time + resize_time + clahe_time + norm_time
        print(f"   📊 Total preprocessing: {total_preprocess_time:.3f}s")
        
        if total_preprocess_time > 1.0:
            print("   🚨 PREPROCESSING IS SLOW!")
            
    except Exception as e:
        print(f"❌ Preprocessing ERROR: {e}")
        return
    
    # 5. Check for file corruption
    print(f"\n🔍 Checking for corruption...")
    try:
        # Try to access all pixel data
        pixel_sum = np.sum(img_cv)
        pixel_mean = np.mean(img_cv)
        print(f"   Pixel statistics: sum={pixel_sum:.0f}, mean={pixel_mean:.2f}")
        
        # Check for unusual values
        if pixel_mean < 10 or pixel_mean > 245:
            print("   ⚠️  Unusual brightness detected")
        
        # Check for all black or all white
        if np.all(img_cv == 0):
            print("   ❌ Image is completely black!")
        elif np.all(img_cv == 255):
            print("   ❌ Image is completely white!")
            
    except Exception as e:
        print(f"   ❌ Corruption check failed: {e}")
    
    # 6. Visual inspection
    print(f"\n👁️  Creating visual inspection...")
    try:
        plt.figure(figsize=(15, 5))
        
        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        plt.title(f"Original\n{width}x{height}")
        plt.axis('off')
        
        # Resized image
        plt.subplot(1, 3, 2)
        plt.imshow(img_resized)
        plt.title("Resized 96x96")
        plt.axis('off')
        
        # Enhanced image
        plt.subplot(1, 3, 3)
        plt.imshow(img_enhanced)
        plt.title("CLAHE Enhanced")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig("image_diagnosis.png", dpi=150, bbox_inches='tight')
        plt.show()
        print("   ✅ Visual inspection saved as 'image_diagnosis.png'")
        
    except Exception as e:
        print(f"   ❌ Visual inspection failed: {e}")
    
    # 7. Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    if file_size_mb > 5:
        print("   🔧 Consider resizing image before processing")
        print("   🔧 Use image compression to reduce file size")
    
    if total_preprocess_time > 0.5:
        print("   🔧 Use ultra-fast preprocessing mode (skip CLAHE)")
        print("   🔧 Resize image to smaller dimensions first")
    
    if 'Copy of Copy of' in image_path:
        print("   🔧 This appears to be a multiple-copy file")
        print("   🔧 Check if original file has better performance")
    
    print(f"\n✅ Diagnosis complete!")

def create_optimized_version(image_path):
    """Create an optimized version of the problematic image"""
    try:
        print(f"\n🛠️ Creating optimized version...")
        
        # Read original
        img = cv2.imread(image_path)
        if img is None:
            print("❌ Cannot read image for optimization")
            return
        
        # Resize to reasonable size first
        height, width = img.shape[:2]
        if max(height, width) > 1024:
            if height > width:
                new_height = 1024
                new_width = int(width * (1024 / height))
            else:
                new_width = 1024
                new_height = int(height * (1024 / width))
            
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
            print(f"   Resized from {width}x{height} to {new_width}x{new_height}")
        
        # Save optimized version
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        optimized_path = f"{base_name}_optimized.jpg"
        
        # Use JPEG compression
        cv2.imwrite(optimized_path, img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        optimized_size = os.path.getsize(optimized_path) / (1024 * 1024)
        original_size = os.path.getsize(image_path) / (1024 * 1024)
        
        print(f"   ✅ Optimized version saved: {optimized_path}")
        print(f"   📉 Size reduction: {original_size:.2f}MB → {optimized_size:.2f}MB")
        
        return optimized_path
        
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        return None

if __name__ == "__main__":
    # Test with the problematic image
    problem_image = r"C:\Users\Azief\Documents\GitHub\Thesis-Project\raw_dataset\healthy_leaf\Copy of Copy of 20251005-005612(5).jpg"
    
    diagnose_image_issues(problem_image)
    create_optimized_version(problem_image)