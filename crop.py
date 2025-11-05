import cv2
import os

def split_image_into_9(image_path, output_folder):
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Cannot read image: {image_path}")
        return

    h, w, _ = img.shape
    print(f"📸 Original image size: {w}x{h}")

    # Create output folder if not exists
    os.makedirs(output_folder, exist_ok=True)

    # Calculate size of each crop
    h_crop = h // 3
    w_crop = w // 3

    # Split and save
    count = 1
    for i in range(3):
        for j in range(3):
            y_start = i * h_crop
            y_end = (i + 1) * h_crop if i < 2 else h  # handle rounding
            x_start = j * w_crop
            x_end = (j + 1) * w_crop if j < 2 else w

            cropped = img[y_start:y_end, x_start:x_end]
            filename = os.path.join(output_folder, f"part_{count}.jpg")
            cv2.imwrite(filename, cropped)
            print(f"✅ Saved {filename}")
            count += 1

    print("🎉 Done splitting image into 9 parts!")

# Example usage
image_path = "C:/Users/Azief/Desktop/1.png"  # change this to your image
output_folder = "C:/Users/Azief/Desktop"   # folder where parts will be saved
split_image_into_9(image_path, output_folder)
