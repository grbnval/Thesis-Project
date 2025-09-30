import os
from PIL import Image

def augment_image(image_path, save_dir, base_name):
    img = Image.open(image_path)
    # Mirror
    img_mirror = img.transpose(Image.FLIP_LEFT_RIGHT)
    img_mirror.save(os.path.join(save_dir, f"{base_name}_mirror.jpg"))
    # Rotate right (90 degrees)
    img_rot_right = img.transpose(Image.ROTATE_270)
    img_rot_right.save(os.path.join(save_dir, f"{base_name}_rotR.jpg"))
    # Rotate left (270 degrees)
    img_rot_left = img.transpose(Image.ROTATE_90)
    img_rot_left.save(os.path.join(save_dir, f"{base_name}_rotL.jpg"))
    # Upside down (180 degrees)
    img_upside = img.transpose(Image.ROTATE_180)
    img_upside.save(os.path.join(save_dir, f"{base_name}_upside.jpg"))

def augment_dataset(raw_dataset_dir):
    for class_name in os.listdir(raw_dataset_dir):
        class_dir = os.path.join(raw_dataset_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                fpath = os.path.join(class_dir, fname)
                base_name, _ = os.path.splitext(fname)
                augment_image(fpath, class_dir, base_name)

if __name__ == "__main__":
    augment_dataset("raw_dataset")
    print("Augmentation complete.")
