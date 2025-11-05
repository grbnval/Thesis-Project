from PIL import Image
import matplotlib.pyplot as plt
import os

# Paths to the images
img_paths = [
    'models/SqueezeNet-inspired_layer_param_counts.png',
    'models/MobileNetV2-baseline_layer_param_counts.png',
    'models/MobileNetV2-Eff-style_layer_param_counts.png'
]
labels = ['SqueezeNet-inspired', 'MobileNetV2-baseline', 'MobileNetV2-Eff-style']

images = [Image.open(p) for p in img_paths]

# Create a new figure for merged display
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, img, label in zip(axes, images, labels):
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(label, fontsize=14)
plt.tight_layout()
plt.savefig('models/merged_layer_param_counts.png', dpi=200)
plt.close()
print('Merged layer parameter count images saved to models/merged_layer_param_counts.png')
