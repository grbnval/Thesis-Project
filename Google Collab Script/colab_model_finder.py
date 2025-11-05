# GOOGLE COLAB MODEL FINDER - Find your model files
# Copy and paste this into a new Colab cell to find your model

import os
from google.colab import drive

print("🔍 FINDING YOUR MODEL FILES")
print("=" * 60)

# Make sure Drive is mounted
drive.mount('/content/drive')

def search_for_models(directory, max_depth=3, current_depth=0):
    """Search for model files recursively"""
    model_files = []
    
    if current_depth >= max_depth:
        return model_files
        
    try:
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            
            if os.path.isfile(item_path):
                # Check if it's a model file
                if item.endswith(('.h5', '.keras', '.pb', '.tflite')):
                    file_size = os.path.getsize(item_path) / (1024 * 1024)  # Size in MB
                    model_files.append({
                        'path': item_path,
                        'name': item,
                        'size_mb': file_size,
                        'type': item.split('.')[-1]
                    })
            elif os.path.isdir(item_path) and not item.startswith('.'):
                # Recursively search subdirectories
                model_files.extend(search_for_models(item_path, max_depth, current_depth + 1))
                
    except PermissionError:
        pass  # Skip directories we can't access
    except Exception as e:
        print(f"Error searching {directory}: {e}")
        
    return model_files

# Search for model files
print("🔍 Searching for model files in Google Drive...")
drive_root = '/content/drive/MyDrive'

model_files = search_for_models(drive_root)

if model_files:
    print(f"\n✅ Found {len(model_files)} model file(s):")
    print("=" * 60)
    
    for i, model in enumerate(model_files, 1):
        print(f"{i}. 📄 {model['name']}")
        print(f"   📁 Path: {model['path']}")
        print(f"   📊 Size: {model['size_mb']:.1f} MB")
        print(f"   🏷️  Type: {model['type']}")
        print()
        
    # Find the most likely candidate
    best_candidates = []
    for model in model_files:
        if ('mobilenet' in model['name'].lower() or 
            'best_model' in model['name'].lower() or
            '96x96' in model['name'].lower()):
            best_candidates.append(model)
    
    if best_candidates:
        print("🎯 BEST CANDIDATES for your project:")
        print("=" * 60)
        for i, model in enumerate(best_candidates, 1):
            print(f"{i}. {model['name']} ({model['size_mb']:.1f} MB)")
            print(f"   Path: {model['path']}")
        
        print(f"\n💡 RECOMMENDED: Use this path in your code:")
        print(f"model_path = '{best_candidates[0]['path']}'")
        
else:
    print("❌ No model files found!")
    print("\n📁 Let's check what's in your Drive root:")
    
    try:
        items = os.listdir(drive_root)
        print(f"\n📂 Contents of {drive_root}:")
        for item in sorted(items)[:20]:  # Show first 20 items
            item_path = os.path.join(drive_root, item)
            if os.path.isdir(item_path):
                print(f"   📁 {item}/")
            else:
                print(f"   📄 {item}")
        
        if len(items) > 20:
            print(f"   ... and {len(items) - 20} more items")
            
    except Exception as e:
        print(f"Error listing Drive contents: {e}")

# Also search for class_info.json
print(f"\n🔍 Searching for class_info.json...")
class_info_files = []

for root, dirs, files in os.walk(drive_root):
    for file in files:
        if file == 'class_info.json':
            file_path = os.path.join(root, file)
            class_info_files.append(file_path)

if class_info_files:
    print(f"✅ Found class_info.json files:")
    for path in class_info_files:
        print(f"   📄 {path}")
else:
    print("❌ No class_info.json found")

print(f"\n🚀 NEXT STEPS:")
print("=" * 60)
print("1. Copy the correct model path from above")
print("2. Update the model_path variable in your diagnostic script")
print("3. Run the diagnostic again with the correct path")

# Generate updated diagnostic code with correct path
if model_files:
    best_model = best_candidates[0] if best_candidates else model_files[0]
    
    print(f"\n📝 UPDATED CODE FOR YOUR DIAGNOSTIC:")
    print("=" * 60)
    print(f"""
# Update these paths in your diagnostic script:
model_path = '{best_model['path']}'

# If you found class_info.json, use its path too:
""")
    
    if class_info_files:
        print(f"class_info_path = '{class_info_files[0]}'")
    else:
        print("# No class_info.json found, will use fallback classes")
        print("class_names = ['early_blight_leaf', 'healthy_leaf', 'late_blight_leaf', 'septoria_leaf', 'unknown']")