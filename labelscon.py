import os
import json

# Replace this with the path to your dataset's train folder
dataset_dir = r'C:\Users\verma\Downloads\Plant disease\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\train'

# Get sorted list of folder names (class labels)
class_names = sorted([folder for folder in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, folder))])

# Create a dictionary with index as key and class name as value
class_dict = {str(i): name for i, name in enumerate(class_names)}

# Save to labels.json
with open("class_labels.json", "w") as f:
    json.dump(class_dict, f, indent=2)

print("✅ class_labels.json file created successfully.")
