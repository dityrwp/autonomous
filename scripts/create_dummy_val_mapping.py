#!/usr/bin/env python
import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm

"""
This script creates a dummy sample_mapping.json file for the validation split
by copying a subset of samples from the training split. This is a temporary
solution to allow training to proceed when validation data is not available.
"""

# Paths
train_dir = '/home/mevi/Documents/bev/test/train'
val_dir = '/home/mevi/Documents/bev/test/val'
train_mapping_path = os.path.join(train_dir, 'sample_mapping.json')

# Create val directory if it doesn't exist
os.makedirs(val_dir, exist_ok=True)

# Check if train mapping exists
if not os.path.exists(train_mapping_path):
    print(f"Error: Train mapping file not found at {train_mapping_path}")
    exit(1)

# Load train mapping
with open(train_mapping_path, 'r') as f:
    train_mapping = json.load(f)

# Create a smaller subset for validation (5% of train samples)
num_val_samples = max(10, len(train_mapping['samples']) // 20)  # At least 10 samples
val_samples = train_mapping['samples'][:num_val_samples]

# Create validation mapping
val_mapping = {
    'version': train_mapping['version'],
    'split': 'val',
    'grid_size': train_mapping['grid_size'],
    'resolution': train_mapping['resolution'],
    'samples': val_samples
}

# Save validation mapping
val_mapping_path = os.path.join(val_dir, 'sample_mapping.json')
with open(val_mapping_path, 'w') as f:
    json.dump(val_mapping, f, indent=2)

print(f"Created dummy validation mapping with {len(val_samples)} samples")
print(f"Saved to {val_mapping_path}")

# Copy all necessary files for the validation samples
print("Copying files for validation samples...")
for sample in tqdm(val_samples):
    scene_token = sample['scene_token']
    sample_token = sample['sample_token']
    
    # Create scene directory in val
    scene_dir = os.path.join(val_dir, scene_token)
    os.makedirs(scene_dir, exist_ok=True)
    
    # Copy JSON metadata file
    source_json = os.path.join(train_dir, scene_token, f"{sample_token}.json")
    target_json = os.path.join(val_dir, scene_token, f"{sample_token}.json")
    
    if os.path.exists(source_json):
        shutil.copy(source_json, target_json)
    else:
        print(f"Warning: Source metadata file not found: {source_json}")
    
    # Copy the corresponding NPY file
    source_npy = os.path.join(train_dir, scene_token, f"{sample_token}.npy")
    target_npy = os.path.join(val_dir, scene_token, f"{sample_token}.npy")
    
    if os.path.exists(source_npy):
        shutil.copy(source_npy, target_npy)
    else:
        print(f"Warning: Source NPY file not found: {source_npy}")

print("Done copying files for validation samples")

# Note: This creates a dummy validation set by copying a subset of training samples.
# This is only for testing the training pipeline and not for proper model evaluation. 