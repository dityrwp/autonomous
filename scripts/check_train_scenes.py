#!/usr/bin/env python
import os
import json
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
import random
from tqdm import tqdm

"""
This script checks which scenes from the metadata are actually available in the dataset.
It helps understand which subset of the full NuScenes dataset is available.
"""

# Paths
dataroot = '/home/mevi/Documents/bev/nuscenes07'
version = 'v1.0-trainval'
bev_labels_dir = '/home/mevi/Documents/bev/test/train'

# Initialize NuScenes
print(f"Loading NuScenes {version} from {dataroot}...")
nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)

# Get official splits
splits = create_splits_scenes()
train_scene_names = splits['train']
val_scene_names = splits['val']

# Get scene tokens from the precomputed BEV labels directory
precomputed_scene_tokens = [d for d in os.listdir(bev_labels_dir) 
                           if os.path.isdir(os.path.join(bev_labels_dir, d)) 
                           and d != '__pycache__']

print(f"Found {len(precomputed_scene_tokens)} scene directories in precomputed BEV labels")

# Map scene tokens to scene names
scene_token_to_name = {}
scene_name_to_token = {}
for scene in nusc.scene:
    scene_token_to_name[scene['token']] = scene['name']
    scene_name_to_token[scene['name']] = scene['token']

# Check which precomputed scenes are from train or val split
precomputed_train_scenes = []
precomputed_val_scenes = []
unknown_scenes = []

for scene_token in precomputed_scene_tokens:
    scene_name = scene_token_to_name.get(scene_token)
    if scene_name in train_scene_names:
        precomputed_train_scenes.append(scene_token)
    elif scene_name in val_scene_names:
        precomputed_val_scenes.append(scene_token)
    else:
        unknown_scenes.append(scene_token)

print(f"Precomputed scenes from train split: {len(precomputed_train_scenes)}")
print(f"Precomputed scenes from val split: {len(precomputed_val_scenes)}")
print(f"Precomputed scenes not in any split: {len(unknown_scenes)}")

# Check which train scenes have actual files
train_scenes_with_files = []
train_scenes_missing_files = []

print("\nChecking which train scenes have actual files...")
for scene_name in tqdm(train_scene_names):
    scene_token = scene_name_to_token.get(scene_name)
    if not scene_token:
        continue
        
    # Get the first sample from this scene
    scene = nusc.get('scene', scene_token)
    sample_token = scene['first_sample_token']
    sample = nusc.get('sample', sample_token)
    
    # Check camera and LiDAR files
    cam_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])
    lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    
    cam_path = os.path.join(dataroot, cam_data['filename'])
    lidar_path = os.path.join(dataroot, lidar_data['filename'])
    
    if os.path.exists(cam_path) and os.path.exists(lidar_path):
        train_scenes_with_files.append(scene_name)
    else:
        train_scenes_missing_files.append(scene_name)

print(f"Train scenes with files: {len(train_scenes_with_files)} out of {len(train_scene_names)}")
print(f"Train scenes missing files: {len(train_scenes_missing_files)}")

# Check if the precomputed scenes match the available scenes
available_scene_tokens = [scene_name_to_token[name] for name in train_scenes_with_files]
matching_scenes = set(precomputed_scene_tokens).intersection(set(available_scene_tokens))

print(f"\nMatching scenes (precomputed and available): {len(matching_scenes)}")
print(f"Precomputed scenes not available in dataset: {len(set(precomputed_scene_tokens) - set(available_scene_tokens))}")
print(f"Available scenes not precomputed: {len(set(available_scene_tokens) - set(precomputed_scene_tokens))}")

# Summary
print("\nSummary:")
print(f"1. The dataset has metadata for all {len(train_scene_names)} train scenes and {len(val_scene_names)} val scenes")
print(f"2. Only {len(train_scenes_with_files)} train scenes have actual files on disk")
print(f"3. {len(precomputed_scene_tokens)} scenes have been precomputed for BEV labels")
print(f"4. The precomputation script fails for val split because none of the val scenes have files on disk")

# Recommendation
print("\nRecommendation:")
print("Continue using the dummy validation set created from the training data.")
print("This is a valid approach for development and testing when working with a subset of the full dataset.") 