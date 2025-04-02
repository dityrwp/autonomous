#!/usr/bin/env python
import os
import json
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
import random
from tqdm import tqdm

"""
This script checks if the validation samples from the metadata actually have
corresponding files on disk. It helps diagnose why the precomputation script
fails for the validation split despite having validation metadata.
"""

# Paths
dataroot = '/home/mevi/Documents/bev/nuscenes07'
version = 'v1.0-trainval'

# Initialize NuScenes
print(f"Loading NuScenes {version} from {dataroot}...")
nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)

# Get official splits
splits = create_splits_scenes()
val_scene_names = splits['val']

# Find validation scenes in the dataset
val_scene_tokens = []
for scene in nusc.scene:
    if scene['name'] in val_scene_names:
        val_scene_tokens.append(scene['token'])

print(f"Found {len(val_scene_tokens)} validation scenes in metadata")

# Check a random validation scene
if val_scene_tokens:
    random_scene_token = random.choice(val_scene_tokens)
    scene = nusc.get('scene', random_scene_token)
    print(f"\nChecking random validation scene: {scene['name']} (token: {random_scene_token})")
    
    # Get the first sample from this scene
    sample_token = scene['first_sample_token']
    sample = nusc.get('sample', sample_token)
    
    # Check camera and LiDAR files
    cam_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])
    lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    
    cam_path = os.path.join(dataroot, cam_data['filename'])
    lidar_path = os.path.join(dataroot, lidar_data['filename'])
    
    print(f"Camera path: {cam_path}")
    print(f"  Exists: {os.path.exists(cam_path)}")
    
    print(f"LiDAR path: {lidar_path}")
    print(f"  Exists: {os.path.exists(lidar_path)}")
    
    # Check more samples from validation scenes
    print("\nChecking 10 random validation samples...")
    
    # Collect all validation samples
    val_samples = []
    for scene_token in val_scene_tokens:
        scene = nusc.get('scene', scene_token)
        sample_token = scene['first_sample_token']
        
        while sample_token:
            val_samples.append(sample_token)
            sample = nusc.get('sample', sample_token)
            sample_token = sample['next']
    
    print(f"Total validation samples in metadata: {len(val_samples)}")
    
    # Check random samples
    random_samples = random.sample(val_samples, min(10, len(val_samples)))
    
    missing_files = 0
    for sample_token in random_samples:
        sample = nusc.get('sample', sample_token)
        
        # Check camera and LiDAR files
        cam_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])
        lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        
        cam_path = os.path.join(dataroot, cam_data['filename'])
        lidar_path = os.path.join(dataroot, lidar_data['filename'])
        
        if not os.path.exists(cam_path) or not os.path.exists(lidar_path):
            missing_files += 1
            print(f"Sample {sample_token}: Missing files")
            if not os.path.exists(cam_path):
                print(f"  Missing camera: {cam_path}")
            if not os.path.exists(lidar_path):
                print(f"  Missing LiDAR: {lidar_path}")
    
    if missing_files == 0:
        print("All checked samples have their files available!")
    else:
        print(f"{missing_files} out of 10 checked samples have missing files")
    
    # Check all validation samples
    print("\nChecking all validation samples for missing files...")
    
    missing_cam_files = 0
    missing_lidar_files = 0
    total_checked = 0
    
    for sample_token in tqdm(val_samples):
        total_checked += 1
        sample = nusc.get('sample', sample_token)
        
        # Check camera and LiDAR files
        cam_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])
        lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        
        cam_path = os.path.join(dataroot, cam_data['filename'])
        lidar_path = os.path.join(dataroot, lidar_data['filename'])
        
        if not os.path.exists(cam_path):
            missing_cam_files += 1
        
        if not os.path.exists(lidar_path):
            missing_lidar_files += 1
    
    print(f"Checked {total_checked} validation samples")
    print(f"Missing camera files: {missing_cam_files} ({missing_cam_files/total_checked*100:.2f}%)")
    print(f"Missing LiDAR files: {missing_lidar_files} ({missing_lidar_files/total_checked*100:.2f}%)")
    
    if missing_cam_files > 0 or missing_lidar_files > 0:
        print("\nThis explains why the precomputation script fails for the validation split:")
        print("The metadata includes validation scenes, but the actual files are not available on disk.")
        print("This is common when working with a subset of the full NuScenes dataset.")
else:
    print("No validation scenes found in metadata!") 