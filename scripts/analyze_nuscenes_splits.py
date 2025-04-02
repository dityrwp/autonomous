#!/usr/bin/env python
import os
import json
from nuscenes.utils.splits import create_splits_scenes

# Get the official splits
splits = create_splits_scenes()

# Load the scene data from the dataset
with open('/home/mevi/Documents/bev/nuscenes07/v1.0-trainval/scene.json', 'r') as f:
    scenes = json.load(f)

# Extract scene names
scene_names = [scene['name'] for scene in scenes]

# Check which scenes belong to which split
train_scenes = [scene for scene in scene_names if scene in splits['train']]
val_scenes = [scene for scene in scene_names if scene in splits['val']]
test_scenes = [scene for scene in scene_names if scene in splits['test']]
other_scenes = [scene for scene in scene_names if scene not in splits['train'] and 
                                                  scene not in splits['val'] and 
                                                  scene not in splits['test']]

# Print statistics
print(f'Total scenes in dataset: {len(scene_names)}')
print(f'Train scenes in dataset: {len(train_scenes)} out of {len(splits["train"])} in full dataset')
print(f'Val scenes in dataset: {len(val_scenes)} out of {len(splits["val"])} in full dataset')
print(f'Test scenes in dataset: {len(test_scenes)} out of {len(splits["test"])} in full dataset')
print(f'Other scenes: {len(other_scenes)}')

# Print some example scene names
print("\nExample train scenes:")
for scene in train_scenes[:5]:
    print(f"  - {scene}")

print("\nExample val scenes:")
for scene in val_scenes[:5]:
    print(f"  - {scene}")

# Check if there are any scenes in the val split
if len(val_scenes) == 0:
    print("\nWARNING: No validation scenes found in the dataset!")
    print("This explains why the precomputation script fails for the val split.") 