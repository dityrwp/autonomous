import matplotlib.pyplot as plt
import numpy as np
import cv2
from nuscenes_dataset import NuScenesBEVDataset

def visualize_bev_label(bev_label, save_path=None):
    """
    Visualizes BEV segmentation labels.

    Args:
        bev_label (np.ndarray): BEV segmentation map, shape (2, 128, 128)
                                - Channel 0: Object Class Labels
                                - Channel 1: Attribute Labels (e.g., moving/stationary)
        save_path (str): Path to save the visualization (optional).

    Returns:
        None
    """
    # Extract class labels
    class_mask = bev_label[0]  # First channel (class segmentation)
    attr_mask = bev_label[1]   # Second channel (attribute segmentation, optional)

    # Define colors for different object classes
    class_colors = {
        0: (0, 0, 0),       # Background - Black
        1: (255, 0, 0),     # Car - Red
        2: (0, 255, 0),     # Truck - Green
        3: (0, 0, 255),     # Trailer - Blue
        4: (255, 255, 0),   # Bus - Yellow
        5: (255, 0, 255),   # Construction Vehicle - Magenta
        6: (0, 255, 255),   # Bicycle - Cyan
        7: (128, 128, 128), # Motorcycle - Gray
        8: (255, 165, 0),   # Pedestrian - Orange
        9: (0, 128, 255),   # Traffic Cone - Light Blue
        10: (128, 0, 128)   # Barrier - Purple
    }

    # Convert class mask to RGB image
    bev_visual = np.zeros((class_mask.shape[0], class_mask.shape[1], 3), dtype=np.uint8)
    for class_id, color in class_colors.items():
        bev_visual[class_mask == class_id] = color

    # Plot the BEV segmentation
    plt.figure(figsize=(6, 6))
    plt.imshow(bev_visual)
    plt.title("BEV Segmentation Map")
    plt.axis("off")

    # Save or show the visualization
    if save_path:
        plt.savefig(save_path)
    plt.show()
# Load one sample
dataset = NuScenesBEVDataset(dataroot="C:\\nuscenes_mini")
sample = dataset.get_data_info(0)

# Extract BEV segmentation map
bev_label = sample['bev_label']  # (2, 128, 128)

# Visualize the BEV segmentation
visualize_bev_label(bev_label, save_path="bev_debug.png")
