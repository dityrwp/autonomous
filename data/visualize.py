import matplotlib.pyplot as plt
import numpy as np
import cv2
from nuscenes.nuscenes import NuScenes
import os
from nuscenes_dataset import NuScenesBEVDataset

def visualize_bev_and_image(dataset, sample_idx, save_path=None):
    """
    Visualizes rear-cropped BEV segmentation labels alongside the front camera image.
    The BEV shows a 50m x 50m area behind the vehicle, with the ego vehicle at the bottom.
    """
    sample = dataset.samples[sample_idx]
    
    # Get front camera image
    cam_token = sample['data'][dataset.front_cam]
    cam_data = dataset.nusc.get('sample_data', cam_token)
    img_path = os.path.join(dataset.dataroot, cam_data['filename'])
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Get BEV segmentation
    data_info = dataset.get_data_info(sample_idx)
    bev_label = data_info['bev_label']
    
    # Extract labels
    semantic_mask = bev_label[0]  # Semantic segmentation
    instance_mask = bev_label[1]  # Instance segmentation
    map_mask = bev_label[2]      # Map layers

    # Define colors with better visibility
    semantic_colors = {
        0: (40, 40, 40),       # Background - Dark Gray
        1: (255, 100, 100),    # Car - Brighter Red
        2: (107, 142, 35),     # Truck - Olive Green
        3: (100, 149, 237),    # Trailer - Cornflower Blue
        4: (255, 191, 0),      # Bus - Amber
        5: (222, 111, 161),    # Construction Vehicle
        6: (95, 158, 160),     # Bicycle
        7: (189, 183, 107),    # Motorcycle
        8: (255, 127, 80),     # Pedestrian - Coral
        9: (86, 180, 233),     # Traffic Cone
        10: (153, 108, 166),   # Barrier
        11: (176, 224, 230)    # Drivable Area - Light Blue
    }

    # Define map layer colors with better visibility
    map_colors = {
        1: (176, 224, 230),    # Drivable Area - Light Blue
        2: (152, 251, 152),    # Road Segment - Pale Green
        3: (255, 182, 193),    # Road Block - Light Pink
        4: (255, 215, 0),      # Lane - Gold
        5: (255, 0, 0),        # Road Divider - Red
        6: (255, 255, 0)       # Lane Divider - Yellow
    }

    # Define class names mapping
    class_names = {
        0: 'background',
        1: 'car (ego)',
        2: 'truck', 
        3: 'trailer',
        4: 'bus',
        5: 'construction_vehicle',
        6: 'bicycle',
        7: 'motorcycle',
        8: 'pedestrian',
        9: 'traffic_cone',
        10: 'barrier',
        11: 'drivable_area'
    }

    # Define map layer names
    map_names = {
        1: 'drivable_area',
        2: 'road_segment',
        3: 'road_block',
        4: 'lane',
        5: 'road_divider',
        6: 'lane_divider'
    }

    # Create figure with subplots
    fig = plt.figure(figsize=(24, 16))
    
    # Plot front camera image
    ax1 = plt.subplot(221)
    ax1.imshow(img)
    ax1.set_title("Front Camera View", fontsize=14)
    ax1.axis("off")

    # Plot semantic segmentation
    ax2 = plt.subplot(222)
    semantic_visual = np.zeros((semantic_mask.shape[0], semantic_mask.shape[1], 3), dtype=np.uint8)
    for class_id, color in semantic_colors.items():
        semantic_visual[semantic_mask == class_id] = color
    ax2.imshow(semantic_visual)
    ax2.set_title("Semantic Segmentation", fontsize=14)
    ax2.axis("off")

    # Plot instance segmentation with random colors
    ax3 = plt.subplot(223)
    instance_visual = np.zeros((instance_mask.shape[0], instance_mask.shape[1], 3), dtype=np.uint8)
    unique_instances = np.unique(instance_mask)[1:]  # Skip background
    for instance_id in unique_instances:
        color = np.random.randint(0, 255, 3)
        instance_visual[instance_mask == instance_id] = color
    ax3.imshow(instance_visual)
    ax3.set_title("Instance Segmentation", fontsize=14)
    ax3.axis("off")

    # Plot map layers with alpha blending
    ax4 = plt.subplot(224)
    map_visual = np.zeros((map_mask.shape[0], map_mask.shape[1], 4), dtype=np.float32)  # RGBA
    
    # Add layers in order (bottom to top)
    layer_order = [1, 2, 4, 5, 6]  # drivable_area, road_segment, lane, road_divider, lane_divider
    for layer_id in layer_order:
        mask = map_mask == layer_id
        if mask.any():  # Only process if layer exists
            color = np.array(map_colors[layer_id]) / 255.0
            alpha = 0.7 if layer_id <= 4 else 0.9  # Higher alpha for dividers
            
            # Create RGBA color
            rgba = np.array([*color, alpha])
            
            # Update pixels where mask is True
            map_visual[mask] = rgba

    # Set background to black
    map_visual[np.all(map_visual == 0, axis=-1)] = [0, 0, 0, 1]

    ax4.imshow(map_visual)
    ax4.set_title("Map Layers", fontsize=14)
    ax4.axis("off")

    # Add grid and distance markers
    for ax in [ax2, ax3, ax4]:
        ax.grid(True, linestyle='--', alpha=0.3)
        distances = [10, 20, 30, 40]  # meters
        pixels_per_meter = semantic_visual.shape[0] / 50  # 50m total range
        for dist in distances:
            y_pos = semantic_visual.shape[0] - dist * pixels_per_meter
            ax.axhline(y=y_pos, color='white', linestyle='--', alpha=0.3)
            ax.text(5, y_pos-5, f'{dist}m', color='white', alpha=0.7, fontsize=10)

    # Add legends
    # Semantic legend
    semantic_legend = [
        plt.Rectangle((0,0), 1, 1, 
                     facecolor=np.array(color)/255,
                     label=class_names[class_id])
        for class_id, color in semantic_colors.items()
        if class_id in np.unique(semantic_mask)
    ]
    ax2.legend(handles=semantic_legend, 
              loc='center left', 
              bbox_to_anchor=(1, 0.5),
              title='Classes',
              fontsize=8,
              title_fontsize=10)

    # Map layers legend
    map_legend = [
        plt.Rectangle((0,0), 1, 1,
                     facecolor=np.array(map_colors[layer_id])/255,
                     alpha=0.7 if layer_id <= 4 else 0.9,
                     label=map_names[layer_id])
        for layer_id in layer_order
        if layer_id in np.unique(map_mask)
    ]
    ax4.legend(handles=map_legend,
              loc='center left',
              bbox_to_anchor=(1, 0.5),
              title='Map Layers',
              fontsize=8,
              title_fontsize=10)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300, facecolor='black')
        print(f"Saved visualization to {save_path}")
    plt.show()

def visualize_point_cloud(points, title="LiDAR Point Cloud"):
    """
    Visualize LiDAR point cloud from top-down view.
    
    Args:
        points (np.ndarray): Point cloud array of shape (N, 4) containing x, y, z, intensity
        title (str): Plot title
    """
    plt.figure(figsize=(8, 8))
    
    # Create top-down view (x-y plane)
    plt.scatter(points[:, 0], points[:, 1], 
                
               c=points[:, 2],  # Color by height (z)
               cmap='viridis',
               s=1)  # Point size
    
    plt.title(title)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.axis('equal')
    plt.colorbar(label='Height (m)')
    plt.show()

def test_dataset_preprocessing():
    """Test the NuScenes dataset preprocessing pipeline."""
    try:
        # Initialize dataset
        dataset = NuScenesBEVDataset(
            dataroot="C:\\nuscenes_mini",
            split='train',
            sample_ratio=1.0
        )
        
        print(f"Dataset initialized with {len(dataset.samples)} samples")
        
        # Test multiple samples
        for sample_idx in range(min(5, len(dataset.samples))):
            print(f"\nProcessing sample {sample_idx}...")
            
            # Visualize BEV and front camera image
            visualize_bev_and_image(
                dataset, 
                sample_idx, 
                save_path=f"bev_visualization_{sample_idx}.png"
            )
            
            # Optional: wait for user input before showing next sample
            input("Press Enter to continue to next sample...")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dataset_preprocessing()
