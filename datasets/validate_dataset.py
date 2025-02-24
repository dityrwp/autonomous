import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import logging
from tqdm import tqdm
from pyquaternion import Quaternion
from shapely.geometry import Polygon
from descartes import PolygonPatch
import random

from nuscenes_data import NuScenesFilteredDataset, NuScenesBEVLabelDataset

class DatasetValidator:
    """Validates NuScenes BEV dataset implementation for trainval dataset"""
    
    def __init__(self, dataroot):
        self.dataroot = dataroot
        
        # Verify trainval dataset exists
        self._verify_dataset_files()
        
        # Initialize datasets
        self.filtered_dataset = NuScenesFilteredDataset(
            dataroot=dataroot,
            version="v1.0-trainval",  # Enforce trainval version
            split="train"
        )
        
        self.bev_dataset = NuScenesBEVLabelDataset(
            filtered_dataset=self.filtered_dataset,
            grid_size=256,
            resolution=0.2
        )
        
        # Color map for visualization
        def hex_to_rgb(hex_color):
            """Convert hex color to RGB values (0-255)."""
            hex_color = hex_color.lstrip('#')
            return [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]
        
        # Using the provided color scheme
        self.class_colors = {
            'background': [50, 50, 50],       # Dark gray background
            'drivable_area': hex_to_rgb('#a6cee3'),
            'road_divider': hex_to_rgb('#cab2d6'),
            'lane_divider': hex_to_rgb('#6a3d9a'),
            'walkway': hex_to_rgb('#e31a1c'),
            'ped_crossing': hex_to_rgb('#fb9a99')
        }
        self.colors = list(self.class_colors.values())
        self.class_names = list(self.class_colors.keys())
        self.cmap = ListedColormap(np.array(self.colors) / 255.0)
        
        logging.info("Dataset validator initialized for trainval dataset")
    
    def _verify_dataset_files(self):
        """Verify that all required trainval dataset files exist."""
        required_files = [
            'maps/expansion/singapore-onenorth.json',
            'maps/expansion/singapore-hollandvillage.json',
            'maps/expansion/singapore-queenstown.json',
            'maps/expansion/boston-seaport.json',
            'v1.0-trainval/scene.json',
            'v1.0-trainval/sample.json',
            'v1.0-trainval/sample_data.json',
            'v1.0-trainval/calibrated_sensor.json',
            'v1.0-trainval/ego_pose.json',
            'v1.0-trainval/log.json',
            'v1.0-trainval/map.json'
        ]
        
        missing_files = []
        for file_path in required_files:
            full_path = os.path.join(self.dataroot, file_path)
            if not os.path.exists(full_path):
                missing_files.append(file_path)
        
        if missing_files:
            raise FileNotFoundError(
                f"Missing required trainval dataset files:\n" + 
                "\n".join(missing_files) +
                "\nPlease download the full trainval dataset."
            )
        
        # Check sensor directories
        sensor_dirs = ['samples/CAM_FRONT', 'samples/LIDAR_TOP']
        for dir_path in sensor_dirs:
            full_path = os.path.join(self.dataroot, dir_path)
            if not os.path.exists(full_path):
                raise FileNotFoundError(
                    f"Sensor directory not found: {full_path}\n"
                    "Please check dataset structure."
                )
            files = os.listdir(full_path)
            if not files:
                raise FileNotFoundError(
                    f"No files found in sensor directory: {full_path}\n"
                    "Please check dataset extraction."
                )
            print(f"\nFound {len(files)} files in {dir_path}")
            print("First 5 files:")
            for f in sorted(files)[:5]:
                print(f"  {f}")
        
        logging.info("All required trainval dataset files found")
    
    def get_random_samples(self, num_samples=3):
        """Get random sample indices from the dataset."""
        total_samples = len(self.bev_dataset)
        if total_samples < num_samples:
            logging.warning(f"Requested {num_samples} samples but only {total_samples} available")
            return list(range(total_samples))
        return random.sample(range(total_samples), num_samples)
    
    def test_calibration(self, num_samples=3):
        """Test 1: Calibration matrix validation (95% confidence)"""
        print("\nTest 1: Calibration Matrix Validation")
        print("-" * 50)
        
        for idx in range(num_samples):
            sample = self.filtered_dataset[idx]
            calib = sample['calib']
            
            print(f"\nSample {idx} Calibration:")
            
            # Check matrix shapes and properties
            print("\nMatrix shapes and properties:")
            for key, mat in calib.items():
                print(f"\n{key}:")
                print(f"Shape: {mat.shape}")
                
                # Verify homogeneous transformation matrices
                if key in ['cam2ego', 'lidar2ego']:
                    # Check shape
                    assert mat.shape == (4, 4), f"{key} should be 4x4"
                    
                    # Check rotation part is orthogonal
                    R = mat[:3, :3]
                    I = np.eye(3)
                    R_error = np.abs(R.T @ R - I).max()
                    print(f"Rotation orthogonality error: {R_error:.6f}")
                    assert R_error < 1e-6, f"{key} rotation is not orthogonal"
                    
                    # Check last row is [0,0,0,1]
                    last_row_error = np.abs(mat[3] - np.array([0,0,0,1])).max()
                    print(f"Last row error: {last_row_error:.6f}")
                    assert last_row_error < 1e-6, f"{key} last row is invalid"
                
                elif key == 'cam_intrinsic':
                    # Verify typical intrinsic matrix properties
                    assert mat[0,1] == 0, "Skew should be 0"
                    assert mat[1,0] == 0 and mat[2,0] == 0 and mat[2,1] == 0, \
                        "Lower triangular elements should be 0"
                    assert mat[2,2] == 1, "Last element should be 1"
                    
                    print(f"Focal lengths: fx={mat[0,0]:.1f}, fy={mat[1,1]:.1f}")
                    print(f"Principal point: cx={mat[0,2]:.1f}, cy={mat[1,2]:.1f}")
    
    def test_data_loading(self, num_samples=5):
        """Test 2: Data loading and path validation (95% confidence)"""
        print("\nTest 2: Data Loading and Path Validation")
        print("-" * 50)
        
        successful_samples = 0
        errors = []
        
        for idx in range(min(len(self.filtered_dataset), num_samples * 2)):  # Try more samples to get enough valid ones
            if successful_samples >= num_samples:
                break
                
            try:
                print(f"\nValidating sample {idx}:")
                sample = self.filtered_dataset[idx]
                
                # Print sample information
                print(f"Token: {sample['sample_token']}")
                print(f"Image path: {os.path.basename(sample['image'])}")
                print(f"LiDAR path: {os.path.basename(sample['lidar'])}")
                
                # Verify paths exist
                if not os.path.exists(sample['image']):
                    raise FileNotFoundError(f"Image not found: {sample['image']}")
                if not os.path.exists(sample['lidar']):
                    raise FileNotFoundError(f"LiDAR not found: {sample['lidar']}")
                
                # Print directory information
                img_dir = os.path.dirname(sample['image'])
                lidar_dir = os.path.dirname(sample['lidar'])
                
                print(f"\nImage directory: {img_dir}")
                if os.path.exists(img_dir):
                    print("Image directory exists")
                    files = sorted(os.listdir(img_dir))[:5]
                    print("First 5 files in image directory:")
                    for f in files:
                        print(f"  {f}")
                else:
                    raise FileNotFoundError(f"Image directory does not exist: {img_dir}")
                
                print(f"\nLiDAR directory: {lidar_dir}")
                if os.path.exists(lidar_dir):
                    print("LiDAR directory exists")
                    files = sorted(os.listdir(lidar_dir))[:5]
                    print("First 5 files in LiDAR directory:")
                    for f in files:
                        print(f"  {f}")
                else:
                    raise FileNotFoundError(f"LiDAR directory does not exist: {lidar_dir}")
                
                # Check BEV label properties
                bev_data = self.bev_dataset[idx]
                label = bev_data['bev_label']
                print(f"\nBEV label shape: {label.shape}")
                print(f"BEV label type: {label.dtype}")
                print(f"Unique labels: {torch.unique(label).numpy()}")
                
                successful_samples += 1
                
            except Exception as e:
                errors.append(f"Error processing sample {idx}: {str(e)}")
                continue
        
        if errors:
            print("\nErrors encountered during validation:")
            for error in errors:
                print(f"  {error}")
        
        if successful_samples == 0:
            raise RuntimeError("No valid samples could be processed. Please check dataset integrity.")
        else:
            print(f"\nSuccessfully validated {successful_samples} samples")
    
    def test_bev_visualization(self, num_samples=12):
        """Test 3: BEV label visualization and raw map layers (90% confidence)"""
        print("\nTest 3: BEV Label and Map Layer Visualization")
        print("-" * 50)
        
        # Get random sample indices
        random_indices = self.get_random_samples(num_samples)
        logging.info(f"Selected random samples: {random_indices}")
        
        # Calculate number of rows and columns for better visualization
        num_cols = min(4, num_samples)  # Maximum 4 columns
        num_rows = (num_samples + num_cols - 1) // num_cols  # Ceiling division
        
        # Create figure with subplots - 2 sets of visualizations (BEV and raw map)
        fig, axes = plt.subplots(2 * num_rows, num_cols, figsize=(5*num_cols, 5*2*num_rows))
        fig.subplots_adjust(right=0.9, wspace=0.3, hspace=0.4)
        
        # Process each sample
        for i, idx in enumerate(random_indices):
            row = (i // num_cols) * 2  # Multiply by 2 because we have two rows per sample
            col = i % num_cols
            
            # Get axes for this sample (both BEV and raw map)
            if num_rows * num_cols > 1:
                bev_ax = axes[row, col]
                map_ax = axes[row + 1, col]
            else:
                bev_ax = axes[0]
                map_ax = axes[1]
            
            # Get sample data
            bev_data = self.bev_dataset[idx]
            sample_token = bev_data['sample_token']
            
            # Get scene location for map data
            sample = self.bev_dataset.nusc.get('sample', sample_token)
            scene = self.bev_dataset.nusc.get('scene', sample['scene_token'])
            log = self.bev_dataset.nusc.get('log', scene['log_token'])
            location = log['location']
            nusc_map = self.bev_dataset.map_cache[location]
            
            # 1. Plot rasterized BEV
            label = bev_data['bev_label'].numpy()
            im = bev_ax.imshow(label, cmap=self.cmap, interpolation='nearest', aspect='equal')
            
        #     # Mark ego vehicle rear position
        #     rear_x = self.bev_dataset.grid_size // 2
        #     rear_y = self.bev_dataset.grid_size - 1
        #     bev_ax.plot(rear_x, rear_y, 'ro', markersize=10, label='Ego Rear')
            
        #     # Add title with statistics
        #     unique_labels = np.unique(label)
        #     label_stats = [f"{self.class_names[l]}: {np.sum(label == l)}" 
        #                   for l in unique_labels if l > 0]
        #     bev_ax.set_title(f"BEV Sample {idx}\n" + "\n".join(label_stats), 
        #                     fontsize=8)
        #     bev_ax.axis('off')
            
        #     # Add grid lines to rasterized view
        #     # bev_ax.grid(True, alpha=0.3, linestyle='--')
        #     # bev_ax.set_xticks(np.arange(0, 256, 32))
        #     # bev_ax.set_yticks(np.arange(0, 256, 32))
            
        #     # 2. Plot raw map layers
        #     map_ax.set_aspect('equal')
            
        #     # Get ego pose for the patch
        #     sample = self.bev_dataset.filtered_dataset[idx]
        #     ego_pose = sample['ego_pose']
        #     patch_size = self.bev_dataset.grid_size * self.bev_dataset.resolution
        #     patch_box = (
        #         ego_pose['translation'][0] - patch_size/2,
        #         ego_pose['translation'][1] - patch_size/2,
        #         ego_pose['translation'][0] + patch_size/2,
        #         ego_pose['translation'][1] + patch_size/2
        #     )
            
        #     # Plot each layer with different colors
        #     for layer_name in self.class_colors.keys():
        #         if layer_name == 'background':
        #             continue
                    
        #         try:
        #             # Get records in patch
        #             records = nusc_map.get_records_in_patch(patch_box, layer_names=[layer_name])
        #             layer_records = records.get(layer_name, [])
                    
        #             color = np.array(self.class_colors[layer_name])/255.0
                    
        #             for token in layer_records:
        #                 if layer_name == 'drivable_area':
        #                     polygons = [nusc_map.extract_polygon(poly_token) 
        #                               for poly_token in nusc_map.get(layer_name, token)['polygon_tokens']]
        #                     for polygon in polygons:
        #                         if not polygon.is_empty:
        #                             patch = PolygonPatch(polygon, color=color, alpha=0.5)
        #                             map_ax.add_patch(patch)
                        
        #                 elif layer_name in ['road_divider', 'lane_divider']:
        #                     line = nusc_map.get(layer_name, token)
        #                     if line['line_token']:
        #                         line_obj = nusc_map.extract_line(line['line_token'])
        #                         if not line_obj.is_empty:
        #                             xs, ys = line_obj.xy
        #                             map_ax.plot(xs, ys, color=color, linewidth=1, alpha=0.7)
                        
        #                 else:  # Other polygon layers
        #                     polygon = nusc_map.extract_polygon(nusc_map.get(layer_name, token)['polygon_token'])
        #                     if not polygon.is_empty:
        #                         patch = PolygonPatch(polygon, color=color, alpha=0.5)
        #                         map_ax.add_patch(patch)
                
        #         except Exception as e:
        #             print(f"Error plotting layer {layer_name}: {str(e)}")
            
        #     # Set map plot limits to patch box
        #     map_ax.set_xlim(patch_box[0], patch_box[2])
        #     map_ax.set_ylim(patch_box[1], patch_box[3])
        #     map_ax.set_title(f"Raw Map - Sample {idx}\n{location}", 
        #                     fontsize=8)
            
        #     # Plot ego vehicle position
        #     map_ax.plot(ego_pose['translation'][0], ego_pose['translation'][1], 
        #                 'ro', markersize=10, label='Ego Position')
        #     map_ax.grid(True)
        
        # # Remove empty subplots if any
        # for i in range(i + 1, num_rows * num_cols):
        #     row = (i // num_cols) * 2
        #     col = i % num_cols
        #     if num_rows * num_cols > 1:
        #         axes[row, col].remove()
        #         axes[row + 1, col].remove()
        
        # # Add colorbar
        # cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        # cbar = plt.colorbar(im, cax=cbar_ax)
        # cbar.set_ticks(range(len(self.class_names)))
        # cbar.set_ticklabels(self.class_names)
        # cbar.ax.tick_params(labelsize=8)
        
        # # Add legend
        # legend_elements = [plt.Rectangle((0,0),1,1, facecolor=np.array(color)/255.0, 
        #                                label=name)
        #                  for name, color in zip(self.class_names[1:], self.colors[1:])]  # Skip background
        # fig.legend(handles=legend_elements, 
        #           loc='lower center', 
        #           ncol=len(self.class_names)//2,
        #           bbox_to_anchor=(0.5, 0),
        #           title="BEV Segmentation Classes",
        #           fontsize=8)
        
        # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        # plt.savefig('bev_visualization_random_samples.png', dpi=300, bbox_inches='tight')
        # plt.close()
        
        # print(f"Saved BEV visualization to bev_visualization_random_samples.png")
    
    def test_coordinate_mapping(self):
        """Test 4: Coordinate mapping validation (100% confidence)"""
        print("\nTest 4: Coordinate Mapping Validation")
        print("-" * 50)
        
        # Get a sample ego pose
        sample = self.filtered_dataset[0]
        ego_pose = sample['ego_pose']
        
        # Test points in ego vehicle coordinates
        test_points = np.array([
            [10.0, 0.0],    # 10m forward
            [0.0, 5.0],     # 5m right
            [-5.0, -5.0],   # 5m back, 5m left
            [20.0, 20.0]    # Far corner
        ])
        
        # Manual computation
        center = self.bev_dataset.grid_size // 2
        resolution = self.bev_dataset.resolution
        
        print("\nTesting coordinate mapping:")
        for i, point in enumerate(test_points):
            # Dataset's method
            grid_coords = self.bev_dataset._world_to_grid(
                point.reshape(1, 2),
                ego_pose
            )[0]
            
            # Manual computation
            rotation = Quaternion(ego_pose['rotation']).rotation_matrix[:2, :2]
            translation = np.array(ego_pose['translation'][:2])
            point_transformed = np.dot(rotation.T, point - translation)
            manual_coords = (point_transformed / resolution + center).astype(int)
            manual_coords = np.clip(manual_coords, 0, self.bev_dataset.grid_size - 1)
            
            print(f"\nTest point {i+1}: {point}")
            print(f"Dataset mapping: {grid_coords}")
            print(f"Manual mapping: {manual_coords}")
            print(f"Difference: {np.abs(grid_coords - manual_coords).max()}")
            
            # Verify mapping is close
            assert np.allclose(grid_coords, manual_coords, atol=1), \
                "Coordinate mapping mismatch"
            
            # Visualize the point on the BEV grid
            fig, ax = plt.subplots(figsize=(5, 5))
            bev_data = self.bev_dataset[0]
            label = bev_data['bev_label'].numpy()
            ax.imshow(label, cmap=self.cmap, interpolation='nearest')
            ax.plot(grid_coords[0], grid_coords[1], 'ro', label='Dataset')
            ax.plot(manual_coords[0], manual_coords[1], 'bx', label='Manual')
            ax.set_xlim(0, self.bev_dataset.grid_size)
            ax.set_ylim(0, self.bev_dataset.grid_size)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title(f"BEV Coordinate Mapping - Point {i+1}")
            ax.grid(True)
            ax.legend()
            plt.savefig(f'coordinate_mapping_{i}.png')
            plt.close()
    
    def test_map_statistics(self):
        """Test 5: Map statistics validation (85% confidence)"""
        print("\nTest 5: Map Statistics Validation")
        print("-" * 50)
        
        # Get map data for first sample
        sample = self.filtered_dataset[0]
        sample_token = sample['sample_token']
        scene_token = self.filtered_dataset.nusc.get('sample', sample_token)['scene_token']
        scene = self.filtered_dataset.nusc.get('scene', scene_token)
        log = self.filtered_dataset.nusc.get('log', scene['log_token'])
        location = log['location']
        
        # Get map
        nusc_map = self.bev_dataset.map_cache[location]
        print(f"\nAnalyzing map for location: {location}")
        
        # Store statistics
        stats = {'layer': [], 'num_polygons': [], 'x_range': [], 'y_range': []}
        
        for layer_name in self.filtered_dataset.class_map.keys():
            try:
                records = nusc_map.get_records_in_patch(
                    box_coords=(-50, -50, 50, 50),  # 100m x 100m patch
                    layer_names=[layer_name]
                )[layer_name]
                
                print(f"\nLayer: {layer_name}")
                print(f"Number of records: {len(records)}")
                
                if records:
                    # Get geometries
                    geometries = []
                    for token in records:
                        if layer_name in ['road_divider', 'lane_divider']:
                            line = nusc_map.get('line', token)
                            coords = nusc_map.extract_line(line['line_token'])
                        else:
                            poly = nusc_map.get(layer_name, token)
                            if layer_name == 'drivable_area':
                                for poly_token in poly['polygon_tokens']:
                                    coords = nusc_map.extract_polygon(poly_token)
                                    geometries.append(coords)
                                continue
                            else:
                                coords = nusc_map.extract_polygon(poly['polygon_token'])
                        geometries.append(coords)
                    
                    print(f"Number of geometries: {len(geometries)}")
                    
                    # Compute statistics
                    coords = np.vstack(geometries)
                    print(f"\n{layer_name}:")
                    print(f"Coordinate range X: [{coords[:, 0].min():.1f}, {coords[:, 0].max():.1f}]")
                    print(f"Coordinate range Y: [{coords[:, 1].min():.1f}, {coords[:, 1].max():.1f}]")
                    
                    # Store for visualization
                    stats['layer'].append(layer_name)
                    stats['num_polygons'].append(len(geometries))
                    stats['x_range'].append(coords[:, 0].max() - coords[:, 0].min())
                    stats['y_range'].append(coords[:, 1].max() - coords[:, 1].min())
            
                else:
                    print("No records found")
            
            except Exception as e:
                logging.warning(f"Error processing {layer_name}: {e}")
        
        print("\nCollected statistics:")
        print(stats)
        
        # Visualize statistics
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Number of polygons
        ax1.bar(stats['layer'], stats['num_polygons'])
        ax1.set_title('Number of Geometries per Layer')
        ax1.set_xticklabels(stats['layer'], rotation=45)
        ax1.grid(True)
        
        # Coordinate ranges
        ax2.bar(stats['layer'], stats['x_range'], label='X Range')
        ax2.bar(stats['layer'], stats['y_range'], bottom=stats['x_range'], label='Y Range')
        ax2.set_title('Coordinate Ranges per Layer')
        ax2.set_xticklabels(stats['layer'], rotation=45)
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('map_statistics.png')
        plt.close()
        
        print("\nSaved map statistics visualization to map_statistics.png")
    
    def test_dataloader(self, batch_size=4):
        """Test 6: DataLoader validation (95% confidence)"""
        print("\nTest 6: DataLoader Validation")
        print("-" * 50)
        
        # Create DataLoader
        loader = DataLoader(
            self.bev_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Use main process to avoid memory issues
            pin_memory=False
        )
        
        print(f"DataLoader initialized with batch size {batch_size}")
        print(f"Number of batches: {len(loader)}")
        
        # Get first batch
        batch = next(iter(loader))
        print("\nFirst batch properties:")
        print(f"BEV label shape: {batch['bev_label'].shape}")
        print(f"BEV label type: {batch['bev_label'].dtype}")
        print(f"Memory layout: {'contiguous' if batch['bev_label'].is_contiguous() else 'non-contiguous'}")
        print(f"Device: {batch['bev_label'].device}")
        
        # Verify batch properties
        assert batch['bev_label'].shape == (batch_size, self.bev_dataset.grid_size, 
                                          self.bev_dataset.grid_size)
        assert batch['bev_label'].dtype == torch.int64
        assert batch['bev_label'].is_contiguous()
        
        # Visualize batch
        fig, axes = plt.subplots(1, batch_size, figsize=(5*batch_size, 5))
        for i in range(batch_size):
            axes[i].imshow(batch['bev_label'][i], cmap=self.cmap)
            axes[i].set_title(f"Batch item {i}")
            axes[i].axis('off')
        
        plt.savefig('batch_visualization.png')
        plt.close()
        
        print("\nSaved batch visualization to batch_visualization.png")
    
    def run_all_tests(self):
        """Run all validation tests"""
        try:
            print("\nRunning validation tests for trainval dataset...")
            #self.test_calibration()
            #self.test_data_loading()
            self.test_bev_visualization()
            #self.test_coordinate_mapping()
            #self.test_map_statistics()
            self.test_dataloader()
            print("\nAll validation tests completed successfully!")
            
        except Exception as e:
            logging.error(f"Validation failed: {e}")
            raise

def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Set dataset path for trainval
    dataroot = "C:/nuscenes07"  # Update this to your trainval dataset path
    
    try:
        # Run validation
        validator = DatasetValidator(dataroot)
        validator.run_all_tests()
        
    except FileNotFoundError as e:
        logging.error(f"Dataset validation failed: {e}")
        logging.error("Please download the full trainval dataset.")
    except Exception as e:
        logging.error(f"Unexpected error during validation: {e}")
        raise

if __name__ == "__main__":
    main() 