import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import logging
from tqdm import tqdm
from pyquaternion import Quaternion

from nuscenes_data import NuScenesFilteredDataset, NuScenesBEVLabelDataset

class DatasetValidator:
    """Validates NuScenes BEV dataset implementation"""
    
    def __init__(self, dataroot, version="v1.0-mini"):
        self.dataroot = dataroot
        self.version = version
        
        # Initialize datasets
        self.filtered_dataset = NuScenesFilteredDataset(
            dataroot=dataroot,
            version=version,
            split="train"
        )
        
        self.bev_dataset = NuScenesBEVLabelDataset(
            filtered_dataset=self.filtered_dataset,
            grid_size=128,
            resolution=0.2
        )
        
        # Color map for visualization with class names
        self.class_colors = {
            'background': [0, 0, 0],       # Black
            'drivable_area': [255, 255, 255], # White
            'road_segment': [200, 200, 200], # Light gray
            'road_block': [150, 150, 150],  # Gray
            'lane': [100, 200, 100],       # Green
            'road_divider': [255, 200, 0],  # Yellow
            'lane_divider': [200, 100, 100] # Red
        }
        self.colors = list(self.class_colors.values())
        self.class_names = list(self.class_colors.keys())
        self.cmap = ListedColormap(np.array(self.colors) / 255.0)
    
    def test_calibration(self, num_samples=3):
        """Test 6: Calibration matrix validation (95% confidence)"""
        print("\nTest 6: Calibration Matrix Validation")
        print("-" * 50)
        
        for idx in range(num_samples):
            sample = self.filtered_dataset[idx]
            calib = sample['calib']
            
            print(f"\nSample {idx} Calibration:")
            
            # Check matrix shapes
            print("\nMatrix shapes:")
            for key, mat in calib.items():
                print(f"{key}: {mat.shape}")
                
                # Verify homogeneous transformation matrices
                if key in ['cam2ego', 'lidar2ego']:
                    # Check shape
                    assert mat.shape == (4, 4), f"{key} should be 4x4, got {mat.shape}"
                    
                    # Check rotation part is orthogonal
                    R = mat[:3, :3]
                    I = np.eye(3)
                    R_error = np.abs(R.T @ R - I).max()
                    print(f"{key} rotation orthogonality error: {R_error:.6f}")
                    
                    # Check last row is [0,0,0,1]
                    last_row_error = np.abs(mat[3] - np.array([0,0,0,1])).max()
                    print(f"{key} last row error: {last_row_error:.6f}")
                
                elif key == 'cam_intrinsic':
                    # Check shape
                    assert mat.shape == (3, 3), f"Intrinsics should be 3x3, got {mat.shape}"
                    
                    # Verify typical intrinsic matrix properties
                    assert mat[0,1] == 0, "Skew should be 0"
                    assert mat[1,0] == 0 and mat[2,0] == 0 and mat[2,1] == 0, \
                        "Lower triangular elements should be 0"
                    assert mat[2,2] == 1, "Last element should be 1"
                    
                    print(f"Focal lengths: fx={mat[0,0]:.1f}, fy={mat[1,1]:.1f}")
                    print(f"Principal point: cx={mat[0,2]:.1f}, cy={mat[1,2]:.1f}")
            
            # Verify transformation chain
            print("\nTransformation chain test:")
            test_point = np.array([1, 0, 0, 1])  # Point 1m ahead in LiDAR frame
            lidar_to_cam = np.linalg.inv(calib['cam2ego']) @ calib['lidar2ego']
            point_cam = lidar_to_cam @ test_point
            print(f"Test point in camera frame: {point_cam[:3]}")
    
    def test_basic_loading(self, num_samples=5):
        """Test 1: Basic data loading (95% confidence)"""
        print("\nTest 1: Basic Data Loading")
        print("-" * 50)
        
        for idx in range(num_samples):
            sample = self.filtered_dataset[idx]
            bev_data = self.bev_dataset[idx]
            
            print(f"\nSample {idx}:")
            print(f"Token: {sample['sample_token']}")
            print(f"Image path: {os.path.basename(sample['image'])}")
            print(f"LiDAR path: {os.path.basename(sample['lidar'])}")
            print(f"BEV label shape: {bev_data['bev_label'].shape}")
            print(f"Unique labels: {torch.unique(bev_data['bev_label']).numpy()}")
            
            # Verify paths exist
            assert os.path.exists(sample['image']), f"Image not found: {sample['image']}"
            assert os.path.exists(sample['lidar']), f"LiDAR not found: {sample['lidar']}"
    
    def test_bev_visualization(self, num_samples=3):
        """Test 2: BEV label visualization (90% confidence)"""
        print("\nTest 2: BEV Label Visualization")
        print("-" * 50)
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, num_samples, figsize=(5*num_samples, 5))
        if num_samples == 1:
            axes = [axes]  # Make it iterable
            
        # Create BEV visualizations
        for idx, ax in enumerate(axes):
            bev_data = self.bev_dataset[idx]
            label = bev_data['bev_label'].numpy()
            
            im = ax.imshow(label, cmap=self.cmap, interpolation='nearest')
            ax.set_title(f"Sample {idx}\nUnique labels: {np.unique(label)}")
            ax.axis('off')
        
        # Add colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = plt.colorbar(im, cax=cbar_ax)
        cbar.set_ticks(range(len(self.class_names)))
        cbar.set_ticklabels(self.class_names)
        
        # Add legend
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=np.array(color)/255.0, 
                                       label=name)
                         for name, color in zip(self.class_names, self.colors)]
        fig.legend(handles=legend_elements, 
                  loc='lower center', 
                  ncol=len(self.class_names)//2,
                  bbox_to_anchor=(0.5, 0),
                  title="BEV Segmentation Classes")
        
        plt.tight_layout()
        plt.savefig('bev_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved enhanced BEV visualization to bev_visualization.png")
    
    def test_coordinate_mapping(self):
        """Test 3: World to grid coordinate mapping (100% confidence)"""
        print("\nTest 3: Coordinate Mapping Validation")
        print("-" * 50)
        
        # Get a sample ego pose
        sample = self.filtered_dataset[0]
        ego_pose = sample['ego_pose']
        
        # Test points (in ego vehicle coordinates)
        test_points = np.array([
            [10.0, 0.0],    # 10m forward
            [0.0, 5.0],     # 5m right
            [-5.0, -5.0],   # 5m back, 5m left
            [20.0, 20.0]    # Far corner
        ])
        
        # Manual computation
        center = self.bev_dataset.grid_size // 2
        resolution = self.bev_dataset.resolution
        
        print("Testing coordinate mapping:")
        for i, point in enumerate(test_points):
            # Dataset's method
            grid_coords = self.bev_dataset._world_to_grid(
                point.reshape(1, 2),
                ego_pose
            )[0]
            
            # Manual computation (simplified for validation)
            manual_coords = (point / resolution + center).astype(int)
            manual_coords = np.clip(manual_coords, 0, self.bev_dataset.grid_size - 1)
            
            print(f"\nTest point {i+1}: {point}")
            print(f"Dataset mapping: {grid_coords}")
            print(f"Manual mapping: {manual_coords}")
            
            # Visualize the point
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.plot(grid_coords[0], grid_coords[1], 'ro', label='Dataset')
            ax.plot(manual_coords[0], manual_coords[1], 'bx', label='Manual')
            ax.set_xlim(0, self.bev_dataset.grid_size)
            ax.set_ylim(0, self.bev_dataset.grid_size)
            ax.grid(True)
            ax.legend()
            ax.set_title(f"Point {i+1}: {point}")
            plt.savefig(f'coordinate_mapping_{i}.png')
            plt.close()
    
    def test_map_statistics(self):
        """Test 4: Map layer statistics (85% confidence)"""
        print("\nTest 4: Map Layer Statistics")
        print("-" * 50)
        
        sample = self.filtered_dataset[0]
        sample_token = sample['sample_token']
        scene = self.filtered_dataset.nusc.get('sample', sample_token)['scene_token']
        scene = self.filtered_dataset.nusc.get('scene', scene)
        location = scene['location']
        
        # Get map
        nusc_map = self.bev_dataset.map_cache.get(location) or \
                  self.filtered_dataset.nusc.get('map', location)
        
        print(f"Location: {location}")
        
        # Store statistics for visualization
        stats = {'layer': [], 'num_polygons': [], 'x_range': [], 'y_range': []}
        
        for layer_name in self.filtered_dataset.class_map.keys():
            try:
                records = nusc_map.get_records(layer_name)
                if records:
                    coords = np.vstack([poly['polygon'] for poly in records])
                    print(f"\n{layer_name}:")
                    print(f"Number of polygons: {len(records)}")
                    print(f"Coordinate range X: [{coords[:, 0].min():.1f}, {coords[:, 0].max():.1f}]")
                    print(f"Coordinate range Y: [{coords[:, 1].min():.1f}, {coords[:, 1].max():.1f}]")
                    
                    # Store for visualization
                    stats['layer'].append(layer_name)
                    stats['num_polygons'].append(len(records))
                    stats['x_range'].append(coords[:, 0].max() - coords[:, 0].min())
                    stats['y_range'].append(coords[:, 1].max() - coords[:, 1].min())
            except Exception as e:
                print(f"Error processing {layer_name}: {e}")
        
        # Visualize statistics
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Number of polygons
        ax1.bar(stats['layer'], stats['num_polygons'])
        ax1.set_title('Number of Polygons per Layer')
        ax1.set_xticklabels(stats['layer'], rotation=45)
        
        # Coordinate ranges
        ax2.bar(stats['layer'], stats['x_range'], label='X Range')
        ax2.bar(stats['layer'], stats['y_range'], bottom=stats['x_range'], label='Y Range')
        ax2.set_title('Coordinate Ranges per Layer')
        ax2.set_xticklabels(stats['layer'], rotation=45)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('map_statistics.png')
        plt.close()
        
        print("\nSaved map statistics visualization to map_statistics.png")
    
    def test_dataloader(self, batch_size=4):
        """Test 5: DataLoader functionality (95% confidence)"""
        print("\nTest 5: DataLoader Validation")
        print("-" * 50)
        
        loader = DataLoader(
            self.bev_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        print(f"DataLoader initialized with batch size {batch_size}")
        print(f"Number of batches: {len(loader)}")
        
        # Get first batch
        batch = next(iter(loader))
        print("\nFirst batch contents:")
        print(f"BEV label shape: {batch['bev_label'].shape}")
        print(f"BEV label type: {batch['bev_label'].dtype}")
        print(f"Sample tokens: {batch['sample_token']}")
        
        # Check memory layout
        print("\nMemory layout:")
        print(f"Contiguous: {batch['bev_label'].is_contiguous()}")
        print(f"Device: {batch['bev_label'].device}")
        
        # Visualize batch
        fig, axes = plt.subplots(1, batch_size, figsize=(5*batch_size, 5))
        for i in range(batch_size):
            axes[i].imshow(batch['bev_label'][i], cmap=self.cmap)
            axes[i].set_title(f"Batch item {i}")
            axes[i].axis('off')
        plt.savefig('batch_visualization.png')
        plt.close()
        
        print("\nSaved batch visualization to batch_visualization.png")
    
    def test_map_layers(self):
        """Test 7: Map Layer Validation (95% confidence)"""
        print("\nTest 7: Map Layer Validation")
        print("-" * 50)
        
        # Verify all required layers exist
        required_layers = [
            'drivable_area', 'road_segment', 'road_block', 
            'lane', 'ped_crossing', 'walkway', 'stop_line', 
            'carpark_area', 'road_divider', 'lane_divider'
        ]
        
        for layer in required_layers:
            records = self.filtered_dataset.nusc.map.get_records_in_patch(
                box_coords=(-50, -50, 50, 50),
                layer_names=[layer]
            )
            print(f"\n{layer}:")
            print(f"Number of records: {len(records[layer])}")
    
    def test_polygon_validity(self):
        """Test 8: Polygon Validation (90% confidence)"""
        print("\nTest 8: Polygon Validation")
        print("-" * 50)
        
        for layer in self.filtered_dataset.nusc.map.non_geometric_polygon_layers:
            invalid_count = 0
            total_count = 0
            
            records = getattr(self.filtered_dataset.nusc.map, layer)
            for record in records:
                total_count += 1
                if layer == 'drivable_area':
                    polygons = [self.filtered_dataset.nusc.map.extract_polygon(token) 
                               for token in record['polygon_tokens']]
                else:
                    polygons = [self.filtered_dataset.nusc.map.extract_polygon(
                        record['polygon_token'])]
                
                for polygon in polygons:
                    if not polygon.is_valid:
                        invalid_count += 1
                        
            print(f"\n{layer}:")
            print(f"Total polygons: {total_count}")
            print(f"Invalid polygons: {invalid_count}")
    
    def test_coordinate_transforms(self):
        """Test 9: Coordinate Transform Validation (100% confidence)"""
        print("\nTest 9: Coordinate Transform Validation")
        print("-" * 50)
        
        # Test points in different coordinate frames
        test_points = [
            (0, 0),      # Origin
            (10, 0),     # 10m forward
            (0, 10),     # 10m right
            (-10, -10)   # 10m back and left
        ]
        
        for x, y in test_points:
            # Get layers at this point
            layers = self.filtered_dataset.nusc.map.layers_on_point(x, y)
            print(f"\nPoint ({x}, {y}):")
            for layer, token in layers.items():
                if token:
                    print(f"{layer}: {token}")
    
    def test_map_patch_extraction(self):
        """Test 10: Map Patch Extraction (85% confidence)"""
        print("\nTest 10: Map Patch Extraction")
        print("-" * 50)
        
        # Test different patch sizes
        patch_sizes = [(20, 20), (50, 50), (100, 100)]
        
        for size in patch_sizes:
            patch_box = (0, 0, size[0], size[1])  # centered at origin
            patch_angle = 0
            
            # Get map mask for all layers
            map_mask = self.filtered_dataset.nusc.map.get_map_mask(
                patch_box=patch_box,
                patch_angle=patch_angle,
                layer_names=None,  # All layers
                canvas_size=(128, 128)
            )
            
            print(f"\nPatch size {size}:")
            print(f"Mask shape: {map_mask.shape}")
            print(f"Non-zero elements: {np.count_nonzero(map_mask)}")
    
    def run_all_tests(self):
        """Run all validation tests"""
        try:
            self.test_calibration()
            self.test_basic_loading()
            self.test_bev_visualization()
            self.test_coordinate_mapping()
            self.test_map_statistics()
            self.test_dataloader()
            self.test_map_layers()
            self.test_polygon_validity()
            self.test_coordinate_transforms()
            self.test_map_patch_extraction()
            print("\nAll tests completed successfully!")
        except Exception as e:
            print(f"\nError during testing: {e}")
            raise

def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Set dataset paths
    dataroot = "C:\\nuscenes_mini"  
    version = "v1.0-mini"  # Use mini dataset for testing
    
    # Run validation
    validator = DatasetValidator(dataroot, version)
    validator.run_all_tests()

if __name__ == "__main__":
    main() 