import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from models.backbones import EfficientNetV2Backbone, SECONDBackbone
import logging
from typing import Dict, List
import torch.nn.functional as F
import time

def visualize_features(features: Dict[str, torch.Tensor], prefix: str):
    """Visualize feature maps and save to disk"""
    for stage_name, feat_map in features.items():
        # Take first sample and compute feature magnitude
        if isinstance(feat_map, torch.Tensor):
            feat_vis = feat_map[0].norm(dim=0)
        else:  # SparseConvTensor
            dense = feat_map.dense()
            feat_vis = dense[0].norm(dim=0).mean(dim=0)  # Average over depth dimension
            
        # Normalize for visualization
        feat_vis = (feat_vis - feat_vis.min()) / (feat_vis.max() - feat_vis.min())
        
        # Plot
        plt.figure(figsize=(8, 8))
        plt.imshow(feat_vis.cpu(), cmap='viridis')
        plt.title(f"{stage_name} Feature Magnitude")
        plt.colorbar()
        plt.savefig(f'{prefix}_{stage_name}_features.png', dpi=300, bbox_inches='tight')
        plt.close()

def print_feature_stats(features: Dict[str, torch.Tensor], prefix: str):
    """Print detailed feature statistics"""
    print(f"\n{prefix} Feature Statistics:")
    for stage_name, feat_map in features.items():
        print(f"\n{stage_name}:")
        if isinstance(feat_map, torch.Tensor):
            print(f"  Shape: {feat_map.shape}")
            print(f"  Memory: {feat_map.element_size() * feat_map.nelement() / (1024*1024):.1f}MB")
            print(f"  Stats:")
            print(f"    Min: {feat_map.min().item():.3f}")
            print(f"    Max: {feat_map.max().item():.3f}")
            print(f"    Mean: {feat_map.mean().item():.3f}")
            print(f"    Std: {feat_map.std().item():.3f}")
            print(f"    Active neurons: {(feat_map != 0).float().mean().item()*100:.1f}%")
        else:  # SparseConvTensor
            print(f"  Indices shape: {feat_map.indices.shape}")
            print(f"  Features shape: {feat_map.features.shape}")
            print(f"  Spatial shape: {feat_map.spatial_shape}")
            print(f"  Memory: {(feat_map.features.element_size() * feat_map.features.nelement() + feat_map.indices.element_size() * feat_map.indices.nelement()) / (1024*1024):.1f}MB")
            print(f"  Stats:")
            print(f"    Min: {feat_map.features.min().item():.3f}")
            print(f"    Max: {feat_map.features.max().item():.3f}")
            print(f"    Mean: {feat_map.features.mean().item():.3f}")
            print(f"    Std: {feat_map.features.std().item():.3f}")
            print(f"    Active voxels: {len(feat_map.features)}")

class BackboneValidator:
    """Validates camera and LiDAR backbone implementations"""
    
    def __init__(self):
        # Configure device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {self.device}")
        
        # Initialize backbones
        self.camera_backbone = EfficientNetV2Backbone(pretrained=True).to(self.device)
        self.lidar_backbone = SECONDBackbone(
            voxel_size=[0.8, 0.8, 0.8],
            point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            max_num_points=5,
            max_voxels=12000
        ).to(self.device)
        
        # Set to eval mode
        self.camera_backbone.eval()
        self.lidar_backbone.eval()
    
    def test_camera_backbone(self, batch_size: int = 8):
        """Test EfficientNetV2 backbone for camera feature extraction"""
        print("\nTesting Camera Backbone (EfficientNetV2):")
        print("-" * 50)
        
        # Create dummy input (B, 3, 224, 224)
        dummy_img = torch.randn(batch_size, 3, 224, 224).to(self.device)
        print(f"\nInput shape: {dummy_img.shape}")
        
        # Test forward pass
        with torch.no_grad(), torch.cuda.amp.autocast():
            try:
                start_time = time.time()
                features = self.camera_backbone(dummy_img)
                torch.cuda.synchronize()
                elapsed = time.time() - start_time
                
                # Print performance metrics
                print(f"\nCamera Backbone Performance:")
                print(f"Batch size: {batch_size}")
                print(f"Forward pass time: {elapsed*1000:.1f}ms")
                print(f"Images processed per second: {batch_size/elapsed:.1f}")
                
                # Memory statistics
                print(f"\nGPU Memory Usage:")
                print(f"Allocated: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
                print(f"Cached: {torch.cuda.memory_reserved()/1024**2:.1f}MB")
                
                print("\nFeature Statistics:")
                print_feature_stats(features, "Camera")
                visualize_features(features, "camera")
                
                print("\nCamera backbone validation successful!")
                return True
                
            except Exception as e:
                print(f"Camera backbone validation failed: {str(e)}")
                return False
    
    def test_lidar_backbone(self, batch_size: int = 4, num_points: int = 50000):
        """Test SECOND backbone for LiDAR feature extraction"""
        print("\nTesting LiDAR Backbone (SECOND):")
        print("-" * 50)
        
        try:
            # Generate random point cloud within expected range
            dummy_points = self._generate_dummy_pointcloud(batch_size, num_points)
            print(f"\nInput point cloud shape: {dummy_points.shape}")
            print(f"Point cloud value ranges:")
            print(f"  X: [{dummy_points[..., 0].min():.1f}, {dummy_points[..., 0].max():.1f}]")
            print(f"  Y: [{dummy_points[..., 1].min():.1f}, {dummy_points[..., 1].max():.1f}]")
            print(f"  Z: [{dummy_points[..., 2].min():.1f}, {dummy_points[..., 2].max():.1f}]")
            print(f"  Intensity: [{dummy_points[..., 3].min():.1f}, {dummy_points[..., 3].max():.1f}]")
            print(f"  Center_X: [{dummy_points[..., 4].min():.1f}, {dummy_points[..., 4].max():.1f}]")
            print(f"  Center_Y: [{dummy_points[..., 5].min():.1f}, {dummy_points[..., 5].max():.1f}]")
            print(f"  Center_Z: [{dummy_points[..., 6].min():.1f}, {dummy_points[..., 6].max():.1f}]")
            print(f"  Std_X: [{dummy_points[..., 7].min():.1f}, {dummy_points[..., 7].max():.1f}]")
            print(f"  Std_Y: [{dummy_points[..., 8].min():.1f}, {dummy_points[..., 8].max():.1f}]")
            print(f"  Std_Z: [{dummy_points[..., 9].min():.1f}, {dummy_points[..., 9].max():.1f}]")
            
            print("\nDebug: Starting voxelization process...")
            
            # Test forward pass
            with torch.no_grad(), torch.cuda.amp.autocast():
                try:
                    # Debug: Print voxelization parameters
                    print("\nDebug: Voxelization parameters:")
                    print(f"Max voxels: {self.lidar_backbone.max_voxels}")
                    print(f"Point cloud range: {self.lidar_backbone.point_cloud_range}")
                    print(f"Voxel size: {self.lidar_backbone.voxel_size}")
                    print(f"Grid size: {self.lidar_backbone.grid_size}")
                    
                    start_time = time.time()
                    features = self.lidar_backbone(dummy_points)
                    torch.cuda.synchronize()
                    elapsed = time.time() - start_time
                    
                    # Print performance metrics
                    print(f"\nLiDAR Backbone Performance:")
                    print(f"Batch size: {batch_size}")
                    print(f"Points per batch: {num_points}")
                    print(f"Forward pass time: {elapsed*1000:.1f}ms")
                    print(f"Points processed per second: {batch_size*num_points/elapsed:.1f}")
                    
                    # Memory statistics
                    print(f"\nGPU Memory Usage:")
                    print(f"Allocated: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
                    print(f"Cached: {torch.cuda.memory_reserved()/1024**2:.1f}MB")
                    
                    # Feature statistics
                    print("\nFeature Statistics:")
                    for stage_name, feat_dict in features.items():
                        if stage_name == 'sparse_features':
                            for level, sparse_feat in feat_dict.items():
                                print(f"\n{stage_name} - {level}:")
                                print(f"  Shape: {sparse_feat.dense().shape}")
                                print(f"  Features: [{sparse_feat.features.min():.3f}, {sparse_feat.features.max():.3f}]")
                                print(f"  Active voxels: {len(sparse_feat.indices)}")
                        elif stage_name == 'bev_features':
                            for level, bev_feat in feat_dict.items():
                                print(f"\n{stage_name} - {level}:")
                                print(f"  Shape: {bev_feat.shape}")
                                print(f"  Range: [{bev_feat.min():.3f}, {bev_feat.max():.3f}]")
                                print(f"  Mean: {bev_feat.mean():.3f}")
                                print(f"  Std: {bev_feat.std():.3f}")
                    
                    # Visualize features
                    visualize_features(features['bev_features'], "lidar")
                    
                    print("\nLiDAR backbone validation successful!")
                    return True
                    
                except Exception as e:
                    print(f"Forward pass failed: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    return False
                
        except Exception as e:
            print(f"LiDAR backbone validation failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _generate_dummy_pointcloud(self, batch_size: int, num_points: int) -> torch.Tensor:
        """
        Generate realistic dummy point cloud data with all required features.
        Returns tensor of shape (batch_size, num_points, 10) with features:
        [x, y, z, intensity, center_x, center_y, center_z, std_x, std_y, std_z]
        """
        # Generate ground plane points (60% of points)
        num_ground = int(0.6 * num_points)
        ground_x = torch.randn(num_ground, device=self.device) * 20.0  # Wider spread
        ground_y = torch.randn(num_ground, device=self.device) * 20.0
        ground_z = torch.zeros(num_ground, device=self.device) + torch.randn(num_ground, device=self.device) * 0.1
        
        # Generate object points (40% of points)
        num_objects = int(0.4 * num_points)
        obj_x = torch.randn(num_objects, device=self.device) * 15.0
        obj_y = torch.randn(num_objects, device=self.device) * 15.0
        obj_z = torch.randn(num_objects, device=self.device) * 1.5 + 1.0  # Above ground
        
        # Combine coordinates
        x = torch.cat([ground_x, obj_x])
        y = torch.cat([ground_y, obj_y])
        z = torch.cat([ground_z, obj_z])
        
        # Calculate distances for intensity
        distances = torch.sqrt(x*x + y*y + z*z)
        surface_noise = torch.rand_like(distances) * 0.2
        intensity = torch.exp(-distances / 50.0) + surface_noise
        
        # Calculate center coordinates (local mean in small radius)
        radius = 2.0
        center_x = x + torch.randn_like(x) * radius
        center_y = y + torch.randn_like(y) * radius
        center_z = z + torch.randn_like(z) * radius
        
        # Calculate standard deviations (local spread)
        std_x = torch.ones_like(x) * 0.5 + torch.rand_like(x) * 0.2
        std_y = torch.ones_like(y) * 0.5 + torch.rand_like(y) * 0.2
        std_z = torch.ones_like(z) * 0.3 + torch.rand_like(z) * 0.1
        
        # Combine all features
        points = torch.stack([x, y, z, intensity, center_x, center_y, center_z, std_x, std_y, std_z], dim=1)
        
        # Clip to valid ranges
        points = torch.clamp(points, min=-50.0, max=50.0)
        points[:, 3] = torch.clamp(points[:, 3], min=0.0, max=1.0)  # Intensity
        
        # Expand to batch dimension and randomize point order
        points = points.unsqueeze(0).expand(batch_size, -1, -1)
        for i in range(batch_size):
            idx = torch.randperm(points.shape[1], device=self.device)
            points[i] = points[i, idx]
        
        return points
    
    def test_activation_patterns(self):
        """Test activation pattern and feature propagation validation"""
        print("\nTesting Activation Patterns and Feature Propagation")
        print("-" * 50)
        
        # Test camera backbone
        print("\nTesting Camera Backbone Activations:")
        dummy_img = torch.randn(8, 3, 224, 224).to(self.device)
        self.camera_backbone.eval()
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            features = self.camera_backbone(dummy_img)
            
            # Analyze activation patterns for each stage
            print("\nCamera Feature Statistics:")
            for stage_name, feat_map in features.items():
                active_neurons = (feat_map != 0).float().mean().item() * 100
                mean_val = feat_map.mean().item()
                std_val = feat_map.std().item()
                max_val = feat_map.max().item()
                
                print(f"\n{stage_name}:")
                print(f"  Active neurons: {active_neurons:.1f}%")
                print(f"  Mean activation: {mean_val:.3f}")
                print(f"  Std deviation: {std_val:.3f}")
                print(f"  Max activation: {max_val:.3f}")
                
                # Visualize activation distribution
                plt.figure(figsize=(8, 4))
                plt.hist(feat_map.flatten().cpu().numpy(), bins=50, density=True)
                plt.title(f"{stage_name} Activation Distribution")
                plt.xlabel("Activation Value")
                plt.ylabel("Density")
                plt.savefig(f'camera_{stage_name}_activation_dist.png')
                plt.close()
        
        # Test LiDAR backbone
        print("\nTesting LiDAR Backbone Activations:")
        dummy_points = self._generate_dummy_pointcloud(4, 50000)
        self.lidar_backbone.eval()
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            features = self.lidar_backbone(dummy_points)
            
            # Analyze sparse feature activation patterns
            print("\nSparse Feature Statistics:")
            for stage_name, feat_map in features['sparse_features'].items():
                active_voxels = len(feat_map.features)
                mean_val = feat_map.features.mean().item()
                std_val = feat_map.features.std().item()
                max_val = feat_map.features.max().item()
                
                print(f"\n{stage_name}:")
                print(f"  Active voxels: {active_voxels}")
                print(f"  Mean activation: {mean_val:.3f}")
                print(f"  Std deviation: {std_val:.3f}")
                print(f"  Max activation: {max_val:.3f}")
                
                # Visualize sparse feature distribution
                plt.figure(figsize=(8, 4))
                plt.hist(feat_map.features.flatten().cpu().numpy(), bins=50, density=True)
                plt.title(f"{stage_name} Sparse Feature Distribution")
                plt.xlabel("Activation Value")
                plt.ylabel("Density")
                plt.savefig(f'lidar_sparse_{stage_name}_activation_dist.png')
                plt.close()
            
            # Analyze BEV feature activation patterns
            print("\nBEV Feature Statistics:")
            for stage_name, feat_map in features['bev_features'].items():
                active_neurons = (feat_map != 0).float().mean().item() * 100
                mean_val = feat_map.mean().item()
                std_val = feat_map.std().item()
                max_val = feat_map.max().item()
                
                print(f"\n{stage_name}:")
                print(f"  Active neurons: {active_neurons:.1f}%")
                print(f"  Mean activation: {mean_val:.3f}")
                print(f"  Std deviation: {std_val:.3f}")
                print(f"  Max activation: {max_val:.3f}")
                
                # Visualize BEV feature distribution
                plt.figure(figsize=(8, 4))
                plt.hist(feat_map.flatten().cpu().numpy(), bins=50, density=True)
                plt.title(f"{stage_name} BEV Feature Distribution")
                plt.xlabel("Activation Value")
                plt.ylabel("Density")
                plt.savefig(f'lidar_bev_{stage_name}_activation_dist.png')
                plt.close()
                
                # Visualize spatial activation patterns
                plt.figure(figsize=(8, 8))
                plt.imshow(feat_map[0].norm(dim=0).cpu(), cmap='viridis')
                plt.colorbar()
                plt.title(f"{stage_name} BEV Spatial Activation Pattern")
                plt.savefig(f'lidar_bev_{stage_name}_spatial_pattern.png')
                plt.close()
        
        print("\nSaved activation visualizations to current directory")
    
    def run_all_tests(self):
        """Run all backbone validation tests"""
        try:
            print("\nRunning backbone validation tests...")
            print("\nGPU Information:")
            print(f"Device: {torch.cuda.get_device_name()}")
            print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory/1024**2:.0f}MB")
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"PyTorch Version: {torch.__version__}")
            
            camera_ok = self.test_camera_backbone()
            lidar_ok = self.test_lidar_backbone()
            
            if camera_ok and lidar_ok:
                print("\nTesting activation patterns...")
                self.test_activation_patterns()
                print("\nAll backbone tests passed successfully!")
                return True
            else:
                print("\nSome backbone tests failed!")
                return False
            
        except Exception as e:
            print(f"Backbone validation failed with error: {str(e)}")
            return False

def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Run validation
    validator = BackboneValidator()
    validator.run_all_tests()

if __name__ == "__main__":
    main() 