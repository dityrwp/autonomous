import torch
import torch.nn as nn
import torchvision.models as models
from typing import List, Dict, Tuple
import numpy as np
import spconv.pytorch as spconv
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.feature_extraction import get_graph_node_names
#from mmdet3d.models.voxel_encoders import DynamicVoxelEncoder
#from spconv.utils import Point2VoxelCPU3d as VoxelGenerator



class EfficientNetV2Backbone(nn.Module):
    """
    EfficientNetV2-S backbone for camera feature extraction.
    Extracts multi-scale features at 5 different resolutions (1/2 to 1/32).
    Uses the official torchvision implementation with proper feature maps.
    Memory-efficient with projection layers.
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        # Load pretrained EfficientNetV2-S
        model = models.efficientnet_v2_s(weights='DEFAULT' if pretrained else None)
        
        # Get the actual node names from the model
        train_nodes, eval_nodes = get_graph_node_names(model)
        #print("Available nodes:", train_nodes)  # Debug print
        
        # Define feature extraction nodes using actual model node names
        # We want features after each stage of the network
        return_nodes = {
            'features.1.1.add': 'stage1',     # Early features
            'features.2.3.add': 'stage2',     # Mid-level features
            'features.4.5.add': 'stage3',     # Higher-level features
            'features.5.8.add': 'stage4',     # Deep features
            'features.6.14.add': 'stage5'     # Final features
        }
        
        # Create feature extractor
        self.feature_extractor = create_feature_extractor(model, return_nodes)
        
        # Original output channels from EfficientNetV2-S at each stage
        # These values are from the actual model output (verified by debug)
        in_channels = {
            'stage1': 24,     # After first MBConv block
            'stage2': 48,     # After second set of MBConv blocks
            'stage3': 128,    # After third set of MBConv blocks
            'stage4': 160,    # After fourth set of MBConv blocks
            'stage5': 256     # After final MBConv block
        }
        
        # Output channels for each stage (powers of 2 for better fusion)
        out_channels = {
            'stage1': 32,     # 24 -> 32
            'stage2': 64,     # 48 -> 64
            'stage3': 128,    # 128 -> 128 (keep same)
            'stage4': 256,    # 160 -> 256
            'stage5': 512     # 256 -> 512
        }
        
        # Add projection layers to reduce channel dimensions for memory efficiency
        self.projections = nn.ModuleDict({
            stage_name: nn.Sequential(
                nn.Conv2d(in_channels[stage_name], out_channels[stage_name], 1, bias=False),
                nn.BatchNorm2d(out_channels[stage_name]),
                nn.ReLU(inplace=True)
            ) for stage_name in return_nodes.values()
        })
        
        # Store output channels for other modules that might need this info
        self.out_channels = out_channels
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the camera backbone
        Args:
            x: Input image tensor of shape (B, 3, H, W)
        Returns:
            Dictionary of multi-scale features with reduced channel dimensions
            Each feature map is projected to a power-of-2 channel dimension
        """
        # Extract features using feature extractor
        features = self.feature_extractor(x)
        
        # Project each feature map to desired dimension
        projected = {
            stage_name: self.projections[stage_name](feat)
            for stage_name, feat in features.items()
        }
                
        return projected

class SECONDBackbone(nn.Module):
    """
    SECOND backbone for LiDAR feature extraction with BEV output.
    Adapted for Velodyne Puck 16-channel LiDAR.
    """
    def __init__(self, 
                 voxel_size: List[float] = [0.8, 0.8, 0.8],
                 point_cloud_range: List[float] = [-51.2, -51.2, -5, 51.2, 51.2, 3],
                 max_num_points: int = 32,
                 max_voxels: int = 4000):
        super().__init__()
        
        # Store parameters
        self.max_num_points = max_num_points
        self.max_voxels = max_voxels
        
        # Calculate grid size dynamically
        self.point_cloud_range = np.array(point_cloud_range)
        self.voxel_size = np.array(voxel_size)
        self.grid_size = np.round(
            (self.point_cloud_range[3:] - self.point_cloud_range[:3]) / self.voxel_size
        ).astype(np.int64)
        self.sparse_shape = np.array([self.grid_size[2], self.grid_size[1], self.grid_size[0]])  # D, H, W
        
        # Create voxel encoder
        self.voxel_encoder = nn.Sequential(
            nn.Linear(4, 16),  # 4 = (x,y,z,intensity)
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True)
        )
        
        # Multi-scale sparse convolution backbone
        self.encoder = spconv.SparseSequential(
            # Initial conv
            self._make_sparse_block(32, 32, 3, padding=1, indice_key="init"),
            
            # Stage 1: 32 -> 32 channels (no downsampling)
            self._make_stage(32, 32, num_blocks=2, stride=1, indice_key="stage1"),
            
            # Stage 2: 32 -> 64 channels
            self._make_stage(32, 64, num_blocks=2, stride=2, indice_key="stage2"),
            
            # Stage 3: 64 -> 128 channels
            self._make_stage(64, 128, num_blocks=2, stride=2, indice_key="stage3")
        )
        
        # BEV projection layers for each stage
        self.bev_projections = nn.ModuleDict({
            'stage1': self._make_bev_projection(32),
            'stage2': self._make_bev_projection(64),
            'stage3': self._make_bev_projection(128)
        })
        
    def _voxelize(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Voxelize point cloud using grid parameters
        Args:
            points: (N, 5) tensor of [batch_idx, x, y, z, intensity]
        Returns:
            voxel_features: (M, 4) tensor of mean features per voxel
            voxel_coords: (M, 4) tensor of voxel coordinates (batch, z, y, x)
        """
        points_xyz = points[:, 1:4].contiguous()
        batch_idx = points[:, 0].contiguous().long()
        features = points[:, [1,2,3,4]]  # x,y,z,intensity
        
        # Convert point cloud range to tensor
        pc_range = torch.tensor(self.point_cloud_range[:3], 
                              dtype=torch.float32, 
                              device=points.device)
        grid_size = torch.tensor(self.grid_size, 
                               dtype=torch.float32, 
                               device=points.device)
        voxel_size = torch.tensor(self.voxel_size, 
                                 dtype=torch.float32, 
                                 device=points.device)
        
        # Convert points to voxel coordinates
        voxel_coords = ((points_xyz - pc_range) / voxel_size).int()
        
        # Filter out points outside range
        valid_mask = torch.all(voxel_coords >= 0, dim=1) & \
                    torch.all(voxel_coords < grid_size, dim=1)
        
        voxel_coords = voxel_coords[valid_mask]
        features = features[valid_mask]
        batch_idx = batch_idx[valid_mask]
        
        # Create unique voxels
        voxel_coords = torch.cat([batch_idx.unsqueeze(1), voxel_coords], dim=1)
        
        # Get unique voxels and their indices
        unique_coords, inverse_indices, counts = torch.unique(voxel_coords, dim=0, 
                                                            return_inverse=True, 
                                                            return_counts=True)
        
        # Compute mean features per voxel
        voxel_features = torch.zeros(len(unique_coords), 4, 
                                   dtype=torch.float32,
                                   device=points.device)
        
        # Use scatter_add_ for more efficient feature accumulation
        for i in range(4):
            voxel_features[:, i].scatter_add_(0, inverse_indices, features[:, i])
        
        # Compute means
        voxel_features = voxel_features / counts.float().unsqueeze(1)
        
        # Limit number of voxels if needed
        if len(unique_coords) > self.max_voxels:
            indices = torch.randperm(len(unique_coords), device=points.device)[:self.max_voxels]
            unique_coords = unique_coords[indices]
            voxel_features = voxel_features[indices]
        
        return voxel_features, unique_coords
        
    def _make_sparse_block(self, in_channels: int, out_channels: int, 
                          kernel_size: int, stride: int = 1, padding: int = 0,
                          indice_key: str = None) -> spconv.SparseSequential:
        """Creates a basic sparse convolution block with BatchNorm and ReLU"""
        return spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, out_channels, kernel_size,
                            stride=stride, padding=padding, bias=False,
                            indice_key=indice_key),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _make_stage(self, in_channels: int, out_channels: int, num_blocks: int,
                    stride: int, indice_key: str) -> spconv.SparseSequential:
        """Creates a stage with multiple sparse convolution blocks"""
        blocks = []
        
        # Downsample block (if stride > 1)
        if stride > 1:
            blocks.append(
                spconv.SparseSequential(
                    spconv.SparseConv3d(in_channels, out_channels, 3,
                                      stride=stride, padding=1, bias=False,
                                      indice_key=f"{indice_key}_down"),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
            current_channels = out_channels
        else:
            current_channels = in_channels
        
        # Residual blocks
        for i in range(num_blocks):
            blocks.append(
                spconv.SparseSequential(
                    self._make_sparse_block(current_channels, out_channels, 3,
                                         padding=1, indice_key=f"{indice_key}_block{i}"),
                    self._make_sparse_block(out_channels, out_channels, 3,
                                         padding=1, indice_key=f"{indice_key}_block{i}_2")
                )
            )
            current_channels = out_channels
            
        return spconv.SparseSequential(*blocks)
    
    def _make_bev_projection(self, channels: int) -> nn.Sequential:
        """Creates a BEV projection module for a specific feature level"""
        return nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
    
    def _sparse_to_bev(self, x: spconv.SparseConvTensor) -> torch.Tensor:
        """Converts sparse 3D features to BEV by max-pooling along z-axis"""
        dense = x.dense()
        dense = dense.permute(0, 4, 2, 3, 1).contiguous()  # [B, C, H, W, D]
        return torch.max(dense, dim=-1)[0]  # [B, C, H, W]
        
    def forward(self, points: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the LiDAR backbone
        Args:
            points: (B, N, 4) tensor of point cloud (x, y, z, intensity)
        Returns:
            Dictionary containing:
                - 'sparse_features': Dict of multi-scale sparse features
                - 'bev_features': Dict of multi-scale BEV features
        """
        batch_size = points.shape[0]
        points_list = []
        
        # Process each batch
        for batch_idx in range(batch_size):
            batch_points = points[batch_idx]
            batch_indices = torch.full((batch_points.shape[0], 1), batch_idx, 
                                    dtype=torch.float32, device=points.device)
            points_with_batch = torch.cat([batch_indices, batch_points], dim=1)
            points_list.append(points_with_batch)
        
        # Combine all batches
        points_with_batch = torch.cat(points_list, dim=0)
        
        # Voxelize and encode points
        voxel_features, voxel_coords = self._voxelize(points_with_batch)
        voxel_features = self.voxel_encoder(voxel_features)
        
        # Create initial sparse tensor
        x = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        
        # Process through encoder
        sparse_features = {}
        bev_features = {}
        
        # Stage 1
        x = self.encoder[0](x)  # Initial conv
        x = self.encoder[1](x)  # Stage 1
        sparse_features['stage1'] = x
        bev = self._sparse_to_bev(x)
        bev_features['stage1'] = self.bev_projections['stage1'](bev)
        
        # Stage 2-3
        for i, stage in enumerate(self.encoder[2:], start=2):
            x = stage(x)
            stage_name = f'stage{i}'
            sparse_features[stage_name] = x
            bev = self._sparse_to_bev(x)
            bev_features[stage_name] = self.bev_projections[stage_name](bev)
        
        return {
            'sparse_features': sparse_features,
            'bev_features': bev_features
        }

def test_backbones():
    """Test function to verify both backbones are working correctly"""
    import matplotlib.pyplot as plt
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. SECONDBackbone requires CUDA for sparse convolutions.")
        return
    
    # Test EfficientNetV2Backbone
    def test_camera_backbone():
        print("\nTesting EfficientNetV2Backbone...")
        
        # Create dummy input with smaller size for debugging
        batch_size, channels, height, width = 2, 3, 224, 224  # Standard ImageNet size
        dummy_img = torch.randn(batch_size, channels, height, width, device=device)
        
        print(f"\nCreated dummy input with shape: {dummy_img.shape}")
        
        # Initialize backbone
        print("\nInitializing EfficientNetV2Backbone...")
        camera_backbone = EfficientNetV2Backbone(pretrained=False).to(device)
        camera_backbone.eval()
        
        # Forward pass
        print("\nPerforming forward pass...")
        with torch.no_grad():
            try:
                features = camera_backbone(dummy_img)
                print("\nForward pass successful!")
            except Exception as e:
                print(f"\nForward pass failed with error: {str(e)}")
                raise
    
    # Test SECONDBackbone
    def test_lidar_backbone():
        print("\nTesting SECONDBackbone...")
        
        # Create dummy point cloud
        batch_size = 1
        num_points = 25000  # Increased number of points
        
        print(f"\nGenerating dummy point cloud:")
        print(f"  Batch size: {batch_size}")
        print(f"  Points per batch: {num_points}")
        
        # Generate random points within the point cloud range
        dummy_points = []
        for b in range(batch_size):
            print(f"\nProcessing batch {b}:")
            
            # Random x, y, z coordinates
            xyz = torch.rand(num_points, 3, device=device) * torch.tensor([[102.4, 102.4, 8.0]], device=device)
            xyz = xyz - torch.tensor([[51.2, 51.2, 5.0]], device=device)  # Center around origin
            
            # Random intensity values
            intensity = torch.rand(num_points, 1, device=device)
            
            # Combine coordinates and intensity
            points = torch.cat([xyz, intensity], dim=1)
            
            print(f"  XYZ range: min={xyz.min().item():.2f}, max={xyz.max().item():.2f}")
            print(f"  Intensity range: min={intensity.min().item():.2f}, max={intensity.max().item():.2f}")
            print(f"  Points shape: {points.shape}")
            
            dummy_points.append(points)
        
        # Stack batches
        dummy_points = torch.stack(dummy_points)  # Shape: (B, N, 4)
        print(f"\nFinal point cloud shape: {dummy_points.shape}")
        
        # Initialize backbone
        print("\nInitializing SECONDBackbone...")
        lidar_backbone = SECONDBackbone().to(device)
        lidar_backbone.eval()
        
        # Forward pass
        print("\nStarting forward pass...")
        with torch.no_grad():
            try:
                print("Processing point cloud...")
                features = lidar_backbone(dummy_points)
                
                print("\nLiDAR Feature Maps:")
                print("\nSparse Features:")
                for stage_name, feat_map in features['sparse_features'].items():
                    print(f"\n{stage_name}:")
                    print(f"  Indices shape: {feat_map.indices.shape}")
                    print(f"  Features shape: {feat_map.features.shape}")
                    print(f"  Spatial shape: {feat_map.spatial_shape}")
                    print(f"  Batch size: {feat_map.batch_size}")
                    print(f"  Features stats:")
                    print(f"    Min: {feat_map.features.min().item():.3f}")
                    print(f"    Max: {feat_map.features.max().item():.3f}")
                    print(f"    Mean: {feat_map.features.mean().item():.3f}")
                    print(f"    Non-zero elements: {(feat_map.features != 0).sum().item()}")
                
                print("\nBEV Features:")
                for stage_name, feat_map in features['bev_features'].items():
                    print(f"\n{stage_name}:")
                    print(f"  Shape: {feat_map.shape}")
                    print(f"  Stats:")
                    print(f"    Min: {feat_map.min().item():.3f}")
                    print(f"    Max: {feat_map.max().item():.3f}")
                    print(f"    Mean: {feat_map.mean().item():.3f}")
                    print(f"    Non-zero elements: {(feat_map != 0).sum().item()}")
                    
                print("\nForward pass completed successfully!")
                        
            except Exception as e:
                print(f"\nError during LiDAR backbone forward pass:")
                print(f"Error type: {type(e).__name__}")
                print(f"Error message: {str(e)}")
                raise
    
    # Run tests
    try:
        test_camera_backbone()
        test_lidar_backbone()
        print("\nAll tests completed successfully!")
    except Exception as e:
        print(f"\nTest failed with error: {str(e)}")

if __name__ == "__main__":
    test_backbones()
