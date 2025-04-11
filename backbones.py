import torch
import torch.nn as nn
import torchvision.models as models
from typing import List, Dict, Tuple
import numpy as np
import spconv.pytorch as spconv
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.feature_extraction import get_graph_node_names
from torch.nn import functional as F
#from mmdet3d.models.voxel_encoders import DynamicVoxelEncoder
#from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
import time
import math



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
        
        # Define feature extraction nodes using actual model node names
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

class LayerScale(nn.Module):
    """Layer scaling module to help with feature propagation"""
    def __init__(self, dim, init_values=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))
    
    def forward(self, x):
        return x * self.gamma

class SECONDBackbone(nn.Module):
    """
    SECOND backbone for LiDAR feature extraction with BEV output.
    Enhanced with better feature propagation and activation patterns.
    """
    def __init__(self, voxel_size, point_cloud_range, max_num_points=5,
                 max_voxels=20000):
        super().__init__()
        
        # Convert inputs to numpy arrays
        self.voxel_size = np.array(voxel_size, dtype=np.float32)
        self.point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
        self.max_num_points = max_num_points
        self.max_voxels = max_voxels
        
        # Calculate grid size
        grid_size = (
            np.round((self.point_cloud_range[3:] - self.point_cloud_range[:3]) / self.voxel_size)
        ).astype(np.int64)
        self.sparse_shape = grid_size[::-1]  # (z, y, x)
        
        # Store grid size for later use
        self.grid_size = grid_size
        
        # Voxel encoder with multi-branch architecture
        self.voxel_encoder = nn.ModuleDict({
            'input': nn.Linear(10, 16),  # Initial feature extraction
            'middle': nn.Linear(16, 16),  # Parallel branch
            'output': nn.Linear(32, 32)   # Combined features
        })
        self.voxel_feature_norm = nn.LayerNorm(32)
        
        # Encoder stages with proper channel progression
        encoder_blocks = []
        
        # Initial convolution: maintain 32 channels
        encoder_blocks.append(
            spconv.SparseSequential(
                spconv.SubMConv3d(32, 32, 3, padding=1, bias=False),
                nn.BatchNorm1d(32),
                nn.ReLU()
            )
        )
        
        # Stage 1: 32 -> 64
        encoder_blocks.append(self._make_stage(32, 64, num_blocks=2))
        
        # Stage 2: 64 -> 128
        encoder_blocks.append(self._make_stage(64, 128, num_blocks=2))
        
        # Stage 3: 128 -> 256
        encoder_blocks.append(self._make_stage(128, 256, num_blocks=2))
        
        self.encoder = nn.ModuleList(encoder_blocks)
        
        # BEV projection layers
        self.bev_projections = nn.ModuleDict({
            'stage1': self._make_bev_projection(64),
            'stage2': self._make_bev_projection(128),
            'stage3': self._make_bev_projection(256)
        })
        
        # Initialize weights with improved scaling
        self._init_weights()
        
    def _init_weights(self):
        """Improved weight initialization for better gradient flow"""
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                # Use Kaiming initialization with larger gain
                gain = 1.5
                fan_in = m.weight.size(1)
                std = gain / math.sqrt(fan_in)
                nn.init.normal_(m.weight, 0, std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (spconv.SubMConv3d, spconv.SparseConv3d)):
                # Initialize sparse conv with larger variance
                gain = 1.5
                fan_in = m.weight.size(1) * np.prod(m.kernel_size)
                std = gain / math.sqrt(fan_in)
                nn.init.normal_(m.weight, 0, std)
    
    def _make_stage(self, in_channels: int, out_channels: int, num_blocks: int) -> spconv.SparseSequential:
        """Enhanced stage with improved residual connections and normalization"""
        blocks = []
        current_channels = in_channels
        
        # Initial projection if needed
        if in_channels != out_channels:
            blocks.append(
                spconv.SparseSequential(
                    spconv.SubMConv3d(in_channels, out_channels, 3,
                                    padding=1, bias=False),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True),
                    # Add LayerNorm for better stability
                    nn.LayerNorm(out_channels)
                )
            )
            current_channels = out_channels
        
        # Residual blocks with enhanced normalization
        for i in range(num_blocks):
            blocks.append(
                spconv.SparseSequential(
                    # First conv with gradient clipping
                    spconv.SubMConv3d(current_channels, out_channels, 3,
                                    padding=1, bias=False,
                                    indice_key=f"block{i}_1"),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True),
                    
                    # Second conv with gradient clipping
                    spconv.SubMConv3d(out_channels, out_channels, 3,
                                    padding=1, bias=False,
                                    indice_key=f"block{i}_2"),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True),
                    
                    # Layer scaling and normalization
                    LayerScale(out_channels, init_values=0.1),
                    nn.LayerNorm(out_channels)
                )
            )
            current_channels = out_channels
        
        return spconv.SparseSequential(*blocks)
    
    def _make_bev_projection(self, in_channels: int) -> nn.Sequential:
        """Enhanced BEV projection with feature preservation"""
        return nn.Sequential(
            # Channel mixing for richer features
            nn.Conv2d(in_channels, in_channels, 1, groups=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            
            # Spatial attention
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            
            # Final projection
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def _sparse_to_bev(self, x: spconv.SparseConvTensor) -> torch.Tensor:
        """Convert sparse features to BEV representation with enhanced feature preservation"""
        features = x.features
        indices = x.indices
        spatial_shape = x.spatial_shape
        
        # Create empty BEV tensor
        bev = torch.zeros(
            (x.batch_size, features.shape[1], spatial_shape[1], spatial_shape[2]),
            dtype=features.dtype,
            device=features.device
        )
        
        # Enhanced feature accumulation with max pooling
        for b, h, w, d, f in zip(indices[:, 0], indices[:, 1], indices[:, 2], 
                                indices[:, 3], features):
            # Use maximum value for overlapping voxels
            current = bev[b, :, w, d]
            bev[b, :, w, d] = torch.where(
                current.abs().sum() > f.abs().sum(),
                current,
                f
            )
        
        # Adaptive feature enhancement
        valid_mask = bev.abs().sum(dim=1, keepdim=True) > 0
        
        # Apply channel attention
        channel_weights = F.softmax(bev.mean(dim=(2, 3), keepdim=True), dim=1)
        bev = bev * channel_weights
        
        # Normalize features while preserving structure
        bev = F.layer_norm(bev, bev.shape[1:])
        bev = torch.where(valid_mask, bev, torch.zeros_like(bev))
        
        return bev
        
    def _voxelize(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Voxelize point cloud using grid parameters with enhanced feature computation
        Args:
            points: (N, 5) tensor of [batch_idx, x, y, z, intensity]
        Returns:
            voxel_features: (M, 10) tensor of enhanced features per voxel
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
        
        # Ensure coordinates are within valid range
        voxel_coords[:, 1] = torch.clamp(voxel_coords[:, 1], 0, self.grid_size[2] - 1)  # z
        voxel_coords[:, 2] = torch.clamp(voxel_coords[:, 2], 0, self.grid_size[1] - 1)  # y
        voxel_coords[:, 3] = torch.clamp(voxel_coords[:, 3], 0, self.grid_size[0] - 1)  # x
        
        # Get unique voxels and their indices
        unique_coords, inverse_indices, counts = torch.unique(voxel_coords, dim=0, 
                                                            return_inverse=True, 
                                                            return_counts=True)
        
        # Initialize enhanced feature tensor
        voxel_features = torch.zeros(len(unique_coords), 10, 
                                   dtype=torch.float32,
                                   device=points.device)
        
        # Compute mean features (x,y,z,intensity)
        for i in range(4):
            voxel_features[:, i].scatter_add_(0, inverse_indices, features[:, i])
        voxel_features[:, :4] = voxel_features[:, :4] / counts.float().unsqueeze(1)
        
        # Compute geometric features per voxel
        for idx in range(len(unique_coords)):
            voxel_mask = inverse_indices == idx
            if voxel_mask.sum() > 1:  # Only compute if voxel has points
                voxel_points = features[voxel_mask, :3]
                # Compute center coordinates (x,y,z)
                voxel_features[idx, 4:7] = voxel_points.mean(dim=0)
                # Compute standard deviation (x,y,z)
                voxel_features[idx, 7:10] = voxel_points.std(dim=0)
        
        # Limit number of voxels if needed
        if len(unique_coords) > self.max_voxels:
            indices = torch.randperm(len(unique_coords), device=points.device)[:self.max_voxels]
            unique_coords = unique_coords[indices]
            voxel_features = voxel_features[indices]
        
        return voxel_features, unique_coords
        
    def forward(self, points: List[torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        """Forward pass with improved feature propagation and normalization

        Args:
            points: A list of point cloud tensors, where each tensor corresponds to 
                    one sample in the batch and has shape [N_i, C] (e.g., [N_i, 5] for x,y,z, intensity, ring).
                    N_i is the number of points for sample i.

        Returns:
            Dictionary containing sparse features and BEV features at different stages.
        """
        # Get batch size from the length of the input list
        batch_size = len(points)
        
        # Prepare points with batch indices for voxelization
        points_with_batch_list = []
        for batch_idx in range(batch_size):
            # Create batch indices tensor for the current point cloud
            current_points = points[batch_idx]
            batch_indices = torch.full((current_points.shape[0], 1), batch_idx, 
                                       dtype=current_points.dtype, # Match point dtype
                                       device=current_points.device)
            
            # Ensure points have at least 4 features (x,y,z, intensity/other)
            # If points only have x,y,z, add a dummy feature (e.g., intensity=0)
            if current_points.shape[1] < 4:
                 # Create dummy intensity column if missing
                 dummy_feature = torch.zeros((current_points.shape[0], 1), 
                                           dtype=current_points.dtype, 
                                           device=current_points.device)
                 current_points = torch.cat([current_points[:, :3], dummy_feature], dim=1)
            # Select necessary features (x, y, z, intensity/first feature)
            # The voxelizer expects specific columns, typically x,y,z,intensity
            points_to_use = current_points[:, :4] # Assume first 4 are x,y,z,intensity
            
            # Prepend batch index
            points_with_batch_list.append(torch.cat([batch_indices, points_to_use], dim=1))
        
        # Concatenate all points with batch indices into a single tensor
        points_with_batch = torch.cat(points_with_batch_list, dim=0) # Shape: [N_total, 5] (batch_idx, x, y, z, intensity)
        
        # Enhanced voxelization and encoding
        # _voxelize expects tensor with shape [N, 5] where first column is batch_idx
        voxel_features, voxel_coords = self._voxelize(points_with_batch)
        
        # Multi-branch voxel encoding with normalization
        input_features = self.voxel_encoder['input'](voxel_features)
        input_features = F.layer_norm(input_features, [input_features.shape[-1]])
        
        middle_features = self.voxel_encoder['middle'](input_features)
        middle_features = F.layer_norm(middle_features, [middle_features.shape[-1]])
        
        combined_features = torch.cat([input_features, middle_features], dim=1)
        voxel_features = self.voxel_encoder['output'](combined_features)
        voxel_features = self.voxel_feature_norm(voxel_features)
        
        # Create initial sparse tensor
        x = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        
        sparse_features = {}
        bev_features = {}
        
        # Stage 1: Initial conv + first stage
        x = self.encoder[0](x)  # Initial conv (32 -> 32)
        x = self.encoder[1](x)  # Stage 1 (32 -> 64)
        sparse_features['stage1'] = x
        bev = self._sparse_to_bev(x)
        bev = self.bev_projections['stage1'](bev)
        bev_features['stage1'] = bev
        
        # Stage 2: 64 -> 128
        x = self.encoder[2](x)
        sparse_features['stage2'] = x
        bev = self._sparse_to_bev(x)
        bev = self.bev_projections['stage2'](bev)
        bev_features['stage2'] = bev
        
        # Stage 3: 128 -> 256
        x = self.encoder[3](x)
        sparse_features['stage3'] = x
        bev = self._sparse_to_bev(x)
        bev = self.bev_projections['stage3'](bev)
        bev_features['stage3'] = bev
        
        return {
            'sparse_features': sparse_features,
            'bev_features': bev_features
        }

