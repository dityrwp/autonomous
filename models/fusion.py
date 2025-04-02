import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional
import math
import matplotlib.pyplot as plt


class DepthNet(nn.Module):
    """Predicts depth distributions from image features"""
    def __init__(self, in_channels: int, depth_channels: int = 64):
        super().__init__()
        self.depth_net = nn.Sequential(
            nn.Conv2d(in_channels, depth_channels, 1),
            nn.BatchNorm2d(depth_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Image features [B, C, H, W]
        Returns:
            Depth distributions [B, D, H, W]
        """
        return self.depth_net(x)

class ModalityProjection(nn.Module):
    """Projects both modalities to the same feature space"""
    def __init__(self, 
                 lidar_channels: int,
                 image_channels: int,
                 output_channels: int = 128):
        super().__init__()
        
        # Projection layers
        self.lidar_proj = nn.Sequential(
            nn.Conv2d(lidar_channels, output_channels, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
        
        self.image_proj = nn.Sequential(
            nn.Conv2d(image_channels, output_channels, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, 
                lidar_features: torch.Tensor,
                image_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project both modalities to same feature space
        Args:
            lidar_features: LiDAR BEV features [B, C1, H, W]
            image_features: Image BEV features [B, C2, H, W]
        Returns:
            Projected features [B, C, H, W] for both modalities
        """
        return self.lidar_proj(lidar_features), self.image_proj(image_features)


class PositionalEncoding(nn.Module):
    """Adds learnable positional encodings to features"""
    def __init__(self, channels: int, height: int, width: int):
        super().__init__()
        
        # Learnable position embeddings
        self.pos_embed = nn.Parameter(
            torch.zeros(1, channels, height, width)
        )
        
        # Initialize
        nn.init.normal_(self.pos_embed, std=0.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encodings
        Args:
            x: Input features [B, C, H, W]
        Returns:
            Features with positional encoding [B, C, H, W]
        """
        return x + self.pos_embed


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion module for combining LiDAR and image features"""
    
    def __init__(
        self,
        lidar_channels: int,
        image_channels: int,
        output_channels: int,
        chunk_size: int = 2048,
        use_reentrant: bool = False
    ):
        super().__init__()
        
        self.lidar_channels = lidar_channels
        self.image_channels = image_channels
        self.output_channels = output_channels
        self.chunk_size = chunk_size
        self.use_reentrant = use_reentrant
        
        # Feature projections
        self.lidar_projection = nn.Sequential(
            nn.Conv2d(lidar_channels, output_channels, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
        
        self.image_projection = nn.Sequential(
            nn.Conv2d(image_channels, output_channels, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
        
        # Learnable modality weights
        self.modality_weights = nn.Parameter(torch.ones(2))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with improved scaling"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def get_modality_weights(self) -> torch.Tensor:
        """Get normalized modality weights"""
        return F.softmax(self.modality_weights, dim=0)
    
    def _attention_block(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Compute cross-attention for a chunk of features"""
        # Reshape inputs
        B, C, H, W = q.shape
        q = q.view(B, C, -1).permute(0, 2, 1)  # B, HW, C
        k = k.view(B, C, -1)  # B, C, HW
        v = v.view(B, C, -1).permute(0, 2, 1)  # B, HW, C
        
        # Compute attention scores
        attn = torch.bmm(q, k) / math.sqrt(C)  # B, HW, HW
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = torch.bmm(attn, v)  # B, HW, C
        out = out.permute(0, 2, 1).view(B, C, H, W)
        
        return out, {'attention_scores': attn}
    
    @torch.amp.autocast(device_type='cuda')
    def forward(self, lidar_features: torch.Tensor, image_features: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass of cross-attention fusion
        
        Args:
            lidar_features: LiDAR BEV features (B, C, H, W)
            image_features: Image features (B, C, H, W)
        
        Returns:
            Tuple of:
                - Fused features tensor
                - Dictionary containing fusion metrics
        """
        # Project features
        lidar_proj = self.lidar_projection(lidar_features)
        image_proj = self.image_projection(image_features)
        
        # Ensure spatial dimensions match
        if image_proj.shape[-2:] != lidar_proj.shape[-2:]:
            image_proj = F.interpolate(
                image_proj,
                size=lidar_proj.shape[-2:],
                mode='bilinear',
                align_corners=True
            )
        
        # Get normalized modality weights
        weights = self.get_modality_weights()
        
        # Process in chunks to save memory
        B, C, H, W = lidar_proj.shape
        num_pixels = H * W
        chunk_outputs = []
        
        for start_idx in range(0, num_pixels, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, num_pixels)
            
            # Extract chunks
            q_chunk = image_proj.view(B, C, -1)[..., start_idx:end_idx].view(B, C, -1, 1)
            k_chunk = lidar_proj.view(B, C, -1)[..., start_idx:end_idx].view(B, C, -1, 1)
            v_chunk = lidar_proj.view(B, C, -1)[..., start_idx:end_idx].view(B, C, -1, 1)
            
            # Process chunk
            chunk_out, _ = torch.utils.checkpoint.checkpoint(
                self._attention_block,
                q_chunk, k_chunk, v_chunk,
                use_reentrant=self.use_reentrant
            )
            chunk_outputs.append(chunk_out)
        
        # Combine chunks
        fused = torch.cat(chunk_outputs, dim=-1).view(B, C, H, W)
        
        # Weighted combination of modalities
        output = weights[0] * image_proj + weights[1] * fused
        
        return output, {'modality_weights': weights.detach()}


class DepthAugmentedBEVLifter(nn.Module):
    def __init__(self, 
                 img_channels: Dict[str, int] = {
                     'stage1': 32,   # 1/2 scale
                     'stage2': 64,   # 1/4 scale
                     'stage3': 128,  # 1/8 scale
                     'stage4': 256,  # 1/16 scale
                     'stage5': 512   # 1/32 scale
                 },
                 bev_size: Tuple[int, int] = (128, 128),
                 depth_channels: int = 64,
                 min_depth: float = 1.0,
                 max_depth: float = 35.0,
                 voxel_size: float = 0.8):
        super().__init__()
        self.bev_h, self.bev_w = bev_size
        self.voxel_size = voxel_size
        
        # Feature reduction with scale-specific processing
        self.feature_reduction = nn.ModuleDict({
            scale: nn.Sequential(
                # Initial projection to handle varying input channels
                nn.Conv2d(channels, 64, 1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                # Scale-specific adaptation
                nn.Conv2d(64, 128, 3, padding=1, bias=False, groups=8),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            ) for scale, channels in img_channels.items()
        })
        
        # Scale-aware depth estimation
        self.depth_net = nn.ModuleDict({
            scale: nn.Sequential(
                # Initial projection to handle varying input channels
                nn.Conv2d(channels, 64, 1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, depth_channels, 1)
            ) for scale, channels in img_channels.items()
        })
        
        # Initialize depth bins
        depth_bins = torch.exp(torch.linspace(
            torch.log(torch.tensor(min_depth)),
            torch.log(torch.tensor(max_depth)),
            depth_channels))
        self.register_buffer('depth_bins', depth_bins)
        
        # Scale-aware confidence estimation
        self.confidence_net = nn.ModuleDict({
            scale: nn.Sequential(
                nn.Conv2d(depth_channels + 128, 64, 3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, 1),
                nn.Sigmoid()
            ) for scale in img_channels.keys()
        })
        
        # Multi-scale fusion
        total_channels = 128 * len(img_channels)
        self.scale_fusion = nn.Sequential(
            nn.Conv2d(total_channels, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 1)
        )
        
        # Cache pixel grids
        self.pixel_grids = {}
        
    def _create_pixel_grid(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        # Create normalized pixel coordinates
        x = torch.linspace(0, W-1, W, device=device)
        y = torch.linspace(0, H-1, H, device=device)
        y, x = torch.meshgrid(y, x, indexing='ij')
        
        # Reshape to [H*W, 2] and normalize
        xy = torch.stack([x, y, torch.ones_like(x)], dim=-1)  # [H, W, 3]
        xy = xy.reshape(-1, 3).T  # [3, H*W]
        
        return xy

    @torch.amp.autocast(device_type='cuda')
    def forward(self, image_features: Dict[str, torch.Tensor], 
                calib: Dict[str, torch.Tensor]) -> torch.Tensor:
        device = list(image_features.values())[0].device
        B = list(image_features.values())[0].shape[0]
        
        # Initialize BEV features for each scale
        scale_features = []
        
        # Pre-compute camera transformations
        K_inv = torch.inverse(calib['intrinsics'])  # [B, 3, 3]
        T = calib['extrinsics'].view(B, 4, 4)  # [B, 4, 4]
        
        # Process each scale
        for scale, features in image_features.items():
            _, _, H, W = features.shape
            
            # Get cached pixel grid
            grid_key = f"{H}_{W}"
            if grid_key not in self.pixel_grids:
                self.pixel_grids[grid_key] = self._create_pixel_grid(H, W, device)
            pixels = self.pixel_grids[grid_key][None].expand(B, -1, -1)  # [B, 3, H*W]
            
            with torch.amp.autocast(enabled=True):
                # 1. Feature reduction
                reduced = self.feature_reduction[scale](features.to(dtype=torch.float16))  # [B, 128, H, W]
                
                # 2. Depth estimation
                depth_logits = self.depth_net[scale](features)  # [B, D, H, W]
                depth_probs = F.softmax(depth_logits * 10.0, dim=1)
                depth_map = (depth_probs * self.depth_bins.view(1, -1, 1, 1)).sum(dim=1)  # [B, H, W]
                
                # 3. Confidence estimation
                confidence = self.confidence_net[scale](
                    torch.cat([depth_logits, reduced], dim=1)
                )  # [B, 1, H, W]
            
            # 4. Unproject to camera space
            depth_map = depth_map.reshape(B, 1, -1)  # [B, 1, H*W]
            cam_pts = depth_map * K_inv.bmm(pixels)  # [B, 3, H*W]
            
            # 5. Transform to ego frame
            cam_pts_h = torch.cat([cam_pts, torch.ones_like(cam_pts[:, :1])], dim=1)  # [B, 4, H*W]
            ego_pts = T.bmm(cam_pts_h)[:, :3]  # [B, 3, H*W]
            
            # 6. Project to BEV grid
            bev_x = ((ego_pts[:, 0] / self.voxel_size) + (self.bev_w // 2)).long()  # [B, H*W]
            bev_y = ((ego_pts[:, 1] / self.voxel_size) + (self.bev_h // 2)).long()  # [B, H*W]
            
            # 7. Filter valid points
            valid_mask = (bev_x >= 0) & (bev_x < self.bev_w) & \
                        (bev_y >= 0) & (bev_y < self.bev_h)  # [B, H*W]
            
            # 8. Weight features
            reduced_flat = reduced.reshape(B, reduced.size(1), -1)  # [B, 128, H*W]
            confidence_flat = confidence.reshape(B, 1, -1)  # [B, 1, H*W]
            weighted = (reduced_flat * confidence_flat).to(dtype=torch.float16)  # [B, 128, H*W]
            weighted = weighted.masked_fill(~valid_mask.unsqueeze(1), 0)
            
            # 9. Accumulate scale features
            scale_bev = torch.zeros(B, reduced.size(1), self.bev_h, self.bev_w, 
                                  dtype=torch.float16, device=device)
            
            # Create linear indices for scatter_add_
            bev_indices = bev_y * self.bev_w + bev_x  # [B, H*W]
            bev_indices = bev_indices.unsqueeze(1).expand(-1, reduced.size(1), -1)  # [B, 128, H*W]
            bev_indices = bev_indices.masked_fill(~valid_mask.unsqueeze(1), 0)
            
            # Reshape and accumulate
            scale_bev = scale_bev.reshape(B, reduced.size(1), -1)  # [B, 128, H*W]
            scale_bev.scatter_add_(2, bev_indices, weighted)
            scale_bev = scale_bev.reshape(B, reduced.size(1), self.bev_h, self.bev_w)
            
            scale_features.append(scale_bev)
        
        # Fuse multi-scale features
        fused = self.scale_fusion(torch.cat(scale_features, dim=1))
        
        return fused.float()


class BEVFusion(nn.Module):
    """Complete BEV fusion module with multi-scale features and memory optimization"""
    
    def __init__(
        self,
        lidar_channels: int = 256,
        image_channels: int = 128,
        output_channels: int = 256,
        spatial_size: Tuple[int, int] = (128, 128),
        chunk_size: int = 2048,
        use_reentrant: bool = False,
        stage_channels: Dict[str, Dict[str, int]] = None
    ):
        super().__init__()
        
        # Store parameters
        self.lidar_channels = lidar_channels
        self.image_channels = image_channels
        self.output_channels = output_channels
        self.spatial_size = spatial_size
        self.chunk_size = chunk_size
        self.use_reentrant = use_reentrant
        
        # Initialize cross-attention fusion modules for each stage
        self.fusion_stages = ['stage1', 'stage2', 'stage3']
        self.fusion_modules = nn.ModuleDict()
        
        # Use provided stage channels or default to output_channels
        if stage_channels is None:
            stage_channels = {
                'stage1': {'lidar': output_channels//2, 'image': output_channels//2},
                'stage2': {'lidar': output_channels, 'image': output_channels},
                'stage3': {'lidar': output_channels*2, 'image': output_channels*2}
            }
        
        # Store stage channels for reference
        self.stage_channels = stage_channels
        
        # Initialize fusion modules and projections for each stage
        for stage in self.fusion_stages:
            # Get channel dimensions for this stage
            stage_lidar_channels = stage_channels[stage]['lidar']
            stage_image_channels = stage_channels[stage]['image']
            stage_output_channels = output_channels
            
            # Create fusion module
            self.fusion_modules[stage] = CrossAttentionFusion(
                lidar_channels=stage_lidar_channels,
                image_channels=stage_image_channels,
                output_channels=stage_output_channels,
                chunk_size=chunk_size,
                use_reentrant=use_reentrant
            )
        
        # Feature aggregation layer
        total_channels = output_channels * len(self.fusion_stages)
        self.feature_aggregation = nn.Sequential(
            nn.Conv2d(total_channels, output_channels*2, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels*2, output_channels, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(size=spatial_size, mode='bilinear', align_corners=True)
        )
    
    @torch.amp.autocast(device_type='cuda')
    def forward(
        self,
        lidar_features: Dict[str, torch.Tensor],
        image_features: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass of BEV fusion module
        
        Args:
            lidar_features: Dictionary of LiDAR BEV features for each stage
            image_features: Dictionary of image features for each stage
        
        Returns:
            Tuple of:
                - Fused features tensor
                - Dictionary containing fusion metrics
        """
        # Initialize containers for fused features and metrics
        fused_features = []
        stage_metrics = {}
        
        # Fuse features for each stage
        for stage in self.fusion_stages:
            # Get features for current stage
            lidar = lidar_features[stage]
            image = image_features[stage]
            
            # Ensure spatial dimensions match
            if image.shape[-2:] != lidar.shape[-2:]:
                image = F.interpolate(
                    image,
                    size=lidar.shape[-2:],
                    mode='bilinear',
                    align_corners=True
                )
            
            # Fuse features
            fused, metrics = self.fusion_modules[stage](lidar, image)
            fused_features.append(fused)
            stage_metrics[stage] = metrics
        
        # Concatenate fused features from all stages
        fused_features = torch.cat(fused_features, dim=1)
        
        # Aggregate features
        output = self.feature_aggregation(fused_features)
        
        # Collect modality weights from all stages
        modality_weights = {}
        for stage in self.fusion_stages:
            modality_weights[stage] = stage_metrics[stage]['modality_weights']
        
        return output, {'modality_weights': modality_weights}

    def test_cross_attention(self, batch_size: int = 4):
        """Test cross-attention fusion module"""
        print("\nTesting Cross-Attention Fusion:")
        print("-" * 50)
        
        try:
            with torch.no_grad():
                # Create dummy inputs
                dummy_img = torch.randn(batch_size, 3, 224, 224).to(self.device)
                dummy_points = torch.randn(batch_size, 50000, 10).to(self.device)
                
                # Get backbone features
                image_features = self.camera_backbone(dummy_img)
                lidar_features = self.lidar_backbone(dummy_points)
                
                # Create calibration
                calib = {
                    'intrinsics': torch.tensor([
                        [1000., 0., 112.],
                        [0., 1000., 112.],
                        [0., 0., 1.]
                    ], device=self.device).expand(batch_size, 3, 3),
                    'extrinsics': torch.eye(4, device=self.device).expand(batch_size, 4, 4)
                }
                
                # Get image BEV features
                image_bev = self.fusion_module.image_lifting(image_features, calib)
                lidar_bev = lidar_features['bev_features']['stage3']
                
                # Project both to common feature space
                lidar_proj, image_proj = self.fusion_module.modality_proj(lidar_bev, image_bev)
                
                print(f"\nFeature shapes:")
                print(f"LiDAR BEV: {lidar_bev.shape}")
                print(f"Image BEV: {image_bev.shape}")
                print(f"Projected LiDAR: {lidar_proj.shape}")
                print(f"Projected Image: {image_proj.shape}")
                
                # Test fusion
                fused_features = self.fusion_module.fusion(lidar_proj, image_proj)
                
                print(f"\nFused features shape: {fused_features.shape}")
                print(f"Value range: [{fused_features.min():.2f}, {fused_features.max():.2f}]")
                print(f"Mean activation: {fused_features.mean():.2f}")
                print(f"Active cells: {(fused_features != 0).float().mean()*100:.1f}%")
                
                # Analyze feature correlations
                print("\nFeature Correlations:")
                lidar_feat = lidar_proj.flatten(2)
                image_feat = image_proj.flatten(2)
                fused_feat = fused_features.flatten(2)
                
                corr_li = F.cosine_similarity(lidar_feat, image_feat).mean().item()
                corr_lf = F.cosine_similarity(lidar_feat, fused_feat).mean().item()
                corr_if = F.cosine_similarity(image_feat, fused_feat).mean().item()
                
                print(f"LiDAR-Image correlation: {corr_li:.3f}")
                print(f"LiDAR-Fused correlation: {corr_lf:.3f}")
                print(f"Image-Fused correlation: {corr_if:.3f}")
                
                # Visualize attention patterns
                plt.figure(figsize=(15, 5))
                
                plt.subplot(1, 3, 1)
                lidar_mag = lidar_proj[0].norm(dim=0).cpu()
                plt.imshow(lidar_mag, cmap='viridis')
                plt.colorbar(label='Magnitude')
                plt.title('Projected LiDAR Features')
                
                plt.subplot(1, 3, 2)
                image_mag = image_proj[0].norm(dim=0).cpu()
                plt.imshow(image_mag, cmap='viridis')
                plt.colorbar(label='Magnitude')
                plt.title('Projected Image Features')
                
                plt.subplot(1, 3, 3)
                fused_mag = fused_features[0].norm(dim=0).cpu()
                plt.imshow(fused_mag, cmap='viridis')
                plt.colorbar(label='Magnitude')
                plt.title('Fused Features')
                
                plt.savefig('fusion_comparison.png')
                plt.close()
                
                print("\nCross-attention fusion test passed!")
                return True
                
        except Exception as e:
            print(f"Cross-attention fusion test failed: {str(e)}")
            return False
