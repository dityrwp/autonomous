import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


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
    """Fuses LiDAR and image features using cross-attention"""
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        assert self.head_dim * num_heads == channels, \
            "channels must be divisible by num_heads"
            
        # Linear projections
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)
        
        # Layer norm
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.ReLU(inplace=True),
            nn.Linear(channels * 4, channels)
        )
        
    def forward(self,
                lidar_features: torch.Tensor,
                image_features: torch.Tensor) -> torch.Tensor:
        """
        Fuse features using cross-attention
        Args:
            lidar_features: LiDAR features [B, C, H, W]
            image_features: Image features [B, C, H, W]
        Returns:
            Fused features [B, C, H, W]
        """
        B, C, H, W = lidar_features.shape
        
        # Reshape to sequence
        q = lidar_features.flatten(2).permute(0, 2, 1)  # [B, HW, C]
        k = image_features.flatten(2).permute(0, 2, 1)  # [B, HW, C]
        v = k
        
        # Multi-head attention
        q = self.q_proj(q).view(B, H*W, self.num_heads, self.head_dim)
        k = self.k_proj(k).view(B, H*W, self.num_heads, self.head_dim)
        v = self.v_proj(v).view(B, H*W, self.num_heads, self.head_dim)
        
        # Transpose for attention
        q = q.transpose(1, 2)  # [B, num_heads, HW, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        
        # Combine heads
        out = (attn @ v).transpose(1, 2).reshape(B, H*W, C)
        out = self.out_proj(out)
        
        # Add & Norm
        out = self.norm1(q.view(B, H*W, C) + out)
        
        # FFN
        out = out + self.ffn(out)
        out = self.norm2(out)
        
        # Reshape back to feature map
        out = out.view(B, H, W, C).permute(0, 3, 1, 2)
        
        return out


class DepthAugmentedBEVLifter(nn.Module):
    def __init__(self, 
                 img_channels: Dict[str, int] = {
                     'stage3': 256,  # 1/16 scale
                     'stage4': 384,  # 1/32 scale
                     'stage5': 512   # 1/64 scale
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
                nn.Conv2d(channels, 128, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                # Scale-specific adaptation
                nn.Conv2d(128, 128, 3, padding=1, bias=False, groups=8),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            ) for scale, channels in img_channels.items()
        })
        
        # Scale-aware depth estimation
        self.depth_net = nn.ModuleDict({
            scale: nn.Sequential(
                nn.Conv2d(channels, 128, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, depth_channels, 1)
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
        
    @torch.jit.ignore
    def _create_pixel_grid(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        """Create pixel coordinate grid (cached for each resolution)"""
        u = torch.linspace(0, width-1, width, device=device)
        v = torch.linspace(0, height-1, height, device=device)
        v, u = torch.meshgrid(v, u, indexing='ij')
        return torch.stack([u, v, torch.ones_like(u)], dim=0)

    @torch.cuda.amp.autocast()
    def forward(self, image_features: Dict[str, torch.Tensor], 
                calib: Dict[str, torch.Tensor]) -> torch.Tensor:
        device = list(image_features.values())[0].device
        B = list(image_features.values())[0].shape[0]
        
        # Initialize BEV features for each scale
        scale_features = []
        
        # Pre-compute camera transformations
        K_inv = torch.inverse(calib['intrinsics'])
        T = calib['extrinsics'].view(B, 4, 4)
        
        # Process each scale
        for scale, features in image_features.items():
            _, _, H, W = features.shape
            
            # Get cached pixel grid
            grid_key = f"{H}_{W}"
            if grid_key not in self.pixel_grids:
                self.pixel_grids[grid_key] = self._create_pixel_grid(H, W, device)
            pixels = self.pixel_grids[grid_key][None].expand(B, -1, -1, -1)
            
            with torch.cuda.amp.autocast(enabled=True):
                # 1. Feature reduction
                reduced = self.feature_reduction[scale](features.to(dtype=torch.float16))
                
                # 2. Depth estimation
                depth_logits = self.depth_net[scale](features)
                depth_probs = F.softmax(depth_logits * 10.0, dim=1)
                depth_map = (depth_probs * self.depth_bins.view(1, -1, 1, 1)).sum(dim=1)
                
                # 3. Confidence estimation
                confidence = self.confidence_net[scale](
                    torch.cat([depth_logits, reduced], dim=1)
                )
            
            # 4. Unproject to camera space
            cam_pts = depth_map.unsqueeze(1) * K_inv.bmm(pixels.reshape(B, 3, -1))
            
            # 5. Transform to ego frame
            cam_pts_h = torch.cat([cam_pts, torch.ones_like(cam_pts[:, :1])], dim=1)
            ego_pts = torch.bmm(T, cam_pts_h).reshape(B, 4, H, W)[:, :3]
            
            # 6. Project to BEV grid
            bev_x = ((ego_pts[:, 0] / self.voxel_size) + (self.bev_w // 2)).long()
            bev_y = ((ego_pts[:, 1] / self.voxel_size) + (self.bev_h // 2)).long()
            
            # 7. Filter valid points
            valid_mask = (bev_x >= 0) & (bev_x < self.bev_w) & \
                        (bev_y >= 0) & (bev_y < self.bev_h)
            valid_mask = valid_mask.unsqueeze(1).expand(-1, 128, -1, -1)
            
            # 8. Weight features
            weighted = (reduced * confidence).to(dtype=torch.float16)
            weighted = weighted.masked_fill(~valid_mask, 0)
            
            # 9. Accumulate scale features
            scale_bev = torch.zeros(B, 128, self.bev_h, self.bev_w, 
                                  dtype=torch.float16, device=device)
            scale_bev.scatter_add_(2,
                bev_y.unsqueeze(1).expand(-1, 128, -1, -1).masked_fill(~valid_mask, 0),
                weighted
            )
            scale_features.append(scale_bev)
        
        # Fuse multi-scale features
        fused = self.scale_fusion(torch.cat(scale_features, dim=1))
        
        return fused.float()


class BEVFusion(nn.Module):
    """Complete BEV fusion module aligned with SECOND backbone"""
    def __init__(self,
                 lidar_channels: int = 128,  # SECOND stage3 channels
                 image_channels: Dict[str, int] = {
                     'stage3': 256,  # 1/16 scale
                     'stage4': 384,  # 1/32 scale
                     'stage5': 512   # 1/64 scale
                 },
                 output_channels: int = 128,
                 bev_height: int = 128,  # Match SECOND's BEV resolution
                 bev_width: int = 128,
                 voxel_size: float = 0.8,  # Match SECOND's voxel size
                 num_heads: int = 4):
        super().__init__()
        
        # Image lifting network aligned with SECOND backbone
        self.image_lifting = DepthAugmentedBEVLifter(
            img_channels=image_channels,
            bev_size=(bev_height, bev_width),
            depth_channels=64,
            min_depth=1.0,
            max_depth=35.0,
            voxel_size=voxel_size
        )
        
        # Modality projection (input channels is now fixed at 128 from DepthAugmentedBEVLifter)
        self.modality_proj = ModalityProjection(
            lidar_channels=lidar_channels,
            image_channels=128,  # Fixed output from DepthAugmentedBEVLifter
            output_channels=output_channels
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            channels=output_channels,
            height=bev_height,
            width=bev_width
        )
        
        # Cross-attention fusion
        self.fusion = CrossAttentionFusion(
            channels=output_channels,
            num_heads=num_heads
        )
        
    def forward(self,
                lidar_features: torch.Tensor,
                image_features: Dict[str, torch.Tensor],
                calib: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Fuse LiDAR and image features in BEV space
        Args:
            lidar_features: LiDAR BEV features from SECOND [B, 128, 128, 128]
            image_features: Dict of multi-scale image features from EfficientNetV2-S
                - stage3: [B, 256, H/16, W/16]
                - stage4: [B, 384, H/32, W/32]
                - stage5: [B, 512, H/64, W/64]
            calib: Camera calibration matrices (camera-to-ego transform)
        Returns:
            Fused BEV features [B, 128, 128, 128]
        """
        # Lift image features to BEV using depth-augmented lifter
        image_bev = self.image_lifting(image_features, calib)  # [B, 128, H, W]
        
        # Project both modalities to common feature space
        lidar_proj, image_proj = self.modality_proj(lidar_features, image_bev)
        
        # Add positional encoding
        lidar_pos = self.pos_encoder(lidar_proj)
        image_pos = self.pos_encoder(image_proj)
        
        # Fuse using cross-attention
        fused = self.fusion(lidar_pos, image_pos)
        
        return fused
