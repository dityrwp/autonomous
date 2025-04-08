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
                 img_channels_dict: Dict[str, int],
                 bev_size: Tuple[int, int] = (128, 128),
                 bev_skip_channels: Dict[str, int] = {
                     'stage1': 32,  # Earlier/high-res features
                     'stage2': 64,
                     'stage3': 128  # Later/semantic features
                 },
                 main_bev_channels: int = 128,
                 main_bev_source_stages: List[str] = ['stage5'],
                 depth_channels: int = 64,
                 min_depth: float = 1.0,
                 max_depth: float = 60.0,
                 voxel_size: Tuple[float, float] = (0.4, 0.4)):
        super().__init__()
        self.bev_h, self.bev_w = bev_size
        self.voxel_size = voxel_size
        self.main_bev_source_stages = main_bev_source_stages
        
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
            ) for scale, channels in img_channels_dict.items()
        })
        
        # Scale-aware depth estimation
        self.depth_net = nn.ModuleDict({
            scale: nn.Sequential(
                nn.Conv2d(channels, 64, 1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, depth_channels, 1)
            ) for scale, channels in img_channels_dict.items()
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
            ) for scale in img_channels_dict.keys()
        })
        
        # Skip feature projections (one per skip stage)
        self.skip_bev_projections = nn.ModuleDict({
            stage: nn.Sequential(
                nn.Conv2d(self.feature_reduction[stage][0].out_channels, 
                         bev_skip_channels[stage], 1, bias=False),
                nn.BatchNorm2d(bev_skip_channels[stage]),
                nn.ReLU(inplace=True)
            ) for stage in bev_skip_channels.keys()
            if stage in self.feature_reduction
        })
        
        # Main BEV feature projection
        # Input channels will be the sum of channels from main source stages
        main_source_channels = sum(
            self.feature_reduction[stage][0].out_channels 
            for stage in main_bev_source_stages 
            if stage in self.feature_reduction
        )
        self.main_bev_projection = nn.Sequential(
            nn.Conv2d(main_source_channels, main_bev_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(main_bev_channels),
            nn.ReLU(inplace=True)
        )
        
        # Cache pixel grids
        self.pixel_grids = {}
    
    def _create_pixel_grid(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Creates normalized pixel coordinates."""
        if f"{H}_{W}" not in self.pixel_grids:
            x = torch.linspace(0, W-1, W, device=device)
            y = torch.linspace(0, H-1, H, device=device)
            y, x = torch.meshgrid(y, x, indexing='ij')
            xy = torch.stack([x, y, torch.ones_like(x)], dim=-1)  # [H, W, 3]
            xy = xy.reshape(-1, 3).T  # [3, H*W]
            self.pixel_grids[f"{H}_{W}"] = xy
        return self.pixel_grids[f"{H}_{W}"]

    @torch.amp.autocast(device_type='cuda')
    def forward(self, image_features: Dict[str, torch.Tensor], 
                calib: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        device = list(image_features.values())[0].device
        B = list(image_features.values())[0].shape[0]
        
        # Initialize accumulators for skip features and main features
        bev_skip_accumulators = {
            stage: torch.zeros(B, self.feature_reduction[stage][0].out_channels, 
                             self.bev_h, self.bev_w, device=device, dtype=torch.float16)
            for stage in self.skip_bev_projections.keys()
        }
        
        # Initialize main feature accumulator
        main_source_channels = sum(
            self.feature_reduction[stage][0].out_channels 
            for stage in self.main_bev_source_stages 
            if stage in self.feature_reduction
        )
        main_bev_accumulator = torch.zeros(B, main_source_channels, self.bev_h, self.bev_w,
                                         device=device, dtype=torch.float16)
        main_channel_offset = 0  # Track channel offset for main accumulator
        
        # Pre-compute camera transformations
        K_inv = torch.inverse(calib['intrinsics'])  # [B, 3, 3]
        T = calib['extrinsics'].view(B, 4, 4)  # [B, 4, 4]
        
        # Process each scale
        for scale, features in image_features.items():
            _, _, H, W = features.shape
            
            # Get cached pixel grid
            pixels = self._create_pixel_grid(H, W, device)[None].expand(B, -1, -1)  # [B, 3, H*W]
            
            with torch.amp.autocast(enabled=True):
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
            depth_map = depth_map.reshape(B, 1, -1)
            cam_pts = depth_map * K_inv.bmm(pixels)
            
            # 5. Transform to ego frame
            cam_pts_h = torch.cat([cam_pts, torch.ones_like(cam_pts[:, :1])], dim=1)
            ego_pts = T.bmm(cam_pts_h)[:, :3]
            
            # 6. Project to BEV grid
            bev_x = ((ego_pts[:, 0] / self.voxel_size[0]) + (self.bev_w // 2)).long()
            bev_y = ((ego_pts[:, 1] / self.voxel_size[1]) + (self.bev_h // 2)).long()
            
            # 7. Filter valid points
            valid_mask = (bev_x >= 0) & (bev_x < self.bev_w) & \
                        (bev_y >= 0) & (bev_y < self.bev_h)
            
            # 8. Weight features
            reduced_flat = reduced.reshape(B, reduced.size(1), -1)
            confidence_flat = confidence.reshape(B, 1, -1)
            weighted = (reduced_flat * confidence_flat).to(dtype=torch.float16)
            weighted = weighted.masked_fill(~valid_mask.unsqueeze(1), 0)
            
            # 9. Accumulate features
            for b in range(B):
                mask_b = valid_mask[b]
                if not mask_b.any():
                    continue
                
                x_b = bev_x[b, mask_b].long()
                y_b = bev_y[b, mask_b].long()
                feat_b = weighted[b, :, mask_b]
                
                # Calculate linear indices for BEV grid
                bev_indices = y_b * self.bev_w + x_b
                
                # Accumulate to skip features if this stage is used for skips
                if scale in self.skip_bev_projections:
                    indices_expanded = bev_indices.unsqueeze(0).expand(feat_b.shape[0], -1)
                    target_flat = bev_skip_accumulators[scale][b].view(feat_b.shape[0], -1)
                    target_flat.scatter_add_(1, indices_expanded, feat_b)
                
                # Accumulate to main features if this stage is a source
                if scale in self.main_bev_source_stages:
                    start_ch = main_channel_offset
                    end_ch = start_ch + feat_b.shape[0]
                    indices_expanded = bev_indices.unsqueeze(0).expand(feat_b.shape[0], -1)
                    target_slice = main_bev_accumulator[b, start_ch:end_ch].view(feat_b.shape[0], -1)
                    target_slice.scatter_add_(1, indices_expanded, feat_b)
            
            # Update main channel offset if this stage contributes to main features
            if scale in self.main_bev_source_stages:
                main_channel_offset += reduced.size(1)
        
        # --- Final Projections ---
        # Project accumulated skip features
        bev_skip_features = {}
        for stage_name, accumulator in bev_skip_accumulators.items():
            bev_skip_features[stage_name] = self.skip_bev_projections[stage_name](
                accumulator.to(torch.float32)
            )
        
        # Project accumulated main features
        main_bev_feature = self.main_bev_projection(main_bev_accumulator.to(torch.float32))
        
        return main_bev_feature, bev_skip_features


class BEVFusion(nn.Module):
    """Complete BEV fusion module with multi-scale features and memory optimization"""
    
    def __init__(
        self,
        lidar_channels: int = 256,
        image_channels: int = 128,
        output_channels: int = 256,
        spatial_size: Tuple[int, int] = (128, 128),
        chunk_size: int = 2048,
        use_reentrant: bool = False
    ):
        super().__init__()
        
        # Main fusion module for final features
        self.main_fusion_module = CrossAttentionFusion(
            lidar_channels=lidar_channels,
            image_channels=image_channels,
            output_channels=output_channels,
            chunk_size=chunk_size,
            use_reentrant=use_reentrant
        )

    def forward(
        self,
        projected_lidar_features: Dict[str, torch.Tensor],
        main_image_bev_feature: torch.Tensor,
        image_bev_skips: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            projected_lidar_features: Dict with final lidar features
            main_image_bev_feature: Main BEV feature from lifter
            image_bev_skips: Dict of BEV skip features matching head's expected keys
        Returns:
            Tuple of:
            - Fused main features
            - Skip features dict (passed through for head)
        """
        # Get final lidar feature
        lidar_final = projected_lidar_features['stage3']  # Assuming this is the final stage

        # Fuse main features
        fused_features, _ = self.main_fusion_module(
            lidar_final,
            main_image_bev_feature
        )

        # Return both fused features and skip features
        # The head will use both for the decoder path
        return fused_features, image_bev_skips
