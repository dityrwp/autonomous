import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional
import math
import matplotlib.pyplot as plt


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion module for combining LiDAR and image features"""
    
    def __init__(
        self,
        lidar_channels: int,
        image_channels: int,
        output_channels: int,
        chunk_size: int = 2048,
        use_reentrant: bool = True  # Enable reentrant by default for better memory efficiency
    ):
        super().__init__()
        
        self.lidar_channels = lidar_channels
        self.image_channels = image_channels
        self.output_channels = output_channels
        self.chunk_size = chunk_size
        self.use_reentrant = use_reentrant
        
        # Remove redundant projections since train.py already provides projected features
        # Only keep a dimension adaptation layer in case input channels don't match output
        self.dim_adapt = None
        if lidar_channels != output_channels or image_channels != output_channels:
            self.dim_adapt = nn.Sequential(
                nn.Conv2d(max(lidar_channels, image_channels), output_channels, 1, bias=False),
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
        B, C, N = q.shape
        q = q.permute(0, 2, 1)  # B, N, C
        k = k  # B, C, N (already in correct shape)
        v = v.permute(0, 2, 1)  # B, N, C
        
        # Compute attention scores with reduced precision for memory savings
        with torch.cuda.amp.autocast(enabled=True):
            attn_scores = torch.bmm(q, k) / math.sqrt(C)  # B, N, N
            attn = F.softmax(attn_scores, dim=-1)
            out = torch.bmm(attn, v)  # B, N, C
        
        out = out.permute(0, 2, 1)  # B, C, N
        
        return out, {'attention_scores': attn_scores.detach()}
    
    def forward(self, lidar_features: torch.Tensor, image_features: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass of cross-attention fusion
        
        Args:
            lidar_features: LiDAR BEV features (B, C, H, W) - already projected
            image_features: Image features (B, C, H, W) - already projected
        
        Returns:
            Tuple of:
                - Fused features tensor
                - Dictionary containing fusion metrics
        """
        # Only adapt dimensions if necessary
        if self.dim_adapt is not None:
            # Handle dimension adaptation if needed
            if lidar_features.shape[1] != self.output_channels:
                lidar_features = self.dim_adapt(lidar_features)
            if image_features.shape[1] != self.output_channels:
                image_features = self.dim_adapt(image_features)
        
        # Ensure spatial dimensions match
        if image_features.shape[-2:] != lidar_features.shape[-2:]:
            image_features = F.interpolate(
                image_features,
                size=lidar_features.shape[-2:],
                mode='bilinear',
                align_corners=True
            )
        
        # Get normalized modality weights
        weights = self.get_modality_weights()
        
        # Process in chunks to save memory
        B, C, H, W = lidar_features.shape
        num_pixels = H * W
        
        # Use adaptive chunking for large feature maps
        effective_chunk_size = min(self.chunk_size, max(1024, num_pixels // 4))
        
        # Flatten spatial dimensions for chunking
        lidar_flat = lidar_features.reshape(B, C, -1)  # B, C, HW
        image_flat = image_features.reshape(B, C, -1)  # B, C, HW
        
        fused_flat = torch.zeros_like(lidar_flat)
        
        # Process chunks
        for start_idx in range(0, num_pixels, effective_chunk_size):
            end_idx = min(start_idx + effective_chunk_size, num_pixels)
            
            # Get chunks directly in flattened form
            q_chunk = image_flat[..., start_idx:end_idx]  # B, C, chunk_size
            k_chunk = lidar_flat[..., start_idx:end_idx]  # B, C, chunk_size
            v_chunk = lidar_flat[..., start_idx:end_idx]  # B, C, chunk_size
            
            # Process chunk with checkpointing to save memory
            chunk_out, _ = torch.utils.checkpoint.checkpoint(
                self._attention_block,
                q_chunk, k_chunk, v_chunk,
                use_reentrant=self.use_reentrant
            )
            
            # Store result in output tensor
            fused_flat[..., start_idx:end_idx] = chunk_out
        
        # Reshape back to spatial dimensions
        fused = fused_flat.reshape(B, C, H, W)
        
        # Weighted combination of modalities with checkpointing
        def weighted_fusion(a, b, w):
            return w[0] * a + w[1] * b
        
        output = torch.utils.checkpoint.checkpoint(
            weighted_fusion,
            image_features, fused, weights,
            use_reentrant=self.use_reentrant
        )
        
        return output, {'modality_weights': weights.detach()}


class DepthAugmentedBEVLifter(nn.Module):
    def __init__(self,
                 img_channels_dict: Dict[str, int],
                 bev_size: Tuple[int, int] = (128, 128),
                 bev_skip_channels: Dict[str, int] = {
                     'stage1': 32,  # Earlier/high-res features (keep for fine details)
                     'stage3': 128  # Later/semantic features (keep for semantic info)
                     # Removed stage2 to reduce memory/compute as it may be redundant
                 },
                 main_bev_channels: int = 128,
                 main_bev_source_stages: List[str] = ['stage5'],
                 depth_channels: int = 48,  # Reduced from 64
                 min_depth: float = 1.0,
                 max_depth: float = 60.0,
                 voxel_size: Tuple[float, float] = (0.4, 0.4)):
        super().__init__()
        self.bev_h, self.bev_w = bev_size
        self.voxel_size = voxel_size
        self.main_bev_source_stages = main_bev_source_stages
        
        # Simplify feature reduction to a single 1x1 conv
        self.feature_reduction = nn.ModuleDict({
            scale: nn.Sequential(
                nn.Conv2d(channels, 64, 1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            ) for scale, channels in img_channels_dict.items()
            if scale in bev_skip_channels or scale in main_bev_source_stages
        })
        
        # Corrected depth estimation - outputs logits for proper softmax later
        self.depth_net = nn.ModuleDict({
            scale: nn.Sequential(
                nn.Conv2d(channels, depth_channels, 1, bias=False),
                nn.BatchNorm2d(depth_channels)
                # No activation - output raw logits for softmax in forward pass
            ) for scale, channels in img_channels_dict.items()
            if scale in bev_skip_channels or scale in main_bev_source_stages
        })
        
        # Initialize depth bins - use log-spaced bins for better depth resolution
        depth_bins = torch.exp(torch.linspace(
            torch.log(torch.tensor(min_depth)),
            torch.log(torch.tensor(max_depth)),
            depth_channels))
        self.register_buffer('depth_bins', depth_bins)
        
        # Skip feature projections (one per skip stage)
        self.skip_bev_projections = nn.ModuleDict({
            stage: nn.Sequential(
                nn.Conv2d(64, # Use fixed input channels from feature_reduction 
                         bev_skip_channels[stage], 1, bias=False),
                nn.BatchNorm2d(bev_skip_channels[stage]),
                nn.ReLU(inplace=True)
            ) for stage in bev_skip_channels.keys()
            if stage in self.feature_reduction
        })
        
        # Main BEV feature projection
        # Calculate input channels based on included stages
        main_source_channels = 64 * len([stage for stage in main_bev_source_stages 
                                       if stage in self.feature_reduction])
        self.main_bev_projection = nn.Sequential(
            nn.Conv2d(main_source_channels, main_bev_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(main_bev_channels),
            nn.ReLU(inplace=True)
        )
        
        # Cache pixel grids for efficiency
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

    def _process_scale(self, features: torch.Tensor, scale: str, 
                       K_inv: torch.Tensor, T: torch.Tensor, batch_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process a single scale with activation checkpointing to save memory"""
        B, _, H, W = features.shape
        device = features.device
        
        # Get cached pixel grid and expand for batch
        pixels = self._create_pixel_grid(H, W, device)[None].expand(B, -1, -1)  # [B, 3, H*W]
        
        # Feature reduction
        reduced = self.feature_reduction[scale](features)
        
        # Depth estimation - apply softmax explicitly for numerical stability
        depth_logits = self.depth_net[scale](features)
        depth_probs = F.softmax(depth_logits * 10.0, dim=1)  # Scale for sharper distribution
        depth_map = (depth_probs * self.depth_bins.view(1, -1, 1, 1)).sum(dim=1)  # Weighted sum
        
        # Reshape for unprojection
        depth_map = depth_map.reshape(B, 1, -1)
        
        # Camera space unprojection
        cam_pts = depth_map * K_inv.bmm(pixels)
        
        # Transform to ego frame
        cam_pts_h = torch.cat([cam_pts, torch.ones_like(cam_pts[:, :1])], dim=1)
        ego_pts = T.bmm(cam_pts_h)[:, :3]
        
        # --- Add Height Filtering --- 
        # Define a plausible height range for relevant BEV features (e.g., -5m to 3m around the ego vehicle)
        min_ego_z = -5.0
        max_ego_z = 3.0
        height_mask = (ego_pts[:, 2] >= min_ego_z) & (ego_pts[:, 2] <= max_ego_z) # Shape: [B, H*W]
        # --- End Height Filtering ---
        
        # Project to BEV grid with safer rounding
        bev_x = ((ego_pts[:, 0] / self.voxel_size[0]) + (self.bev_w // 2)).floor()
        bev_y = ((ego_pts[:, 1] / self.voxel_size[1]) + (self.bev_h // 2)).floor()
        
        # Combine valid mask (in BEV grid) with height mask (in ego frame)
        valid_mask = (bev_x >= 0) & (bev_x < self.bev_w) & \
                    (bev_y >= 0) & (bev_y < self.bev_h) & \
                    height_mask # Add height constraint
        
        # Reshape features and compute weights using feature activations
        reduced_flat = reduced.reshape(B, reduced.size(1), -1)
        
        # Scale features by distance for better near-field representation
        # Points closer to the ego vehicle get higher weights
        z_ego = ego_pts[:, 2].reshape(B, 1, -1)
        distance_weight = torch.exp(-0.05 * z_ego.abs())  # Exponential decay with distance
        
        # Apply weighting and masking
        weighted = reduced_flat * distance_weight
        weighted = weighted.masked_fill(~valid_mask.unsqueeze(1), 0)
        
        return bev_x, bev_y, valid_mask, weighted

    def forward(self, image_features: Dict[str, torch.Tensor], 
                calib: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass with memory optimizations"""
        device = list(image_features.values())[0].device
        B = list(image_features.values())[0].shape[0]
        
        # Initialize accumulators for skip features
        bev_skip_accumulators = {
            stage: torch.zeros(B, 64,  # Fixed 64 channels from feature reduction
                             self.bev_h, self.bev_w, device=device, dtype=torch.float16)
            for stage in self.skip_bev_projections.keys()
        }
        
        # Initialize main feature accumulator
        main_source_channels = 64 * len([s for s in self.main_bev_source_stages if s in self.feature_reduction])
        main_bev_accumulator = torch.zeros(B, main_source_channels, self.bev_h, self.bev_w,
                                         device=device, dtype=torch.float16)
        main_channel_offset = 0
        
        # Pre-compute camera transformations
        K_inv = torch.inverse(calib['intrinsics'])  # [B, 3, 3]
        T = calib['extrinsics'].view(B, 4, 4)  # [B, 4, 4]
        
        # Only process needed scales (skip stages + main source stages)
        needed_scales = list(set(list(self.skip_bev_projections.keys()) + self.main_bev_source_stages))
        scales_to_process = [s for s in needed_scales if s in image_features]
        
        # Process each needed scale
        for scale in scales_to_process:
            features = image_features[scale]
            
            # Skip scales not in feature_reduction
            if scale not in self.feature_reduction:
                continue
                
            # Use checkpointing for the scale processing to save memory
            process_func = lambda feat, s, ki, t: self._process_scale(feat, s, ki, t, 0)
            bev_x, bev_y, valid_mask, weighted = torch.utils.checkpoint.checkpoint(
                process_func, 
                features, 
                scale, 
                K_inv, 
                T,
                use_reentrant=True
            )
            
            # Accumulate features for each batch separately
            for b in range(B):
                mask_b = valid_mask[b]
                if not mask_b.any():
                    continue
                
                x_b = bev_x[b, mask_b].long()
                y_b = bev_y[b, mask_b].long()
                feat_b = weighted[b, :, mask_b]
                
                # Ensure feature dtype matches accumulator dtype (float16)
                feat_b_fp16 = feat_b.to(torch.float16)
                
                # Calculate linear indices for faster accumulation
                bev_indices = y_b * self.bev_w + x_b
                
                # Accumulate to skip features
                if scale in self.skip_bev_projections:
                    indices_expanded = bev_indices.unsqueeze(0).expand(feat_b_fp16.shape[0], -1)
                    target_flat = bev_skip_accumulators[scale][b].reshape(feat_b_fp16.shape[0], -1)
                    # Use scatter_add_ for accumulation
                    target_flat.scatter_add_(1, indices_expanded, feat_b_fp16)
                
                # Accumulate to main features
                if scale in self.main_bev_source_stages:
                    start_ch = main_channel_offset
                    end_ch = start_ch + feat_b_fp16.shape[0]
                    indices_expanded = bev_indices.unsqueeze(0).expand(feat_b_fp16.shape[0], -1)
                    target_slice = main_bev_accumulator[b, start_ch:end_ch].reshape(feat_b_fp16.shape[0], -1)
                    target_slice.scatter_add_(1, indices_expanded, feat_b_fp16)
            
            # Update main channel offset
            if scale in self.main_bev_source_stages:
                main_channel_offset += 64  # Fixed channel size from feature reduction
        
        # --- Final Projections ---
        # Project accumulated skip features (removed checkpointing for stability)
        bev_skip_features = {}
        for stage_name, accumulator in bev_skip_accumulators.items():
            # Ensure accumulator is float32 before projection
            bev_skip_features[stage_name] = self.skip_bev_projections[stage_name](accumulator.to(torch.float32))
        
        # Project accumulated main features (removed checkpointing for stability)
        # Ensure accumulator is float32 before projection
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
        use_reentrant: bool = True
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
        lidar_features: torch.Tensor,
        main_image_bev_feature: torch.Tensor,
        image_bev_skips: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            lidar_features: Final lidar BEV features [B, C, H, W]
            main_image_bev_feature: Main BEV feature from lifter [B, C, H, W]
            image_bev_skips: Dict of BEV skip features matching head's expected keys
        Returns:
            Tuple of:
            - Fused main features [B, C, H, W]
            - Skip features dict (passed through for head)
        """
        # Fuse main features
        fused_features, fusion_info = self.main_fusion_module(
            lidar_features,
            main_image_bev_feature
        )

        # Return both fused features and skip features
        # The head will use both for the decoder path
        return fused_features, image_bev_skips
