from typing import Dict, List, Optional, Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2,
    reduction: str = "none"
) -> torch.Tensor:
    """
    Compute core focal loss for binary segmentation without alpha weighting.
    Class weighting should be applied externally.
    
    Args:
        inputs: [B, C, H, W] raw logits
        targets: [B, C, H, W] binary targets
        gamma: Focusing parameter
        reduction: 'none', 'mean', 'sum'
    Returns:
        Focal loss value
    """
    inputs = inputs.float()
    targets = targets.float()
    
    # Compute sigmoid probabilities
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    
    # For numerical stability, clamp p_t values away from 0 and 1
    p_t = p * targets + (1 - p) * (1 - targets)
    p_t = torch.clamp(p_t, min=1e-7, max=1-1e-7)
    
    # Compute focal weights (core focal term)
    focal_weight = ((1 - p_t) ** gamma).clamp(min=1e-7)
    loss = ce_loss * focal_weight

    # Ensure loss is non-negative
    loss = torch.clamp(loss, min=0.0)

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


class DecoderBlock(nn.Module):
    """Decoder block for U-Net style architecture with skip connections"""
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, scale_factor: float = 2.0):
        super().__init__()
        self.scale_factor = scale_factor
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        # Concat channels: in_channels (from prev layer) + skip_channels (from skip connection)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with upsampling and skip connection
        Args:
            x: Input tensor from previous decoder stage
            skip: Skip connection tensor from encoder
        Returns:
            Processed feature tensor at scale_factor spatial resolution
        """
        x = self.upsample(x)
        # Ensure skip connection matches upsampled feature size
        if x.shape[-2:] != skip.shape[-2:]:
            skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class BEVSegmentationHead(nn.Module):
    """
    U-Net style segmentation head optimized for BEV segmentation.
    Features:
    - Multi-scale decoder with skip connections from image BEV features
    - Efficient combined Focal and Dice loss with class weighting
    - Proper upsampling path to recover spatial resolution
    """
    def __init__(
        self,
        in_channels: int,           # Channels from the BEVFusion module output
        skip_channels: Dict[str, int], # Channels of PROJECTED IMAGE BEV skip features
                                      # Only requires 'stage1' for simplified architecture
        decoder_channels: Tuple[int, int, int] = (128, 64, 32),
        num_classes: int = 6,
        dropout: float = 0.1,
        use_focal_loss: bool = True,
        focal_gamma: float = 1.5,
        class_weights: Optional[List[float]] = None,
        use_dice_loss: bool = True,
        dice_weight: float = 0.5,
        dice_smooth: float = 10.0,  # Increased from 1.0 to 100.0 for numerical stability
        label_smoothing: float = 0.05,
        output_size: Tuple[int, int] = (128, 128)  # Match BEV lifter's output size
    ) -> None:
        super().__init__()
        
        # Store parameters
        self.num_classes = num_classes
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
        self.use_dice_loss = use_dice_loss
        self.dice_weight = dice_weight
        self.dice_smooth = dice_smooth
        self.label_smoothing = label_smoothing
        self.output_size = output_size
        
        # Validate skip stage keys - only require stage1 now
        expected_stages = {'stage1'}
        if not all(stage in skip_channels for stage in expected_stages):
            raise ValueError(f"skip_channels must contain all required stages: {expected_stages}")
        
        # Register class weights
        if class_weights is not None:
            if len(class_weights) != num_classes:
                raise ValueError(f"class_weights must have length {num_classes}")
            self.register_buffer('class_weights', 
                               torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.register_buffer('class_weights', 
                               torch.ones(num_classes, dtype=torch.float32))
        
        # --- Decoder Architecture ---
        self.skip_channels = skip_channels
        skip1_ch = self.skip_channels['stage1']  # Early features (highest resolution) - only skip we need
        
        dec_ch1, dec_ch2, dec_ch3 = decoder_channels
        
        # Bottleneck conv to reduce channels before decoder
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, dec_ch1, 3, padding=1, bias=False),
            nn.BatchNorm2d(dec_ch1),
            nn.ReLU(inplace=True)
        )
        
        # Simplified decoder for stage1-only architecture:
        # Decoder blocks with a direct path using only stage1 skip connection
        
        # First upsampling block (no skip connection yet)
        self.upsample1 = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=True),
            nn.Conv2d(dec_ch1, dec_ch2, 3, padding=1, bias=False),
            nn.BatchNorm2d(dec_ch2),
            nn.ReLU(inplace=True)
        )
        
        # Second upsampling block with stage1 skip connection
        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=True),
        )
        
        # Skip connection fusion - combine upsampled features with stage1 skip
        self.skip_fusion = nn.Sequential(
            nn.Conv2d(dec_ch2 + skip1_ch, dec_ch3, 3, padding=1, bias=False),
            nn.BatchNorm2d(dec_ch3),
            nn.ReLU(inplace=True),
            nn.Conv2d(dec_ch3, dec_ch3, 3, padding=1, bias=False),
            nn.BatchNorm2d(dec_ch3),
            nn.ReLU(inplace=True)
        )
        
        # Final upsampling to desired output size if needed
        self.final_upsample = nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=True)
        
        # Final layers
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.final_conv = nn.Conv2d(dec_ch3, num_classes, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _compute_losses(self, logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute losses separately to ensure tensors are properly combined for backpropagation."""
        losses = {}
        total_loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        # Check for NaNs in inputs
        if torch.isnan(logits).any() or torch.isnan(targets).any():
            print("WARNING: NaN detected in inputs to loss function")
            # Replace NaNs with zeros in logits and targets
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1.0, neginf=-1.0)
            targets = torch.nan_to_num(targets, nan=0.0, posinf=1.0, neginf=0.0)
        
        if self.use_focal_loss:
            # Compute focal loss for each class
            focal_loss_pixelwise = sigmoid_focal_loss(logits, targets, gamma=self.focal_gamma, reduction='none')
            # Apply class weights
            focal_loss_pixelwise = focal_loss_pixelwise * self.class_weights.view(1, -1, 1, 1)
            # Compute per-class focal loss
            class_focal_loss = focal_loss_pixelwise.mean(dim=(0, 2, 3))
            # Reduce to tensor for combination
            focal_loss = focal_loss_pixelwise.mean()
            losses['focal'] = focal_loss
            losses['class_focal'] = class_focal_loss
            
            # Add to total loss tensor (not scalar) for backpropagation
            if self.use_dice_loss:
                # Scale by weight if we're combining losses
                total_loss = total_loss + ((1.0 - self.dice_weight) * focal_loss)
            else:
                total_loss = total_loss + focal_loss
        
        if self.use_dice_loss:
            # Compute dice loss with increased smoothing factor
            self.dice_smooth = 10.0  # Increase from 1.0 to 100.0 to prevent division by small numbers
            
            probs = torch.sigmoid(logits)
            numerator = 2 * (probs * targets).sum(dim=(2, 3))
            denominator = (probs + targets).sum(dim=(2, 3)) + self.dice_smooth
            dice_loss_per_class = 1 - (numerator / denominator)
            
            # Check for NaNs in dice loss
            if torch.isnan(dice_loss_per_class).any():
                print("WARNING: NaN detected in dice loss calculation")
                dice_loss_per_class = torch.nan_to_num(dice_loss_per_class, nan=0.5, posinf=1.0, neginf=0.0)
            
            # Apply class weights
            weighted_dice_loss = dice_loss_per_class * self.class_weights
            # Get scalar-valued losses for logging
            class_dice_loss = weighted_dice_loss.mean(dim=0)
            dice_loss = weighted_dice_loss.mean()
            
            # Store for logging
            losses['dice'] = dice_loss
            losses['class_dice'] = class_dice_loss
            
            # Add to total loss tensor for backpropagation
            if self.use_focal_loss:
                # Scale by weight if we're combining losses
                total_loss = total_loss + (self.dice_weight * dice_loss)
            else:
                total_loss = total_loss + dice_loss
        
        # Final NaN check on total loss
        if torch.isnan(total_loss):
            print("WARNING: Total loss is NaN, replacing with small positive value")
            total_loss = torch.tensor(0.1, device=logits.device, requires_grad=True)
        
        # Store total_loss for logging
        losses['total_loss'] = total_loss
        
        return losses
    
    def forward(
        self,
        x: torch.Tensor,
        skips: Dict[str, torch.Tensor],
        targets: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass of segmentation head
        Args:
            x: Input tensor from fusion [B, C, H, W]
            skips: Dictionary of skip features from image BEV projection
                  Must contain: 'stage1' only
            targets: Optional target tensor for training [B, H, W] or [B, C, H, W]
        Returns:
            During training: Dict containing losses with tensor values for backpropagation
            During inference: Predicted segmentation probabilities
        """
        # Validate required skip connection keys - only need stage1 now
        if 'stage1' not in skips:
            raise ValueError("Missing required skip connection: 'stage1'")
        
        # Convert input to float32 for consistent processing
        x = x.float()
        skips = {k: v.float() for k, v in skips.items()}
        
        # Get the stage1 skip connection
        skip1 = skips['stage1']
        
        # Apply bottleneck
        x = self.bottleneck(x)
        
        # Apply first upsampling (no skip yet)
        x = self.upsample1(x)
        
        # Apply second upsampling
        x = self.upsample2(x)
        
        # Ensure skip connection matches upsampled feature size
        if x.shape[-2:] != skip1.shape[-2:]:
            skip1 = F.interpolate(skip1, size=x.shape[-2:], mode='bilinear', align_corners=True)
        
        # Concatenate with stage1 skip connection and apply fusion
        x = torch.cat([x, skip1], dim=1)
        x = self.skip_fusion(x)
        
        # Apply final upsampling
        x = self.final_upsample(x)
        
        # Final processing
        x = self.dropout(x)
        logits = self.final_conv(x)
        
        # Ensure output size matches BEV lifter's output size
        if logits.shape[-2:] != self.output_size:
            logits = F.interpolate(logits, size=self.output_size, mode='bilinear', align_corners=True)
        
        if targets is not None:
            # Convert targets to float32
            targets = targets.float()
            
            # Convert single-channel targets to one-hot encoding if needed
            if targets.dim() == 3:  # [B, H, W]
                targets = F.one_hot(targets.long(), num_classes=self.num_classes).permute(0, 3, 1, 2).float()
            
            # Apply label smoothing if enabled
            if self.label_smoothing > 0:
                targets = targets * (1 - self.label_smoothing) + self.label_smoothing / self.num_classes
            
            # Compute losses using tensor operations to ensure proper backpropagation
            return self._compute_losses(logits, targets)
        
        # During inference, return probabilities
        return torch.sigmoid(logits)
