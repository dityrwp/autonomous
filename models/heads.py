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
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
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
            Processed feature tensor at 2x spatial resolution
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
                                      # Must match keys from DepthAugmentedBEVLifter
                                      # Expected: {'stage1': C1, 'stage2': C2, 'stage3': C3}
        decoder_channels: Tuple[int, int, int] = (128, 64, 32),
        num_classes: int = 6,
        dropout: float = 0.1,
        use_focal_loss: bool = True,
        focal_gamma: float = 1.5,
        class_weights: Optional[List[float]] = None,
        use_dice_loss: bool = True,
        dice_weight: float = 0.5,
        dice_smooth: float = 1.0,
        label_smoothing: float = 0.05
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
        
        # Validate skip stage keys
        expected_stages = {'stage1', 'stage2', 'stage3'}
        if not all(stage in skip_channels for stage in expected_stages):
            raise ValueError(f"skip_channels must contain all stages: {expected_stages}")
        
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
        skip1_ch = self.skip_channels['stage1']  # Early features (highest resolution)
        skip2_ch = self.skip_channels['stage2']  # Mid-level features
        skip3_ch = self.skip_channels['stage3']  # Deep features
        dec_ch1, dec_ch2, dec_ch3 = decoder_channels
        
        # Bottleneck conv to reduce channels before decoder
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(inplace=True)
        )
        
        # Decoder blocks with skip connections (bottom-up path)
        self.dec_block1 = DecoderBlock(in_channels//2, skip3_ch, dec_ch1)  # Deep features
        self.dec_block2 = DecoderBlock(dec_ch1, skip2_ch, dec_ch2)         # Mid-level
        self.dec_block3 = DecoderBlock(dec_ch2, skip1_ch, dec_ch3)         # Fine details
        
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
                  Must contain keys: 'stage1', 'stage2', 'stage3'
            targets: Optional target tensor for training
        Returns:
            During training: Dict containing losses
            During inference: Predicted segmentation probabilities
        """
        # Validate skip connection keys
        for stage in ['stage1', 'stage2', 'stage3']:
            if stage not in skips:
                raise ValueError(f"Missing required skip connection: {stage}")
        
        # Decoder path (bottom-up)
        x = self.bottleneck(x)
        x = self.dec_block1(x, skips['stage3'])  # 2x upsampling
        x = self.dec_block2(x, skips['stage2'])  # 4x upsampling
        x = self.dec_block3(x, skips['stage1'])  # 8x upsampling (final resolution)
        
        x = self.dropout(x)
        logits = self.final_conv(x)
        
        if targets is not None:
            return self._compute_loss(logits, targets)
        
        return torch.sigmoid(logits)
    
    def _compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute the combined loss with class weighting."""
        losses = {}
        
        # Convert targets to one-hot if needed
        if targets.dim() == 3 or (targets.dim() == 4 and targets.size(1) == 1):
            targets = targets.squeeze(1) if targets.dim() == 4 else targets
            targets_one_hot = F.one_hot(targets, num_classes=self.num_classes)
            targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
        else:
            targets_one_hot = targets
        
        # Apply label smoothing if enabled
        if self.label_smoothing > 0:
            targets_one_hot = targets_one_hot * (1 - self.label_smoothing) + \
                            self.label_smoothing / self.num_classes
        
        # Get probabilities
        probs = torch.sigmoid(logits)
        
        # Focal Loss (with external class weighting)
        if self.use_focal_loss:
            focal_loss = sigmoid_focal_loss(
                logits, targets_one_hot,
                gamma=self.focal_gamma,
                reduction='none'
            )
            # Apply class weights after focal loss calculation
            focal_loss = focal_loss * self.class_weights.view(1, -1, 1, 1)
            focal_loss = focal_loss.mean()
            losses['focal_loss'] = focal_loss
        
        # Dice Loss
        if self.use_dice_loss:
            dice_loss = self._dice_loss(probs, targets_one_hot)
            losses['dice_loss'] = dice_loss
        
        # Combine losses
        total_loss = 0.0
        if self.use_focal_loss:
            total_loss += (1 - self.dice_weight) * focal_loss
        if self.use_dice_loss:
            total_loss += self.dice_weight * dice_loss
        
        losses['total_loss'] = total_loss
        return losses
    
    def _dice_loss(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Dice loss with class weights."""
        B, C = probs.shape[:2]
        
        # Flatten predictions and targets
        probs = probs.view(B, C, -1)
        targets = targets.view(B, C, -1)
        
        # Compute intersection and cardinalities
        intersection = (probs * targets).sum(dim=2)
        cardinality = probs.sum(dim=2) + targets.sum(dim=2)
        
        # Compute Dice coefficient with smoothing
        dice = (2. * intersection + self.dice_smooth) / (cardinality + self.dice_smooth)
        
        # Apply class weights and compute mean
        weighted_dice = dice * self.class_weights.view(1, -1)
        return 1. - weighted_dice.mean()
