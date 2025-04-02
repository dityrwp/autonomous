from typing import Dict, List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = -1,
    gamma: float = 2,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    Compute focal loss for binary segmentation
    Args:
        inputs: [B, C, H, W] raw logits
        targets: [B, C, H, W] binary targets
        alpha: Weighting factor for rare class
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
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


class BEVSegmentationHead(nn.Module):
    """
    Memory-efficient BEV segmentation head optimized for Jetson Orin NX.
    Features:
    - Lightweight architecture with grouped convolutions
    - FP16 support for intermediate features
    - Efficient loss computation
    - No grid transformation (assumes BEV-aligned input)
    """
    def __init__(
        self,
        in_channels: int = 128,
        hidden_channels: int = 128,
        num_classes: int = 3,
        dropout: float = 0.1,
        use_focal_loss: bool = True,
        focal_gamma: float = 2.0,
        learnable_alpha: bool = True,
        initial_alpha: float = 0.25,
        class_weights: Optional[List[float]] = None,
        use_dynamic_weighting: bool = True
    ) -> None:
        super().__init__()
        
        self.num_classes = num_classes
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
        self.dropout = dropout
        
        # Learnable class weights for focal loss
        if learnable_alpha:
            self.register_parameter(
                'focal_alpha',
                nn.Parameter(torch.full((num_classes,), initial_alpha))
            )
        else:
            self.register_buffer(
                'focal_alpha',
                torch.full((num_classes,), initial_alpha)
            )
            
        # Class weights for addressing imbalance
        if class_weights is not None:
            self.register_buffer(
                'class_weights',
                torch.tensor(class_weights, dtype=torch.float32)
            )
        else:
            # Default: inverse frequency weights (to be updated during training)
            self.register_buffer(
                'class_weights', 
                torch.ones(num_classes, dtype=torch.float32)
            )
        
        # Dynamic class weighting based on frequency
        self.use_dynamic_weighting = use_dynamic_weighting
        self.class_count = torch.zeros(num_classes)
        self.total_pixels = 0
        self.ema_factor = 0.99  # Exponential moving average factor
        
        # Optimized feature processing with more dropout for regularization
        self.features = nn.Sequential(
            # Spatial context with grouped 3x3
            nn.Conv2d(in_channels, hidden_channels, 3, 
                     padding=1, bias=False, groups=8),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            
            # Channel mixing with 1x1
            nn.Conv2d(hidden_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout * 1.5),  # Increased dropout for regularization
        )
        
        # Class prediction (keep in fp32 for stability)
        self.classifier = nn.Conv2d(hidden_channels, num_classes, 1)
        
        # Initialize weights
        self._init_weights()
        
        # Debug counters
        self.register_buffer('running_mean_preds', 
                           torch.zeros(num_classes))
        self.register_buffer('running_count', torch.tensor(0))
        
        # Track class distribution for dynamic class weight adjustment
        self.register_buffer('class_distribution', 
                           torch.zeros(num_classes))
        self.register_buffer('samples_seen', torch.tensor(0))
    
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Adjust fan-in for grouped convs
                if m.groups > 1:
                    fan_in = m.in_channels // m.groups
                else:
                    fan_in = m.in_channels
                bound = 1 / (fan_in ** 0.5)
                nn.init.uniform_(m.weight, -bound, bound)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def update_class_weights(self, targets):
        """Update class weights based on observed class frequencies"""
        if not self.use_dynamic_weighting:
            return
        
        # Count class occurrences in this batch
        batch_counts = torch.zeros(self.num_classes, device=targets.device)
        
        if targets.dim() == 4 and targets.size(1) > 1:  # One-hot encoded
            batch_counts = targets.sum(dim=(0, 2, 3))
        else:  # Class indices
            if targets.dim() == 4:
                targets = targets.squeeze(1)  # [B, 1, H, W] -> [B, H, W]
            for c in range(self.num_classes):
                batch_counts[c] = (targets == c).sum()
        
        # Update total counts with EMA
        total_pixels = batch_counts.sum()
        if self.total_pixels == 0:  # First batch
            self.class_count = batch_counts.float().to(self.class_count.device)
            self.total_pixels = total_pixels.to(self.class_count.device)
        else:
            self.class_count = self.ema_factor * self.class_count + (1 - self.ema_factor) * batch_counts.float().to(self.class_count.device)
            self.total_pixels = self.ema_factor * self.total_pixels + (1 - self.ema_factor) * total_pixels.to(self.class_count.device)
        
        # Calculate class frequencies
        class_freq = self.class_count / max(self.total_pixels, 1)
        
        # Inverse frequency weighting with smoothing to prevent extreme values
        smoothed_weights = 1.0 / (class_freq + 0.05)
        
        # Normalize weights to mean of 1.0
        smoothed_weights = smoothed_weights * (self.num_classes / smoothed_weights.sum())
        
        # Clamp weights to reasonable range
        self.class_weights = smoothed_weights.clamp(0.5, 3.0)
    
    @torch.amp.autocast(device_type='cuda')
    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        
        # Feature extraction (with FP16)
        feat = self.features(x)
        
        # Class prediction
        logits = self.classifier(feat)
        
        # Update running statistics in training
        if self.training:
            with torch.no_grad():
                probs = torch.sigmoid(logits.detach())
                self.running_mean_preds = (
                    self.running_mean_preds * self.running_count + 
                    probs.mean(dim=(0, 2, 3))
                ) / (self.running_count + 1)
                self.running_count += 1
        
        if targets is not None:
            losses = {}
            
            # Convert targets to one-hot format if they are class indices
            if targets.dim() == 3 or (targets.dim() == 4 and targets.size(1) == 1):
                # Handle both [B, H, W] and [B, 1, H, W] formats
                if targets.dim() == 4:
                    targets = targets.squeeze(1)  # [B, 1, H, W] -> [B, H, W]
                
                # Convert class indices to one-hot
                one_hot_targets = torch.zeros(
                    (targets.size(0), self.num_classes, targets.size(1), targets.size(2)),
                    dtype=torch.float32,
                    device=targets.device
                )
                for c in range(self.num_classes):
                    one_hot_targets[:, c] = (targets == c).float()
                targets = one_hot_targets
            
            # Update class weights if using dynamic weighting
            if self.use_dynamic_weighting:
                with torch.no_grad():
                    self.update_class_weights(targets)
            
            # Debug info
            with torch.no_grad():
                batch_means = logits.sigmoid().mean(dim=(0, 2, 3))
                for i in range(self.num_classes):
                    losses[f'class_{i}_mean'] = batch_means[i].item()
                    losses[f'class_{i}_running_mean'] = self.running_mean_preds[i].item()
                    losses[f'class_{i}_weight'] = self.class_weights[i].item()
            
            # Use simplified focal loss calculation to avoid negative values
            if self.use_focal_loss:
                # Base binary cross entropy loss
                bce_loss = F.binary_cross_entropy_with_logits(
                    logits, targets, reduction='none'
                )
                
                # Get probabilities for focal weight calculation
                p = torch.sigmoid(logits)
                pt = p * targets + (1 - p) * (1 - targets)
                
                # Apply focal loss scaling (avoid potential numerical issues)
                focal_weight = torch.pow((1 - pt).clamp(min=1e-7), self.focal_gamma)
                
                # Apply class weights
                class_weights = self.class_weights.to(bce_loss.device)
                class_weight_map = torch.zeros_like(targets)
                for c in range(self.num_classes):
                    class_weight_map[:, c] = class_weights[c]
                
                # Calculate alpha weights for pos/neg samples
                alpha_weights = self.focal_alpha.to(bce_loss.device)
                alpha_weight_map = torch.zeros_like(targets)
                for c in range(self.num_classes):
                    alpha_weight_map[:, c] = alpha_weights[c]
                
                # Combine all weighting factors
                weighted_loss = bce_loss * focal_weight * class_weight_map
                pos_loss = alpha_weight_map * targets * weighted_loss
                neg_loss = (1 - alpha_weight_map) * (1 - targets) * weighted_loss
                
                # Final loss with guaranteed positive values
                total_loss = (pos_loss + neg_loss).mean()
                
                # Extra safety: ensure loss is strictly positive (should never happen with correct implementation)
                if total_loss < 0:
                    print(f"WARNING: Negative loss detected before clamping: {total_loss.item()}")
                    total_loss = torch.clamp(total_loss, min=0.0)
                
                losses['total_loss'] = total_loss
                
                # Add detailed diagnostics for debugging
                with torch.no_grad():
                    losses['bce_raw'] = bce_loss.mean().item()
                    losses['focal_weight_mean'] = focal_weight.mean().item()
                    losses['focal_weight_min'] = focal_weight.min().item()
                    losses['focal_weight_max'] = focal_weight.max().item()
                    losses['pt_min'] = pt.min().item()
                    losses['pt_max'] = pt.max().item()
            else:
                # Standard CE with class weights
                class_weights = self.class_weights.to(logits.device)
                loss = F.cross_entropy(logits, targets.argmax(dim=1) if targets.dim() == 4 else targets, 
                                      weight=class_weights)
                losses['total_loss'] = loss
            
            return losses
        
        else:
            return torch.sigmoid(logits)  # [B, num_classes, H, W]
    
    def get_debug_info(self) -> Dict[str, float]:
        """Get debug information about class predictions"""
        return {
            f'class_{i}_running_mean': self.running_mean_preds[i].item()
            for i in range(self.num_classes)
        }
