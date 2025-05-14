from typing import Dict, List, Optional, Union
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
        Focal loss value (pixel-wise if reduction='none')
    """
    inputs = inputs.float()
    targets = targets.float()
    
    # Compute sigmoid probabilities
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    
    # Calculate pt for focal weight
    p_t = p * targets + (1 - p) * (1 - targets)
    p_t = torch.clamp(p_t, min=1e-7, max=1 - 1e-7)
    
    # Compute focal weights (core focal term)
    focal_weight = ((1.0 - p_t) ** gamma).clamp(min=1e-7)
    loss = focal_weight * ce_loss

    # Ensure loss is non-negative
    loss = torch.clamp(loss, min=0.0)

    # Apply reduction if specified (otherwise return pixel-wise loss)
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


class BEVSegmentationHead(nn.Module):
    """
    Memory-efficient BEV segmentation head optimized for Jetson Orin NX.
    Uses static class weights for Focal and Dice loss.
    """
    def __init__(
        self,
        in_channels: int = 128,
        hidden_channels: int = 128,
        num_classes: int = 6,
        dropout: float = 0.1,
        use_focal_loss: bool = True,
        focal_gamma: float = 2.0,
        class_weights: List[float] = [0.75, 0.5, 5.0, 5.0, 2.0, 1.0],
        label_smoothing: float = 0.05,
        use_dice_loss: bool = True,
        dice_weight: float = 0.5,
        dice_smooth: float = 1.0
    ) -> None:
        super().__init__()
        
        self.num_classes = num_classes
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        
        # Dice loss parameters
        self.use_dice_loss = use_dice_loss
        self.dice_weight = dice_weight
        self.dice_smooth = dice_smooth
        
        # Validate and register static class weights
        if len(class_weights) != num_classes:
            raise ValueError(f"Length of class_weights ({len(class_weights)}) must match num_classes ({num_classes})")
        self.register_buffer(
            'class_weights',
            torch.tensor(class_weights, dtype=torch.float32)
        )
            
        # Optimized feature processing
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, 
                     padding=1, bias=False, groups=8),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout), 
        )
        
        # Class prediction
        self.classifier = nn.Conv2d(hidden_channels, num_classes, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.groups > 1:
                    fan_in = m.in_channels // m.groups
                else:
                    fan_in = m.in_channels
                bound = 1 / (fan_in ** 0.5)
                nn.init.uniform_(m.weight, -bound, bound)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _dice_loss(self, probs, targets):
        """
        Compute Dice loss using static class weights.
        """
        # Reshape tensors
        B, C, H, W = probs.shape
        probs = probs.view(B, C, -1)
        targets = targets.view(B, C, -1)
        
        # Compute intersection and cardinalities
        intersection = (probs * targets).sum(dim=2)
        cardinality = probs.sum(dim=2) + targets.sum(dim=2)
        
        # Compute Dice score
        dice = (2.0 * intersection + self.dice_smooth) / (cardinality + self.dice_smooth)
        
        # Compute Dice loss
        dice_loss = 1.0 - dice  # [B, C]
        
        # Apply STATIC class weights
        weighted_dice_loss = dice_loss * self.class_weights.to(dice_loss.device)
        
        # Return mean across batches and classes
        return weighted_dice_loss.mean()

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        
        # Feature extraction (with FP16)
        feat = self.features(x.float()) # feat might be float16 if autocast is active

        # 2. Classification
        logits = self.classifier(feat.float())
        if targets is not None:
            losses = {}
            
            # Convert targets to one-hot format if needed
            if targets.dim() == 3 or (targets.dim() == 4 and targets.size(1) == 1):
                if targets.dim() == 4:
                    targets = targets.squeeze(1)
                one_hot_targets = F.one_hot(targets.long(), num_classes=self.num_classes).permute(0, 3, 1, 2).float()
                targets = one_hot_targets
            
            # Ensure targets are float for calculations
            targets = targets.float()
            
            # Apply label smoothing if enabled
            if self.label_smoothing > 0 and self.training:
                targets = targets * (1 - self.label_smoothing) + self.label_smoothing / self.num_classes
            
            # Calculate losses using static weights
            focal_loss = torch.tensor(0.0, device=logits.device)
            dice_loss = torch.tensor(0.0, device=logits.device)
            
            # --- Calculate Focal Loss (if enabled) ---
            if self.use_focal_loss:
                # Compute pixel-wise focal loss (reduction='none')
                focal_loss_pixelwise = sigmoid_focal_loss(
                    logits, targets, gamma=self.focal_gamma, reduction='none'
                )
                
                # Apply STATIC class weights
                weights = self.class_weights.to(focal_loss_pixelwise.device).view(1, -1, 1, 1)
                weighted_focal_loss = focal_loss_pixelwise * weights
                
                # Mean reduction
                focal_loss = weighted_focal_loss.mean()
                focal_loss = torch.clamp(focal_loss, min=0.0)
                losses['focal_loss'] = focal_loss.item()
            
            # --- Calculate Dice Loss (if enabled) ---
            if self.use_dice_loss:
                p = torch.sigmoid(logits)
                p = torch.clamp(p, min=1e-6, max=1-1e-6)
                dice_loss = self._dice_loss(p, targets)
                losses['dice_loss'] = dice_loss.item()
            
            # --- Combine losses ---
            if self.use_focal_loss and self.use_dice_loss:
                total_loss = (1.0 - self.dice_weight) * focal_loss + self.dice_weight * dice_loss
            elif self.use_dice_loss:
                total_loss = dice_loss
            elif self.use_focal_loss:
                total_loss = focal_loss
            else:
                # Fallback to standard weighted CE
                class_weights = self.class_weights.to(logits.device)
                target_indices = targets.argmax(dim=1) if targets.dim() == 4 else targets.long()
                total_loss = F.cross_entropy(logits, target_indices, weight=class_weights)
                print("Warning: No primary loss (Focal/Dice) enabled. Using CrossEntropy.")
            
            losses['total_loss'] = total_loss
            
            return losses
        
        else:
            # Inference: return probabilities
            return torch.sigmoid(logits)