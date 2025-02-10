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
        initial_alpha: float = 0.25
    ) -> None:
        super().__init__()
        
        self.num_classes = num_classes
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
        
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
        
        # Optimized feature processing
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
            nn.Dropout2d(dropout)
        )
        
        # Class prediction (keep in fp32 for stability)
        self.classifier = nn.Conv2d(hidden_channels, num_classes, 1)
        
        # Initialize weights
        self._init_weights()
        
        # Debug counters
        self.register_buffer('running_mean_preds', 
                           torch.zeros(num_classes))
        self.register_buffer('running_count', torch.tensor(0))
    
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
    
    @torch.cuda.amp.autocast()
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
        
        if self.training and targets is not None:
            losses = {}
            
            # Debug info
            with torch.no_grad():
                batch_means = logits.sigmoid().mean(dim=(0, 2, 3))
                for i in range(self.num_classes):
                    losses[f'class_{i}_mean'] = batch_means[i].item()
                    losses[f'class_{i}_running_mean'] = self.running_mean_preds[i].item()
            
            # Compute loss for each class
            total_loss = 0
            for i in range(self.num_classes):
                if self.use_focal_loss:
                    loss = sigmoid_focal_loss(
                        logits[:, i:i+1],
                        targets[:, i:i+1],
                        alpha=self.focal_alpha[i].item(),
                        gamma=self.focal_gamma
                    )
                else:
                    loss = F.binary_cross_entropy_with_logits(
                        logits[:, i:i+1],
                        targets[:, i:i+1],
                        reduction='mean'
                    )
                losses[f'class_{i}_loss'] = loss.item()  # Store float for logging
                total_loss += loss
            
            losses['total_loss'] = total_loss
            return losses
        
        else:
            return torch.sigmoid(logits)  # [B, num_classes, H, W]
    
    def get_debug_info(self) -> Dict[str, float]:
        """Get debug information about class predictions"""
        return {
            f'class_{i}_running_mean': self.running_mean_preds[i].item()
            for i in range(self.num_classes)
        }
