import torch
import numpy as np
from typing import Dict, Optional, Tuple


class SegmentationMetrics:
    """
    Compute and accumulate segmentation metrics for BEV predictions.
    Handles binary and multi-class segmentation with support for:
    - Per-class IoU
    - Mean IoU
    - Precision and Recall per class
    - Confusion Matrix
    """
    def __init__(self, num_classes: int, device: str = 'cuda'):
        self.num_classes = num_classes
        self.device = device
        self.reset()
        
    def reset(self):
        """Reset accumulated statistics"""
        self.confusion_matrix = torch.zeros(
            (self.num_classes, self.num_classes),
            device=self.device
        )
        
    @torch.no_grad()
    def update(self, 
               predictions: torch.Tensor,
               targets: torch.Tensor,
               threshold: float = 0.5):
        """
        Update metrics with new predictions
        Args:
            predictions: [B, C, H, W] prediction probabilities
            targets: [B, C, H, W] binary ground truth
            threshold: Classification threshold for predictions
        """
        if predictions.shape != targets.shape:
            raise ValueError(
                f"Shape mismatch: predictions {predictions.shape}, "
                f"targets {targets.shape}"
            )
            
        # Threshold predictions
        preds = (predictions > threshold).long()
        targets = targets.long()
        
        # Update confusion matrix
        for i in range(self.num_classes):
            pred_i = preds[:, i].reshape(-1)
            target_i = targets[:, i].reshape(-1)
            
            # Binary confusion matrix for this class
            tp = (pred_i * target_i).sum()
            fp = (pred_i * (1 - target_i)).sum()
            fn = ((1 - pred_i) * target_i).sum()
            tn = ((1 - pred_i) * (1 - target_i)).sum()
            
            self.confusion_matrix[i, i] += tp
            self.confusion_matrix[i, 1-i] += fp
            self.confusion_matrix[1-i, i] += fn
            self.confusion_matrix[1-i, 1-i] += tn
    
    def compute_iou(self) -> Tuple[torch.Tensor, float]:
        """
        Compute IoU for each class and mean IoU
        Returns:
            per_class_iou: IoU for each class
            mean_iou: Mean IoU across all classes
        """
        tp = torch.diag(self.confusion_matrix)
        fp = self.confusion_matrix.sum(dim=0) - tp
        fn = self.confusion_matrix.sum(dim=1) - tp
        
        # Compute IoU per class
        iou = tp / (tp + fp + fn + 1e-7)  # Add epsilon to avoid division by zero
        mean_iou = iou.mean().item()
        
        return iou, mean_iou
    
    def compute_precision_recall(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute precision and recall for each class
        Returns:
            precision: Precision per class
            recall: Recall per class
        """
        tp = torch.diag(self.confusion_matrix)
        fp = self.confusion_matrix.sum(dim=0) - tp
        fn = self.confusion_matrix.sum(dim=1) - tp
        
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        
        return precision, recall
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get all computed metrics
        Returns:
            Dictionary containing:
            - IoU per class
            - Mean IoU
            - Precision per class
            - Recall per class
        """
        iou, mean_iou = self.compute_iou()
        precision, recall = self.compute_precision_recall()
        
        metrics = {
            'mean_iou': mean_iou,
        }
        
        # Add per-class metrics
        for i in range(self.num_classes):
            metrics[f'class_{i}_iou'] = iou[i].item()
            metrics[f'class_{i}_precision'] = precision[i].item()
            metrics[f'class_{i}_recall'] = recall[i].item()
        
        return metrics


class MetricsLogger:
    """
    Log and accumulate metrics over training/validation
    Supports:
    - Running average for loss values
    - Best metric tracking
    - Epoch-wise metric history
    """
    def __init__(self):
        self.current_metrics = {}
        self.best_metrics = {}
        self.history = []
        self.running_loss = 0.0
        self.running_count = 0
        
    def update_loss(self, loss: float):
        """Update running loss average"""
        self.running_loss += loss
        self.running_count += 1
        
    def update_metrics(self, metrics: Dict[str, float]):
        """Update current metrics"""
        self.current_metrics.update(metrics)
        
        # Update best metrics
        for key, value in metrics.items():
            if key not in self.best_metrics or value > self.best_metrics[key]:
                self.best_metrics[key] = value
    
    def get_running_loss(self) -> float:
        """Get current running loss average"""
        if self.running_count == 0:
            return 0.0
        return self.running_loss / self.running_count
    
    def epoch_end(self):
        """
        End of epoch processing:
        - Save current metrics to history
        - Reset running loss
        """
        metrics = {
            'loss': self.get_running_loss(),
            **self.current_metrics
        }
        self.history.append(metrics)
        
        # Reset running values
        self.running_loss = 0.0
        self.running_count = 0
        self.current_metrics = {}
        
    def get_current_metrics(self) -> Dict[str, float]:
        """Get all current metrics including loss"""
        return {
            'loss': self.get_running_loss(),
            **self.current_metrics
        }
    
    def get_best_metrics(self) -> Dict[str, float]:
        """Get best values for each metric"""
        return self.best_metrics.copy()

