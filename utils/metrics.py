import torch
import numpy as np
from typing import Dict, Optional, Tuple, List
import time
import json


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
            targets: [B, C, H, W] binary ground truth or [B, H, W] class indices
            threshold: Classification threshold for predictions
        """
        # Print shapes for debugging
        print(f"  SegmentationMetrics.update input shapes: predictions {predictions.shape}, targets {targets.shape}")
        
        # Handle case where targets are class indices [B, H, W] instead of one-hot [B, C, H, W]
        if targets.dim() == 3 and predictions.dim() == 4:
            # Convert predictions to class indices if they're probabilities
            pred_classes = predictions.argmax(dim=1)  # [B, H, W]
            
            # Flatten both tensors for confusion matrix calculation
            pred_flat = pred_classes.reshape(-1)
            target_flat = targets.reshape(-1)
            
            # Create a mask for valid target indices (in case there are out-of-range values)
            valid_mask = (target_flat >= 0) & (target_flat < self.num_classes) & \
                         (pred_flat >= 0) & (pred_flat < self.num_classes)
            
            pred_flat = pred_flat[valid_mask]
            target_flat = target_flat[valid_mask]
            
            # Skip update if no valid pixels
            if len(target_flat) == 0:
                print("  No valid pixels found for confusion matrix update")
                return
                
            # Create a temporary confusion matrix for this batch
            batch_cm = torch.zeros((self.num_classes, self.num_classes), 
                                  device=self.device, dtype=torch.float32)
            
            # Update confusion matrix using bincount for efficiency
            for t_idx in range(self.num_classes):
                # Get predictions where the target is this class
                t_mask = (target_flat == t_idx)
                if not t_mask.any():
                    continue
                    
                # Count occurrences of each predicted class for this target class
                pred_for_class = pred_flat[t_mask]
                for p_idx in range(self.num_classes):
                    batch_cm[t_idx, p_idx] = (pred_for_class == p_idx).sum()
            
            # Add to the global confusion matrix
            self.confusion_matrix += batch_cm
            
        else:
            # For one-hot encoded targets
            if predictions.shape != targets.shape:
                raise ValueError(
                    f"Shape mismatch: predictions {predictions.shape}, "
                    f"targets {targets.shape}"
                )
                
            # Threshold predictions for binary case
            preds = (predictions > threshold).float()
            
            # Move to CPU for calculation
            B, C, H, W = predictions.shape
            
            # Create a temporary confusion matrix for this batch
            batch_cm = torch.zeros((self.num_classes, self.num_classes), 
                                  device=self.device, dtype=torch.float32)
            
            # Update confusion matrix for each class
            for i in range(self.num_classes):
                pred_i = preds[:, i].reshape(-1)  # Predictions for class i
                target_i = targets[:, i].reshape(-1)  # Targets for class i
                
                # True positives: predicted class i and actual class i
                tp = (pred_i * target_i).sum()
                batch_cm[i, i] = tp
                
                # False positives: predicted class i but not actual class i
                # These are distributed across other classes based on actual class
                for j in range(self.num_classes):
                    if j != i:
                        # Predicted class i but actual class j
                        fp_ij = (pred_i * targets[:, j].reshape(-1)).sum()
                        batch_cm[j, i] = fp_ij
            
            # Add to the global confusion matrix
            self.confusion_matrix += batch_cm
        
        print("  SegmentationMetrics.update completed")
    
    def compute(self):
        """
        Compute metrics from confusion matrix
        Returns:
            dict: Dictionary of metrics
        """
        print("Computing metrics from confusion matrix")
        print(f"Confusion matrix shape: {self.confusion_matrix.shape}")
        print(f"Confusion matrix sum: {self.confusion_matrix.sum().item()}")
        
        # Ensure we have data to compute metrics
        if self.confusion_matrix.sum() == 0:
            print("Warning: Confusion matrix is empty, returning zero metrics")
            return {
                "iou": torch.zeros(self.num_classes, device=self.device),
                "dice": torch.zeros(self.num_classes, device=self.device),
                "accuracy": torch.zeros(1, device=self.device),
                "precision": torch.zeros(self.num_classes, device=self.device),
                "recall": torch.zeros(self.num_classes, device=self.device),
            }
        
        # Extract values from confusion matrix
        tp = torch.diag(self.confusion_matrix)  # True positives for each class
        
        # Calculate class-wise metrics
        # Sum over columns (axis=1) gives all predictions of this class (TP + FP)
        # Sum over rows (axis=0) gives all targets of this class (TP + FN)
        pred_sum = self.confusion_matrix.sum(dim=0)  # TP + FP (sum over rows)
        target_sum = self.confusion_matrix.sum(dim=1)  # TP + FN (sum over columns)
        
        # IoU = TP / (TP + FP + FN)
        # Add small epsilon to avoid division by zero
        epsilon = 1e-6
        union = pred_sum + target_sum - tp + epsilon
        iou = tp / union
        
        # Clamp IoU values to be between 0 and 1
        iou = torch.clamp(iou, min=0.0, max=1.0)
        
        # Dice = 2*TP / (2*TP + FP + FN) = 2*TP / (TP + FP + TP + FN)
        dice_denominator = pred_sum + target_sum + epsilon
        dice = 2 * tp / dice_denominator
        dice = torch.clamp(dice, min=0.0, max=1.0)
        
        # Precision = TP / (TP + FP)
        precision = tp / (pred_sum + epsilon)
        precision = torch.clamp(precision, min=0.0, max=1.0)
        
        # Recall = TP / (TP + FN)
        recall = tp / (target_sum + epsilon)
        recall = torch.clamp(recall, min=0.0, max=1.0)
        
        # Accuracy = (TP + TN) / (TP + TN + FP + FN)
        # For multi-class, we use the sum of diagonal (all TPs) divided by total sum
        accuracy = tp.sum() / (self.confusion_matrix.sum() + epsilon)
        accuracy = torch.clamp(accuracy, min=0.0, max=1.0)
        
        # Print metrics for debugging
        print(f"IoU: {iou}")
        print(f"Dice: {dice}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"Accuracy: {accuracy}")
        
        return {
            "iou": iou,
            "dice": dice,
            "accuracy": accuracy.unsqueeze(0),
            "precision": precision,
            "recall": recall,
        }
    
    def compute_iou(self) -> Tuple[torch.Tensor, float]:
        """
        Compute IoU for each class and mean IoU
        Returns:
            per_class_iou: IoU for each class
            mean_iou: Mean IoU across all classes
        """
        # Debug prints
        print(f"Confusion matrix:\n{self.confusion_matrix}")
        
        # Extract values from confusion matrix
        tp = torch.diag(self.confusion_matrix)  # True positives (diagonal elements)
        
        # Ensure all values are non-negative
        tp = torch.clamp(tp, min=0.0)
        
        # Calculate false positives and false negatives
        fp = torch.clamp(self.confusion_matrix.sum(dim=0) - tp, min=0.0)  # Column sum - TP
        fn = torch.clamp(self.confusion_matrix.sum(dim=1) - tp, min=0.0)  # Row sum - TP
        
        # Debug prints
        print(f"True positives: {tp}")
        print(f"False positives: {fp}")
        print(f"False negatives: {fn}")
        
        # Compute IoU per class: TP / (TP + FP + FN)
        # Add epsilon to avoid division by zero
        denominator = tp + fp + fn + 1e-7
        iou = tp / denominator
        
        # Ensure IoU values are between 0 and 1
        iou = torch.clamp(iou, min=0.0, max=1.0)
        
        # Debug prints
        print(f"IoU per class: {iou}")
        
        # Calculate mean IoU, ignoring NaN values
        valid_iou = iou[~torch.isnan(iou)]
        if len(valid_iou) > 0:
            mean_iou = valid_iou.mean().item()
        else:
            mean_iou = 0.0
            
        print(f"Mean IoU: {mean_iou}")
        
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
    - Learning rate history
    - Performance metrics (throughput, batch time)
    - Confusion matrix tracking
    """
    def __init__(self):
        self.current_metrics = {}
        self.best_metrics = {}
        self.history = []
        self.running_loss = 0.0
        self.running_count = 0
        
        # Learning rate history for each parameter group
        self.lr_history = {}
        
        # Performance metrics
        self.batch_times = []
        self.throughputs = []
        self.gpu_memory = []
        
        # Epoch timestamps for ETA calculation
        self.epoch_timestamps = []
        
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
    
    def update_lr(self, optimizer):
        """Update learning rate history from optimizer"""
        for i, param_group in enumerate(optimizer.param_groups):
            if i not in self.lr_history:
                self.lr_history[i] = []
            self.lr_history[i].append(param_group['lr'])
    
    def update_performance(self, batch_time: float, throughput: float, gpu_memory: float = None):
        """Update performance metrics"""
        self.batch_times.append(batch_time)
        self.throughputs.append(throughput)
        if gpu_memory is not None:
            self.gpu_memory.append(gpu_memory)
    
    def get_running_loss(self) -> float:
        """Get current running loss average"""
        if self.running_count == 0:
            return 0.0
        return self.running_loss / self.running_count
    
    def get_lr_history(self, group_idx: int = 0) -> List[float]:
        """Get learning rate history for a parameter group"""
        return self.lr_history.get(group_idx, [])
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        if not self.batch_times:
            return {}
            
        return {
            'avg_batch_time': sum(self.batch_times) / len(self.batch_times),
            'avg_throughput': sum(self.throughputs) / len(self.throughputs),
            'avg_gpu_memory': sum(self.gpu_memory) / len(self.gpu_memory) if self.gpu_memory else 0,
            'min_batch_time': min(self.batch_times),
            'max_batch_time': max(self.batch_times),
            'min_throughput': min(self.throughputs),
            'max_throughput': max(self.throughputs)
        }
    
    def estimate_eta(self, current_epoch: int, total_epochs: int) -> float:
        """Estimate time remaining based on epoch timestamps"""
        if len(self.epoch_timestamps) < 2:
            return float('inf')
            
        # Calculate average time per epoch
        epoch_times = [self.epoch_timestamps[i+1] - self.epoch_timestamps[i] 
                      for i in range(len(self.epoch_timestamps)-1)]
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        
        # Estimate remaining time
        remaining_epochs = total_epochs - current_epoch
        return avg_epoch_time * remaining_epochs
    
    def epoch_end(self, optimizer=None):
        """
        End of epoch processing:
        - Save current metrics to history
        - Reset running loss
        - Update learning rate history
        - Record timestamp
        """
        metrics = {
            'loss': self.get_running_loss(),
            **self.current_metrics
        }
        
        # Add performance metrics
        if self.batch_times:
            perf_stats = self.get_performance_stats()
            metrics.update({
                'avg_batch_time': perf_stats['avg_batch_time'],
                'avg_throughput': perf_stats['avg_throughput']
            })
        
        self.history.append(metrics)
        
        # Update learning rate history if optimizer provided
        if optimizer:
            self.update_lr(optimizer)
        
        # Record timestamp for ETA calculation
        self.epoch_timestamps.append(time.time())
        
        # Reset running values
        self.running_loss = 0.0
        self.running_count = 0
        self.current_metrics = {}
        self.batch_times = []
        self.throughputs = []
        self.gpu_memory = []
        
    def get_current_metrics(self) -> Dict[str, float]:
        """Get all current metrics including loss"""
        metrics = {
            'loss': self.get_running_loss(),
            **self.current_metrics
        }
        
        # Add performance metrics if available
        if self.batch_times:
            perf_stats = self.get_performance_stats()
            metrics.update({
                'avg_batch_time': perf_stats['avg_batch_time'],
                'avg_throughput': perf_stats['avg_throughput']
            })
            
        return metrics
    
    def get_best_metrics(self) -> Dict[str, float]:
        """Get best values for each metric"""
        return self.best_metrics.copy()
    
    def get_metric_history(self, metric_name: str) -> List[float]:
        """Get history of a specific metric across epochs"""
        return [epoch_metrics.get(metric_name, float('nan')) 
                for epoch_metrics in self.history]
    
    def save_to_file(self, filepath: str):
        """Save metrics history to a JSON file"""
        data = {
            'history': self.history,
            'best_metrics': self.best_metrics,
            'lr_history': self.lr_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_from_file(self, filepath: str):
        """Load metrics history from a JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        self.history = data.get('history', [])
        self.best_metrics = data.get('best_metrics', {})
        self.lr_history = {int(k): v for k, v in data.get('lr_history', {}).items()}

