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
        print(f"SegmentationMetrics.update called with:")
        print(f"  predictions shape: {predictions.shape}")
        print(f"  targets shape: {targets.shape}")
        print(f"  predictions type: {predictions.dtype}")
        print(f"  targets type: {targets.dtype}")
        print(f"  num_classes: {self.num_classes}")
        
        # Handle case where targets are class indices [B, H, W] instead of one-hot [B, C, H, W]
        if targets.dim() == 3 and predictions.dim() == 4:
            print("  Case: targets are class indices [B, H, W]")
            # Convert class indices to one-hot encoding
            B, H, W = targets.shape
            C = predictions.shape[1]
            
            # Check class distribution in targets
            target_counts = []
            for c in range(self.num_classes):
                count = (targets == c).sum().item()
                target_counts.append(count)
                print(f"  Class {c} count in targets: {count} ({count/(B*H*W)*100:.2f}%)")
            
            # Get predicted class indices
            pred_classes = predictions.argmax(dim=1)  # [B, H, W]
            print(f"  pred_classes shape: {pred_classes.shape}")
            print(f"  pred_classes min/max: {pred_classes.min().item()}/{pred_classes.max().item()}")
            
            # Check class distribution in predictions
            pred_counts = []
            for c in range(self.num_classes):
                count = (pred_classes == c).sum().item()
                pred_counts.append(count)
                print(f"  Class {c} count in predictions: {count} ({count/(B*H*W)*100:.2f}%)")
            
            # Flatten both tensors for confusion matrix calculation
            pred_flat = pred_classes.reshape(-1).cpu()
            target_flat = targets.reshape(-1).cpu()
            print(f"  pred_flat shape: {pred_flat.shape}")
            print(f"  target_flat shape: {target_flat.shape}")
            print(f"  pred_flat min/max: {pred_flat.min().item()}/{pred_flat.max().item()}")
            print(f"  target_flat min/max: {target_flat.min().item()}/{target_flat.max().item()}")
            
            # Move confusion matrix to CPU for calculation
            conf_matrix = self.confusion_matrix.cpu()
            
            # Create a mask for valid target indices (in case there are out-of-range values)
            valid_mask = (target_flat >= 0) & (target_flat < self.num_classes)
            print(f"  valid_mask sum: {valid_mask.sum().item()}/{len(valid_mask)}")
            pred_flat = pred_flat[valid_mask]
            target_flat = target_flat[valid_mask]
            
            # Update confusion matrix using bincount for efficiency
            if len(target_flat) > 0:  # Only proceed if we have valid targets
                try:
                    # Instead of using bincount, directly update the confusion matrix
                    for i in range(len(target_flat)):
                        t = target_flat[i].item()
                        p = pred_flat[i].item()
                        conf_matrix[t, p] += 1
                    
                    print("  Confusion matrix updated successfully")
                except Exception as e:
                    print(f"  Error in confusion matrix calculation: {str(e)}")
                    print(f"  Error type: {type(e).__name__}")
                    import traceback
                    traceback.print_exc()
            
            # Move back to device
            self.confusion_matrix = conf_matrix.to(self.device)
            
            # Debug: Print confusion matrix sum
            print(f"  Confusion matrix sum: {self.confusion_matrix.sum().item()}")
            print(f"  Confusion matrix diagonal sum: {torch.diag(self.confusion_matrix).sum().item()}")
        else:
            print("  Case: targets are one-hot encoded [B, C, H, W]")
            # For one-hot encoded targets
            if predictions.shape != targets.shape:
                raise ValueError(
                    f"Shape mismatch: predictions {predictions.shape}, "
                    f"targets {targets.shape}"
                )
                
            # Threshold predictions for binary case
            preds = (predictions > threshold).long()
            targets = targets.long()
            
            # Move to CPU for calculation
            preds_cpu = preds.cpu()
            targets_cpu = targets.cpu()
            conf_matrix = self.confusion_matrix.cpu()
            
            # Update confusion matrix
            for i in range(self.num_classes):
                pred_i = preds_cpu[:, i].reshape(-1)
                target_i = targets_cpu[:, i].reshape(-1)
                
                # Binary confusion matrix for this class
                tp = (pred_i * target_i).sum().item()
                fp = (pred_i * (1 - target_i)).sum().item()
                fn = ((1 - pred_i) * target_i).sum().item()
                tn = ((1 - pred_i) * (1 - target_i)).sum().item()
                
                conf_matrix[i, i] += tp
                conf_matrix[i, 1-i] += fp
                conf_matrix[1-i, i] += fn
                conf_matrix[1-i, 1-i] += tn
            
            # Move back to device
            self.confusion_matrix = conf_matrix.to(self.device)
            
            # Debug: Print confusion matrix sum
            print(f"  Confusion matrix sum: {self.confusion_matrix.sum().item()}")
            print(f"  Confusion matrix diagonal sum: {torch.diag(self.confusion_matrix).sum().item()}")
        
        print("  SegmentationMetrics.update completed")
    
    def compute_iou(self) -> Tuple[torch.Tensor, float]:
        """
        Compute IoU for each class and mean IoU
        Returns:
            per_class_iou: IoU for each class
            mean_iou: Mean IoU across all classes
        """
        # Debug prints
        print(f"Confusion matrix:\n{self.confusion_matrix}")
        
        tp = torch.diag(self.confusion_matrix)
        fp = self.confusion_matrix.sum(dim=0) - tp
        fn = self.confusion_matrix.sum(dim=1) - tp
        
        # Debug prints
        print(f"True positives: {tp}")
        print(f"False positives: {fp}")
        print(f"False negatives: {fn}")
        
        # Compute IoU per class
        iou = tp / (tp + fp + fn + 1e-7)  # Add epsilon to avoid division by zero
        
        # Debug prints
        print(f"IoU per class: {iou}")
        
        mean_iou = iou.mean().item()
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

