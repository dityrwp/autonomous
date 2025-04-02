import numpy as np
import torch
import logging
from typing import Dict, Optional, Union, List, Any

class EarlyStopping:
    """Early stopping utility to prevent overfitting.
    
    This class monitors validation metrics and stops training when the model
    starts overfitting. It also provides functionality to save the best model
    based on a specified metric.
    """
    
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.001,
        mode: str = 'max',
        metric_name: str = 'mean_iou',
        save_path: Optional[str] = None,
        verbose: bool = True,
    ):
        """Initialize early stopping utility.
        
        Args:
            patience: Number of epochs with no improvement after which training will stop
            min_delta: Minimum change in the monitored metric to qualify as an improvement
            mode: 'min' or 'max' - whether to monitor for metric minimization or maximization
            metric_name: Name of the metric to monitor
            save_path: Path to save the best model checkpoint
            verbose: Whether to print logging information
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.metric_name = metric_name
        self.save_path = save_path
        self.verbose = verbose
        
        # Initialize counters and best values
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_metrics_min = np.Inf if mode == 'min' else -np.Inf
        
        # Initialize tracking 
        self.train_history = []
        self.val_history = []
        self.divergence_ratios = []
        self.best_epoch = 0
        
        # Validation function based on mode
        self.improved = self._improved_min if mode == 'min' else self._improved_max
    
    def _improved_min(self, score: float, best_score: float) -> bool:
        """Check if score improved in min mode."""
        return score < best_score - self.min_delta
    
    def _improved_max(self, score: float, best_score: float) -> bool:
        """Check if score improved in max mode."""
        return score > best_score + self.min_delta
    
    def track_metrics(
        self, 
        train_metrics: Dict[str, float], 
        val_metrics: Dict[str, float],
        model: torch.nn.Module,
        epoch: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        extra_save_info: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Track metrics and check for early stopping condition.
        
        Args:
            train_metrics: Dictionary of training metrics
            val_metrics: Dictionary of validation metrics
            model: Model to save if improved
            epoch: Current epoch
            optimizer: Optimizer to save in checkpoint
            scheduler: Learning rate scheduler to save in checkpoint
            extra_save_info: Additional information to save in checkpoint
            
        Returns:
            Whether training should stop
        """
        # Extract the monitored metric
        val_score = val_metrics.get(self.metric_name, None)
        train_score = train_metrics.get(self.metric_name, None)
        
        if val_score is None:
            logging.warning(f"Metric '{self.metric_name}' not found in validation metrics. Skipping early stopping check.")
            return False
        
        # Track history
        self.train_history.append((epoch, train_metrics))
        self.val_history.append((epoch, val_metrics))
        
        # Calculate train/val divergence (indicator of overfitting)
        if train_score is not None:
            divergence = train_score / max(val_score, 1e-10) if val_score > 0 else float('inf')
            self.divergence_ratios.append((epoch, divergence))
            
            # Log if there's significant divergence
            if divergence > 3.0 and len(self.divergence_ratios) > 2:
                divergence_increase = divergence / self.divergence_ratios[-2][1]
                if divergence_increase > 1.5:
                    logging.warning(
                        f"Potential overfitting detected: train/val ratio = {divergence:.2f} "
                        f"(increased by {divergence_increase:.2f}x)"
                    )
        
        # Check if score improved
        if self.best_score is None:
            self.best_score = val_score
            self.save_checkpoint(model, val_metrics, epoch, optimizer, scheduler, extra_save_info)
        elif self.improved(val_score, self.best_score):
            if self.verbose:
                self._log_improvement(epoch, val_score)
                
            self.best_score = val_score
            self.best_epoch = epoch
            self.counter = 0
            self.save_checkpoint(model, val_metrics, epoch, optimizer, scheduler, extra_save_info)
        else:
            self.counter += 1
            if self.verbose:
                logging.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def save_checkpoint(
        self, 
        model: torch.nn.Module,
        metrics: Dict[str, float],
        epoch: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        extra_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """Save model checkpoint."""
        if self.save_path is None:
            return
        
        if self.verbose:
            logging.info(f"Saving best model to {self.save_path}")
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'metrics': metrics,
            'best_score': self.best_score,
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        if extra_info is not None:
            checkpoint.update(extra_info)
        
        torch.save(checkpoint, self.save_path)
    
    def _log_improvement(self, epoch: int, score: float) -> None:
        """Log improvement in the monitored metric."""
        change = score - self.best_score if self.mode == 'max' else self.best_score - score
        change_pct = 100 * change / abs(self.best_score) if self.best_score != 0 else float('inf')
        
        logging.info(
            f"Epoch {epoch}: {self.metric_name} improved from {self.best_score:.6f} to {score:.6f} "
            f"(Δ {change:.6f}, {change_pct:.2f}%)"
        )
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of training history."""
        return {
            'best_epoch': self.best_epoch,
            'best_score': self.best_score,
            'patience': self.patience,
            'stopped_early': self.early_stop,
            'epochs_without_improvement': self.counter,
            'train_history': self.train_history,
            'val_history': self.val_history,
            'divergence_ratios': self.divergence_ratios,
        } 