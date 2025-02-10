import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import cv2


class Logger:
    """
    Comprehensive logger for training and evaluation
    Supports:
    - Console logging
    - TensorBoard logging
    - Metric tracking
    - Image saving
    """
    def __init__(
        self,
        log_dir: str,
        name: str = "train",
        use_tensorboard: bool = True,
        config: Optional[Dict] = None
    ):
        self.log_dir = Path(log_dir)
        self.name = name
        
        # Create log directory
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Setup file logging
        self.setup_file_logging()
        
        # Setup TensorBoard
        self.use_tensorboard = use_tensorboard
        if use_tensorboard:
            self.writer = SummaryWriter(str(self.log_dir / 'tensorboard'))
            if config is not None:
                self.writer.add_text('config', str(config))
        
        # Initialize metric tracking
        self.reset_metrics()
        
        # Log start time
        self.start_time = time.time()
        logging.info(f"=== Starting {name} ===")
        
    def setup_file_logging(self):
        """Setup file logging"""
        log_file = self.log_dir / f"{self.name}.log"
        
        # Configure logging format
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(str(log_file)),
                logging.StreamHandler()
            ]
        )
    
    def reset_metrics(self):
        """Reset metric tracking"""
        self.metrics = {}
        self.metric_counts = {}
    
    def update_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Update running metrics
        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number for TensorBoard
        """
        # Update running averages
        for name, value in metrics.items():
            if name not in self.metrics:
                self.metrics[name] = 0.0
                self.metric_counts[name] = 0
            
            self.metrics[name] += value
            self.metric_counts[name] += 1
        
        # Log to TensorBoard
        if self.use_tensorboard and step is not None:
            for name, value in metrics.items():
                self.writer.add_scalar(f"metrics/{name}", value, step)
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current metric averages"""
        return {
            name: self.metrics[name] / self.metric_counts[name]
            for name in self.metrics
        }
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics to console and TensorBoard
        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number for TensorBoard
        """
        # Log to console
        log_str = []
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                log_str.append(f"{name}: {value:.4f}")
        
        if log_str:
            logging.info(" | ".join(log_str))
        
        # Log to TensorBoard
        if self.use_tensorboard and step is not None:
            for name, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"metrics/{name}", value, step)
    
    def log_images(
        self,
        images: Dict[str, torch.Tensor],
        step: int,
        max_images: int = 4
    ):
        """
        Log images to TensorBoard
        Args:
            images: Dictionary of image names and tensors
            step: Step number
            max_images: Maximum number of images to log
        """
        if not self.use_tensorboard:
            return
        
        for name, image_batch in images.items():
            if len(image_batch.shape) == 4:  # BCHW format
                # Convert to numpy and clip to valid range
                image_batch = image_batch.detach().cpu().float().numpy()
                image_batch = np.clip(image_batch, 0, 1)
                
                # Log up to max_images
                for i in range(min(image_batch.shape[0], max_images)):
                    self.writer.add_image(
                        f"images/{name}_{i}",
                        image_batch[i],
                        step,
                        dataformats='CHW'
                    )
    
    def save_predictions(
        self,
        predictions: torch.Tensor,
        save_dir: str,
        prefix: str = "pred",
        step: Optional[int] = None
    ):
        """
        Save prediction visualizations
        Args:
            predictions: Prediction tensor [B, C, H, W]
            save_dir: Directory to save visualizations
            prefix: Filename prefix
            step: Optional step number for filename
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Convert predictions to numpy
        preds = predictions.detach().cpu().float().numpy()
        
        for i in range(preds.shape[0]):
            # Create visualization
            vis = np.zeros((preds.shape[2], preds.shape[3], 3), dtype=np.uint8)
            
            # Color code for each class
            colors = [
                (255, 0, 0),    # Red for class 0
                (0, 255, 0),    # Green for class 1
                (0, 0, 255)     # Blue for class 2
            ]
            
            # Add each class prediction
            for c in range(preds.shape[1]):
                mask = preds[i, c] > 0.5
                vis[mask] = colors[c]
            
            # Save visualization
            if step is not None:
                filename = f"{prefix}_{step:06d}_{i:02d}.png"
            else:
                filename = f"{prefix}_{i:02d}.png"
            
            cv2.imwrite(
                os.path.join(save_dir, filename),
                cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
            )
    
    def close(self):
        """Close logger and cleanup"""
        if self.use_tensorboard:
            self.writer.close()
        
        # Log total time
        total_time = time.time() - self.start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        
        logging.info(f"=== {self.name} completed in {hours:02d}:{minutes:02d}:{seconds:02d} ===")


class MetricTracker:
    """
    Track and compute running statistics for metrics
    Supports:
    - Running averages
    - Best value tracking
    - Smoothing
    """
    def __init__(self, smoothing: float = 0.9):
        self.smoothing = smoothing
        self.reset()
    
    def reset(self):
        """Reset all tracking"""
        self.current = {}
        self.best = {}
        self.running = {}
        self.history = []
        self.smooth = {}
    
    def update(self, metrics: Dict[str, float]):
        """
        Update metrics
        Args:
            metrics: Dictionary of metric values
        """
        self.current = metrics.copy()
        
        # Update running averages
        for name, value in metrics.items():
            if name not in self.running:
                self.running[name] = value
                self.smooth[name] = value
            else:
                self.running[name] = (
                    self.smoothing * self.running[name] +
                    (1 - self.smoothing) * value
                )
                self.smooth[name] = value
        
        # Update best values
        for name, value in metrics.items():
            if name not in self.best or value > self.best[name]:
                self.best[name] = value
        
        # Add to history
        self.history.append(metrics)
    
    def get_current(self) -> Dict[str, float]:
        """Get current metric values"""
        return self.current.copy()
    
    def get_running(self) -> Dict[str, float]:
        """Get running averages"""
        return self.running.copy()
    
    def get_best(self) -> Dict[str, float]:
        """Get best values"""
        return self.best.copy()
    
    def get_smooth(self) -> Dict[str, float]:
        """Get smoothed values"""
        return self.smooth.copy() 