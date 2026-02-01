"""
Enhanced metrics tracking for TwinBrain training.
Tracks loss components, gradients, and other training metrics.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


class MetricsTracker:
    """Track and log training metrics over epochs."""
    
    def __init__(self, output_dir: Optional[Path] = None, enabled: bool = True):
        """
        Initialize metrics tracker.
        
        Args:
            output_dir: Directory to save metrics
            enabled: Whether tracking is enabled
        """
        self.enabled = enabled
        self.output_dir = Path(output_dir) if output_dir else None
        self.metrics_history = defaultdict(list)
        self.current_epoch = 0
        
        if self.output_dir and self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Metrics tracking enabled, saving to {self.output_dir}")
    
    def log_epoch(self, epoch: int, metrics: Dict[str, float]):
        """
        Log metrics for an epoch.
        
        Args:
            epoch: Epoch number
            metrics: Dictionary of metric name -> value
        """
        if not self.enabled:
            return
        
        self.current_epoch = epoch
        
        # Store each metric
        for name, value in metrics.items():
            if isinstance(value, (int, float, np.number)):
                self.metrics_history[name].append({
                    'epoch': epoch,
                    'value': float(value)
                })
    
    def log_loss_components(
        self, 
        epoch: int, 
        recon_loss: float,
        temp_loss: float,
        align_loss: Optional[float] = None,
        total_loss: Optional[float] = None,
        **kwargs
    ):
        """
        Log individual loss components.
        
        Args:
            epoch: Epoch number
            recon_loss: Reconstruction loss
            temp_loss: Temporal loss
            align_loss: Alignment loss (optional)
            total_loss: Total loss (optional)
            **kwargs: Additional loss components
        """
        if not self.enabled:
            return
        
        components = {
            'loss/reconstruction': recon_loss,
            'loss/temporal': temp_loss,
        }
        
        if align_loss is not None:
            components['loss/alignment'] = align_loss
        
        if total_loss is not None:
            components['loss/total'] = total_loss
        
        # Add any additional components
        for name, value in kwargs.items():
            components[f'loss/{name}'] = value
        
        self.log_epoch(epoch, components)
    
    def log_gradient_stats(
        self,
        epoch: int,
        grad_norm: float,
        grad_max: Optional[float] = None,
        grad_min: Optional[float] = None
    ):
        """
        Log gradient statistics.
        
        Args:
            epoch: Epoch number
            grad_norm: Gradient norm
            grad_max: Maximum gradient value
            grad_min: Minimum gradient value
        """
        if not self.enabled:
            return
        
        stats = {'gradient/norm': grad_norm}
        
        if grad_max is not None:
            stats['gradient/max'] = grad_max
        if grad_min is not None:
            stats['gradient/min'] = grad_min
        
        self.log_epoch(epoch, stats)
    
    def get_metric_history(self, metric_name: str) -> List[Dict[str, Any]]:
        """
        Get history for a specific metric.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            List of {epoch, value} dictionaries
        """
        return self.metrics_history.get(metric_name, [])
    
    def get_latest_metrics(self) -> Dict[str, float]:
        """
        Get latest value for all tracked metrics.
        
        Returns:
            Dictionary of metric name -> latest value
        """
        latest = {}
        for name, history in self.metrics_history.items():
            if history:
                latest[name] = history[-1]['value']
        return latest
    
    def save_metrics(self, filename: str = "metrics_history.json"):
        """
        Save metrics history to JSON file.
        
        Args:
            filename: Output filename
        """
        if not self.enabled or not self.output_dir:
            return
        
        filepath = self.output_dir / filename
        
        try:
            with open(filepath, 'w') as f:
                json.dump(dict(self.metrics_history), f, indent=2)
            logger.info(f"Metrics saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
    
    def print_summary(self, last_n_epochs: int = 10):
        """
        Print summary of recent metrics.
        
        Args:
            last_n_epochs: Number of recent epochs to summarize
        """
        if not self.enabled:
            return
        
        logger.info("=" * 80)
        logger.info(f"Metrics Summary (Last {last_n_epochs} epochs)")
        logger.info("=" * 80)
        
        for metric_name, history in sorted(self.metrics_history.items()):
            if not history:
                continue
            
            recent = history[-last_n_epochs:]
            values = [h['value'] for h in recent]
            
            avg = np.mean(values)
            std = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)
            latest = values[-1]
            
            logger.info(
                f"{metric_name:30s} | "
                f"Latest: {latest:8.4f} | "
                f"Avg: {avg:8.4f} | "
                f"Std: {std:8.4f} | "
                f"Min: {min_val:8.4f} | "
                f"Max: {max_val:8.4f}"
            )
        
        logger.info("=" * 80)
    
    def get_best_epoch(self, metric_name: str, mode: str = 'min') -> Optional[int]:
        """
        Get epoch with best value for a metric.
        
        Args:
            metric_name: Name of the metric
            mode: 'min' or 'max'
            
        Returns:
            Epoch number with best value, or None if metric not found
        """
        history = self.metrics_history.get(metric_name, [])
        if not history:
            return None
        
        values = [h['value'] for h in history]
        
        if mode == 'min':
            best_idx = np.argmin(values)
        else:
            best_idx = np.argmax(values)
        
        return history[best_idx]['epoch']


class TrainingMonitor:
    """Monitor training progress and detect issues."""
    
    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 1e-4,
        check_interval: int = 5
    ):
        """
        Initialize training monitor.
        
        Args:
            patience: Number of checks without improvement before warning
            min_delta: Minimum change to consider as improvement
            check_interval: Check every N epochs
        """
        self.patience = patience
        self.min_delta = min_delta
        self.check_interval = check_interval
        self.best_loss = float('inf')
        self.epochs_without_improvement = 0
        self.warnings = []
    
    def check_progress(self, epoch: int, loss: float) -> Dict[str, Any]:
        """
        Check training progress and return status.
        
        Args:
            epoch: Current epoch
            loss: Current loss value
            
        Returns:
            Dictionary with status information
        """
        status = {
            'should_stop': False,
            'warnings': [],
            'improved': False
        }
        
        # Only check at intervals
        if epoch % self.check_interval != 0:
            return status
        
        # Check for improvement
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.epochs_without_improvement = 0
            status['improved'] = True
        else:
            self.epochs_without_improvement += self.check_interval
        
        # Check for stagnation
        if self.epochs_without_improvement >= self.patience:
            warning = f"No improvement for {self.epochs_without_improvement} epochs"
            status['warnings'].append(warning)
            logger.warning(warning)
        
        # Check for NaN/Inf
        if not np.isfinite(loss):
            warning = f"Loss is not finite: {loss}"
            status['warnings'].append(warning)
            status['should_stop'] = True
            logger.error(warning)
        
        return status
