"""
Stimulation Simulator
=====================

Simulate virtual stimulation and perturbations to brain regions.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass


@dataclass
class StimulationConfig:
    """Configuration for brain stimulation."""
    target_regions: List[int]  # Region IDs to stimulate
    amplitude: float = 0.5  # Stimulation amplitude
    duration: int = 10  # Duration in time steps
    pattern: str = "pulse"  # pulse, sine, ramp, continuous
    frequency: float = 10.0  # Frequency for sine pattern (Hz)
    spatial_spread: float = 5.0  # Spatial spread radius (mm)
    onset: int = 0  # Onset time step


class StimulationSimulator:
    """
    Simulate effects of virtual stimulation on brain activity.
    
    Supports different stimulation patterns:
    - Pulse: Brief activation
    - Sine: Rhythmic stimulation (e.g., tACS)
    - Ramp: Gradually increasing stimulation
    - Continuous: Sustained activation
    """
    
    def __init__(
        self,
        n_regions: int,
        region_positions: Optional[np.ndarray] = None,
        connectivity: Optional[np.ndarray] = None
    ):
        """
        Initialize stimulation simulator.
        
        Args:
            n_regions: Number of brain regions
            region_positions: [N_regions, 3] array of region coordinates
            connectivity: [N_regions, N_regions] connectivity matrix
        """
        self.n_regions = n_regions
        self.region_positions = region_positions
        self.connectivity = connectivity
    
    def apply_stimulation(
        self,
        brain_state: torch.Tensor,
        config: StimulationConfig,
        current_time: int
    ) -> torch.Tensor:
        """
        Apply stimulation to brain state at current time.
        
        Args:
            brain_state: [N_regions, T, F] brain activity tensor
            config: Stimulation configuration
            current_time: Current time step
        
        Returns:
            Modified brain state with stimulation applied
        """
        # Check if stimulation is active at current time
        if not self._is_active(config, current_time):
            return brain_state
        
        # Clone to avoid modifying original
        stimulated_state = brain_state.clone()
        
        # Generate stimulation pattern
        stim_amplitude = self._generate_pattern(config, current_time)
        
        # Apply direct stimulation
        for region_id in config.target_regions:
            if region_id < self.n_regions:
                stimulated_state[region_id] += stim_amplitude
        
        # Apply spatial spread
        if self.region_positions is not None and config.spatial_spread > 0:
            stimulated_state = self._apply_spatial_spread(
                stimulated_state, config, stim_amplitude
            )
        
        # Apply network propagation
        if self.connectivity is not None:
            stimulated_state = self._apply_network_propagation(
                stimulated_state, config, stim_amplitude
            )
        
        return stimulated_state
    
    def _is_active(self, config: StimulationConfig, current_time: int) -> bool:
        """Check if stimulation is active at current time."""
        return config.onset <= current_time < (config.onset + config.duration)
    
    def _generate_pattern(
        self,
        config: StimulationConfig,
        current_time: int
    ) -> float:
        """Generate stimulation amplitude based on pattern."""
        relative_time = current_time - config.onset
        
        if config.pattern == "pulse":
            # Brief pulse at onset
            return config.amplitude if relative_time == 0 else 0.0
        
        elif config.pattern == "sine":
            # Sinusoidal pattern
            phase = 2 * np.pi * config.frequency * relative_time
            return config.amplitude * np.sin(phase)
        
        elif config.pattern == "ramp":
            # Gradually increasing
            progress = relative_time / config.duration
            return config.amplitude * progress
        
        elif config.pattern == "continuous":
            # Constant stimulation
            return config.amplitude
        
        else:
            raise ValueError(f"Unknown pattern: {config.pattern}")
    
    def _apply_spatial_spread(
        self,
        brain_state: torch.Tensor,
        config: StimulationConfig,
        amplitude: float
    ) -> torch.Tensor:
        """Apply spatial spread based on distance."""
        for target_id in config.target_regions:
            if target_id >= self.n_regions:
                continue
            
            target_pos = self.region_positions[target_id]
            
            # Calculate distances to all regions
            distances = np.linalg.norm(
                self.region_positions - target_pos,
                axis=1
            )
            
            # Apply Gaussian spread
            spread_weights = np.exp(-distances**2 / (2 * config.spatial_spread**2))
            
            # Apply to brain state
            for region_id in range(self.n_regions):
                if region_id != target_id:
                    weight = spread_weights[region_id]
                    brain_state[region_id] += amplitude * weight
        
        return brain_state
    
    def _apply_network_propagation(
        self,
        brain_state: torch.Tensor,
        config: StimulationConfig,
        amplitude: float
    ) -> torch.Tensor:
        """Apply stimulation propagation through network connections."""
        # Create stimulation vector
        stim_vector = torch.zeros(self.n_regions, device=brain_state.device)
        for target_id in config.target_regions:
            if target_id < self.n_regions:
                stim_vector[target_id] = amplitude
        
        # Propagate through connectivity
        connectivity_tensor = torch.tensor(
            self.connectivity,
            dtype=brain_state.dtype,
            device=brain_state.device
        )
        
        propagated = connectivity_tensor @ stim_vector
        
        # Add propagated effect to brain state
        brain_state += propagated.unsqueeze(1).unsqueeze(2)
        
        return brain_state
    
    def simulate_response(
        self,
        initial_state: torch.Tensor,
        config: StimulationConfig,
        n_steps: int,
        model: Optional[nn.Module] = None
    ) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        """
        Simulate brain response to stimulation over time.
        
        Args:
            initial_state: [N_regions, T, F] initial brain state
            config: Stimulation configuration
            n_steps: Number of time steps to simulate
            model: Optional neural network model for dynamics
        
        Returns:
            trajectory: [n_steps, N_regions, T, F] state trajectory
            metrics: List of per-step metrics
        """
        trajectory = []
        metrics = []
        
        current_state = initial_state.clone()
        
        for t in range(n_steps):
            # Apply stimulation
            stimulated_state = self.apply_stimulation(
                current_state, config, t
            )
            
            # If model provided, use it to predict next state
            if model is not None:
                with torch.no_grad():
                    next_state = model.predict_next(stimulated_state)
            else:
                # Simple dynamics: slight decay
                next_state = stimulated_state * 0.95
            
            # Record
            trajectory.append(next_state.clone())
            
            # Compute metrics
            step_metrics = self._compute_step_metrics(
                next_state, config, t
            )
            metrics.append(step_metrics)
            
            # Update current state
            current_state = next_state
        
        trajectory_tensor = torch.stack(trajectory)
        
        return trajectory_tensor, metrics
    
    def _compute_step_metrics(
        self,
        state: torch.Tensor,
        config: StimulationConfig,
        time_step: int
    ) -> Dict[str, Any]:
        """Compute metrics for a single time step."""
        # Global activity
        global_activity = state.mean().item()
        
        # Target region activity
        target_activity = 0.0
        if len(config.target_regions) > 0:
            valid_targets = [r for r in config.target_regions if r < self.n_regions]
            if valid_targets:
                target_activity = state[valid_targets].mean().item()
        
        return {
            "time_step": time_step,
            "global_activity": global_activity,
            "target_activity": target_activity,
            "stimulation_active": self._is_active(config, time_step)
        }
    
    def design_inverse_stimulation(
        self,
        initial_state: torch.Tensor,
        target_state: torch.Tensor,
        max_amplitude: float = 1.0,
        n_iterations: int = 100,
        learning_rate: float = 0.01
    ) -> StimulationConfig:
        """
        Design stimulation to reach target brain state (inverse problem).
        
        Args:
            initial_state: Starting brain state
            target_state: Desired brain state
            max_amplitude: Maximum allowed stimulation amplitude
            n_iterations: Number of optimization iterations
            learning_rate: Learning rate for optimization
        
        Returns:
            Optimized stimulation configuration
        """
        # Initialize stimulation parameters
        stim_amplitudes = torch.zeros(
            self.n_regions,
            requires_grad=True,
            device=initial_state.device
        )
        
        optimizer = torch.optim.Adam([stim_amplitudes], lr=learning_rate)
        
        for iteration in range(n_iterations):
            optimizer.zero_grad()
            
            # Apply stimulation
            stimulated = initial_state.clone()
            stimulated += stim_amplitudes.unsqueeze(1).unsqueeze(2)
            
            # Loss: difference from target
            loss = torch.nn.functional.mse_loss(stimulated, target_state)
            
            # Regularization: prefer sparse and small amplitudes
            l1_reg = torch.sum(torch.abs(stim_amplitudes))
            l2_reg = torch.sum(stim_amplitudes ** 2)
            total_loss = loss + 0.1 * l1_reg + 0.01 * l2_reg
            
            # Optimize
            total_loss.backward()
            optimizer.step()
            
            # Project to feasible region
            with torch.no_grad():
                stim_amplitudes.clamp_(-max_amplitude, max_amplitude)
        
        # Extract target regions (those with significant amplitude)
        threshold = 0.1 * max_amplitude
        target_regions = torch.where(torch.abs(stim_amplitudes) > threshold)[0]
        target_regions = target_regions.cpu().numpy().tolist()
        
        # Get mean amplitude of selected regions
        if len(target_regions) > 0:
            amplitude = float(torch.abs(stim_amplitudes[target_regions]).mean().item())
        else:
            amplitude = 0.0
        
        config = StimulationConfig(
            target_regions=target_regions,
            amplitude=amplitude,
            duration=10,
            pattern="continuous"
        )
        
        return config
    
    def to_json(self, config: StimulationConfig) -> Dict[str, Any]:
        """Convert stimulation config to JSON-compatible dict."""
        return {
            "active": True,
            "target_regions": config.target_regions,
            "amplitude": config.amplitude,
            "duration": config.duration,
            "pattern": config.pattern,
            "frequency": config.frequency,
            "spatial_spread": config.spatial_spread,
            "onset": config.onset
        }
