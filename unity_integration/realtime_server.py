"""
Real-time Brain Visualization Server
=====================================

WebSocket server for real-time communication with Unity frontend.
"""

import asyncio
import json
import logging
from typing import Dict, Any, Set, Optional
from pathlib import Path
from datetime import datetime

# Try to import websockets, but make it optional
try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    logging.warning("websockets package not available. Install with: pip install websockets")

# Import ModelServer
try:
    from .model_server import ModelServer
    MODEL_SERVER_AVAILABLE = True
except ImportError:
    MODEL_SERVER_AVAILABLE = False
    logging.warning("ModelServer not available")


class BrainVisualizationServer:
    """
    WebSocket server for real-time brain visualization.
    
    Provides endpoints for:
    - Getting current brain state
    - Predicting future states
    - Simulating stimulation effects
    - Streaming brain activity
    """
    
    def __init__(
        self,
        model = None,
        exporter = None,
        simulator = None,
        model_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        host: str = "0.0.0.0",
        port: int = 8765
    ):
        """
        Initialize server.
        
        Args:
            model: Trained neural network model (legacy, use model_path instead)
            exporter: BrainStateExporter instance
            simulator: StimulationSimulator instance
            model_path: Path to trained model file
            output_dir: Output directory for predictions
            host: Server host
            port: Server port
        """
        if not WEBSOCKETS_AVAILABLE:
            raise ImportError("websockets package is required. Install with: pip install websockets")
        
        self.model = model
        self.exporter = exporter
        self.simulator = simulator
        self.host = host
        self.port = port
        
        # Initialize ModelServer if available
        self.model_server = None
        if MODEL_SERVER_AVAILABLE and model_path:
            self.model_server = ModelServer(
                model_path=model_path,
                output_dir=output_dir or "unity_project/brain_data/model_output"
            )
            self.logger = logging.getLogger(__name__)
            self.logger.info("✓ ModelServer initialized")
        elif MODEL_SERVER_AVAILABLE and model:
            # Create ModelServer without loading (use existing model)
            self.model_server = ModelServer(
                output_dir=output_dir or "unity_project/brain_data/model_output"
            )
            self.model_server.model = model
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logging.getLogger(__name__)
        
        self.clients: Set = set()
    
    async def register_client(self, websocket):
        """Register a new client connection."""
        self.clients.add(websocket)
        self.logger.info(f"Client connected: {websocket.remote_address}")
        
        # Send welcome message
        welcome = {
            "type": "welcome",
            "message": "Connected to TwinBrain server",
            "version": "1.0"
        }
        await websocket.send(json.dumps(welcome))
    
    async def unregister_client(self, websocket):
        """Unregister a client connection."""
        self.clients.discard(websocket)
        self.logger.info(f"Client disconnected: {websocket.remote_address}")
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients."""
        if self.clients:
            message_str = json.dumps(message)
            await asyncio.gather(
                *[client.send(message_str) for client in self.clients],
                return_exceptions=True
            )
    
    async def handle_client(self, websocket, path):
        """Handle client connection and requests."""
        await self.register_client(websocket)
        
        try:
            async for message in websocket:
                try:
                    request = json.loads(message)
                    response = await self.process_request(request)
                    await websocket.send(json.dumps(response))
                except json.JSONDecodeError:
                    error = {"type": "error", "message": "Invalid JSON"}
                    await websocket.send(json.dumps(error))
                except Exception as e:
                    self.logger.error(f"Error processing request: {e}")
                    error = {"type": "error", "message": str(e)}
                    await websocket.send(json.dumps(error))
        finally:
            await self.unregister_client(websocket)
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process client request and return response."""
        request_type = request.get("type", "unknown")
        
        # Log the request
        self.logger.info(f"Processing request: {request_type}")
        
        try:
            if request_type == "get_state":
                return await self.handle_get_state(request)
            
            elif request_type == "predict":
                return await self.handle_predict(request)
            
            elif request_type == "simulate":
                return await self.handle_simulate(request)
            
            elif request_type == "stream_start":
                return await self.handle_stream_start(request)
            
            elif request_type == "stream_stop":
                return {"type": "stream_stopped", "success": True}
            
            elif request_type == "convert_cache":
                return await self.handle_convert_cache(request)
            
            else:
                self.logger.warning(f"Unknown request type: {request_type}")
                return {
                    "type": "error",
                    "success": False,
                    "message": f"Unknown request type: {request_type}"
                }
        except Exception as e:
            self.logger.error(f"Error processing {request_type}: {e}", exc_info=True)
            return {
                "type": "error",
                "success": False,
                "message": f"Server error: {str(e)}"
            }
    
    async def handle_get_state(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle request for current brain state."""
        try:
            # Generate example brain state
            import torch
            import numpy as np
            
            n_regions = 200
            # Generate example fMRI data
            fmri_data = torch.randn(n_regions, 1, 1)
            eeg_data = torch.randn(n_regions, 1, 1)
            
            brain_activity = {
                'fmri': fmri_data,
                'eeg': eeg_data
            }
            
            # Generate connectivity
            connectivity_matrix = np.random.rand(n_regions, n_regions)
            connectivity_matrix = (connectivity_matrix + connectivity_matrix.T) / 2
            connectivity_matrix[connectivity_matrix < 0.7] = 0
            connectivity = {'structural': connectivity_matrix}
            
            # Export to JSON format
            if self.exporter:
                brain_state = self.exporter.export_brain_state(
                    brain_activity=brain_activity,
                    connectivity=connectivity,
                    time_point=0,
                    subject_id="realtime"
                )
                
                return {
                    "type": "brain_state",
                    "success": True,
                    "data": brain_state
                }
            else:
                return {
                    "type": "brain_state",
                    "success": False,
                    "message": "Exporter not available"
                }
                
        except Exception as e:
            self.logger.error(f"Error getting state: {e}")
            return {
                "type": "error",
                "message": f"Failed to get state: {str(e)}"
            }
    
    async def handle_predict(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle prediction request."""
        n_steps = request.get("n_steps", 10)
        
        # Input validation
        original_n_steps = n_steps
        if not isinstance(n_steps, int) or n_steps <= 0 or n_steps > 1000:
            self.logger.warning(f"Invalid n_steps: {n_steps}, using default 10")
            n_steps = 10
        
        # Inform client if parameter was adjusted
        parameter_adjusted = (n_steps != original_n_steps)
        
        try:
            # Use ModelServer if available
            if self.model_server:
                self.logger.info(f"Using ModelServer for prediction ({n_steps} steps)")
                predictions = self.model_server.predict_future(
                    n_steps=n_steps,
                    subject_id="prediction"
                )
                
                return {
                    "type": "prediction",
                    "success": True,
                    "n_steps": n_steps,
                    "predictions": predictions,
                    "saved_to": str(self.model_server.output_dir),
                    "parameter_adjusted": parameter_adjusted,
                    "warning": "n_steps was adjusted to valid range" if parameter_adjusted else None
                }
            
            # Fallback to simple prediction generation
            import torch
            import numpy as np
            
            n_regions = 200
            
            # Generate prediction sequence
            predictions = []
            for t in range(n_steps):
                # Generate predicted brain state
                fmri_data = torch.randn(n_regions, 1, 1) * (0.5 + t * 0.05)
                eeg_data = torch.randn(n_regions, 1, 1) * (0.5 + t * 0.05)
                
                brain_activity = {
                    'fmri': fmri_data,
                    'eeg': eeg_data
                }
                
                if self.exporter:
                    brain_state = self.exporter.export_brain_state(
                        brain_activity=brain_activity,
                        time_point=t,
                        time_second=float(t),
                        subject_id="prediction"
                    )
                    predictions.append(brain_state)
            
            return {
                "type": "prediction",
                "success": True,
                "n_steps": n_steps,
                "predictions": predictions,
                "parameter_adjusted": parameter_adjusted,
                "warning": "n_steps was adjusted to valid range" if parameter_adjusted else None
            }
            
        except Exception as e:
            self.logger.error(f"Error in prediction: {e}")
            return {
                "type": "error",
                "message": f"Prediction failed: {str(e)}"
            }
    
    async def handle_simulate(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle stimulation simulation request."""
        stimulation = request.get("stimulation", {})
        
        # Input validation
        if not stimulation or not isinstance(stimulation, dict):
            return {
                "type": "error",
                "success": False,
                "message": "Invalid stimulation parameters"
            }
        
        try:
            import torch
            import numpy as np
            
            # Parse stimulation parameters
            target_regions = stimulation.get("target_regions", [])
            amplitude = stimulation.get("amplitude", 0.5)
            pattern = stimulation.get("pattern", "sine")
            frequency = stimulation.get("frequency", 10.0)
            duration = stimulation.get("duration", 20)
            
            # Validate target regions
            if not isinstance(target_regions, list) or len(target_regions) == 0:
                return {
                    "type": "error",
                    "success": False,
                    "message": "target_regions must be a non-empty list"
                }
            
            # Validate and clamp values
            amplitude = float(max(0.0, min(10.0, amplitude)))
            frequency = float(max(0.1, min(100.0, frequency)))
            duration = int(max(1, min(1000, duration)))
            
            # Filter valid region IDs
            n_regions = 200
            valid_regions = [r for r in target_regions if isinstance(r, int) and 0 <= r < n_regions]
            
            if len(valid_regions) == 0:
                return {
                    "type": "error",
                    "success": False,
                    "message": f"No valid target regions (must be 0-{n_regions-1})"
                }
            
            if len(valid_regions) < len(target_regions):
                self.logger.warning(f"Filtered {len(target_regions) - len(valid_regions)} invalid region IDs")
                target_regions = valid_regions
            
            n_steps = 50
            
            # Use ModelServer if available
            if self.model_server:
                self.logger.info(f"Using ModelServer for stimulation simulation")
                responses = self.model_server.simulate_stimulation(
                    target_regions=target_regions,
                    amplitude=amplitude,
                    pattern=pattern,
                    frequency=frequency,
                    duration=duration,
                    subject_id="stimulation"
                )
                
                return {
                    "type": "simulation",
                    "success": True,
                    "n_steps": len(responses),
                    "stimulation": stimulation,
                    "responses": responses,
                    "saved_to": str(self.model_server.output_dir)
                }
            
            # Fallback to using simulator
            from .stimulation_simulator import StimulationConfig
            
            # Create stimulation config
            stim_config = StimulationConfig(
                target_regions=target_regions,
                amplitude=amplitude,
                duration=duration,
                pattern=pattern,
                frequency=frequency
            )
            
            # Generate initial state
            initial_state = torch.randn(n_regions, 1, 1)
            
            # Simulate response
            if self.simulator:
                trajectory, metrics = self.simulator.simulate_response(
                    initial_state=initial_state,
                    config=stim_config,
                    n_steps=n_steps
                )
                
                # Export trajectory as sequence
                responses = []
                for t, state in enumerate(trajectory):
                    if len(state.shape) == 2:
                        state = state.unsqueeze(1)
                    
                    brain_activity = {'fmri': state}
                    
                    # Add stimulation info if active
                    stim_info = None
                    if metrics[t].get('stimulation_active', False):
                        stim_info = {
                            "active": True,
                            "target_regions": target_regions,
                            "amplitude": amplitude,
                            "pattern": pattern
                        }
                    
                    if self.exporter:
                        brain_state = self.exporter.export_brain_state(
                            brain_activity=brain_activity,
                            time_point=t,
                            time_second=float(t),
                            subject_id="simulation",
                            stimulation=stim_info
                        )
                        responses.append(brain_state)
                
                return {
                    "type": "simulation",
                    "success": True,
                    "n_steps": n_steps,
                    "stimulation": stimulation,
                    "responses": responses,
                    "metrics": metrics
                }
            else:
                # Simple simulation without simulator
                responses = []
                for t in range(n_steps):
                    # Enhanced activity in target regions
                    fmri_data = torch.randn(n_regions, 1, 1)
                    for region_id in target_regions:
                        if region_id < n_regions:
                            fmri_data[region_id] += amplitude * np.sin(2 * np.pi * frequency * t / n_steps)
                    
                    brain_activity = {'fmri': fmri_data}
                    stim_info = {
                        "active": t < duration,
                        "target_regions": target_regions,
                        "amplitude": amplitude
                    }
                    
                    if self.exporter:
                        brain_state = self.exporter.export_brain_state(
                            brain_activity=brain_activity,
                            time_point=t,
                            subject_id="simulation",
                            stimulation=stim_info
                        )
                        responses.append(brain_state)
                
                return {
                    "type": "simulation",
                    "success": True,
                    "n_steps": n_steps,
                    "stimulation": stimulation,
                    "responses": responses
                }
                
        except Exception as e:
            self.logger.error(f"Error in simulation: {e}")
            return {
                "type": "error",
                "message": f"Simulation failed: {str(e)}"
            }
    
    async def handle_convert_cache(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle cache to JSON conversion request.
        
        Request format:
        {
            "type": "convert_cache",
            "cache_dir": "/path/to/cache",
            "output_dir": "/path/to/output"
        }
        """
        try:
            cache_dir = request.get("cache_dir")
            output_dir = request.get("output_dir")
            
            if not cache_dir or not output_dir:
                return {
                    "type": "error",
                    "message": "cache_dir and output_dir are required"
                }
            
            cache_path = Path(cache_dir)
            output_path = Path(output_dir)
            
            if not cache_path.exists():
                return {
                    "type": "error",
                    "message": f"Cache directory does not exist: {cache_dir}"
                }
            
            # Create output directory if it doesn't exist
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Find cache files (only .pt/.pth files are the actual cache format)
            import glob
            cache_files = []
            for ext in ['*.pt', '*.pth']:
                cache_files.extend(glob.glob(str(cache_path / ext)))
            
            if not cache_files:
                return {
                    "type": "error",
                    "message": f"No cache files found in {cache_dir}"
                }
            
            self.logger.info(f"Found {len(cache_files)} cache files to convert")
            
            # Process each cache file
            converted_count = 0
            errors = []
            
            for cache_file in cache_files:
                try:
                    # Load cache file
                    import torch
                    import numpy as np
                    
                    file_path = Path(cache_file)
                    ext = file_path.suffix
                    
                    # Only .pt files are supported (the actual cache format)
                    if ext not in ['.pt', '.pth']:
                        self.logger.warning(f"Skipping unsupported file format: {file_path.name}")
                        continue
                    
                    self.logger.info(f"Loading cache file: {file_path.name}")
                    data = torch.load(file_path, map_location='cpu', weights_only=False)
                    
                    # Determine cache file type and convert accordingly
                    # Cache files are: eeg_data.pt or hetero_graphs.pt
                    # Both are Dict[str, ...] where keys are task names
                    
                    if 'eeg_data' in file_path.stem:
                        # eeg_data.pt format: Dict[task_name, Dict["on"|"off", HeteroData]]
                        self.logger.info(f"Processing EEG data cache: {file_path.name}")
                        converted = self._convert_eeg_data_cache(data, output_path, file_path.stem)
                        converted_count += converted
                        
                    elif 'hetero_graphs' in file_path.stem:
                        # hetero_graphs.pt format: Dict[task_name, List[HeteroData]]
                        self.logger.info(f"Processing hetero graphs cache: {file_path.name}")
                        converted = self._convert_hetero_graphs_cache(data, output_path, file_path.stem)
                        converted_count += converted
                        
                    else:
                        # Unknown format - try to extract data generically
                        self.logger.warning(f"Unknown cache format for {file_path.name}, attempting generic conversion")
                        if isinstance(data, dict):
                            for task_name, task_data in data.items():
                                try:
                                    brain_activity = self._extract_brain_activity_from_hetero(task_data)
                                    if brain_activity:
                                        output_file = output_path / f"brain_state_{file_path.stem}_{task_name}.json"
                                        self._export_brain_state_json(brain_activity, output_file, task_name)
                                        converted_count += 1
                                except Exception as e:
                                    error_msg = f"Error converting task {task_name}: {str(e)}"
                                    self.logger.error(error_msg)
                                    errors.append(error_msg)
                        
                except Exception as e:
                    error_msg = f"Error processing {cache_file}: {str(e)}"
                    self.logger.error(error_msg)
                    errors.append(error_msg)
            
            if converted_count > 0:
                return {
                    "type": "convert_cache_response",
                    "success": True,
                    "message": f"Successfully converted {converted_count} cache files",
                    "converted_count": converted_count,
                    "errors": errors if errors else None,
                    "output_dir": str(output_path)
                }
            else:
                return {
                    "type": "error",
                    "success": False,
                    "message": "No files were converted",
                    "errors": errors
                }
                
        except Exception as e:
            self.logger.error(f"Error in handle_convert_cache: {e}")
            return {
                "type": "error",
                "success": False,
                "message": str(e)
            }
    
    def _extract_brain_activity_from_hetero(self, hetero_data):
        """
        Extract brain activity tensors from HeteroData objects.
        
        Args:
            hetero_data: Can be HeteroData, Dict, or List[HeteroData]
            
        Returns:
            Dict with 'fmri' and/or 'eeg' tensors, or None if extraction fails
        """
        import torch
        brain_activity = {}
        
        try:
            # Handle different input types
            if hasattr(hetero_data, 'node_types'):  # HeteroData object
                # Extract from HeteroData
                if 'fmri' in hetero_data.node_types:
                    fmri_node = hetero_data['fmri']
                    # x_seq is the typical attribute for sequence data
                    if hasattr(fmri_node, 'x_seq'):
                        brain_activity['fmri'] = fmri_node.x_seq
                    elif hasattr(fmri_node, 'x'):
                        brain_activity['fmri'] = fmri_node.x
                
                if 'eeg' in hetero_data.node_types:
                    eeg_node = hetero_data['eeg']
                    if hasattr(eeg_node, 'x_seq'):
                        brain_activity['eeg'] = eeg_node.x_seq
                    elif hasattr(eeg_node, 'x'):
                        brain_activity['eeg'] = eeg_node.x
                        
            elif isinstance(hetero_data, dict):
                # Dict with 'on'/'off' keys (EEG data format)
                if 'on' in hetero_data:
                    on_activity = self._extract_brain_activity_from_hetero(hetero_data['on'])
                    if on_activity and 'eeg' in on_activity:
                        brain_activity['eeg'] = on_activity['eeg']
                        
            elif isinstance(hetero_data, list) and len(hetero_data) > 0:
                # List of HeteroData objects - use first one
                first_activity = self._extract_brain_activity_from_hetero(hetero_data[0])
                if first_activity:
                    brain_activity = first_activity
                    
        except Exception as e:
            self.logger.error(f"Error extracting brain activity: {e}")
            return None
        
        return brain_activity if brain_activity else None
    
    def _convert_eeg_data_cache(self, data, output_path, base_name):
        """
        Convert eeg_data.pt cache file.
        Format: Dict[task_name, Dict["on"|"off", HeteroData]]
        """
        converted = 0
        
        if not isinstance(data, dict):
            self.logger.error(f"EEG data cache is not a dictionary: {type(data)}")
            return 0
        
        for task_name, task_data in data.items():
            try:
                if not isinstance(task_data, dict):
                    continue
                
                # Process "on" and "off" separately
                for condition in ['on', 'off']:
                    if condition not in task_data:
                        continue
                    
                    hetero_data = task_data[condition]
                    brain_activity = self._extract_brain_activity_from_hetero(hetero_data)
                    
                    if brain_activity:
                        output_file = output_path / f"brain_state_{base_name}_{task_name}_{condition}.json"
                        self._export_brain_state_json(
                            brain_activity, 
                            output_file, 
                            f"{task_name}_{condition}"
                        )
                        converted += 1
                        self.logger.info(f"  Converted: {task_name}/{condition} -> {output_file.name}")
                        
            except Exception as e:
                self.logger.error(f"Error converting EEG task {task_name}: {e}")
        
        return converted
    
    def _convert_hetero_graphs_cache(self, data, output_path, base_name):
        """
        Convert hetero_graphs.pt cache file.
        Format: Dict[task_name, List[HeteroData]]
        """
        converted = 0
        
        if not isinstance(data, dict):
            self.logger.error(f"Hetero graphs cache is not a dictionary: {type(data)}")
            return 0
        
        for task_name, graph_list in data.items():
            try:
                if not isinstance(graph_list, list) or len(graph_list) == 0:
                    continue
                
                # Convert first few graphs (to avoid generating too many files)
                max_graphs = min(10, len(graph_list))
                for idx in range(max_graphs):
                    hetero_data = graph_list[idx]
                    brain_activity = self._extract_brain_activity_from_hetero(hetero_data)
                    
                    if brain_activity:
                        output_file = output_path / f"brain_state_{base_name}_{task_name}_{idx:03d}.json"
                        self._export_brain_state_json(
                            brain_activity,
                            output_file,
                            f"{task_name}_frame{idx}"
                        )
                        converted += 1
                        
                if max_graphs < len(graph_list):
                    self.logger.info(f"  Converted {max_graphs}/{len(graph_list)} graphs for task {task_name}")
                else:
                    self.logger.info(f"  Converted all {len(graph_list)} graphs for task {task_name}")
                    
            except Exception as e:
                self.logger.error(f"Error converting hetero graph task {task_name}: {e}")
        
        return converted
    
    def _export_brain_state_json(self, brain_activity, output_file, subject_id):
        """
        Export brain activity to JSON file.
        
        Args:
            brain_activity: Dict with 'fmri' and/or 'eeg' tensors
            output_file: Path to save JSON
            subject_id: Subject identifier for metadata
        """
        import torch
        
        try:
            # Use exporter if available
            if self.exporter:
                self.exporter.export_brain_state(
                    brain_activity=brain_activity,
                    time_point=0,
                    time_second=0.0,
                    subject_id=subject_id,
                    output_path=output_file
                )
            else:
                # Fallback: create minimal JSON
                json_data = {
                    "version": "2.0",
                    "timestamp": datetime.now().isoformat(),
                    "subject_id": subject_id,
                    "brain_state": {
                        "time_point": 0,
                        "regions": []
                    }
                }
                
                # Extract basic statistics for each modality
                for modality, tensor in brain_activity.items():
                    if isinstance(tensor, torch.Tensor):
                        json_data[f"{modality}_shape"] = list(tensor.shape)
                        json_data[f"{modality}_mean"] = float(tensor.mean())
                        json_data[f"{modality}_std"] = float(tensor.std())
                
                with open(output_file, 'w') as f:
                    json.dump(json_data, f, indent=2)
                    
        except Exception as e:
            self.logger.error(f"Error exporting to JSON: {e}")
            raise
    
    async def handle_stream_start(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle stream start request."""
        fps = request.get("fps", 10)
        duration = request.get("duration", 60)
        
        # Input validation
        fps = int(max(1, min(60, fps)))
        duration = int(max(1, min(3600, duration)))
        
        # Start streaming in background
        asyncio.create_task(self.stream_brain_activity(fps, duration))
        
        return {
            "type": "stream_started",
            "success": True,
            "fps": fps,
            "duration": duration
        }
    
    async def stream_brain_activity(self, fps: int = 10, duration: int = 60):
        """Stream brain activity to all connected clients."""
        n_frames = int(fps * duration)
        
        try:
            import torch
            import numpy as np
            
            n_regions = 200
            
            for frame_idx in range(n_frames):
                # Generate dynamic brain state
                # Simulate wave-like activity patterns
                phase = 2 * np.pi * frame_idx / (fps * 5)  # 5 second cycle
                
                fmri_data = torch.randn(n_regions, 1, 1)
                # Add wave pattern
                for i in range(n_regions):
                    fmri_data[i] += 0.5 * np.sin(phase + i * 0.1)
                
                eeg_data = torch.randn(n_regions, 1, 1)
                
                brain_activity = {
                    'fmri': fmri_data,
                    'eeg': eeg_data
                }
                
                # Export brain state
                if self.exporter:
                    brain_state = self.exporter.export_brain_state(
                        brain_activity=brain_activity,
                        time_point=frame_idx,
                        time_second=frame_idx / fps,
                        subject_id="stream"
                    )
                    
                    frame_data = {
                        "type": "stream_frame",
                        "frame": frame_idx,
                        "time": frame_idx / fps,
                        "data": brain_state
                    }
                else:
                    frame_data = {
                        "type": "stream_frame",
                        "frame": frame_idx,
                        "time": frame_idx / fps,
                        "data": {}
                    }
                
                # Broadcast to all clients
                await self.broadcast(frame_data)
                
                # Control frame rate
                await asyncio.sleep(1.0 / fps)
            
            # Send stream end message
            await self.broadcast({"type": "stream_ended", "n_frames": n_frames})
            
        except Exception as e:
            self.logger.error(f"Error in streaming: {e}")
            await self.broadcast({
                "type": "error",
                "message": f"Streaming failed: {str(e)}"
            })
    
    def start(self):
        """Start the WebSocket server."""
        if not WEBSOCKETS_AVAILABLE:
            raise ImportError("websockets package is required")
        
        self.logger.info(f"Starting WebSocket server on {self.host}:{self.port}")
        
        start_server = websockets.serve(
            self.handle_client,
            self.host,
            self.port
        )
        
        asyncio.get_event_loop().run_until_complete(start_server)
        self.logger.info("Server started successfully")
        
        # Run forever
        asyncio.get_event_loop().run_forever()
    
    async def start_async(self):
        """Start server asynchronously (for use in existing event loop)."""
        if not WEBSOCKETS_AVAILABLE:
            raise ImportError("websockets package is required")
        
        self.logger.info(f"Starting WebSocket server on {self.host}:{self.port}")
        
        async with websockets.serve(self.handle_client, self.host, self.port):
            self.logger.info("Server started successfully")
            await asyncio.Future()  # Run forever


# Standalone server function
def start_server(
    model=None,
    exporter=None,
    simulator=None,
    host: str = "0.0.0.0",
    port: int = 8765
):
    """
    Start the brain visualization server.
    
    Usage:
        from unity_integration.realtime_server import start_server
        start_server(model, exporter, simulator)
    """
    server = BrainVisualizationServer(
        model=model,
        exporter=exporter,
        simulator=simulator,
        host=host,
        port=port
    )
    
    try:
        server.start()
    except KeyboardInterrupt:
        print("\nServer stopped by user")
