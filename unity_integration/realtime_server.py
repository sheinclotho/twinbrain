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
            return {
                "type": "error",
                "message": f"Unknown request type: {request_type}"
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
                    "saved_to": str(self.model_server.output_dir)
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
                "predictions": predictions
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
        
        try:
            import torch
            import numpy as np
            
            # Parse stimulation parameters
            target_regions = stimulation.get("target_regions", [])
            amplitude = stimulation.get("amplitude", 0.5)
            pattern = stimulation.get("pattern", "sine")
            frequency = stimulation.get("frequency", 10.0)
            duration = stimulation.get("duration", 20)
            
            n_regions = 200
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
            
            # Find cache files
            import glob
            cache_files = []
            for ext in ['*.pkl', '*.npy', '*.npz', '*.pt', '*.pth']:
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
                    import pickle
                    
                    file_path = Path(cache_file)
                    ext = file_path.suffix
                    
                    if ext == '.pkl':
                        with open(file_path, 'rb') as f:
                            data = pickle.load(f)
                    elif ext in ['.npy', '.npz']:
                        data = np.load(file_path, allow_pickle=True)
                        if isinstance(data, np.lib.npyio.NpzFile):
                            data = dict(data)
                    elif ext in ['.pt', '.pth']:
                        data = torch.load(file_path, map_location='cpu')
                    else:
                        continue
                    
                    # Convert to brain state JSON
                    if self.exporter:
                        # Extract fMRI/EEG data
                        brain_activity = {}
                        if isinstance(data, dict):
                            if 'fmri' in data:
                                brain_activity['fmri'] = torch.tensor(data['fmri']) if not isinstance(data['fmri'], torch.Tensor) else data['fmri']
                            if 'eeg' in data:
                                brain_activity['eeg'] = torch.tensor(data['eeg']) if not isinstance(data['eeg'], torch.Tensor) else data['eeg']
                        
                        if brain_activity:
                            # Generate JSON output
                            output_file = output_path / f"brain_state_{file_path.stem}.json"
                            self.exporter.export_brain_state(
                                brain_activity=brain_activity,
                                output_path=output_file
                            )
                            converted_count += 1
                            self.logger.info(f"Converted: {file_path.name} -> {output_file.name}")
                    else:
                        # Fallback: simple JSON export
                        output_file = output_path / f"brain_state_{file_path.stem}.json"
                        # Create minimal JSON structure
                        simple_json = {
                            "version": "2.0",
                            "timestamp": datetime.now().isoformat(),
                            "source_file": file_path.name,
                            "note": "Converted from cache without exporter"
                        }
                        with open(output_file, 'w') as f:
                            json.dump(simple_json, f, indent=2)
                        converted_count += 1
                        
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
    
    async def handle_stream_start(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle stream start request."""
        fps = request.get("fps", 10)
        duration = request.get("duration", 60)
        
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
