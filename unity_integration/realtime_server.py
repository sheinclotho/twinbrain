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

# Try to import websockets, but make it optional
try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    logging.warning("websockets package not available. Install with: pip install websockets")


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
        host: str = "0.0.0.0",
        port: int = 8765
    ):
        """
        Initialize server.
        
        Args:
            model: Trained neural network model
            exporter: BrainStateExporter instance
            simulator: StimulationSimulator instance
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
        
        self.clients: Set = set()
        self.logger = logging.getLogger(__name__)
    
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
        
        else:
            return {
                "type": "error",
                "message": f"Unknown request type: {request_type}"
            }
    
    async def handle_get_state(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle request for current brain state."""
        # This is a placeholder - actual implementation would get state from model
        return {
            "type": "brain_state",
            "success": True,
            "message": "State retrieval not yet implemented",
            "data": {}
        }
    
    async def handle_predict(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle prediction request."""
        n_steps = request.get("n_steps", 10)
        
        # Placeholder implementation
        return {
            "type": "prediction",
            "success": True,
            "n_steps": n_steps,
            "message": "Prediction not yet implemented",
            "data": {}
        }
    
    async def handle_simulate(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle stimulation simulation request."""
        stimulation = request.get("stimulation", {})
        
        # Placeholder implementation
        return {
            "type": "simulation",
            "success": True,
            "message": "Simulation not yet implemented",
            "stimulation": stimulation,
            "data": {}
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
        
        for frame_idx in range(n_frames):
            # Create frame data (placeholder)
            frame_data = {
                "type": "stream_frame",
                "frame": frame_idx,
                "time": frame_idx / fps,
                "data": {}  # Actual brain state would go here
            }
            
            # Broadcast to all clients
            await self.broadcast(frame_data)
            
            # Control frame rate
            await asyncio.sleep(1.0 / fps)
        
        # Send stream end message
        await self.broadcast({"type": "stream_ended", "n_frames": n_frames})
    
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
