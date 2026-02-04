"""
Unity Integration Module
========================

Provides tools for exporting brain data and interacting with Unity frontend:
- BrainStateExporter: Export brain states to JSON
- StimulationSimulator: Simulate virtual stimulation
- BrainVisualizationServer: Real-time WebSocket communication
- WorkflowManager: Automated workflow for Unity integration
- BrainOBJGenerator: Generate 3D OBJ models
"""

from .brain_state_exporter import BrainStateExporter
from .stimulation_simulator import StimulationSimulator, StimulationConfig
from .workflow_manager import WorkflowManager, WorkflowConfig, run_unity_workflow
from .obj_generator import BrainOBJGenerator

# Import server with error handling
try:
    from .realtime_server import BrainVisualizationServer
    SERVER_AVAILABLE = True
except ImportError:
    BrainVisualizationServer = None
    SERVER_AVAILABLE = False

__all__ = [
    'BrainStateExporter',
    'BrainOBJGenerator',
    'BrainVisualizationServer',
    'StimulationSimulator',
    'StimulationConfig',
    'WorkflowManager',
    'WorkflowConfig',
    'run_unity_workflow',
    'SERVER_AVAILABLE',
]
