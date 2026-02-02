# Unity Integration Examples

This directory contains Unity C# scripts for visualizing TwinBrain brain states in Unity 3D.

## Files

- `BrainVisualization.cs` - Main visualization component
- `BrainConfigLoader.cs` - Auto-load configuration from unity_config.json
- `WebSocketClient.cs` - WebSocket client for real-time updates

## Quick Start

### 1. Setup Unity Project

1. Create a new Unity project (Unity 2020.3 or later)
2. Install Newtonsoft.Json package via Package Manager:
   - Window → Package Manager
   - Click "+" → Add package from git URL
   - Enter: `com.unity.nuget.newtonsoft-json`

### 2. Add Scripts to Unity

1. Copy all `.cs` files from `unity_examples/` to your Unity project's `Assets/Scripts/` folder:
   - `BrainVisualization.cs` - Main visualization
   - `BrainConfigLoader.cs` - Auto-configuration loader
   - `WebSocketClient.cs` - Real-time communication (optional)

2. Create an empty GameObject in the scene (GameObject → Create Empty)
3. Rename it to "BrainVisualization"
4. Add the `BrainVisualization` component to it
5. Add the `BrainConfigLoader` component to it (optional, for auto-configuration)

### 3. Export Data from TwinBrain

Run the Python workflow to generate all necessary files:

```bash
cd /path/to/twinbrain
python example_unity_workflow.py
```

This will create a complete export with:
- `output/*/json/` - JSON brain states
- `output/*/obj/` - 3D models (optional)
- `output/*/unity_config.json` - Unity configuration
- `output/*/materials/` - Material configurations
- `output/*/workflow_report.json` - Execution report

**Quick export:**
```python
from unity_integration import run_unity_workflow, WorkflowConfig

config = WorkflowConfig(
    output_dir='output/my_export',
    export_formats=['json'],
    time_step=5
)
results = run_unity_workflow(config)
```

### 4. Configure in Unity

#### Option A: Auto-Configuration (Recommended)

If you added `BrainConfigLoader`:

1. Select the BrainVisualization GameObject
2. In the BrainConfigLoader component:
   - Config Path: `path/to/output/unity_config.json`
   - Auto Load: ✓ (checked)
3. Press Play - settings will be loaded automatically!

#### Option B: Manual Configuration

Select the BrainVisualization GameObject and configure manually:

#### For Single State:
- JSON Path: `path/to/output/brain_state_t100.json`
- Load Sequence: ✗ (unchecked)

#### For Animation:
- JSON Path: `path/to/output/brain_sequence`
- Load Sequence: ✓ (checked)
- FPS: 10
- Auto Play: ✓ (checked)

#### Visualization Settings:
- Region Prefab: Drag a sphere prefab (optional)
- Connection Material: Create a material with alpha blending
- Region Scale: 1.0
- Activity Threshold: 0.3
- Show Connections: ✓
- Connection Threshold: 0.5

#### Colors:
- Low Activity Color: Blue (RGB: 0, 0, 255)
- High Activity Color: Red (RGB: 255, 0, 0)

### 5. Run

Press Play in Unity. You should see:
- Brain regions as colored spheres (color = activity level)
- Connections as lines between regions
- Animation if using sequence mode

### Controls

- **Space**: Play/Pause animation
- **R**: Reload current state

## JSON Format

The script expects JSON files in this format:

```json
{
  "version": "2.0",
  "metadata": {
    "subject": "sub-01",
    "atlas": "Schaefer200",
    "time_point": 100
  },
  "brain_state": {
    "regions": [
      {
        "id": 0,
        "label": "Region_1",
        "position": {"x": -5, "y": -85, "z": 5},
        "activity": {
          "fmri": {"amplitude": 0.75}
        }
      }
    ],
    "connections": [
      {
        "source": 0,
        "target": 1,
        "strength": 0.65,
        "type": "structural"
      }
    ]
  }
}
```

See [系统使用指南](../TwinBrain系统使用指南.md) for detailed format specification.

## Advanced Usage

### Custom Region Prefabs

Create a prefab with:
1. Mesh (e.g., sphere, brain region shape)
2. Material with shader that supports colors
3. Optional: Add glow effect for high activity

Assign to "Region Prefab" field.

### Custom Connection Materials

Create a material with:
1. Shader: Particles/Standard Unlit or custom shader
2. Rendering Mode: Fade or Transparent
3. Enable alpha blending

Assign to "Connection Material" field.

### Interaction

Add interactivity by extending the script:

```csharp
// In BrainVisualization.cs

void OnRegionClick(int regionId)
{
    Debug.Log($"Clicked region {regionId}");
    // Show region details
    // Highlight connections
    // etc.
}
```

## Real-time Communication

### Setup WebSocket Connection

1. Add `WebSocketClient` component to a GameObject
2. Start the TwinBrain server:

```bash
python -c "
from unity_integration import BrainVisualizationServer
server = BrainVisualizationServer(port=8765)
server.start()
"
```

3. In Unity, configure WebSocketClient:
   - Server URL: `ws://localhost:8765`
   - Auto Connect: ✓

### Use WebSocket Client

```csharp
// Get reference
WebSocketClient client = GetComponent<WebSocketClient>();

// Register event handlers
client.OnBrainStateReceived += (state) => {
    Debug.Log($"Received brain state at time {state.metadata.time_second}");
    // Update visualization...
};

// Request current state
client.GetBrainState();

// Request prediction
client.RequestPrediction(nSteps: 20);

// Simulate stimulation
client.SimulateStimulation(
    targetRegions: new int[] {10, 15, 20},
    amplitude: 0.5f,
    pattern: "sine"
);

// Start streaming
client.StartStream(fps: 10, duration: 60);
```

## Troubleshooting

### Regions not visible

- Check Activity Threshold setting (lower it)
- Verify JSON file path is correct
- Check console for errors

### Connections not showing

- Enable "Show Connections"
- Lower "Connection Threshold"
- Verify connectivity data in JSON

### Animation not playing

- Check "Auto Play" is enabled
- Verify sequence_index.json exists in directory
- Check FPS setting (try lower value first)

### Performance issues

- Reduce number of connections (increase threshold)
- Disable connections for large networks
- Use simpler region prefabs
- Lower frame rate

## Examples

### Example 1: Static Visualization

Display a single brain state:
1. Set JSON Path to single JSON file
2. Uncheck Load Sequence
3. Play

### Example 2: Time Series Animation

Animate brain activity over time:
1. Set JSON Path to sequence directory
2. Check Load Sequence
3. Set FPS to 10
4. Check Auto Play
5. Play

### Example 3: Stimulation Response

Visualize response to stimulation:
1. Set JSON Path to stimulation_response directory
2. Check Load Sequence
3. FPS: 10
4. Play
5. Observe activity change when stimulation is active

## Next Steps

1. Customize colors and materials for your needs
2. Add UI controls for threshold adjustments
3. Implement region highlighting and selection
4. Add network analysis visualizations
5. Integrate with WebSocket for real-time updates

## Support

For issues or questions:
- Check [系统使用指南](../TwinBrain系统使用指南.md)
- See [优化方向](../TwinBrain优化方向和研究思路.md) for advanced features
- Submit issues on GitHub

## License

[Same as main project]
