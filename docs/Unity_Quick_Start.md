# TwinBrain Unity Quick Start Guide

Get TwinBrain Unity visualization running in 5 minutes.

## Prerequisites

- Python 3.7+
- Unity 2020.3 or higher
- Basic command line experience

## Step 1: Install Python Dependencies

```bash
cd /path/to/twinbrain
pip install -r requirements.txt
pip install websockets  # Optional, for real-time server
```

## Step 2: Generate Unity Assets

Run the one-click automation script:

```bash
python unity_automation.py --mode export
```

This generates in `unity_output/`:

```
unity_output/
├── json/                   # Brain state JSON files (40+ files)
├── obj/                    # 3D model files
├── materials/              # Material configurations
├── UnityScripts/           # Unity C# scripts (7 files)
├── unity_config.json       # Unity configuration
├── unity_scene_config.json # Scene configuration
├── unity_prefab_config.json# Prefab configuration
└── README_UNITY.md        # Detailed instructions
```

⏱️ Estimated time: 30-60 seconds

## Step 3: Create Unity Project

1. Open Unity Hub
2. Create new 3D project named `TwinBrain_Visualization`
3. Wait for project creation

## Step 4: Install Newtonsoft.Json Package

In Unity:

1. Window → Package Manager
2. Click "+" → Add package from git URL
3. Enter: `com.unity.nuget.newtonsoft-json`
4. Wait for installation

⏱️ Estimated time: 1-2 minutes

## Step 5: Import Assets

1. In Unity project's Assets folder, create `Scripts` and `Data` folders
2. Copy all `.cs` files from `unity_output/UnityScripts/` to `Assets/Scripts/`
3. Copy `unity_output/json/` to `Assets/Data/JSON/`
4. Copy `unity_output/obj/` to `Assets/Data/OBJ/`
5. Wait for Unity to compile scripts

⏱️ Estimated time: 1-2 minutes

## Step 6: Configure Scene

1. In Hierarchy, create empty GameObject named `BrainSystem`
2. Select `BrainSystem`, add components in Inspector:
   - `BrainVisualization`
   - `BrainConfigLoader`
   - `BrainInteractionController`
   - `WebSocketClient` (optional)

3. Configure `BrainVisualization` component:
   - JSON Path: `Assets/Data/JSON/`
   - Load Sequence: ✓ (checked)
   - FPS: 10
   - Auto Play: ✓ (checked)
   - Region Scale: 1.0
   - Activity Threshold: 0.3
   - Show Connections: ✓ (checked)

4. Configure `BrainConfigLoader` component:
   - Config Path: `Assets/Data/unity_config.json` (if copied)
   - Auto Load: ✓ (checked)

⏱️ Estimated time: 2-3 minutes

## Step 7: Run!

Click Unity's Play button ▶️

You should see:
- 200 brain regions as colored spheres
- Colors represent activity intensity (blue=low, red=high)
- Auto-playing animation showing brain activity changes
- Connection lines between regions

### Basic Controls

- **Spacebar**: Play/Pause animation
- **R key**: Reload
- **Mouse hover**: Highlight region
- **Left click**: Select region

## (Optional) Step 8: Start Backend Server

For real-time features (prediction, stimulation simulation), start backend server:

```bash
python unity_automation.py --mode server
```

Server starts at `ws://localhost:8765`

Then in Unity's `WebSocketClient` component:
- Server URL: `ws://localhost:8765`
- Auto Connect: ✓

Rerun Unity project - it will auto-connect to server.

## Troubleshooting

### Issue 1: Can't see any brain regions

**Solution:**
- Lower Activity Threshold (e.g., to 0.1)
- Check JSON Path is correct
- Check Unity Console for errors

### Issue 2: Script compilation errors

**Solution:**
- Confirm Newtonsoft.Json package is installed
- Restart Unity
- Check all 7 scripts are copied

### Issue 3: Connections not showing

**Solution:**
- Confirm Show Connections is checked
- Lower Connection Threshold
- Confirm JSON files contain connections data

### Issue 4: Performance issues/lag

**Solution:**
- Increase Activity Threshold (show fewer regions)
- Uncheck Show Connections
- Lower FPS
- Use simpler Region Prefab

## Advanced Features

### Virtual Stimulation

1. Add to `BrainSystem` in Hierarchy:
   - `BrainRegionSelector`
   - `StimulationController`

2. Create UI Canvas and control buttons

3. Connect UI to components

4. Ensure backend server is running

5. In Unity:
   - Click to select target regions
   - Set stimulation parameters
   - Click "Apply Stimulation"
   - Observe predicted brain activity changes

### Prediction Comparison

1. Add `PredictionVisualizer` component

2. Press M key to cycle visualization modes:
   - Real Only: Show only real data
   - Prediction Only: Show only prediction
   - Side by Side: Compare side-by-side
   - Overlay: Overlay both

## Next Steps

- Read `unity_output/README_UNITY.md` for detailed features
- Customize colors and materials
- Add UI control panels
- Try VR mode
- Export visualization videos

## Get Help

- See [Detailed Documentation](unity_output/README_UNITY.md)
- See [System User Guide](docs/TwinBrain系统使用指南.md)
- See [Unity Workflow Guide](docs/Unity工作流说明.md)
- Submit [GitHub Issue](https://github.com/sheinclotho/twinbrain/issues)

## Total Time

- Basic visualization: ~5-10 minutes
- With real-time features: ~10-15 minutes

Enjoy! 🧠✨
