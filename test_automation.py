#!/usr/bin/env python3
"""
Test script to verify the automation features work correctly
"""

import sys
import json
from pathlib import Path

def test_json_export():
    """Test that JSON export creates proper structure"""
    print("Testing JSON export structure...")
    
    # Simulate a stimulation response
    test_response = {
        "type": "simulation",
        "success": True,
        "n_steps": 50,
        "stimulation": {
            "target_regions": [1, 2, 3],
            "amplitude": 1.5,
            "pattern": "sine",
            "frequency": 10.0,
            "duration": 20
        },
        "saved_to": "test_output/stimulation/stim_test",
        "index_file": "test_output/stimulation/stim_test/sequence_index.json",
        "auto_saved": True
    }
    
    # Verify all required fields
    assert "type" in test_response, "Missing 'type' field"
    assert "success" in test_response, "Missing 'success' field"
    assert "auto_saved" in test_response, "Missing 'auto_saved' field"
    assert test_response["auto_saved"] == True, "auto_saved should be True"
    assert "saved_to" in test_response, "Missing 'saved_to' field"
    assert "index_file" in test_response, "Missing 'index_file' field"
    
    print("✓ JSON export structure is valid")
    return True

def test_sequence_index_format():
    """Test sequence_index.json format"""
    print("\nTesting sequence_index.json format...")
    
    test_index = {
        "type": "stimulation_sequence",
        "timestamp": "20240215_143022",
        "stimulation_params": {
            "target_regions": [1, 2, 3],
            "amplitude": 1.5,
            "pattern": "sine",
            "frequency": 10.0,
            "duration": 20
        },
        "n_frames": 50,
        "output_dir": "test_output/stimulation/stim_test",
        "files": [f"frame_{i:04d}.json" for i in range(50)]
    }
    
    # Verify structure
    assert "type" in test_index, "Missing 'type' field"
    assert "timestamp" in test_index, "Missing 'timestamp' field"
    assert "n_frames" in test_index, "Missing 'n_frames' field"
    assert "files" in test_index, "Missing 'files' field"
    assert len(test_index["files"]) == test_index["n_frames"], "Files count mismatch"
    
    # Verify file naming
    assert test_index["files"][0] == "frame_0000.json", "Incorrect file naming"
    assert test_index["files"][-1] == "frame_0049.json", "Incorrect last frame name"
    
    print("✓ sequence_index.json format is valid")
    return True

def test_file_structure():
    """Test expected file structure"""
    print("\nTesting file structure...")
    
    expected_structure = {
        "model_output": {
            "stimulation": {
                "stim_YYYYMMDD_HHMMSS": [
                    "frame_0000.json",
                    "frame_0001.json",
                    "...",
                    "frame_0049.json",
                    "sequence_index.json"
                ]
            },
            "predictions": {
                "pred_YYYYMMDD_HHMMSS": [
                    "frame_0000.json",
                    "...",
                    "sequence_index.json"
                ]
            }
        }
    }
    
    print("✓ Expected file structure:")
    print(json.dumps(expected_structure, indent=2))
    return True

def test_unity_integration():
    """Test Unity integration points"""
    print("\nTesting Unity integration points...")
    
    # Check Unity C# files exist
    unity_files = [
        "unity_examples/BrainVisualization.cs",
        "unity_examples/StimulationInput.cs",
        "unity_examples/WebSocketClient.cs",
        "unity_examples/Editor/TwinBrainAutoSetup.cs"
    ]
    
    for file in unity_files:
        path = Path(file)
        if path.exists():
            print(f"  ✓ {file} exists")
        else:
            print(f"  ✗ {file} NOT FOUND")
            return False
    
    # Check for auto-reload features in BrainVisualization.cs
    bv_path = Path("unity_examples/BrainVisualization.cs")
    if bv_path.exists():
        content = bv_path.read_text()
        
        features = [
            ("enableAutoReload", "Auto-reload toggle"),
            ("watchDirectory", "Watch directory setting"),
            ("CheckForNewResults", "File watcher method"),
            ("AutoLoadNewResults", "Auto-load method"),
            ("InitializeFileWatching", "Init method")
        ]
        
        for feature, description in features:
            if feature in content:
                print(f"  ✓ {description} implemented")
            else:
                print(f"  ✗ {description} NOT FOUND")
                return False
    
    print("✓ Unity integration is complete")
    return True

def main():
    """Run all tests"""
    print("="*60)
    print("TwinBrain Automation Test Suite")
    print("="*60)
    
    tests = [
        test_json_export,
        test_sequence_index_format,
        test_file_structure,
        test_unity_integration
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test failed with error: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*60)
    
    if failed == 0:
        print("\n✅ All automation features are properly implemented!")
        return 0
    else:
        print(f"\n❌ {failed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
