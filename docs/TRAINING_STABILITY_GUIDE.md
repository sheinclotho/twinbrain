# TwinBrain Training Stability Guide

This guide explains the stability improvements made to TwinBrain training and provides best practices for stable training runs.

## Problem Overview

Training could previously hang silently after the "Starting stage: Warmup Stage" message with:
- No error messages or stack traces
- No computation activity (silent CPU/GPU)
- Manual process termination required
- No indication of what went wrong

## Solution Architecture

The training stability system consists of five key components:

### 1. Pre-Training Validation

Before training begins, the system validates:

```python
✓ Data list contains N batches
✓ Model and GraphEncoder initialized
✓ Device (CPU/CUDA) availability
✓ CUDA memory available
✓ First batch structure inspection
  - Node types present
  - Edge types present
  - Tensor shapes and dtypes
```

**Why this helps**: Catches configuration and data issues before computation starts, providing immediate feedback.

### 2. Heartbeat Monitoring

A background thread monitors training progress:

```python
# TrainingHeartbeat class
- Logs heartbeat every 30 seconds
- Warns if no activity for 60 seconds  
- Updates on every 5th batch
- Automatically detects stalls
```

**Why this helps**: Provides continuous proof-of-life even during long computations. Automatically detects when training has stalled.

### 3. Progress Logging

Detailed progress information at multiple levels:

```python
[Epoch 1/100] Processing batch 1/3...  # Every batch in epoch 1
[Epoch 2/100] Processing batch 10/3... # Every 10th batch thereafter
[Heartbeat] Training active (last activity 3.2s ago)
```

**Why this helps**: Users can monitor training progress and identify which stage is slow.

### 4. Comprehensive Error Handling

All critical operations wrapped in try-catch blocks:

```python
# GraphEncoder forward
try:
    enc_out = self.graph_encoder(data)
    torch.cuda.synchronize()  # Ensure GPU completes
except Exception as e:
    logger.error(f"GraphEncoder failed at epoch {epoch}, batch {idx}: {e}")
    raise RuntimeError(f"GraphEncoder forward pass failed: {e}")
```

**Operations protected**:
- Data transfer to device
- GraphEncoder forward pass
- Model forward pass
- Alignment loss computation
- Prediction loss computation
- Temporal prediction loss
- Backward pass and optimization

**Why this helps**: All errors are caught with context (epoch, batch) and reported immediately instead of hanging silently.

### 5. CUDA Synchronization

Explicit synchronization after GPU operations:

```python
if self.device.type == 'cuda':
    torch.cuda.synchronize()
```

**Applied after**:
- data.to(device)
- GraphEncoder forward
- Model forward
- Backward pass

**Why this helps**: Ensures GPU kernels complete before continuing, catching GPU errors immediately instead of hanging.

## Reading Training Logs

### Normal Training

```
INFO | Validating training setup...
INFO |   ✓ Data list contains 3 batches
INFO |   ✓ Model and GraphEncoder initialized
INFO |   ✓ Using CUDA device: NVIDIA GeForce RTX 3080
INFO |   ✓ First batch inspection:
INFO |     - fmri.x_seq shape: (200, 384, 200), dtype: torch.float32
INFO |     - eeg.x_seq shape: (62, 500, 62), dtype: torch.float32
INFO | ✓ Heartbeat monitor started
INFO | [Epoch 1/5] Processing batch 1/3...
INFO | [Epoch 1] Running GraphEncoder on batch 0...
INFO | [Epoch 1] GraphEncoder completed successfully
INFO | [Epoch 1] Running model forward pass...
INFO | [Epoch 1] Model forward completed successfully
INFO | [Heartbeat] Training active (last activity 3.2s ago)
INFO | [Epoch  1] total=2.456 align=0.345 temp=0.567 recon=1.234 time=12.3s
```

### Stall Detected

```
INFO | [Heartbeat] Training active (last activity 28.3s ago)
WARN | [Heartbeat] No activity for 61.2s - possible stall detected!
```

### Error Occurred

```
ERROR | GraphEncoder failed at epoch 1, batch 0: CUDA out of memory
ERROR | Data type: <class 'torch_geometric.data.hetero_data.HeteroData'>, Device: cuda:0
RuntimeError: GraphEncoder forward pass failed: CUDA out of memory
```

## Troubleshooting Guide

### Issue: "CUDA out of memory"

**Cause**: Batch too large for GPU memory

**Solutions**:
1. Reduce batch size in config
2. Enable gradient accumulation:
   ```yaml
   training:
     gradient_accumulation_steps: 4
   ```
3. Reduce model hidden_dim
4. Use CPU instead: `device: cpu` in config

### Issue: "No activity for 60s - possible stall detected"

**Cause**: Very long computation or actual hang

**Investigation**:
1. Check GPU/CPU usage with `nvidia-smi` or `htop`
2. If usage is high → computation is running (just slow)
3. If usage is 0% → actual hang, check error logs

**Solutions**:
- If computation is just slow, increase heartbeat interval
- If actual hang, check the error message before the warning

### Issue: "data_list is empty or not initialized"

**Cause**: Data loading failed

**Investigation**:
1. Check data paths in config
2. Verify data files exist
3. Check file permissions
4. Review data loading logs

### Issue: "Model or GraphEncoder not initialized"

**Cause**: Model creation failed

**Investigation**:
1. Check model parameters in config
2. Verify all required modules are installed
3. Check for import errors in logs

### Issue: Slow first epoch

**This is normal!** The first epoch includes:
- JIT compilation of CUDA kernels
- Memory allocation and optimization
- One-time initialization

Subsequent epochs should be faster.

## Best Practices

### 1. Monitor Training Logs

Always watch the logs during training start:
```bash
# Real-time log monitoring
tail -f path/to/logfile.log
```

### 2. Start Small

For new datasets or configurations:
1. Test with 1-2 epochs first
2. Test with small batch size
3. Scale up after validation

### 3. Use Validation Runs

Before full training:
```yaml
training:
  warmup_epochs: 1
  main_epochs: 2
  finetune_epochs: 0
```

### 4. Check Data First

Run diagnostic mode if available:
```python
workflow = TrainingWorkflow(config, base_dir)
workflow._run_diagnostics(trainer)  # Before training
```

### 5. GPU Memory Management

Monitor GPU memory:
```bash
watch -n 1 nvidia-smi
```

Set memory limits if needed:
```python
torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of GPU
```

## Configuration Examples

### Minimal Stable Config

```yaml
model:
  hidden_dim: 64  # Smaller than default

training:
  warmup_epochs: 2
  main_epochs: 10
  finetune_epochs: 5
  gradient_accumulation_steps: 1

diagnostics:
  enabled: true  # Always enable for debugging
```

### Large Dataset Config

```yaml
model:
  hidden_dim: 128

training:
  warmup_epochs: 5
  main_epochs: 60
  finetune_epochs: 30
  gradient_accumulation_steps: 4  # Simulate larger batches

data:
  use_cache: true  # Cache preprocessed data
```

### Debug Config

```yaml
training:
  warmup_epochs: 1
  main_epochs: 1
  finetune_epochs: 0

diagnostics:
  enabled: true
  save_plots: true

metrics:
  enabled: true
```

## Understanding Heartbeat Messages

| Message | Meaning | Action |
|---------|---------|--------|
| `✓ Heartbeat monitor started` | Monitoring active | Normal |
| `Training active (last activity Xs ago)` | Training progressing | Normal |
| `No activity for 60s - possible stall!` | Potential problem | Check GPU/CPU usage |
| `✓ Training completed successfully` | Training finished | Success! |
| `✓ Heartbeat monitor stopped` | Cleanup complete | Normal shutdown |

## Advanced Debugging

### Enable Debug Logging

```python
import logging
logging.getLogger("DynamicHeteroTrainer").setLevel(logging.DEBUG)
```

### Profile GPU Operations

```python
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    trainer.train(num_epochs=1)
print(prof.key_averages().table(sort_by="cuda_time_total"))
```

### Memory Profiling

```python
import torch
torch.cuda.memory_summary()
```

## Summary

The training stability improvements provide:

✅ **Prevention**: Pre-training validation catches issues early  
✅ **Detection**: Heartbeat monitor detects stalls automatically  
✅ **Diagnosis**: Detailed error messages with context  
✅ **Recovery**: Graceful error handling and cleanup  
✅ **Transparency**: Continuous progress feedback  

These improvements make training more reliable and easier to debug when issues occur.

## Support

If you encounter issues not covered in this guide:

1. Check the CHANGELOG.md for recent updates
2. Enable diagnostic mode
3. Check GPU/CPU usage during hang
4. Share relevant log excerpts
5. Include configuration file

## Related Documentation

- `CHANGELOG.md` - Update history
- `README.md` - General usage
- `config/README.md` - Configuration guide
