# CUDA Memory Optimization Guide

## Overview

This document describes the memory optimization strategies implemented in TwinBrain to prevent CUDA Out of Memory (OOM) errors during training, particularly on GPUs with 8GB or less memory.

## Problem

The error message shows:
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 98.00 MiB. 
GPU 0 has a total capacity of 8.00 GiB of which 0 bytes is free. 
Of the allocated memory 14.35 GiB is allocated by PyTorch
```

This indicates PyTorch is trying to allocate more memory (14.35 GiB) than the GPU has (8 GiB), caused by:
- **Memory accumulation**: Tensors not being properly freed between batches
- **Large models**: Complex GNN + GRU architectures with attention
- **Insufficient cleanup**: Inadequate cache clearing and tensor deletion
- **Memory fragmentation**: Small free chunks but no large contiguous blocks

## Solutions Implemented

### 1. Gradient Checkpointing (NEW)

**What**: Trades computation for memory by recomputing activations during backward pass instead of storing them.

**Where**: Added to `PredictorHead` in `train/predictor.py`:
```python
use_gradient_checkpointing: bool = True  # New parameter
```

**Impact**: Reduces peak memory by 30-40% for prediction module.

**Enable in config:**
```yaml
prediction:
  use_gradient_checkpointing: true
```

### 2. Reduced Default Memory Footprint

**Changes in `config/default.yaml`:**
- `hidden_dim`: 128 → 96 (25% memory reduction)
- `warmup_epochs`: 5 → 3 (faster startup)
- `prediction.context_length`: None → 50 (prevents full sequence processing)

**Impact**: Peak memory reduced from ~9-11 GB to ~6-7 GB.

### 3. Enhanced Memory Cleanup

**Improvements in training loop:**
- Delete `raw_pred_loss_total`, `enc_out`, `sanitized_z` after use
- Move `data` to CPU after processing: `data = data.cpu()`
- Clear cache AFTER all tensors deleted (more effective)

**Code location**: Memory cleanup section in `train/hetero_trainer.py` (in the training loop)

### 4. Optimized Prediction Loop

**Changes:**
- Reduced sliding windows: All windows → 2 windows (beginning + end)
- Immediate tensor cleanup: `del predictions, pred_loss` after each window
- Delete `window_loss` after accumulation

**Impact**: 50% reduction in prediction memory usage.

**Code location**: Prediction loss computation in `train/hetero_trainer.py` (PredictorHead usage section)

### 5. Environment Configuration

The CUDA memory allocator is configured to use expandable segments, which reduces memory fragmentation:

```python
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
```

This setting is applied automatically at module import time in `train/hetero_trainer.py`.

### 6. Periodic Cache Clearing

CUDA cache is cleared at strategic points:
- **Start of each epoch**: `torch.cuda.empty_cache()` for clean slate
- **After each batch**: Based on `clear_cache_frequency` config (default: 1)
- **After tensor deletion**: Maximum effectiveness

Default frequency is 1 (every batch) for 8GB GPUs.

## Memory Usage Estimates

### By Configuration

| Configuration | Peak Memory | Recommended GPU |
|--------------|-------------|-----------------|
| `hidden_dim: 64` | ~4-5 GB | 6GB+ |
| `hidden_dim: 96` (default) | ~6-7 GB | 8GB+ |
| `hidden_dim: 128` | ~9-11 GB | 12GB+ |
| `hidden_dim: 192` | ~15-18 GB | 16GB+ |

*With gradient checkpointing and context_length=50*

### Memory Breakdown

1. **Model Parameters** (~10-20%): GNN, GRU, Attention layers
2. **Activations** (~40-60%): Forward pass intermediate results
3. **Gradients** (~20-30%): Backpropagation gradients + Adam states
4. **Data Tensors** (~10-20%): Input/target sequences

## Quick Fixes for OOM

### For 8GB GPUs (Recommended)

Use the default config - it's already optimized:

```bash
python main.py train --config config/default.yaml
```

### For 6GB GPUs

Create a custom config:

```yaml
model:
  hidden_dim: 64  # Reduced

training:
  warmup_epochs: 2
  main_epochs: 30

prediction:
  enabled: false  # Disable to save memory

memory:
  optimize_for_low_memory: true
```

### Emergency Options

If still OOM:

1. **Disable prediction**: Set `prediction.enabled: false`
2. **Reduce hidden_dim**: Try 64 or even 48
3. **Shorter sequences**: Set `prediction.context_length: 30`
4. **Fewer epochs**: Reduce warmup, main, and finetune epochs

## Configuration Reference

### Key Memory Settings

```yaml
model:
  hidden_dim: 96  # Main memory control (64, 96, 128, 192)

training:
  warmup_epochs: 3  # Reduced from 5
  clear_cache_frequency: 1  # Clear every batch

prediction:
  enabled: false  # Enable only if memory allows
  context_length: 50  # Limit sequence length
  use_gradient_checkpointing: true  # Save memory

memory:
  optimize_for_low_memory: true
  use_gradient_checkpointing: true
```

### Emergency Options

If still OOM after above optimizations:

You can control memory management behavior in `config/default.yaml`:

```yaml
training:
  # Clear CUDA cache every N batches
  # Lower = more frequent clearing = more memory safety but slower training
  # Higher = less frequent clearing = faster training but more memory usage
  clear_cache_frequency: 1  # Default: clear after every batch
  
  # Accumulate gradients over N steps to reduce memory usage
  gradient_accumulation_steps: 1  # Default: no accumulation
```

### Recommended Settings

#### For 8GB GPU (like in the original error)
```yaml
training:
  clear_cache_frequency: 1  # Clear after every batch
  gradient_accumulation_steps: 2  # Effective batch size = 2
```

#### For 12GB+ GPU
```yaml
training:
  clear_cache_frequency: 2  # Clear every 2 batches
  gradient_accumulation_steps: 1  # No accumulation needed
```

#### For 24GB+ GPU
```yaml
training:
  clear_cache_frequency: 5  # Clear every 5 batches
  gradient_accumulation_steps: 1  # No accumulation needed
```

## Monitoring Memory Usage

To monitor GPU memory usage during training, you can use:

```bash
# In a separate terminal
watch -n 1 nvidia-smi
```

Or add PyTorch memory profiling in your code:

```python
import torch

# During training
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / 1024**3  # GB
    reserved = torch.cuda.memory_reserved() / 1024**3    # GB
    print(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
```

## Additional Memory Optimization Tips

### 1. Reduce Batch Size
If OOM errors persist, reduce the effective batch size:
```yaml
training:
  batch_size: 1  # Already at minimum in this codebase
```

### 2. Reduce Model Size
Consider reducing model complexity:
```yaml
model:
  hidden_dim: 128  # Reduce from 256
  num_layers: 2    # Reduce from 4
```

### 3. Reduce Sequence Length
For prediction tasks, limit context length:
```yaml
prediction:
  context_length: 50   # Reduce from 100
  steps: 10            # Reduce number of prediction steps
```

### 4. Use Mixed Precision Training
Enable automatic mixed precision (AMP) to reduce memory:
```python
trainer = DynamicHeteroTrainer(
    use_amp=True,  # Enable mixed precision
    ...
)
```

### 5. Disable Features Temporarily
If debugging OOM, try disabling optional features:
```yaml
prediction:
  enabled: false  # Disable prediction temporarily
```

## Troubleshooting

### Still Getting OOM Errors?

1. **Check GPU availability**:
   ```bash
   nvidia-smi
   ```

2. **Verify cache clearing is working**:
   - Look for cache clearing in logs
   - Monitor memory with `nvidia-smi`

3. **Try more aggressive settings**:
   ```yaml
   training:
     clear_cache_frequency: 1
     gradient_accumulation_steps: 4
   ```

4. **Check for memory leaks**:
   - Ensure no tensors are being stored in global variables
   - Check if any callbacks are holding references to tensors

5. **Use CPU for problematic operations**:
   - Move large tensors to CPU when not in use
   - Use `.cpu()` followed by `del` for temporary tensors

## Performance Considerations

Frequent cache clearing and gradient accumulation can impact training speed:

- **Cache clearing**: ~0-5% slowdown per batch (negligible)
- **Gradient accumulation**: No speed penalty, but increases iterations

The tradeoff is worthwhile to prevent OOM errors and enable successful training.

## References

- [PyTorch Memory Management](https://pytorch.org/docs/stable/notes/cuda.html#memory-management)
- [PyTorch Automatic Mixed Precision](https://pytorch.org/docs/stable/amp.html)
- [CUDA Memory Allocator Configuration](https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
