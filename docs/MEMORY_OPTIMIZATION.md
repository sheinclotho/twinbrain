# CUDA Memory Optimization Guide

## Overview

This document describes the memory optimization strategies implemented in TwinBrain to prevent CUDA Out of Memory (OOM) errors during training.

## Problem

Training deep learning models with limited GPU memory can lead to OOM errors, particularly when:
- Processing large batch sizes or long sequences
- Using complex architectures with many parameters
- Accumulating gradients without proper cleanup
- Memory fragmentation occurs over time

## Solution

TwinBrain implements several memory management strategies to minimize GPU memory usage:

### 1. Environment Configuration

The CUDA memory allocator is configured to use expandable segments, which reduces memory fragmentation:

```python
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
```

This setting is applied automatically at module import time (before any CUDA operations).

### 2. Periodic Cache Clearing

CUDA cache is cleared at strategic points during training:
- **Start of each epoch**: Ensures clean slate for new epoch
- **After each batch**: Reclaims unused memory (frequency configurable)
- **End of each epoch**: Prevents accumulation across epochs

### 3. Explicit Tensor Cleanup

Large intermediate tensors are explicitly deleted after use:
- Loss tensors (total, align, temp, recon, etc.)
- Model outputs (z_dict, gru_seq_dict, proj_seq_dict, etc.)
- Prediction window tensors (in sliding window loop)

**Why explicit deletion?** While Python's garbage collector will eventually collect these tensors, GPU memory is managed separately by CUDA. Explicit deletion before calling `torch.cuda.empty_cache()` allows immediate GPU memory reclamation rather than waiting for the next GC cycle.

### 4. Gradient Accumulation

Gradient accumulation allows training with larger effective batch sizes while using less memory:

```yaml
training:
  gradient_accumulation_steps: 2  # Accumulate gradients over 2 steps
```

This splits the batch into smaller chunks, processes them separately, and accumulates gradients before updating weights.

## Configuration

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
