# Llama 3 Performance Tuning Guide

This guide explains the optimizations made to improve the performance of Llama 3 models on your dual NVIDIA RTX A4000 GPUs.

## Optimizations Implemented

### 1. Tensor Parallelism for Multi-GPU Usage

Your system has two RTX A4000 GPUs with 16GB VRAM each. We've configured the model to split evenly across both GPUs using tensor parallelism:

```python
main_gpu=0,               # Primary GPU
tensor_split=[0.5, 0.5],  # Even 50/50 split between GPUs
```

This effectively gives you 32GB of combined VRAM, allowing for larger models and batch sizes.

### 2. Context and Batch Size Optimization

We've adjusted these parameters based on model size:

```python
n_ctx=2048 if not is_large_model else 1024,  # Context window size
n_batch=512 if not is_large_model else 256,  # Batch size for processing
```

- For 8B model: 2048 token context with 512 batch size
- For 70B model: 1024 token context with 256 batch size

This balances memory usage with performance. You can adjust these values based on your specific needs.

### 3. Memory Efficiency Settings

We've added memory optimization flags:

```python
use_mlock=True,  # Prevents the OS from swapping memory to disk
use_mmap=True,   # Uses memory mapping for efficient loading
f16_kv=True,     # Half-precision for key/value cache
```

These significantly reduce memory usage and improve loading times.

### 4. RoPE Scaling for Enhanced Context Handling

We've enabled YARN RoPE scaling:

```python
rope_scaling={"type": "yarn", "factor": 1.0}
```

This improves the model's ability to handle context, especially for longer texts.

## Model Quantization Guide

The download script allows you to choose between different quantization levels:

| Quantization | Description | VRAM Usage (per GPU) | Use Case |
|--------------|-------------|----------------------|----------|
| Q4_K_M       | Fastest, smallest | ~2-3GB for 8B model | Speed-critical applications |
| Q5_K_M       | Balanced | ~3-4GB for 8B model | General purpose, good balance |
| Q8_0         | Highest quality | ~4-6GB for 8B model | Quality-critical applications |

With your dual RTX A4000 GPUs (16GB each), you can comfortably run:
- The 8B model with any quantization level
- The 70B model with Q4_K_M or Q5_K_M quantization

## Performance Tuning Tips

1. **Monitor GPU Usage**: Use `nvidia-smi` to monitor GPU memory usage during inference.

2. **Adjust Context Size**: If you need longer context, increase `n_ctx` but be aware it will use more VRAM.

3. **Batch Size Tuning**: For throughput-intensive workloads, try increasing `n_batch`. For latency-sensitive applications, decrease it.

4. **Temperature**: Lower temperature values (0.1-0.4) make responses more deterministic and faster but less creative.

5. **Quantization Selection**:
   - For speed: Use Q4_K_M
   - For quality: Use Q8_0
   - For balance: Use Q5_K_M

## Using the 70B Model

If you want to use the 70B model, the configuration has been optimized to split it across both GPUs. You should:

1. Use Q4_K_M quantization for best performance
2. Reduce the context size (n_ctx) to 1024 or lower if needed
3. Monitor GPU memory during usage to ensure it's not getting close to the limit

## Troubleshooting

1. **Out of Memory Errors**:
   - Reduce context size (`n_ctx`)
   - Use more aggressive quantization (Q4_K_M)
   - Reduce batch size (`n_batch`)

2. **Slow Inference**:
   - Check GPU utilization with `nvidia-smi`
   - Increase batch size for throughput
   - Try a different quantization level

3. **Models Not Loading**:
   - Ensure the model path is correct
   - Verify you have proper file permissions
   - Use the download script to get the correct model files 