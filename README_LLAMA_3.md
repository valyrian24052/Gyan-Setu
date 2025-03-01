# Using Llama 3 with the Teacher Training Simulator

This guide explains how to use Llama 3 models locally with the Teacher Training Simulator application. The application now supports two methods for running Llama 3:

1. Using Ollama (recommended)
2. Using LlamaCpp with a downloaded GGUF model file

## Method 1: Using Ollama (Recommended)

[Ollama](https://ollama.ai/) is an easy-to-use tool for running LLMs locally. It handles downloading, setup, and optimization automatically.

### Step 1: Install Ollama

Visit [ollama.ai](https://ollama.ai/) and download the installer for your operating system.

### Step 2: Download Llama 3 Models

Open a terminal and run one of the following commands:

```bash
# Pull the 8B model (faster, requires less VRAM)
ollama pull llama3:8b

# Or pull the 70B model (higher quality, requires more VRAM)
ollama pull llama3:70b
```

### Step 3: Configure the Application

1. Launch the Teacher Training Simulator application
2. In the sidebar, select either "Llama 3 8B (Local)" or "Llama 3 70B (Local)" from the model dropdown
3. You don't need to change the Model Path if using Ollama

## Method 2: Using LlamaCpp with GGUF Files

If you prefer to use LlamaCpp directly, you'll need to download the GGUF model files.

### Step 1: Download GGUF Model Files

You can download Llama 3 GGUF files from [HuggingFace](https://huggingface.co/models?search=llama+3+gguf).

For example:
- [Llama 3 8B GGUF](https://huggingface.co/TheBloke/Llama-3-8B-GGUF)
- [Llama 3 70B GGUF](https://huggingface.co/TheBloke/Llama-3-70B-GGUF)

Download the appropriate quantized version based on your GPU VRAM:
- 4-bit quantization (Q4_K_M) for lower VRAM requirements
- 8-bit quantization (Q8_0) for higher quality with more VRAM

### Step 2: Place Model Files

Create a `models` directory in your project root and place the downloaded GGUF file there:

```bash
mkdir -p models
mv path/to/downloaded/llama-3-8b.Q4_K_M.gguf models/llama-3-8b.gguf
```

### Step 3: Configure the Application

1. Launch the Teacher Training Simulator application
2. In the sidebar, select either "Llama 3 8B (Local)" or "Llama 3 70B (Local)" from the model dropdown
3. Enter the path to your GGUF file in the "Model Path" field (e.g., `./models/llama-3-8b.gguf`)

## System Requirements

The hardware requirements depend on the model size:

### Llama 3 8B
- Minimum: 8GB RAM, integrated GPU or 4GB VRAM 
- Recommended: 16GB RAM, 6GB+ VRAM

### Llama 3 70B
- Minimum: 16GB RAM, 12GB VRAM
- Recommended: 32GB RAM, 24GB+ VRAM (split across multiple GPUs)

## Troubleshooting

### Common Issues

1. **"Model not found" error**:
   - Make sure you've either pulled the model with Ollama or specified the correct path to your GGUF file

2. **"Out of memory" error**:
   - Try a smaller model or a more aggressively quantized version
   - Close other applications consuming memory

3. **Very slow responses**:
   - Your hardware might not be powerful enough for the chosen model
   - Try using the 8B model instead of 70B
   - Try a more quantized version (e.g., Q4_K_M instead of Q8_0)

4. **"Failed to initialize Ollama" error**:
   - Ensure Ollama is installed and running
   - Try restarting the Ollama service
   - Verify the model name matches exactly (llama-3-8b or llama-3-70b)

For additional help, please file an issue on the project's GitHub repository. 