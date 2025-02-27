"""
Language Model Interface and Processing Module

This module provides a comprehensive interface for interacting with Language
Models (LLMs) in the teaching simulation system. It handles all communication
with the LLM backend while providing context management, error handling, and
pedagogical processing capabilities.

Key Features:
    - Standardized LLM communication
    - Multi-GPU tensor parallelism for larger models
    - Advanced quantization techniques
    - Context-aware prompt building
    - Pedagogical language processing
    - Teaching response analysis
    - Student reaction generation
    - Error handling and recovery
    - Configurable model parameters

Components:
    - EnhancedLLMInterface: Multi-GPU optimized interface for LLM interaction
    - PedagogicalLanguageProcessor: Educational language processor
    - TeachingContext: Data class for maintaining teaching context
    - Prompt Management: Context-aware prompt construction

Dependencies:
    - transformers: For model loading and pipeline management
    - torch: For GPU operations and tensor manipulation
    - bitsandbytes: For quantization support
    - typing: For type hints

Example:
    llm = EnhancedLLMInterface()
    response = llm.generate(
        prompt="Explain addition to a second grader",
        context={"grade_level": "2nd", "subject": "math"}
    )
    
    processor = PedagogicalLanguageProcessor()
    analysis = processor.analyze_teaching_response("Let's use blocks to count", context)

---------------------------------------------------------------------------------------
AVAILABLE MODELS AND QUANTIZATION OPTIONS
---------------------------------------------------------------------------------------

The following models can be used with the EnhancedLLMInterface, along with recommended
quantization settings for the UTTA server's dual NVIDIA 4000 series GPUs (32GB total VRAM):

1. LLAMA MODELS:
   - Llama-2-7b: ~7B parameters
     * FP16: ~14GB VRAM
     * 8-bit: ~7GB VRAM
     * 4-bit: ~3.5GB VRAM
     * Suitable for: General text generation, single-GPU deployment
     
   - Llama-2-13b: ~13B parameters
     * FP16: ~26GB VRAM
     * 8-bit: ~13GB VRAM
     * 4-bit: ~6.5GB VRAM
     * Suitable for: Enhanced reasoning, dual-GPU with 8-bit quantization
   
   - Llama-3-8b: ~8B parameters
     * FP16: ~16GB VRAM
     * 8-bit: ~8GB VRAM
     * 4-bit: ~4GB VRAM
     * Suitable for: Better performance than Llama-2-7b with similar size

2. MISTRAL MODELS:
   - Mistral-7B: ~7B parameters
     * FP16: ~14GB VRAM
     * 8-bit: ~7GB VRAM
     * 4-bit: ~3.5GB VRAM
     * Suitable for: Excellent reasoning with smaller size
   
   - Mixtral-8x7B: ~45B parameters (MoE architecture)
     * 4-bit with offloading: ~23GB VRAM
     * Suitable for: Advanced tasks with extensive CPU offloading

3. PHI/MICROSOFT MODELS:
   - Phi-2: ~2.7B parameters
     * FP16: ~5.4GB VRAM
     * 8-bit: ~2.7GB VRAM
     * Suitable for: Lightweight deployment, high efficiency
   
   - Phi-3-mini: ~3.8B parameters
     * FP16: ~7.6GB VRAM
     * 8-bit: ~3.8GB VRAM
     * Suitable for: Improved capabilities over Phi-2

4. GEMMA MODELS:
   - Gemma-2B: ~2B parameters
     * FP16: ~4GB VRAM
     * Suitable for: Very lightweight deployment
   
   - Gemma-7B: ~7B parameters
     * FP16: ~14GB VRAM
     * 8-bit: ~7GB VRAM
     * Suitable for: Good balance of performance and efficiency

5. FALCON MODELS:
   - Falcon-7B: ~7B parameters
     * FP16: ~14GB VRAM
     * 8-bit: ~7GB VRAM
     * Suitable for: General purpose tasks

QUANTIZATION METHODS AND TRADEOFFS:
-----------------------------------
1. FULL PRECISION (FP32):
   - Memory Usage: Highest (~4 bytes per parameter)
   - Quality: Best possible quality
   - Speed: Slowest
   - When to use: Almost never for LLMs due to memory constraints

2. HALF PRECISION (FP16):
   - Memory Usage: High (~2 bytes per parameter)
   - Quality: Excellent, virtually indistinguishable from FP32
   - Speed: Faster than FP32, optimal on most modern GPUs
   - When to use: When you have sufficient VRAM and want best quality

3. 8-BIT QUANTIZATION (INT8):
   - Memory Usage: Medium (~1 byte per parameter)
   - Quality: Very good, minimal degradation for most use cases
   - Speed: Comparable to FP16 on supported hardware
   - When to use: Standard choice for balancing quality and memory usage

4. 4-BIT QUANTIZATION (INT4):
   - Memory Usage: Low (~0.5 bytes per parameter)
   - Quality: Good for most tasks, may degrade for specialized domains
   - Speed: Can be slower than 8-bit on some operations
   - When to use: When memory is severely constrained

5. MIXED PRECISION STRATEGIES:
   - GPTQ: Post-training quantization method, good balance of speed/quality
   - AWQ: Activation-aware quantization, better quality than GPTQ in some cases
   - QLoRA: Allows fine-tuning while keeping weights quantized

MULTI-GPU DISTRIBUTION OPTIONS:
------------------------------
1. TENSOR PARALLELISM:
   - Splits individual tensors across multiple GPUs
   - Best for: Models that barely fit in combined VRAM
   - Tradeoff: Some communication overhead

2. PIPELINE PARALLELISM:
   - Splits model layers across GPUs
   - Best for: Very deep models
   - Tradeoff: Higher latency, complex implementation

3. DATA PARALLELISM:
   - Processes multiple inputs in parallel
   - Best for: Batch processing, training
   - Tradeoff: Doesn't help with model size constraints

This implementation uses tensor parallelism via HuggingFace's "device_map=auto"
to automatically distribute model weights across available GPUs.
"""

import os
import requests
import json
import time
import torch
import signal
import subprocess
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

@dataclass
class TeachingContext:
    """
    Data class for maintaining teaching context and state.
    
    This class encapsulates all relevant information about the current
    teaching situation, student characteristics, and interaction history.
    It serves as a central point for context management in the teaching
    simulation.
    
    Attributes:
        subject (str): The subject being taught (e.g., "math", "reading")
        grade_level (str): The grade level of the student (e.g., "2nd", "3rd")
        learning_objectives (List[str]): Specific goals for the lesson
        student_characteristics (Dict[str, Any]): Student traits including:
            - learning_style: Preferred learning method
            - attention_span: Attention capacity
            - strengths: Areas of proficiency
            - challenges: Areas needing support
        previous_interactions (Optional[List[Dict[str, str]]]): History of
            recent teacher-student interactions
    
    Example:
        context = TeachingContext(
            subject="mathematics",
            grade_level="2nd",
            learning_objectives=["Add two-digit numbers"],
            student_characteristics={
                "learning_style": "visual",
                "attention_span": "moderate"
            }
        )
    """
    subject: str
    grade_level: str
    learning_objectives: List[str]
    student_characteristics: Dict[str, Any]
    previous_interactions: Optional[List[Dict[str, str]]] = None

class EnhancedLLMInterface:
    """
    Enhanced interface for running large language models across multiple GPUs
    with precise control over quantization and memory optimization.
    
    Features:
        - Tensor parallelism across multiple GPUs
        - Advanced quantization techniques
        - Memory-efficient attention mechanisms
        - Configurable model loading parameters
        - Optimized inference pipelines
    
    Attributes:
        model_name (str): HuggingFace model identifier
        max_tokens (int): Maximum response length
        temperature (float): Response randomness (0-1)
        gpu_count (int): Number of available GPUs
    
    Example:
        llm = EnhancedLLMInterface()
        response = llm.generate(
            prompt="How would you explain fractions?",
            context={"student_level": "beginner"}
        )
    """
    
    # List of open-access models that don't require authentication
    OPEN_ACCESS_MODELS = {
        "microsoft/phi-2": {
            "parameters": 2.7e9,
            "parameters_str": "2.7 billion",
            "architecture": "Phi-2",
            "context_length": 2048
        },
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0": {
            "parameters": 1.1e9,
            "parameters_str": "1.1 billion",
            "architecture": "TinyLlama",
            "context_length": 2048
        },
        "google/gemma-2b": {
            "parameters": 2e9,
            "parameters_str": "2 billion",
            "architecture": "Gemma",
            "context_length": 8192
        }
    }
    
    # Define a default fallback model that is guaranteed to work
    DEFAULT_FALLBACK_MODEL = "microsoft/phi-2"
    
    # Gated models requiring Hugging Face authentication
    GATED_MODELS = {
        "meta-llama/Llama-2-7b-hf": {
            "parameters": 7e9,
            "parameters_str": "7 billion",
            "architecture": "Llama 2",
            "context_length": 4096
        },
        "meta-llama/Llama-2-13b-hf": {
            "parameters": 13e9,
            "parameters_str": "13 billion",
            "architecture": "Llama 2",
            "context_length": 4096
        },
        "meta-llama/Llama-3-8b-hf": {
            "parameters": 8e9,
            "parameters_str": "8 billion",
            "architecture": "Llama 3",
            "context_length": 8192
        },
        "mistralai/Mistral-7B-v0.1": {
            "parameters": 7e9,
            "parameters_str": "7 billion",
            "architecture": "Mistral",
            "context_length": 8192
        }
    }
    
    def __init__(self, model_name=None, quantization="8-bit", token=None):
        """
        Initialize the enhanced LLM interface with multi-GPU optimization
        
        This method sets up the configuration for running models across multiple GPUs
        with optimized memory usage. It does not load the model immediately (lazy loading)
        to conserve resources until generation is actually needed.
        
        Args:
            model_name (str, optional): The HuggingFace model identifier. If None, a default
                                      open-access model will be selected based on available resources.
            quantization (str): Quantization level to use - "FP16", "8-bit", or "4-bit"
            token (str, optional): Hugging Face access token for accessing gated models. If None, will try
                                to use HF_TOKEN environment variable.
        """
        # Store Hugging Face access token
        self.hf_token = token or os.environ.get("HF_TOKEN", None)
        
        # Detect available GPU resources
        # The code automatically adjusts to the number of GPUs available in the system
        self.gpu_count = torch.cuda.device_count()
        if self.gpu_count < 2:
            print("Warning: Only one GPU detected. Multi-GPU optimizations won't be applied.")
        
        # Calculate available GPU memory in GB
        total_gpu_memory = 0
        for i in range(self.gpu_count):
            try:
                total_gpu_memory += torch.cuda.get_device_properties(i).total_memory / (1024**3)
            except:
                pass
        
        # Define default model configuration - automatically select based on available resources
        # If the user didn't specify a model, choose an appropriate one based on available GPU memory
        if model_name is None:
            # Default to Phi-2 which is reliable and works well in most cases
            self.model_name = "microsoft/phi-2"  # 2.7B model, good performance for size
            
            # For systems with more memory, suggest larger models but don't auto-select
            if total_gpu_memory > 24:  # Lots of GPU memory
                print(f"Using microsoft/phi-2 model. With {total_gpu_memory:.1f} GB GPU memory, you could use larger models.")
                print("Consider specifying a larger model like 'meta-llama/Llama-3-8b-hf' (requires authentication)")
            else:
                print(f"Using microsoft/phi-2 model based on available GPU memory ({total_gpu_memory:.1f} GB)")
        else:
            self.model_name = model_name
        
        # Response generation parameters
        self.max_tokens = 2000   # Controls maximum length of generated text
        self.temperature = 0.7   # Controls randomness: 0=deterministic, 1=creative
        
        # Track loaded models
        # These are initialized as None for lazy loading
        # Models will only be loaded when first required for generation
        self.loaded_model = None
        self.loaded_tokenizer = None
        self.generation_pipeline = None
        self.quantization_mode = quantization  # Set quantization based on parameter
        
        # Check if model requires authentication
        is_gated = self.model_name in self.GATED_MODELS
        if is_gated:
            if self.hf_token:
                print(f"⚠️  Note: {self.model_name} is a gated model. Authentication token has been provided.")
            else:
                print(f"⚠️  Note: {self.model_name} is a gated model requiring Hugging Face authentication.")
                print("   If you encounter authentication errors, you may need to:")
                print("   1. Create a Hugging Face account at https://huggingface.co/")
                print("   2. Accept the model's license terms on the model's page")
                print("   3. Run `huggingface-cli login` in your terminal and follow the prompts")
                print("   4. Or use an open-access model instead")
                print(f"   Available open-access models: {', '.join(self.OPEN_ACCESS_MODELS.keys())}")
        
        # Print detailed model configuration information
        self.print_model_info(detailed=False)
        
        # Log initialization
        print(f"EnhancedLLMInterface initialized. Will use {self.gpu_count} GPUs for model distribution.")
    
    def set_token(self, token):
        """
        Set Hugging Face access token after initialization
        
        Args:
            token (str): Hugging Face access token for accessing gated models
            
        Returns:
            bool: True if token was set successfully, False otherwise
        """
        if not token or token.strip() == "":
            print("⚠️ No token provided. Gated models will not be accessible.")
            self.hf_token = None
            return False
            
        # Store the token
        self.hf_token = token
        print("✅ Hugging Face token set successfully.")
        
        # Check if current model is gated
        is_gated = self.model_name in self.GATED_MODELS
        if is_gated:
            # Verify the token works with this model
            is_accessible, _, _ = self._check_model_accessibility(self.model_name)
            if is_accessible:
                print(f"✅ Token successfully verified for {self.model_name}")
            else:
                print(f"⚠️ Token authentication failed for {self.model_name}. The model may be inaccessible.")
                print(f"You may need to accept the model terms at: https://huggingface.co/{self.model_name}")
            
            print(f"Current model {self.model_name} is gated. Token will be used for authentication.")
        else:
            print(f"Note: Current model {self.model_name} is open-access and doesn't require authentication.")
            print(f"Token will be used if you switch to a gated model.")
            
        return True
        
    def _check_model_accessibility(self, model_name):
        """
        Check if a model is accessible before attempting to load it
        
        Args:
            model_name: Hugging Face model identifier
            
        Returns:
            tuple: (is_accessible, fallback_model, error_message)
        """
        import requests
        
        # First check if model is in our known good lists
        is_open_access = model_name in self.OPEN_ACCESS_MODELS
        is_gated = model_name in self.GATED_MODELS
        
        # If not in either list, we should be cautious about its existence
        if not (is_open_access or is_gated):
            try:
                # Try to see if the model exists at all by checking the model page
                response = requests.head(f"https://huggingface.co/{model_name}", timeout=5)
                if response.status_code != 200:
                    return False, self.DEFAULT_FALLBACK_MODEL, f"Model {model_name} does not appear to exist on Hugging Face"
            except Exception:
                # If there's any error, assume model might not exist
                return False, self.DEFAULT_FALLBACK_MODEL, f"Could not verify if model {model_name} exists"
                
        # Check if model config is accessible - this is a lightweight check
        try:
            headers = {}
            if self.hf_token:
                headers["Authorization"] = f"Bearer {self.hf_token}"
                
            config_url = f"https://huggingface.co/{model_name}/resolve/main/config.json"
            response = requests.head(config_url, headers=headers, timeout=5)
            
            if response.status_code == 200:
                return True, None, None
            elif response.status_code == 401:  # Authentication required
                if self.hf_token:
                    return False, self.DEFAULT_FALLBACK_MODEL, f"Authentication failed for {model_name}. Your token may not have access to this model."
                
                # If it's a known gated model, suggest similar sized model
                if is_gated:
                    # Find an open-access model with similar parameters
                    model_info = self._get_model_info(model_name)
                    params = model_info.get('parameters', 0)
                    
                    # Find closest match based on parameter count
                    fallback_model = self.DEFAULT_FALLBACK_MODEL
                    min_diff = float('inf')
                    
                    for open_model, info in self.OPEN_ACCESS_MODELS.items():
                        diff = abs(info['parameters'] - params)
                        if diff < min_diff:
                            min_diff = diff
                            fallback_model = open_model
                    
                    error_msg = f"Authentication required for {model_name}. Please provide a token with set_token()."
                    return False, fallback_model, error_msg
                else:
                    # If it's not in our lists, use default fallback
                    return False, self.DEFAULT_FALLBACK_MODEL, f"Model {model_name} requires authentication but no token was provided."
            else:
                # For other errors, use default fallback
                return False, self.DEFAULT_FALLBACK_MODEL, f"HTTP error {response.status_code} accessing {model_name}"
        
        except Exception as e:
            return False, self.DEFAULT_FALLBACK_MODEL, f"Error checking model accessibility: {str(e)}"
    
    def print_model_info(self, detailed=False):
        """
        Print detailed information about the current model configuration
        
        Args:
            detailed (bool): Whether to print detailed information including
                           memory requirements and hardware recommendations
        """
        # Get model details based on ID (if available)
        model_info = self._get_model_info(self.model_name)
        
        if detailed:
            print("\n" + "="*80)
            print("ENHANCED LLM CONFIGURATION".center(80))
            print("="*80 + "\n")
            
            # Print basic model information
            print(f"🤖 MODEL:           {self.model_name}")
            
            # Print parameter count if available
            if model_info and 'parameters_str' in model_info:
                print(f"📏 PARAMETERS:      {model_info['parameters_str']}")
            
            # Print quantization mode and estimated memory usage
            print(f"💾 QUANTIZATION:    {self.quantization_mode}")
            
            # Print architecture if available
            if model_info and 'architecture' in model_info:
                print(f"🧠 ARCHITECTURE:    {model_info['architecture']}")
            
            # Print context length if available
            if model_info and 'context_length' in model_info:
                print(f"🔄 CONTEXT LENGTH:  {model_info['context_length']} tokens")
            
            # Calculate and print memory usage based on parameters and quantization
            params = model_info.get('parameters', 0) if model_info else 0
            if params > 0:
                # Memory usage estimates in different precisions
                if self.quantization_mode == "FP16":
                    mem_weights = params * 2 / 1e9  # 2 bytes per parameter
                elif self.quantization_mode == "8-bit":
                    mem_weights = params * 1 / 1e9  # 1 byte per parameter 
                elif self.quantization_mode == "4-bit":
                    mem_weights = params * 0.5 / 1e9  # 0.5 bytes per parameter
                else:
                    mem_weights = params * 2 / 1e9  # Default to FP16
                
                # Add estimate for activations
                mem_activations = mem_weights * 0.15  # Rough estimate: activations ~15% of weights
                mem_total = mem_weights + mem_activations
                
                print(f"📊 MEMORY USAGE:    ~{mem_total:.1f} GB total")
                print(f"                    • Model weights: ~{mem_weights:.1f} GB")
                print(f"                    • Activations:   ~{mem_activations:.1f} GB")
            
            # Print generation settings
            print("\n⚙️  GENERATION SETTINGS:")
            print(f"   • Max tokens:        {self.max_tokens}")
            print(f"   • Temperature:       {self.temperature}")
            print(f"   • Distribution mode: Tensor Parallelism")
            print(f"   • Authentication:    {'Using token' if self.hf_token else 'Open access'} ({chr(9989) if self.hf_token else chr(10060)} {'Token provided' if self.hf_token else 'No token'})")
            
            # Print model status
            print(f"\n📋 MODEL STATUS:     {'Loaded' if self.loaded_model is not None else 'Not loaded (lazy loading)'}")
            print("="*80)
        else:
            # Print just basic info for non-detailed mode
            model_type = "gated" if self.model_name in self.GATED_MODELS else "open-access"
            param_str = f" ({model_info['parameters_str']} parameters)" if model_info and 'parameters_str' in model_info else ""
            print(f"Model: {self.model_name}{param_str} [{model_type}]")
            print(f"Quantization: {self.quantization_mode}")
            print(f"Authentication: {'Token provided' if self.hf_token else 'No token'}")
    
    def _get_model_info(self, model_name):
        """
        Get information about a model from our predefined lists
        
        Args:
            model_name: The HuggingFace model identifier
            
        Returns:
            dict: Model information or empty dict if not found
        """
        # Check open access models
        if model_name in self.OPEN_ACCESS_MODELS:
            return self.OPEN_ACCESS_MODELS[model_name]
            
        # Check gated models
        if model_name in self.GATED_MODELS:
            return self.GATED_MODELS[model_name]
            
        # Unknown model - collect basic info from name
        if "/" in model_name:
            architecture = model_name.split("/")[1]
            return {
                "architecture": architecture
            }
            
        return {}
    
    def _load_model(self):
        """
        Load model across multiple GPUs with optimized memory usage
        
        This method:
        1. Configures advanced quantization for memory efficiency
        2. Loads the specified model with tensor parallelism across available GPUs
        3. Sets up an optimized text generation pipeline
        
        The method uses several advanced techniques:
        - 8-bit or 4-bit quantization: Reduces memory usage vs FP16 with minimal quality loss
        - Tensor parallelism: Automatically distributes model across GPUs
        - Nested quantization: Further optimizes memory usage
        - Mixed precision: Uses FP16 for outlier weights to preserve quality
        - CPU offloading: Can offload parts to CPU if needed
        """
        # Skip loading if model is already loaded
        if self.loaded_model is not None:
            return
        
        print(f"\nLoading {self.model_name} across {self.gpu_count} GPUs with {self.quantization_mode} quantization...")
        print("This may take a few minutes. Loading large language model...")
        
        # Update quantization mode and print detailed information before loading
        self.print_model_info(detailed=True)
        
        # Check if model is accessible and get fallback if needed
        is_accessible, fallback_model, error_msg = self._check_model_accessibility(self.model_name)
        
        if not is_accessible:
            if fallback_model:
                print(f"\n⚠️  Warning: {error_msg}")
                print(f"🔄 Using fallback model: {fallback_model}")
                self.model_name = fallback_model
                # Print updated model info
                self.print_model_info(detailed=False)
            else:
                print(f"\n❌ Error: {error_msg}")
                print("Please ensure you have proper authentication for this model.")
                print("To authenticate with Hugging Face:")
                print("1. Create an account at https://huggingface.co/")
                print("2. Accept the model license terms on the model's page")
                print("3. Set your Hugging Face token with llm.set_token('your_token')")
                print("4. Alternatively, choose an open-access model")
                available_models = ", ".join(self.OPEN_ACCESS_MODELS.keys())
                print(f"Available open-access models: {available_models}")
                raise ValueError(f"Cannot access model {self.model_name}. Authentication may be required.")
            
        # Configure quantization based on specified mode
        quantization_config = None
        
        if self.quantization_mode == "4-bit":
            # Configure 4-bit quantization - most memory efficient
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,                    # 4-bit quantization
                bnb_4bit_compute_dtype=torch.float16, # Compute in half precision
                bnb_4bit_use_double_quant=True,       # Use nested quantization
                bnb_4bit_quant_type="nf4",            # Use normalized float 4-bit
            )
        elif self.quantization_mode == "8-bit":
            # Configure 8-bit quantization - good balance of quality and efficiency
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,                    # 8-bit quantization
                llm_int8_threshold=6.0,               # Threshold for outlier detection
                llm_int8_has_fp16_weight=True,        # Keep outlier weights in fp16
                bnb_8bit_compute_dtype=torch.float16, # Compute in half precision
                bnb_8bit_use_double_quant=True,       # Use nested quantization
            )
        # Otherwise use FP16 (no quantization config needed)
        
        # Measure loading time
        start_time = time.time()
        
        try:
            # Load tokenizer first - required before model loading
            # The tokenizer is responsible for converting text to tokens and vice versa
            # use_fast=True uses the optimized Rust implementation for faster processing
            print("Loading tokenizer...")
            self.loaded_tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_fast=True,  # Use faster Rust-based tokenizer when available
                token=self.hf_token  # Use stored token (might be None)
            )
            
            # Load model with tensor parallelism across GPUs
            # device_map="auto" automatically distributes the model weights across available GPUs
            # for the UTTA server with dual GPUs, this will split the model across both GPUs
            print("Loading model with tensor parallelism...")
            
            model_args = {
                "device_map": "auto",         # Automatically distribute across available GPUs
                "torch_dtype": torch.float16, # Use half precision for non-quantized operations
                "low_cpu_mem_usage": True,    # Optimize CPU memory usage during loading
                "offload_folder": "offload_folder",  # Folder for potential CPU offloading
                "offload_state_dict": True,   # Offload state dict to CPU during loading
                "trust_remote_code": True,    # Allow model implementations with custom code
                "token": self.hf_token        # Use stored token (might be None)
            }
            
            # Add quantization config if using 4-bit or 8-bit
            if quantization_config:
                model_args["quantization_config"] = quantization_config
                
            self.loaded_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_args
            )
            
            # Create optimized pipeline for text generation
            # This pipeline integrates the model and tokenizer into a single interface for text generation
            # It allows for advanced configurations while providing a simple API
            print("Creating generation pipeline...")
            self.generation_pipeline = pipeline(
                "text-generation",                # Task type
                model=self.loaded_model,          # The loaded model
                tokenizer=self.loaded_tokenizer,  # The loaded tokenizer
                device_map="auto",                # Use same device mapping as model
            )
            
            # Calculate loading time
            loading_time = time.time() - start_time
            
            # Print post-loading info including actual memory usage
            total_gpu_memory = 0
            for i in range(self.gpu_count):
                total_gpu_memory += torch.cuda.memory_allocated(i) / (1024**3)
            
            print(f"\n✅ Model loaded successfully in {loading_time:.2f} seconds!")
            print(f"   - Actual GPU memory used: {total_gpu_memory:.2f} GB")
            
            # Get model details like hidden size, num layers if available
            model_details = ""
            try:
                config = self.loaded_model.config
                if hasattr(config, 'hidden_size'):
                    model_details += f", hidden size: {config.hidden_size}"
                if hasattr(config, 'num_hidden_layers'):
                    model_details += f", layers: {config.num_hidden_layers}"
                if hasattr(config, 'vocab_size'):
                    model_details += f", vocab: {config.vocab_size}"
            except:
                pass
                
            print(f"   - Model details: {self.model_name}{model_details}")
            print(f"   - Device mapping: {self.loaded_model.hf_device_map if hasattr(self.loaded_model, 'hf_device_map') else 'auto'}")
            
        except Exception as e:
            error_message = str(e)
            
            # Handle authentication error specifically
            if "401" in error_message or "gated repo" in error_message:
                print(f"\n❌ Authentication Error: Cannot access {self.model_name}")
                print("This model requires Hugging Face authentication.")
                print("To authenticate with Hugging Face:")
                print("1. Create an account at https://huggingface.co/")
                print("2. Accept the model license terms on the model's page")
                print("3. Run `huggingface-cli login` and enter your token")
                print("\nAlternatively, use an open-access model:")
                
                # List available open-access models with parameter sizes
                for model, info in self.OPEN_ACCESS_MODELS.items():
                    print(f"   - {model} ({info['parameters_str']} parameters)")
                    
                print("\nTo use an open-access model, initialize with:")
                print("   processor = PedagogicalLanguageProcessor(model_name='google/gemma-2b')")
                
            # Re-raise the exception for the caller to handle
            raise
    
    def generate(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a response from the LLM using optimized multi-GPU inference
        
        This method handles the complete process of:
            1. Building the context-aware prompt
            2. Loading the model across GPUs if not already loaded
            3. Processing the response with optimized pipeline
            4. Handling any errors
        
        Args:
            prompt (str): The main instruction or question for the LLM
            context (Optional[Dict[str, Any]]): Additional context including:
                - student_info: Student characteristics
                - subject_area: Current subject
                - grade_level: Student's grade
                - previous_context: Prior interactions
            **kwargs: Additional parameters including:
                - temperature: Response randomness (0-1)
                - max_tokens: Maximum response length
        
        Returns:
            Dict[str, Any]: A structured response containing:
                - text: The LLM's response
                - status: Success or error status
                - model: Name of the model used
                - error: Error message if applicable
        """
        try:
            # Check if model needs to be loaded first time
            # This implements lazy loading - model is only loaded when first needed
            if self.loaded_model is None:
                # First try loading the model
                try:
                    self._load_model()
                except Exception as load_error:
                    # If model loading fails, return informative error
                    print(f"Error loading model: {str(load_error)}")
                    return {
                        "text": f"I apologize, but there was an error loading the language model. Error: {str(load_error)}",
                        "status": "error",
                        "error": str(load_error)
                    }
                
            # Combine prompt with context if provided
            # This creates a properly formatted prompt that includes all relevant context
            full_prompt = self._build_prompt(prompt, context) if context else prompt
            
            # Set up generation parameters
            # These parameters control the behavior of the text generation process
            gen_kwargs = {
                "max_new_tokens": kwargs.get("max_tokens", self.max_tokens),  # Max tokens to generate
                "temperature": kwargs.get("temperature", self.temperature),   # Randomness parameter
                "do_sample": kwargs.get("temperature", self.temperature) > 0, # Enable sampling for temperature > 0
                "top_p": kwargs.get("top_p", 0.95),                        # Nucleus sampling parameter
                "top_k": kwargs.get("top_k", 50),                          # Top-k sampling parameter
                "repetition_penalty": kwargs.get("repetition_penalty", 1.1),  # Penalty for repeating tokens
                "pad_token_id": self.loaded_tokenizer.eos_token_id,        # Use EOS token for padding
            }
            
            # Generate response using the pipeline
            # The pipeline handles tokenization, model forward pass, and token generation
            outputs = self.generation_pipeline(
                full_prompt,
                **gen_kwargs
            )
            
            # Extract the generated text, removing the input prompt
            # This ensures we only return the newly generated content
            generated_text = outputs[0]['generated_text']
            response_text = generated_text[len(full_prompt):] if generated_text.startswith(full_prompt) else generated_text
            
            return {
                "text": response_text.strip(),   # Return cleaned response text
                "status": "success",             # Indicate successful generation
                "model": self.model_name         # Include model information
            }
            
        except Exception as e:
            # Robust error handling
            print(f"Error generating LLM response: {e}")
            return {
                "text": "I apologize, but I'm having trouble processing that request. Error: " + str(e),
                "status": "error",
                "error": str(e)
            }
    
    def _build_prompt(self, prompt: str, context: Dict[str, Any]) -> str:
        """
        Build a complete prompt by combining the input with context.
        
        This method creates a structured prompt that includes both the
        main instruction and any relevant context information in a
        format optimized for LLM understanding.
        
        Args:
            prompt (str): The main instruction or question
            context (Dict[str, Any]): Contextual information including:
                - Student characteristics
                - Learning objectives
                - Previous interactions
                - Environmental factors
            
        Returns:
            str: A formatted prompt string that combines:
                - Context information
                - Main prompt
                - Any necessary formatting or structure
        """
        # Format context as key-value pairs, one per line
        context_str = "\n".join(f"{k}: {v}" for k, v in context.items())
        
        # Create a structured prompt with clear section separation
        return f"Context:\n{context_str}\n\nPrompt:\n{prompt}" 
        
    def generate_response(self, teacher_response: str, scenario: dict = None) -> str:
        """
        Generate a student response to a teacher's action in a scenario.
        
        This method is specifically used by the Enhanced Teacher Training Agent
        to generate realistic student responses based on the teacher's input
        and the current scenario context.
        
        Args:
            teacher_response (str): The teacher's action or statement
            scenario (dict, optional): The current teaching scenario containing:
                - description: The scenario description
                - knowledge_sources: Sources used to create the scenario
                - recommended_strategy: Suggested teaching approach
        
        Returns:
            str: A realistic student response that reflects:
                - Understanding of the teaching context
                - Appropriate reaction to the teacher's approach
                - Age-appropriate language and behavior
        """
        try:
            # Handle the case when scenario is None or not provided
            # This provides a fallback default scenario
            if scenario is None:
                scenario = {"scenario": "Classroom management situation"}
            
            # Build a prompt specifically for student response generation
            # This prompt provides clear instructions to generate an authentic student response
            prompt = f"""
            As an AI simulating a student's response in a teacher training scenario:
            
            SCENARIO:
            {scenario.get('scenario', 'Classroom management situation')}
            
            TEACHER RESPONSE:
            "{teacher_response}"
            
            Please generate a realistic, brief response from the student that would make sense
            in this context. The response should reflect how a real student might react to the
            teacher's approach. Keep the response under 100 words and make it authentic to a
            student's perspective.
            
            STUDENT RESPONSE:
            """
            
            # Generate response
            response = self.generate(prompt=prompt)
            
            # Return just the text part of the response
            return response.get("text", "I'm not sure how to respond to that.")
            
        except Exception as e:
            # Provide fallback response if generation fails
            print(f"Error generating student response: {e}")
            return "Hmm... I'm not sure what to say to that."

    def set_model(self, model_name: str):
        """
        Change the model being used. Will unload any previously loaded model.
        
        This allows for dynamically switching between different models based on task requirements.
        For example, using a larger model for complex reasoning and a smaller, more efficient
        model for simple responses.
        
        The method properly unloads the current model to free GPU memory before changing
        the model name. The new model will be loaded on the next generation request.
        
        Args:
            model_name: The HuggingFace model identifier
                
        Alternative models that can be used:
        - "meta-llama/Llama-2-7b-hf": Smaller model, uses less memory
        - "meta-llama/Llama-3-8b-hf": Newer architecture, better quality/size ratio
        - "mistralai/Mistral-7B-v0.1": Excellent quality at smaller size
        - "microsoft/phi-2": Very small (2.7B) but surprisingly capable
        """
        # Unload current model to free GPU memory
        # This is important to avoid CUDA out-of-memory errors
        if self.loaded_model is not None:
            self.loaded_model = None
            self.loaded_tokenizer = None
            self.generation_pipeline = None
            torch.cuda.empty_cache()  # Clear CUDA cache to release GPU memory
        
        # Update model name
        self.model_name = model_name
        
        # Print new model information
        print(f"Model changed to {model_name}.")
        self.print_model_info(detailed=False)
        print("It will be loaded on the next generation request.")

    def check_server(self):
        """
        Legacy compatibility method. Always returns True since we're not using a server anymore.
        
        This method exists to maintain compatibility with code that expects the old
        Ollama-based interface.
        """
        return True

class PedagogicalLanguageProcessor:
    """
    Processes natural language in educational contexts using LLMs.
    
    This class serves as the main interface for all language-related
    operations in the teaching simulation. It maintains educational
    context while processing language, ensuring responses are
    pedagogically appropriate and age-suitable.
    
    Key Features:
        - Teaching response analysis
        - Student reaction generation
        - Scenario creation
        - Real-time feedback
        - Context maintenance
        - Model switching for different tasks
    
    Components:
        - Enhanced LLM Interface: Handles communication with distributed models
        - Prompt Templates: Manages educational prompt patterns
        - Context Management: Maintains teaching situation state
        - Response Processing: Analyzes and generates responses
    """
    
    def __init__(self, model_name=None, quantization="8-bit", token=None):
        """
        Initialize the processor with enhanced LLM interface and prompt templates.
        
        This constructor:
        1. Creates an EnhancedLLMInterface for model interaction
        2. Attempts to load prompt templates
        3. Falls back to default prompts if templates aren't available
        
        Args:
            model_name (str, optional): The HuggingFace model identifier. If None, a default
                                      open-access model will be selected based on available resources.
            quantization (str): Quantization level to use - "FP16", "8-bit", or "4-bit"
            token (str, optional): Hugging Face access token for accessing gated models
        """
        # Use the enhanced LLM interface for multi-GPU optimization
        print("\n" + "="*80)
        print("INITIALIZING PEDAGOGICAL LANGUAGE PROCESSOR".center(80))
        print("="*80)
        print("\nCreating enhanced LLM interface with multi-GPU support...")
        
        self.llm = EnhancedLLMInterface(model_name=model_name, quantization=quantization, token=token)
        
        # Try to load prompt templates, fall back gracefully if not available
        try:
            from prompt_templates import PromptTemplates
            self.templates = PromptTemplates()
            print("✅ Successfully loaded prompt templates.")
        except ImportError:
            print("⚠️  Warning: PromptTemplates not found. Using default prompts.")
            self.templates = None
            
        print(f"✅ PedagogicalLanguageProcessor initialized and ready.\n")
        print("="*80)
        
    def analyze_teaching_response(
        self, 
        teacher_input: str, 
        context: TeachingContext
    ) -> Dict[str, Any]:
        """
        Analyze the effectiveness of a teacher's response.
        
        This method performs a detailed analysis of teaching effectiveness by:
        1. Switching to the Llama-3-8B model for complex analysis
        2. Using either template-based or default prompts
        3. Evaluating multiple dimensions of teaching quality
        
        Args:
            teacher_input: The teacher's response or action
            context: Current teaching context including student info
            
        Returns:
            Dictionary containing analysis results:
            - effectiveness_score: Float between 0 and 1
            - identified_strengths: List of effective elements
            - improvement_areas: List of areas for improvement
            - suggested_strategies: Alternative approaches
        """
        # For complex analysis tasks, use the Llama 3 model
        # Teaching analysis requires deeper reasoning capabilities
        self.llm.set_model("meta-llama/Llama-3-8b-hf")
        
        # Use template if available, otherwise use default prompt
        prompt = self._get_template("analysis") if self.templates else f"""
        Analyze the following teaching response for effectiveness. Consider:
        1. Alignment with educational goals
        2. Age-appropriateness
        3. Clarity and structure
        4. Engagement level
        5. Potential for student growth
        
        Teaching response: {teacher_input}
        """
        
        # Generate analysis using the LLM
        return self.llm.generate(
            prompt=prompt,
            context=self._context_to_dict(context)
        )
        
    def generate_student_reaction(
        self, 
        context: TeachingContext, 
        effectiveness: float
    ) -> str:
        """
        Generate an age-appropriate student reaction.
        
        This method:
        1. Uses an efficient model for simple response generation
        2. Creates a prompt based on effectiveness score and student age
        3. Generates a realistic student reaction
        
        Args:
            context: Current teaching context
            effectiveness: Float between 0 and 1 indicating teaching effectiveness
            
        Returns:
            A string containing the simulated student's reaction
        """
        # For student reactions, we can use a smaller efficient model
        # Student responses are simpler and don't require as much reasoning depth
        # Try to use Phi-2 as it's efficient for this task
        self.llm.set_model("microsoft/phi-2")
        
        # Use template if available, otherwise use default prompt
        prompt = self._get_template("student_reaction") if self.templates else f"""
        Generate a realistic student reaction to a teaching approach with effectiveness
        score of {effectiveness} (0-1, where 1 is most effective).
        
        The reaction should be age-appropriate for a {context.grade_level} student.
        """
        
        # Generate the student reaction
        response = self.llm.generate(
            prompt=prompt,
            context=self._context_to_dict(context)
        )
        
        # Extract and return just the text
        return response.get("text", "I'm confused about that.")
        
    def create_scenario(
        self, 
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a teaching scenario based on given parameters.
        
        This method:
        1. Uses the Llama-3-8B model for complex scenario creation
        2. Generates a detailed teaching scenario with context and objectives
        
        Args:
            parameters: Dictionary containing scenario parameters:
                - subject: Subject area
                - difficulty: Difficulty level
                - student_profile: Student characteristics
                
        Returns:
            Dictionary containing the complete scenario
        """
        # For complex scenario creation, use the Llama 3 model
        # Scenario creation requires detailed reasoning and planning
        self.llm.set_model("meta-llama/Llama-3-8b-hf")
        
        # Use template if available, otherwise use default prompt
        prompt = self._get_template("scenario_creation") if self.templates else """
        Create a detailed teaching scenario based on the given parameters.
        Include student background, learning objectives, and potential challenges.
        """
        
        # Generate the scenario
        return self.llm.generate(
            prompt=prompt,
            context=parameters
        )

    def _get_template(self, template_name: str) -> str:
        """
        Helper to get a template or return None if templates not available
        
        Args:
            template_name: Name of the template to retrieve
            
        Returns:
            Template string if available, None otherwise
        """
        if self.templates:
            return self.templates.get(template_name)
        return None
        
    def _context_to_dict(self, context: TeachingContext) -> Dict[str, Any]:
        """
        Convert TeachingContext to dictionary for prompt context
        
        This method transforms the structured TeachingContext dataclass into
        a dictionary format suitable for inclusion in prompts.
        
        Args:
            context: The TeachingContext to convert
            
        Returns:
            Dictionary representation of the context
        """
        return {
            "subject": context.subject,
            "grade_level": context.grade_level,
            "learning_objectives": ", ".join(context.learning_objectives),
            "student_characteristics": json.dumps(context.student_characteristics),
            "previous_interactions": json.dumps(context.previous_interactions) if context.previous_interactions else "None"
        }
    
    def check_server(self):
        """
        Legacy compatibility method.
        
        Returns:
            bool: Always True since we're using direct model loading now
        """
        return True
        
    def ensure_server_running(self):
        """
        Legacy compatibility method. No server needed with direct model loading.
        
        This method prints detailed LLM information for testing purposes.
        
        Returns:
            bool: Always True
        """
        print("\n" + "="*80)
        print("LLM SYSTEM INFORMATION".center(80))
        print("="*80)
        
        # Print PyTorch and CUDA information
        print(f"\n🔧 PYTORCH VERSION: {torch.__version__}")
        print(f"🔌 CUDA AVAILABLE:  {'Yes' if torch.cuda.is_available() else 'No'}")
        print(f"🖥️  CUDA VERSION:    {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
        print(f"📊 GPU COUNT:       {torch.cuda.device_count()}")
        
        # Print model information from LLM interface
        print("\n== LLM CONFIGURATION ==")
        self.llm.print_model_info(detailed=True)
        
        # Print library versions
        try:
            import transformers
            import bitsandbytes
            import accelerate
            
            print("\n== LIBRARY VERSIONS ==")
            print(f"🤗 Transformers:    {transformers.__version__}")
            print(f"🔢 BitsAndBytes:    {bitsandbytes.__version__}")
            print(f"🚀 Accelerate:      {accelerate.__version__}")
        except:
            pass
            
        print("="*80 + "\n")
        return True
        
    def switch_to_larger_model(self):
        """
        Switch to a larger model for more complex tasks
        
        This is a convenience method for tasks that require deeper reasoning
        capabilities, such as complex scenario generation or detailed analysis.
        """
        self.llm.set_model("meta-llama/Llama-3-8b-hf")
        
    def switch_to_efficient_model(self):
        """
        Switch to a smaller, more efficient model for routine tasks
        
        This is a convenience method for simpler tasks that don't require
        the full reasoning capabilities of the larger model, such as
        basic student reactions or simple information retrieval.
        
        Uses microsoft/phi-2 which is an efficient and reliable open-access model.
        """
        # Phi-2 is small, efficient, and open-access (no auth required)
        self.llm.set_model("microsoft/phi-2")

# Add backward compatibility for code that imports LLMInterface
# This ensures that existing code referencing the old interface will continue to work
LLMInterface = EnhancedLLMInterface 