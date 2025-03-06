# HuggingFace LoRA Fine-Tuning Tutorial

This tutorial provides a step-by-step guide to implementing HuggingFace fine-tuning with LoRA (Low-Rank Adaptation) for educational question-answering. You'll learn how to prepare your data, set up the training environment, implement LoRA fine-tuning, and use your fine-tuned model.

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- Basic understanding of PyTorch and transformers
- The LLM Fine-Tuning Toolkit installed

## Environment Setup

1. Create a virtual environment and install the required packages:

```bash
conda create -n utta python=3.8
conda activate utta
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets peft tqdm evaluate
```

2. Verify CUDA is available:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Current device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

## Dataset Preparation

### 1. Create a Sample Dataset

Create `create_dataset.py`:

```python
import pandas as pd
from datasets import Dataset

def create_sample_dataset():
    data = {
        'instruction': [
            "Answer the following question about science:",
            "Explain the following concept:",
        ],
        'input': [
            "What is photosynthesis?",
            "What is the water cycle?",
        ],
        'output': [
            "Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen. This process is essential for life on Earth as it provides food for plants and oxygen for all living things.",
            "The water cycle, also known as the hydrologic cycle, is the continuous movement of water on Earth. It involves processes like evaporation (water turning into vapor), condensation (vapor forming clouds), precipitation (rain or snow), and collection (water gathering in bodies of water or groundwater)."
        ]
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Convert to HuggingFace Dataset
    dataset = Dataset.from_pandas(df)
    
    # Split into train and validation
    dataset = dataset.train_test_split(test_size=0.2)
    
    # Save datasets
    dataset['train'].save_to_disk('data/train')
    dataset['test'].save_to_disk('data/validation')

if __name__ == "__main__":
    create_sample_dataset()
```

### 2. Create the Data Processing Script

Create `process_data.py`:

```python
from transformers import AutoTokenizer
from datasets import load_from_disk

def process_data(model_name="meta-llama/Llama-2-7b-hf"):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_function(examples):
        # Combine instruction, input, and output
        prompt = [f"### Instruction: {instruction}\n### Input: {input}\n### Response: {output}"
                 for instruction, input, output in zip(examples['instruction'], 
                                                     examples['input'], 
                                                     examples['output'])]
        
        # Tokenize
        return tokenizer(prompt, truncation=True, padding="max_length", 
                        max_length=512, return_tensors="pt")
    
    # Load and process datasets
    train_dataset = load_from_disk('data/train')
    val_dataset = load_from_disk('data/validation')
    
    # Apply tokenization
    train_tokenized = train_dataset.map(tokenize_function, batched=True)
    val_tokenized = val_dataset.map(tokenize_function, batched=True)
    
    # Save processed datasets
    train_tokenized.save_to_disk('data/train_tokenized')
    val_tokenized.save_to_disk('data/val_tokenized')

if __name__ == "__main__":
    process_data()
```

## Fine-Tuning Implementation

### 1. Create the LoRA Configuration

Create `lora_config.py`:

```python
from peft import LoraConfig, TaskType

def get_lora_config():
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,  # Rank of LoRA update matrices
        lora_alpha=32,  # Alpha scaling factor
        lora_dropout=0.1,
        # Target modules for LoRA
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]
    )
```

### 2. Create the Training Script

Create `train_lora.py`:

```python
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import get_peft_model
from datasets import load_from_disk
from lora_config import get_lora_config

def train():
    # Load model and tokenizer
    model_name = "meta-llama/Llama-2-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Apply LoRA config
    lora_config = get_lora_config()
    model = get_peft_model(model, lora_config)
    
    # Load processed datasets
    train_dataset = load_from_disk('data/train_tokenized')
    val_dataset = load_from_disk('data/val_tokenized')
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="output",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        logging_steps=10,
        learning_rate=2e-4,
        weight_decay=0.01,
        fp16=True,
        bf16=False,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none"
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # Train
    trainer.train()
    
    # Save the final model
    trainer.save_model("final_model")

if __name__ == "__main__":
    train()
```

### 3. Create the Inference Script

Create `inference.py`:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def load_model(base_model_name, adapter_path):
    # Load base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(model, adapter_path)
    return model, tokenizer

def generate_response(model, tokenizer, instruction, input_text):
    # Format prompt
    prompt = f"### Instruction: {instruction}\n### Input: {input_text}\n### Response:"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1
        )
    
    # Decode and return
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("### Response:")[-1].strip()

def main():
    # Load model
    base_model = "meta-llama/Llama-2-7b-hf"
    adapter_path = "final_model"
    model, tokenizer = load_model(base_model, adapter_path)
    
    # Example usage
    instruction = "Answer the following question about science:"
    input_text = "What is cellular respiration?"
    
    response = generate_response(model, tokenizer, instruction, input_text)
    print(f"Question: {input_text}")
    print(f"Response: {response}")

if __name__ == "__main__":
    main()
```

## Running the Fine-Tuning Process

1. Create and process the dataset:
```bash
python create_dataset.py
python process_data.py
```

2. Start the training:
```bash
python train_lora.py
```

3. Test the model:
```bash
python inference.py
```

## Best Practices and Tips

1. **Model Selection**:
   - Choose an appropriate base model size
   - Consider using smaller models for faster iteration
   - Ensure your GPU has enough VRAM

2. **LoRA Parameters**:
   - Adjust rank (r) based on task complexity
   - Experiment with different target modules
   - Balance between adaptation and stability

3. **Training Optimization**:
   - Use gradient accumulation for larger batch sizes
   - Monitor training loss and validation metrics
   - Implement early stopping if needed

4. **Memory Management**:
   - Use mixed precision training (fp16)
   - Adjust batch size based on available GPU memory
   - Enable gradient checkpointing for large models

## Troubleshooting

Common issues and solutions:

1. **Out of Memory Errors**:
   ```python
   RuntimeError: CUDA out of memory
   ```
   Solutions:
   - Reduce batch size
   - Enable gradient checkpointing
   - Use a smaller base model

2. **Training Instability**:
   ```python
   Loss becomes NaN
   ```
   Solutions:
   - Reduce learning rate
   - Adjust gradient clipping
   - Check for data preprocessing issues

3. **Slow Training**:
   Solutions:
   - Enable mixed precision training
   - Increase batch size (if memory allows)
   - Use gradient accumulation

## Advanced Topics

1. **Custom Dataset Creation**:
```python
from datasets import Dataset

def create_custom_dataset(data_path):
    # Load your custom data
    # Format it appropriately
    # Convert to HuggingFace Dataset
    dataset = Dataset.from_dict(your_data)
    return dataset
```

2. **Custom Training Loop**:
```python
def custom_training_loop(model, dataloader, optimizer, epochs):
    for epoch in range(epochs):
        for batch in dataloader:
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
```

3. **Evaluation Metrics**:
```python
from evaluate import load

def compute_metrics(pred):
    rouge = load('rouge')
    predictions = tokenizer.batch_decode(pred.predictions)
    references = tokenizer.batch_decode(pred.label_ids)
    return rouge.compute(predictions=predictions, references=references)
```

## Next Steps

1. Experiment with different LoRA configurations
2. Implement custom evaluation metrics
3. Try different base models
4. Add model quantization
5. Implement model merging

## Resources

- [PEFT Documentation](https://huggingface.co/docs/peft)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [PyTorch Documentation](https://pytorch.org/docs) 