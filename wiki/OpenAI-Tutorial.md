# OpenAI Fine-Tuning Tutorial

This tutorial will guide you through fine-tuning an OpenAI model for educational question-answering. You'll learn how to prepare your data, run the fine-tuning process, and use your fine-tuned model.

## Prerequisites

- Completed [Environment Setup](Environment-Setup.md)
- OpenAI API key
- Basic Python knowledge
- Prepared dataset (see [Dataset Preparation](Dataset-Preparation.md))

## Step 1: Setup

```bash
# Ensure OpenAI package is installed
pip install openai tiktoken

# Set up API key
export OPENAI_API_KEY="your-api-key-here"
```

Create `openai_setup.py`:
```python
from openai import OpenAI
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_setup():
    """Verify OpenAI setup and available models"""
    try:
        client = OpenAI()
        models = client.models.list()
        logger.info("Available models:")
        for model in models:
            if "gpt" in model.id:
                logger.info(f"- {model.id}")
        return True
    except Exception as e:
        logger.error(f"Setup error: {e}")
        return False

if __name__ == "__main__":
    verify_setup()
```

## Step 2: Data Preparation

Create `prepare_training_data.py`:
```python
import json
import tiktoken
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingDataPreparator:
    def __init__(self):
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.max_tokens = 4096
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))
    
    def format_qa_pair(self, question: str, answer: str) -> dict:
        """Format Q&A pair for fine-tuning"""
        return {
            "messages": [
                {"role": "system", "content": "You are a helpful educational assistant."},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]
        }
    
    def validate_example(self, example: dict) -> bool:
        """Validate a single training example"""
        total_tokens = sum(
            self.count_tokens(msg["content"]) 
            for msg in example["messages"]
        )
        return total_tokens <= self.max_tokens
    
    def prepare_data(self, input_file: str, output_file: str):
        """Prepare training data for fine-tuning"""
        logger.info(f"Reading data from {input_file}")
        
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        valid_examples = []
        stats = {"total": 0, "valid": 0, "invalid": 0}
        
        for item in data:
            stats["total"] += 1
            example = self.format_qa_pair(item["question"], item["answer"])
            
            if self.validate_example(example):
                valid_examples.append(example)
                stats["valid"] += 1
            else:
                stats["invalid"] += 1
        
        # Save formatted data
        output_path = Path(output_file)
        with open(output_path, 'w') as f:
            for example in valid_examples:
                f.write(json.dumps(example) + '\n')
        
        logger.info(f"Statistics: {stats}")
        logger.info(f"Saved {stats['valid']} examples to {output_file}")

if __name__ == "__main__":
    preparator = TrainingDataPreparator()
    preparator.prepare_data('processed_data/dspy_data.json', 'training_data.jsonl')
```

## Step 3: Fine-Tuning Process

Create `run_finetuning.py`:
```python
from openai import OpenAI
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FineTuner:
    def __init__(self):
        self.client = OpenAI()
    
    def create_job(self, training_file: str, model: str = "gpt-3.5-turbo"):
        """Create fine-tuning job"""
        try:
            # Upload training file
            with open(training_file, 'rb') as f:
                upload = self.client.files.create(
                    file=f,
                    purpose="fine-tune"
                )
            logger.info(f"File uploaded: {upload.id}")
            
            # Create fine-tuning job
            job = self.client.fine_tuning.jobs.create(
                training_file=upload.id,
                model=model
            )
            logger.info(f"Fine-tuning job created: {job.id}")
            
            return job.id
        
        except Exception as e:
            logger.error(f"Error creating job: {e}")
            return None
    
    def monitor_progress(self, job_id: str):
        """Monitor fine-tuning progress"""
        while True:
            try:
                job = self.client.fine_tuning.jobs.retrieve(job_id)
                logger.info(f"Status: {job.status}")
                
                if job.status in ["succeeded", "failed"]:
                    return job
                
                time.sleep(60)  # Check every minute
            
            except Exception as e:
                logger.error(f"Error monitoring job: {e}")
                return None

if __name__ == "__main__":
    tuner = FineTuner()
    job_id = tuner.create_job('training_data.jsonl')
    if job_id:
        final_job = tuner.monitor_progress(job_id)
        if final_job and final_job.status == "succeeded":
            logger.info(f"Fine-tuning complete! Model: {final_job.fine_tuned_model}")
```

## Step 4: Testing and Usage

Create `test_model.py`:
```python
from openai import OpenAI
import json
import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTester:
    def __init__(self, model_id: str):
        self.client = OpenAI()
        self.model_id = model_id
    
    def generate_answer(self, question: str) -> str:
        """Generate answer using fine-tuned model"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": "You are a helpful educational assistant."},
                    {"role": "user", "content": question}
                ]
            )
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return None
    
    def run_test_cases(self, test_cases: List[Dict]) -> Dict:
        """Run multiple test cases"""
        results = []
        
        for case in test_cases:
            question = case["question"]
            expected = case["answer"]
            generated = self.generate_answer(question)
            
            results.append({
                "question": question,
                "expected": expected,
                "generated": generated
            })
        
        return results

def main():
    # Load your fine-tuned model ID
    model_id = "ft:gpt-3.5-turbo:..."  # Replace with your model ID
    
    # Create test cases
    test_cases = [
        {
            "question": "What is photosynthesis?",
            "answer": "Photosynthesis is the process by which plants convert sunlight into energy."
        },
        {
            "question": "Explain Newton's first law.",
            "answer": "Newton's first law states that an object remains at rest or in motion unless acted upon by an external force."
        }
    ]
    
    # Run tests
    tester = ModelTester(model_id)
    results = tester.run_test_cases(test_cases)
    
    # Save results
    with open('test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("Test results saved to test_results.json")

if __name__ == "__main__":
    main()
```

## Running the Tutorial

1. **Setup and Verification**:
   ```bash
   python openai_setup.py
   ```

2. **Prepare Training Data**:
   ```bash
   python prepare_training_data.py
   ```

3. **Run Fine-Tuning**:
   ```bash
   python run_finetuning.py
   ```

4. **Test Model**:
   ```bash
   # Update model ID in test_model.py
   python test_model.py
   ```

## Cost Estimation

- Training: ~$0.008 per 1K tokens
- Usage: ~$0.012 per 1K tokens
- Example cost for 1000 QA pairs:
  - Training: ~$10-20
  - Usage: ~$0.012 per generation

## Best Practices

1. **Data Quality**:
   - Use consistent formatting
   - Include diverse examples
   - Balance question types
   - Validate token counts

2. **Fine-Tuning**:
   - Start with small dataset
   - Monitor training progress
   - Save model checkpoints
   - Test incrementally

3. **Usage**:
   - Implement rate limiting
   - Handle API errors
   - Log responses
   - Monitor costs

## Troubleshooting

1. **API Errors**:
   ```python
   try:
       response = client.chat.completions.create(...)
   except openai.RateLimitError:
       time.sleep(60)
   except openai.APIError as e:
       logger.error(f"API error: {e}")
   ```

2. **Token Limits**:
   - Use tiktoken to count tokens
   - Split long texts
   - Monitor usage

3. **Cost Control**:
   - Set usage limits
   - Monitor spending
   - Use smaller datasets initially

## Next Steps

1. **Advanced Topics**:
   - [Hyperparameter Tuning](Hyperparameter-Tuning.md)
   - [Model Evaluation](Evaluation-Metrics.md)
   - [Production Deployment](Model-Deployment.md)

2. **Alternative Methods**:
   - [DSPy Tutorial](DSPy-Tutorial.md)
   - [HuggingFace Tutorial](HuggingFace-Tutorial.md)

## Resources

- [OpenAI Fine-Tuning Guide](https://platform.openai.com/docs/guides/fine-tuning)
- [OpenAI Cookbook](https://github.com/openai/openai-cookbook)
- [OpenAI Pricing](https://openai.com/pricing)
- [Best Practices](https://platform.openai.com/docs/guides/fine-tuning/best-practices) 