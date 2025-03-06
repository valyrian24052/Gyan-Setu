# OpenAI Fine-Tuning Tutorial

This tutorial provides a step-by-step guide to implementing OpenAI fine-tuning for educational question-answering. You'll learn how to prepare your data, format it correctly for OpenAI, run the fine-tuning process, and use your fine-tuned model.

## Prerequisites

- Python 3.10+
- OpenAI API key with billing set up
- Basic understanding of Python and JSON
- The LLM Fine-Tuning Toolkit installed

## Cost Estimates

Before starting, understand the potential costs:
- GPT-3.5 Turbo fine-tuning: ~$0.008 per 1K tokens
- GPT-3.5 Turbo usage: ~$0.002 per 1K tokens
- Example cost for 1000 QA pairs (~500K tokens):
  - Training: ~$4.00
  - Usage: ~$1.00 per 1000 queries

## Environment Setup

1. First, create a virtual environment and install the required packages:

```bash
# Create and activate environment
conda create -n utta python=3.10
conda activate utta

# Install dependencies
pip install openai==1.0.0 pandas numpy tqdm tiktoken

# Verify installation
python -c "import openai; print(openai.__version__)"
```

2. Set up your OpenAI API key:

```python
# Option 1: Environment variable (recommended)
export OPENAI_API_KEY="your-api-key-here"

# Option 2: In your script (not recommended for production)
import os
os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# Verify setup
import openai
client = openai.OpenAI()
try:
    models = client.models.list()
    print("API key is valid!")
except Exception as e:
    print(f"API key error: {e}")
```

## Project Structure

Create a project directory with this structure:
```
fine-tuning-project/
├── data/
│   ├── raw/
│   └── processed/
├── scripts/
│   ├── create_dataset.py
│   ├── format_data.py
│   └── openai_finetune.py
├── requirements.txt
└── README.md
```

## Dataset Preparation

### 1. Create a Sample Dataset

Create `scripts/create_dataset.py`:

```python
import json
import pandas as pd
from pathlib import Path

def create_sample_dataset():
    # Create data directory if it doesn't exist
    data_dir = Path('data')
    raw_dir = data_dir / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    data = [
        {
            "messages": [
                {"role": "system", "content": "You are a helpful educational assistant."},
                {"role": "user", "content": "What is photosynthesis?"},
                {"role": "assistant", "content": "Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen. This process is essential for life on Earth as it provides food for plants and oxygen for all living things."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a helpful educational assistant."},
                {"role": "user", "content": "Explain the water cycle."},
                {"role": "assistant", "content": "The water cycle, also known as the hydrologic cycle, is the continuous movement of water on Earth. It involves processes like evaporation (water turning into vapor), condensation (vapor forming clouds), precipitation (rain or snow), and collection (water gathering in bodies of water or groundwater)."}
            ]
        }
    ]
    
    # Save as JSONL file
    output_file = raw_dir / 'training_data.jsonl'
    with open(output_file, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Dataset created at {output_file}")
    return output_file

if __name__ == "__main__":
    create_sample_dataset()
```

### 2. Format and Validate Data

Create `scripts/format_data.py`:

```python
import json
import tiktoken
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataFormatter:
    def __init__(self):
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.max_tokens = 4096
        
    def num_tokens_from_messages(self, messages):
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                num_tokens += len(self.encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is omitted
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    
    def validate_message(self, message):
        required_keys = ['role', 'content']
        valid_roles = ['system', 'user', 'assistant']
        
        # Check required keys
        if not all(key in message for key in required_keys):
            return False, "Missing required keys"
        
        # Check role
        if message['role'] not in valid_roles:
            return False, f"Invalid role: {message['role']}"
        
        # Check content
        if not isinstance(message['content'], str) or len(message['content'].strip()) == 0:
            return False, "Empty or invalid content"
        
        return True, None
    
    def format_and_validate_data(self, input_file, output_file):
        processed_dir = Path('data/processed')
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = processed_dir / output_file
        stats = {'total': 0, 'valid': 0, 'invalid': 0}
        
        with open(input_file, 'r') as f_in, open(output_path, 'w') as f_out:
            for line in f_in:
                stats['total'] += 1
                try:
                    example = json.loads(line)
                    
                    # Validate structure
                    if 'messages' not in example:
                        logger.warning(f"Example {stats['total']}: Missing messages key")
                        stats['invalid'] += 1
                        continue
                    
                    # Validate messages
                    all_valid = True
                    for msg in example['messages']:
                        valid, error = self.validate_message(msg)
                        if not valid:
                            logger.warning(f"Example {stats['total']}: {error}")
                            all_valid = False
                            break
                    
                    if not all_valid:
                        stats['invalid'] += 1
                        continue
                    
                    # Check token count
                    tokens = self.num_tokens_from_messages(example['messages'])
                    if tokens > self.max_tokens:
                        logger.warning(f"Example {stats['total']}: Exceeds token limit")
                        stats['invalid'] += 1
                        continue
                    
                    # Write valid example
                    f_out.write(json.dumps(example) + '\n')
                    stats['valid'] += 1
                    
                except json.JSONDecodeError:
                    logger.warning(f"Example {stats['total']}: Invalid JSON")
                    stats['invalid'] += 1
        
        logger.info(f"Processing complete: {stats['valid']} valid, {stats['invalid']} invalid examples")
        return output_path

if __name__ == "__main__":
    formatter = DataFormatter()
    input_file = Path('data/raw/training_data.jsonl')
    formatter.format_and_validate_data(input_file, 'formatted_training_data.jsonl')
```

## Fine-Tuning Process

### 1. Create the Fine-Tuning Script

Create `scripts/openai_finetune.py`:

```python
from openai import OpenAI
import os
import time
import logging
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenAIFineTuner:
    def __init__(self, api_key=None):
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        self.client = OpenAI()
        self.model_id = None
    
    def validate_api_access(self):
        try:
            self.client.models.list()
            return True
        except Exception as e:
            logger.error(f"API validation failed: {e}")
            return False
    
    def estimate_cost(self, file_path):
        total_tokens = 0
        with open(file_path, 'r') as f:
            for line in f:
                example = json.loads(line)
                messages = example['messages']
                total_tokens += sum(len(msg['content'].split()) * 1.3 for msg in messages)  # Rough estimate
        
        estimated_cost = (total_tokens / 1000) * 0.008  # $0.008 per 1K tokens
        return estimated_cost

    def create_fine_tune(self, training_file_path, model="gpt-3.5-turbo"):
        # Estimate cost
        estimated_cost = self.estimate_cost(training_file_path)
        logger.info(f"Estimated training cost: ${estimated_cost:.2f}")
        
        # Upload file
        logger.info("Uploading training file...")
        with open(training_file_path, 'rb') as file:
            training_file = self.client.files.create(
                file=file,
                purpose='fine-tune'
            )
        logger.info(f"File uploaded with ID: {training_file.id}")

        # Create job
        logger.info("Creating fine-tuning job...")
        job = self.client.fine_tuning.jobs.create(
            training_file=training_file.id,
            model=model
        )
        logger.info(f"Fine-tuning job created with ID: {job.id}")

        return job.id

    def monitor_fine_tune(self, job_id):
        logger.info("Monitoring fine-tuning progress...")
        while True:
            job = self.client.fine_tuning.jobs.retrieve(job_id)
            status = job.status
            
            if hasattr(job, 'trained_tokens'):
                logger.info(f"Status: {status} | Trained tokens: {job.trained_tokens}")
            else:
                logger.info(f"Status: {status}")
            
            if status in ['succeeded', 'failed']:
                if status == 'succeeded':
                    self.model_id = job.fine_tuned_model
                    logger.info(f"Fine-tuning completed! Model ID: {self.model_id}")
                else:
                    logger.error("Fine-tuning failed!")
                return job
            
            time.sleep(60)

    def test_fine_tuned_model(self, test_questions):
        if not self.model_id:
            logger.error("No fine-tuned model available!")
            return
        
        logger.info("Testing fine-tuned model...")
        results = []
        for question in test_questions:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": "You are a helpful educational assistant."},
                    {"role": "user", "content": question}
                ]
            )
            results.append({
                'question': question,
                'answer': response.choices[0].message.content
            })
        
        return results

def main():
    # Initialize fine-tuner
    fine_tuner = OpenAIFineTuner()
    
    # Validate API access
    if not fine_tuner.validate_api_access():
        logger.error("Failed to validate API access. Exiting...")
        return
    
    # Start fine-tuning
    training_file = Path('data/processed/formatted_training_data.jsonl')
    job_id = fine_tuner.create_fine_tune(training_file)
    
    # Monitor progress
    final_job = fine_tuner.monitor_fine_tune(job_id)
    
    if final_job.status == 'succeeded':
        # Test the model
        test_questions = [
            "What is cellular respiration?",
            "Explain the process of mitosis.",
            "How does the nervous system work?"
        ]
        
        results = fine_tuner.test_fine_tuned_model(test_questions)
        
        # Save results
        output_dir = Path('data/results')
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / 'test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Test results saved to {output_dir / 'test_results.json'}")

if __name__ == "__main__":
    main()
```

## Running the Process

1. Set up your environment:
```bash
# Create project structure
mkdir -p fine-tuning-project/data/{raw,processed,results}
mkdir -p fine-tuning-project/scripts
cd fine-tuning-project

# Copy scripts
# ... copy the above scripts to their respective locations ...

# Install dependencies
pip install -r requirements.txt
```

2. Create and format your dataset:
```bash
python scripts/create_dataset.py
python scripts/format_data.py
```

3. Start fine-tuning:
```bash
python scripts/openai_finetune.py
```

## Using Your Fine-Tuned Model

Create a simple application using your fine-tuned model:

```python
from openai import OpenAI
import logging
import time
from typing import List, Dict

class EducationalAssistant:
    def __init__(self, model_id: str):
        self.client = OpenAI()
        self.model_id = model_id
        self.logger = logging.getLogger(__name__)
        
    def ask_question(self, question: str, max_retries: int = 3) -> str:
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[
                        {"role": "system", "content": "You are a helpful educational assistant."},
                        {"role": "user", "content": question}
                    ],
                    temperature=0.7,
                    max_tokens=500
                )
                return response.choices[0].message.content
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise
    
    def batch_questions(self, questions: List[str]) -> List[Dict[str, str]]:
        results = []
        for question in questions:
            try:
                answer = self.ask_question(question)
                results.append({
                    'question': question,
                    'answer': answer,
                    'status': 'success'
                })
            except Exception as e:
                results.append({
                    'question': question,
                    'error': str(e),
                    'status': 'error'
                })
        return results

# Example usage
if __name__ == "__main__":
    # Initialize assistant
    model_id = "ft:gpt-3.5-turbo-0613:your-org:your-model-name:abc123"  # Replace with your model ID
    assistant = EducationalAssistant(model_id)
    
    # Single question
    question = "What is the theory of relativity?"
    answer = assistant.ask_question(question)
    print(f"Q: {question}\nA: {answer}\n")
    
    # Batch questions
    questions = [
        "What is quantum mechanics?",
        "Explain the process of evolution.",
        "How does gravity work?"
    ]
    results = assistant.batch_questions(questions)
    
    # Print results
    for result in results:
        print(f"Status: {result['status']}")
        if result['status'] == 'success':
            print(f"Q: {result['question']}")
            print(f"A: {result['answer']}\n")
        else:
            print(f"Error: {result['error']}\n")
```

## Best Practices and Tips

1. **Data Quality**:
   - Ensure consistent formatting
   - Use clear, concise answers
   - Include diverse examples
   - Maintain consistent style

2. **Cost Management**:
   - Start with small datasets
   - Monitor token usage
   - Use appropriate model size
   - Implement rate limiting

3. **Error Handling**:
   - Implement retries
   - Use exponential backoff
   - Log all errors
   - Validate responses

4. **Production Deployment**:
   - Use environment variables
   - Implement rate limiting
   - Add monitoring
   - Handle errors gracefully

## Troubleshooting

Common issues and solutions:

1. **API Key Errors**:
   ```python
   openai.error.AuthenticationError: No API key provided
   ```
   Solution:
   ```bash
   # Check if key is set
   echo $OPENAI_API_KEY
   
   # Set key if needed
   export OPENAI_API_KEY="your-key-here"
   ```

2. **File Format Errors**:
   ```python
   openai.error.InvalidRequestError: File is not in correct format
   ```
   Solution:
   ```python
   # Validate JSONL format
   import json
   
   def validate_jsonl(file_path):
       with open(file_path, 'r') as f:
           for i, line in enumerate(f, 1):
               try:
                   json.loads(line)
               except json.JSONDecodeError:
                   print(f"Error on line {i}")
   ```

3. **Token Limit Errors**:
   ```python
   openai.error.InvalidRequestError: Token limit exceeded
   ```
   Solution:
   ```python
   # Implement token counting
   import tiktoken
   
   def count_tokens(text):
       encoding = tiktoken.get_encoding("cl100k_base")
       return len(encoding.encode(text))
   ```

## Next Steps

1. **Experiment with Parameters**:
   - Try different system messages
   - Adjust temperature and top_p
   - Test different model sizes

2. **Enhance Dataset**:
   - Add more examples
   - Include edge cases
   - Improve answer quality

3. **Implement Monitoring**:
   - Track usage metrics
   - Monitor error rates
   - Analyze response quality

4. **Add Features**:
   - Implement caching
   - Add rate limiting
   - Create API endpoints

## Resources

- [OpenAI Fine-tuning Guide](https://platform.openai.com/docs/guides/fine-tuning)
- [OpenAI Cookbook](https://github.com/openai/openai-cookbook)
- [OpenAI Python Library](https://github.com/openai/openai-python)
- [Best Practices](https://platform.openai.com/docs/guides/fine-tuning/best-practices) 