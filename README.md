# LLM Chatbot Framework

This repository contains the LLM Chatbot Framework, a flexible framework for building and evaluating educational LLM-powered chatbots with fine-tuning capabilities.

## Repository Structure

The main project is located in the `llm-chatbot-framework` directory. Please navigate there for the full project:

```cd llm-chatbot-framework
```

## Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/llm-chatbot-framework.git

# Navigate to the project directory
cd llm-chatbot-framework

# Set up the environment
conda create -n utta python=3.10
conda activate utta

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Model Access and Authentication

The framework supports various LLM models, including:
- LLaMA 2 models (7B, 13B, 70B variants)
- Mistral models
- Other Hugging Face models

To use these models:

1. Create a Hugging Face account at https://huggingface.co
2. Generate an access token from your Hugging Face account settings
3. Set up your credentials using one of these secure methods:
   ```bash
   # Method 1: Use the CLI tool (recommended)
   huggingface-cli login
   
   # Method 2: The framework will prompt for your token when needed
   # and store it securely
   ```
4. Request access to specific models (e.g., LLaMA 2) through their respective model pages on Hugging Face

## LLM Evaluation Framework

The project includes a comprehensive evaluation framework for assessing chatbot performance:

### Supported Models
- LLaMA 2 series (7B, 13B, 70B)
- Mistral-7B
- Other Hugging Face compatible models

### Automated Metrics

The framework employs several industry-standard metrics to evaluate response quality:

#### Text Overlap Metrics
- **ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**
  - ROUGE-N: Measures n-gram overlap between model output and reference
  - ROUGE-L: Captures longest common subsequence, preserving word order
  - ROUGE-W: Weighted version that favors consecutive matches
  - Useful for: Measuring content preservation and summary quality

#### Semantic Similarity
- **BERTScore**
  - Uses contextual embeddings from BERT to compute similarity
  - Better than exact matches as it captures semantic meaning
  - Correlates well with human judgments
  - Handles synonyms and paraphrasing effectively
  - Provides precision, recall, and F1 scores

#### Translation Quality Metrics
- **BLEU (Bilingual Evaluation Understudy)**
  - Measures precision of n-gram matches
  - Applies brevity penalty for short outputs
  - Industry standard for translation tasks
  
- **METEOR (Metric for Evaluation of Translation with Explicit ORdering)**
  - Handles stemming and synonyms
  - Considers word order and alignment
  - More robust for shorter texts
  - Better correlation with human judgments than BLEU

#### Conversation Evaluation
The framework supports two evaluation modes:
1. **Single Response Mode**
   - Evaluates individual responses in isolation
   - Useful for testing specific capabilities
   
2. **Conversation Mode**
   - Evaluates entire conversation threads
   - Considers context and coherence
   - Tracks metrics across multiple turns
   - Measures consistency and engagement

### Human Evaluation

The framework provides a comprehensive human evaluation system:

#### Structured Evaluation Forms
- **Customizable Templates**
  - JSON/YAML-based form definitions
  - Configurable rating scales (1-5, 1-10, custom ranges)
  - Support for multiple question types:
    - Likert scales
    - Multiple choice
    - Free-form text
    - Binary yes/no
  - Option to include reference answers and guidelines

#### Assessment Criteria
- **Response Relevance**
  - Topic adherence
  - Context awareness
  - Query satisfaction
  - Information completeness

- **Response Accuracy**
  - Factual correctness
  - Source consistency
  - Technical precision
  - Citation accuracy

- **Response Quality**
  - Language fluency
  - Logical coherence
  - Clarity of explanation
  - Professional tone
  - Appropriate detail level

#### Score Aggregation
- **Statistical Analysis**
  - Inter-rater reliability metrics
  - Confidence intervals
  - Agreement analysis
  - Outlier detection
  
- **Aggregation Methods**
  - Weighted averaging
  - Majority voting
  - Bayesian aggregation
  - Custom scoring formulas

#### Feedback Collection
- **Structured Feedback**
  - Timestamped annotations
  - Error categorization
  - Improvement suggestions
  - Priority levels
  
- **Quality Control**
  - Expert validation
  - Cross-checking mechanisms
  - Feedback verification
  - Evaluator qualification tracking

### Benchmark Suite

#### Test Case Management
- **YAML Configuration**
  ```yaml
  test_suite:
    name: "Educational Content Generation"
    version: "1.0"
    categories:
      - knowledge_check
      - explanation_generation
      - feedback_provision
    test_cases:
      - id: "KC001"
        type: "knowledge_check"
        input: "Explain photosynthesis"
        expected_elements:
          - "light energy"
          - "chlorophyll"
          - "carbon dioxide"
        scoring:
          method: "element_coverage"
          min_score: 0.8
  ```
- **Version Control Integration**
  - Git-based test case versioning
  - Change tracking
  - Collaborative editing
  - History maintenance

#### Automated Benchmarking
- **Execution Pipeline**
  - Parallel test execution
  - Batch processing
  - Failure recovery
  - Progress monitoring
  
- **Resource Management**
  - Dynamic resource allocation
  - Load balancing
  - Memory optimization
  - Timeout handling

#### Result Analysis
- **Comparison Tools**
  - Version-to-version comparison
  - Model-to-model comparison
  - Baseline deviation analysis
  - Performance trending
  
- **Visualization**
  - Interactive dashboards
  - Performance graphs
  - Error distribution charts
  - Resource usage plots

#### Predefined Test Sets
- **Educational Scenarios**
  - Content explanation
  - Student assessment
  - Feedback generation
  - Concept clarification
  
- **Domain-Specific Tests**
  - STEM subjects
  - Language learning
  - Professional training
  - Skill assessment

#### Resource Monitoring
- **GPU Metrics**
  - VRAM usage tracking
  - Temperature monitoring
  - Power consumption
  - Utilization rates
  
- **Performance Optimization**
  - Dynamic batch sizing
  - Memory management
  - Cache optimization
  - Load distribution

### Running Evaluations

The framework provides several ways to run evaluations:

#### Using the CLI

```bash
# Run a complete evaluation suite
python -m src.evaluation.run_eval \
    --model="meta-llama/Llama-2-13b-hf" \
    --suite="education_suite" \
    --output="evaluation_results/results.json"

# Run specific test categories
python -m src.evaluation.run_eval \
    --model="meta-llama/Llama-2-13b-hf" \
    --categories="knowledge_check,explanation_generation" \
    --config="config/eval_config.yaml"

# Run with resource constraints
python -m src.evaluation.run_eval \
    --model="meta-llama/Llama-2-13b-hf" \
    --max-gpu-mem="20GB" \
    --batch-size=4 \
    --quantization="4bit" \
    --device="cuda:0"

# Generate detailed reports
python -m src.evaluation.generate_report \
    --results="evaluation_results/results.json" \
    --output="evaluation_results/report" \
    --include-visualizations
```

#### Using Python API

```python
from src.evaluation import Evaluator
from src.llm import ModelManager

# Initialize the model
model_manager = ModelManager()
model = model_manager.load_model("meta-llama/Llama-2-13b-hf")

# Create evaluator
evaluator = Evaluator(
    model=model,
    config_path="config/eval_config.yaml"
)

# Run evaluation
results = evaluator.run_suite("education_suite")

# Generate report
evaluator.generate_report(
    results=results,
    output_path="evaluation_results/report"
)
```

#### Configuration

Evaluation settings can be configured through YAML files:

```yaml
# config/eval_config.yaml
evaluation:
  metrics:
    - rouge
    - bleu
    - bertscore
    - meteor
  
  human_evaluation:
    enabled: true
    num_evaluators: 3
    criteria:
      - relevance
      - accuracy
      - quality
  
  resource_monitoring:
    enabled: true
    track_gpu: true
    track_memory: true
    log_interval: 60  # seconds
```

For detailed documentation and examples, see the [LLM Evaluation Wiki](UTTA.wiki/LLM-Evaluation.md).

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on:
- Code style and standards
- Pull request process
- Development setup
- Testing requirements

## License

This project is licensed under the MIT License - see the LICENSE file for details.