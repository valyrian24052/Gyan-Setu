# LLM Fine-Tuning

Welcome to the comprehensive documentation for the LLM Fine-Tuning Toolkit! This wiki provides detailed explanations, instructions, and examples for fine-tuning large language models using three different approaches.

## 📚 Contents

### Introduction
- [Overview of Fine-Tuning](Overview-of-Fine-Tuning.md)
- [Choosing the Right Method](Choosing-the-Right-Method.md)
- [System Requirements](System-Requirements.md)

### Fine-Tuning Methods
- [DSPy Optimization](DSPy-Optimization.md)
- [OpenAI Fine-Tuning](OpenAI-Fine-Tuning.md)
- [HuggingFace Fine-Tuning with LoRA](HuggingFace-LoRA.md)

### Dataset Preparation
- [Educational QA Dataset](Educational-QA-Dataset.md)
- [Dataset Formats and Requirements](Dataset-Formats.md)
- [Data Cleaning and Preprocessing](Data-Preprocessing.md)

### Tutorials
- [Setting Up Your Environment](Environment-Setup.md)
- [DSPy Step-by-Step Tutorial](DSPy-Tutorial.md)
- [OpenAI Step-by-Step Tutorial](OpenAI-Tutorial.md)
- [HuggingFace LoRA Step-by-Step Tutorial](HuggingFace-Tutorial.md)

### Advanced Topics
- [Evaluation Metrics](Evaluation-Metrics.md)
- [Hyperparameter Tuning](Hyperparameter-Tuning.md)
- [Model Deployment](Model-Deployment.md)
- [Troubleshooting](Troubleshooting.md)

## 🔍 Quick Method Comparison

| Feature | DSPy | OpenAI | HuggingFace |
|---------|------|--------|-------------|
| What changes | Prompts | Model weights (cloud) | Model weights (local) |
| Training data needed | Small (10s) | Medium (100s-1000s) | Medium/Large (1000s+) |
| Cost | API calls only | API + training | One-time compute |
| Setup difficulty | Simple | Simple | Complex |
| Control | Limited | Medium | Full |
| Hardware required | None | None | GPU recommended |
| Deployment | API calls | API calls | Self-host |
| Data privacy | Data shared with API | Data shared with API | Data stays local |

## 🚀 Getting Started

New to LLM fine-tuning? Start with:
1. [Overview of Fine-Tuning](Overview-of-Fine-Tuning.md)
2. [Setting Up Your Environment](Environment-Setup.md)
3. Choose a method tutorial based on your needs

## 📘 References

For academic papers, official documentation, and other resources, see our [References](References.md) page. 