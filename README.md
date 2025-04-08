# UTTA (Utah Teacher Training Assistant)

An AI-powered teaching assistant enhanced by Expert Tuning, our specialized system that leverages expert feedback for continuous improvement using DSPy for efficient small-dataset optimization.

## 🌟 Key Features

### 1. Interactive Teaching Practice
- **Simulated Student Interactions:** Practice teaching in a safe environment
- **Real-time Feedback:** Get immediate guidance on your teaching approach
- **Scenario Variety:** Wide range of teaching situations and student profiles

### 2. Expert Tuning Platform
Our innovative expert feedback collection and model optimization system:
- **DSPy Integration:** Efficient fine-tuning with small datasets
  - Optimized for expert feedback
  - Rapid iteration and improvement
  - Low resource requirements
- **Expert Review System:** Review and rate AI-generated teaching responses
- **Teaching Example Submission:** Share your expertise as training data
- **Structured Data Collection:**
  - **Quality Metrics:** Numerical ratings across teaching dimensions
  - **Improved Responses:** Expert-provided better alternatives
  - **Training Examples:** Real-world teaching scenarios
  - **Training Dataset:** Automatically formatted for DSPy optimization

### 3. Technical Features
- **Model Optimization:** 
  - DSPy-powered language model interactions
  - Continuous improvement through expert feedback
  - Automated enhancement pipeline
  - Quality-weighted training data
  - Version-controlled improvements
- **Knowledge Base:** Context-aware responses enhanced by expert feedback
- **Modern Interface:** Built with Streamlit for ease of use
- **Data Pipeline:**
  - Structured feedback collection
  - DSPy-optimized preprocessing
  - Quality filtering
  - Version control

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Conda package manager
- Git
- Access to Expert Tuning platform (for contributors)

### Quick Start
```bash
# Clone repository
git clone https://github.com/UVU-AI-Innovate/UTTA.git
cd UTTA

# Create and activate conda environment
conda create -n utta python=3.10
conda activate utta

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run src/web/app.py
```

## 📁 Project Structure

```
UTTA/
├── src/
│   ├── core/                # Core functionality
│   ├── llm/                # Language model integration
│   │   ├── dspy/           # DSPy framework integration
│   │   └── expert_tuning/  # Expert Tuning pipeline
│   └── web/                # Web interface
├── docs/
│   ├── expert_tuning/      # Expert Tuning documentation
│   │   ├── overview.md     # System overview
│   │   ├── quickstart.md   # Quick start guide
│   │   └── dspy.md        # DSPy integration guide
├── training_datasets/      # Training data
│   ├── expert_feedback/    # Raw expert feedback
│   ├── processed/          # DSPy-processed training data
│   └── versions/           # Dataset versions
└── knowledge_base/         # Teaching resources
```

## 🎯 Usage

### Basic Mode
1. Start the application
2. Choose a teaching scenario
3. Interact with simulated student
4. Receive feedback and guidance

### Expert Mode
1. Navigate to the Expert Tuning dashboard
2. Choose from three main activities:
   - **Review Teaching Examples:** Rate AI responses and provide improvements
   - **Submit Your Solutions:** Share your teaching expertise as training data
   - **View Impact:** Track how your contributions improve the system

### Contributing as an Expert
- Review AI-generated responses and provide improvements
- Rate responses across teaching dimensions
- Share your own teaching examples
- Provide detailed explanations
- See how your feedback helps improve the AI through DSPy optimization

### Data Quality and Impact
- High-quality contributions are automatically added to training datasets
- DSPy efficiently processes small datasets for maximum impact
- Track how your feedback influences performance
- Earn recognition for helpful feedback
- View impact metrics
- Participate in focused improvement campaigns

## 🤝 Contributing

We welcome contributions! See our [Contributing Guide](docs/CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Utah Valley University
- Contributing teaching experts
- DSPy development team
- Open source community

---
Built with ❤️ by the UVU AI Innovation Lab Team