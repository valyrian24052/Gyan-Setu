# Chapter 1: Development Infrastructure and Environment Setup

[![Status](https://img.shields.io/badge/status-complete-green.svg)]() 
[![Last Updated](https://img.shields.io/badge/last%20updated-April%202024-blue.svg)]()

## Learning Objectives

By the end of this chapter, you'll be able to:
- Set up a complete development environment for GenAI applications
- Configure and connect to AI servers for model training and inference
- Implement secure access and data management practices
- Troubleshoot common infrastructure issues in GenAI projects

## 1.1 Introduction to GenAI Infrastructure

Before diving into the technical details of language models and applications, establishing the right infrastructure is crucial. This chapter covers the essential hardware, software, and configuration needed to develop, train, and deploy generative AI applications effectively.

### 1.1.1 The GenAI Technology Stack

Modern generative AI development requires a well-structured technology stack:

```
├── Hardware Layer
│   ├── CPU/GPU/TPU resources
│   ├── Memory requirements
│   └── Storage systems
├── System Software
│   ├── Operating systems
│   ├── Drivers and libraries
│   └── Containerization
├── AI Frameworks
│   ├── PyTorch/TensorFlow
│   ├── Hugging Face Transformers
│   └── Vector databases
└── Application Layer
    ├── APIs and services
    ├── Web interfaces
    └── Monitoring tools
```

Each layer of this stack presents its own requirements and considerations, which we'll explore throughout this textbook.

### 1.1.2 Local vs. Server Development

GenAI development typically involves both local and server environments:

| Environment | Advantages | Limitations | Best For |
|-------------|------------|-------------|----------|
| **Local** | Immediate feedback<br>No connectivity requirements<br>Full control | Limited computational power<br>Resource constraints | Development<br>Small models<br>Testing |
| **Server** | Powerful hardware<br>Scalable resources<br>Centralized data | Network dependency<br>Setup complexity<br>Shared resources | Training<br>Large models<br>Production |

For the UTTA project (our case study), we use a hybrid approach: local development for interface design and prototyping, with server resources for model training and knowledge base operations.

## 1.2 Development Environment Prerequisites

Before building knowledge-enhanced GenAI applications, ensure your local environment has the necessary tools:

### 1.2.1 Local Development Requirements

- **Python 3.10 or higher**: Required for compatibility with modern ML libraries
- **Git**: For version control and collaboration
- **Docker and Docker Compose**: For containerized applications
- **Node.js 18+**: For frontend development 
- **VSCode or PyCharm**: Recommended IDEs with ML/AI extensions

**Installation guide for Ubuntu/Debian systems:**
```bash
# Update package lists
sudo apt update

# Install Python 3.10 and dependencies
sudo apt install python3.10 python3.10-venv python3-pip

# Install Git
sudo apt install git

# Install Docker and Docker Compose
sudo apt install docker.io docker-compose
sudo usermod -aG docker $USER  # Add user to docker group

# Install Node.js 18+
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install nodejs

# Install development tools
sudo apt install build-essential
```

**Installation guide for macOS (using Homebrew):**
```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python 3.10
brew install python@3.10

# Install Git
brew install git

# Install Docker Desktop
brew install --cask docker

# Install Node.js 18+
brew install node@18

# Install development tools
brew install cmake
```

### 1.2.2 Hardware Recommendations

| Component | Minimum | Recommended for Local Inference |
|-----------|---------|--------------------------------|
| CPU | 4 cores | 8+ cores |
| RAM | 8GB | 16GB+ |
| Storage | 10GB SSD | 20GB+ SSD |
| GPU | None | CUDA-compatible (8GB+ VRAM) |

For the UTTA project, we found that 16GB RAM and an NVIDIA GPU with at least 8GB VRAM provides a good balance for local development with smaller LLMs (7B parameters or less).

### 1.2.3 Python Environment Setup

Creating isolated Python environments is essential for effective development:

```bash
# Create a virtual environment
python -m venv venv

# Activate virtual environment
# On Unix/macOS:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

A typical requirements.txt file for GenAI development includes:

```
torch>=2.0.0
transformers>=4.30.0
datasets>=2.12.0
sentencepiece>=0.1.99
tensorboard>=2.12.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
tqdm>=4.65.0
requests>=2.28.0
python-dotenv>=1.0.0
```

## 1.3 AI Server Infrastructure

For training and serving large models, a dedicated AI server provides necessary computational resources:

### 1.3.1 Server Specifications

The UTTA project utilizes the following server configuration:

- **Server OS**: Ubuntu 24.04 LTS
- **Hostname**: d19559
- **Purpose**: Model training, vector database hosting, and inference
- **Access Method**: SSH via secure VPN

### 1.3.2 Hardware Configuration

Our reference AI server is configured with:

- **CPU**: 32-core AMD EPYC processor
- **RAM**: 128GB DDR5
- **Storage**: 2TB NVMe SSD
- **GPU**: 2x NVIDIA 4000 series (16GB each)
- **Storage**: 20TB 
- **Network**: 10Gbps connection

This configuration is sufficient for fine-tuning models up to 70B parameters and serving multiple concurrent instances of 13B parameter models.

### 1.3.3 Server Software Stack

The AI server comes pre-configured with:

```
├── Base System
│   ├── Ubuntu 24.04 LTS
│   ├── CUDA 12.2 + cuDNN
│   ├── Docker + NVIDIA Container Toolkit
│   └── Python 3.10
├── ML/AI Software
│   ├── PyTorch 2.1
│   ├── Transformers 4.35+
│   ├── Sentence Transformers
│   ├── LangChain
│   └── FAISS/Chroma vector stores
└── Databases
    ├── PostgreSQL
    ├── Redis
    └── MongoDB
```

### 1.3.4 Server Deployment Options

Several deployment architectures can be considered for GenAI applications:

1. **Single Server Deployment**:
   - Simplest configuration
   - All components on one machine
   - Suitable for smaller projects and teams

2. **Distributed Deployment**:
   - Separate model servers and application servers
   - Dedicated database servers
   - Load balancing for high availability

3. **Containerized Deployment**:
   - Docker containers for each component
   - Kubernetes for orchestration
   - Scalable and reproducible

4. **Cloud-Based Deployment**:
   - Leveraging cloud providers' AI services
   - Elastic scaling based on demand
   - Managed services for reduced maintenance

For UTTA, we use a hybrid approach with containerized deployment on dedicated hardware, which balances performance needs with management simplicity.

## 1.4 Server Access Configuration

### 1.4.1 VPN Access

Secure access to the AI server requires VPN credentials:

1. Request VPN access through your organization's IT department
2. For UVU students/staff: [UVU VPN Service](https://www.uvu.edu/itservices/information-security/vpn_campus.html)
3. Contact support: (801) 863-8888
4. Note: VPN access typically requires renewal each semester

### 1.4.2 User Accounts

Each development team receives dedicated user accounts:

| Team | Username | Initial Password | Workspace |
|------|----------|-----------------|-----------|
| Team 1 | team1 | Team2ndGrade12024! | team1_data |
| Team 2 | team2 | Team2ndGrade22024! | team2_data |
| Team 3 | team3 | Team2ndGrade32024! | team3_data |
| Team 4 | team4 | Team2ndGrade42024! | team4_data |
| Team 5 | team5 | Team2ndGrade52024! | team5_data |
| Team 6 | team6 | Team2ndGrade62024! | team6_data |

⚠️ **IMPORTANT**: Change password on first login!

### 1.4.3 SSH Configuration

For efficient access to remote servers, configure SSH properly:

```bash
# Create SSH config file if it doesn't exist
touch ~/.ssh/config
chmod 600 ~/.ssh/config

# Add server configuration
cat << EOF >> ~/.ssh/config
Host utta-server
    HostName d19559
    User teamX  # Replace X with your team number
    Port 22
    ForwardAgent yes
    ServerAliveInterval 60
EOF
```

This allows simple connection with:
```bash
ssh utta-server
```

## 1.5 Initial Project Setup

### 1.5.1 Repository Configuration

```bash
# Clone repository
git clone https://github.com/your-org/knowledge-rag-app.git
cd knowledge-rag-app

# Set up Git configuration
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

### 1.5.2 Environment Variables

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your settings
# Required variables:
# - DATABASE_URL
# - AI_MODEL_PATH
# - API_KEY
# - VECTOR_DB_PATH
```

### 1.5.3 Database Configuration

```bash
# Start database container
docker-compose up -d db

# Run migrations
python manage.py migrate

# Create vector database
python scripts/init_vector_db.py
```

## 1.6 AI Server Workspace

### 1.6.1 Connection Process

```bash
# 1. Connect to VPN
# 2. SSH to server
ssh teamX@d19559  # Replace X with your team number

# 3. Change password on first login
passwd
```

### 1.6.2 Directory Structure

Each team's workspace follows this structure:

```
/home/teamX/
├── data/              # Training and vector data
│   ├── documents/     # Source documents
│   ├── embeddings/    # Vector embeddings
│   └── models/        # Fine-tuned models
├── code/              # Application code
├── databases/         # Database files
│   ├── vector/        # Vector DB files
│   └── metadata/      # Metadata DB files
├── logs/              # System and application logs
└── backups/           # Regular backups
```

### 1.6.3 Running Large Models

The AI server supports multiple options for running large models:

1. **Local LLM**:
```bash
# Run Llama 3 8B model locally
python -m knowledge_app.serve.llm_server \
  --model /shared/models/llama3-8b \
  --port 8080
```

2. **Quantized Models**:
```bash
# Run 4-bit quantized model for lower memory usage
python -m knowledge_app.serve.llm_server \
  --model /shared/models/llama3-8b \
  --quantize 4bit \
  --port 8080
```

3. **API Integration**:
```bash
# Use external API (configured in .env)
python -m knowledge_app.serve.api_proxy \
  --port 8081
```

## 1.7 Security Best Practices

### 1.7.1 Password Requirements

- Minimum 12 characters
- Mix of uppercase and lowercase letters
- Include numbers and special characters
- No dictionary words
- Change every 90 days

### 1.7.2 Data Security

When working with knowledge bases and large models:

- Keep sensitive data on server
- No unauthorized data transfers
- Follow backup procedures
- Encrypt sensitive files using:

```bash
# Encrypt sensitive file
gpg -c --cipher-algo AES256 sensitive_data.json

# Decrypt when needed
gpg sensitive_data.json.gpg
```

### 1.7.3 API Key Management

Store and access API keys securely:

```python
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access API keys
api_key = os.getenv("OPENAI_API_KEY")
hf_token = os.getenv("HF_TOKEN")

# Never hard-code keys in your application
```

### 1.7.4 Model and Data Security Considerations

When working with LLMs and training data, security is crucial:

- **Model Access Control**: Limit access to model weights and API endpoints
- **Inference Logging**: Log queries and responses for security auditing
- **Data Sanitization**: Sanitize training data to remove PII and sensitive information
- **Prompt Injection Prevention**: Implement safeguards against prompt injection attacks
- **Output Filtering**: Filter harmful outputs using content moderation

## 1.8 Troubleshooting Common Issues

### 1.8.1 VPN Connection Problems

1. **Connection Failed**
   - Check VPN credentials
   - Verify network connection
   - Contact IT if persistent: (801) 863-8888

2. **SSH Access Denied**
   - Verify username/password
   - Check VPN connection status
   - Ensure account is active

### 1.8.2 Model Loading Issues

1. **Out of Memory Errors**
   ```
   CUDA out of memory. Tried to allocate 2.00 GiB
   ```
   
   Solutions:
   - Use model quantization: `--quantize 4bit`
   - Reduce batch size: `--batch-size 1`
   - Use CPU offloading: `--cpu-offload`

2. **Model Not Found**
   ```
   FileNotFoundError: [Errno 2] No such file or directory
   ```
   
   Solutions:
   - Check model path
   - Verify permissions
   - Download missing models: 
     ```bash
     python scripts/download_models.py --model llama3-8b
     ```

### 1.8.3 Environment Issues

Common environment issues and solutions:

| Issue | Symptoms | Solution |
|-------|----------|----------|
| Module not found | `ImportError: No module named 'x'` | Activate virtual environment and install requirements |
| CUDA version mismatch | `CUDA error: CUBLAS_STATUS_ALLOC_FAILED` | Update CUDA or use compatible torch version |
| Permission denied | `PermissionError: [Errno 13]` | Check file/directory permissions |
| Memory limit | Application crashes without error | Increase swap space or reduce model size |

## 1.9 Hands-On Exercise: Setting Up Vector Search

Let's practice setting up a vector search environment on the AI server:

```bash
# Connect to the server
ssh teamX@d19559

# Navigate to your workspace
cd ~/code

# Clone the vector search repository
git clone https://github.com/your-org/vector-search-demo.git
cd vector-search-demo

# Install dependencies
pip install -r requirements.txt

# Initialize the vector database
python init_db.py --data_path ~/data/documents

# Start the search service
python serve.py --host 0.0.0.0 --port 8000
```

Access the search UI at: `http://d19559:8000` (through VPN)

## 1.10 UTTA Case Study: Infrastructure for Teacher Training

The UTTA project demonstrates how infrastructure decisions impact AI application development. Let's examine the specific requirements for this educational application:

### 1.10.1 Technical Requirements

For the teacher training assistant, the infrastructure needs to support:

1. **Large Knowledge Base**: Storage and processing of educational materials
2. **Real-Time Conversations**: Low-latency response generation
3. **User Authentication**: Secure access for teachers and administrators
4. **Content Management**: Tools for educators to add and modify content
5. **Evaluation Tools**: Systems to assess model performance on educational tasks

### 1.10.2 Implementation Choices

The UTTA team made the following infrastructure decisions:

- **Model Selection**: Deployed quantized 13B parameter models optimized for teacher-student interactions
- **Vector Database**: Used Chroma with PostgreSQL backend for educational content
- **Deployment**: Containerized application with separate containers for the API, database, and frontend
- **Monitoring**: Implemented detailed logging of teacher interactions for continuous improvement

### 1.10.3 Lessons Learned

Key infrastructure insights from the UTTA project:

1. **Start Small**: Begin with smaller models locally before scaling to larger server models
2. **Modular Design**: Separate the vector database, model service, and application logic
3. **Progressive Enhancement**: Build functionality incrementally, starting with core features
4. **Continual Monitoring**: Implement comprehensive logging to identify performance bottlenecks

## 1.11 Key Takeaways

- A properly configured development environment is essential for GenAI projects
- AI servers provide the computational resources needed for large models
- Secure practices protect sensitive data and API credentials
- Understanding troubleshooting processes saves development time
- Local vs. server development requires different configuration approaches

## 1.12 Chapter Project: Development Environment Setup

For this chapter's project, you'll configure a complete development environment:

1. Set up your local development tools
2. Connect to the AI server and configure your workspace
3. Create a simple knowledge base application that runs both locally and on the server
4. Document the differences in configuration between local and server deployment
5. Implement a backup strategy for your vector database

## References

- Ubuntu Server Documentation: [https://ubuntu.com/server/docs](https://ubuntu.com/server/docs)
- NVIDIA CUDA Documentation: [https://docs.nvidia.com/cuda/](https://docs.nvidia.com/cuda/)
- Docker Documentation: [https://docs.docker.com/](https://docs.docker.com/)
- Hugging Face Model Deployment: [https://huggingface.co/docs/optimum/en/serving/overview](https://huggingface.co/docs/optimum/en/serving/overview)

## Further Reading

- [Chapter 2: Natural Language Processing Fundamentals](NLP-Fundamentals)
- [Chapter 4: Embeddings and Vector Representations](Knowledge-Base-Structure)
- [Chapter 5: Large Language Models](Knowledge-LLM-Integration)
- [Chapter 10: Deployment and Evaluation](Evaluation-Testing)