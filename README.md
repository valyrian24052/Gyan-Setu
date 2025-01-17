# Utah Elementary Teacher Training Assistant (UTAH-TTA)

An LLM-powered educational chatbot designed specifically for training elementary school teachers in Utah. The system simulates authentic classroom scenarios to help teachers develop and refine their classroom management, student interaction, and response strategies aligned with Utah's elementary education program standards.

## 🎯 Project Overview

UTAH-TTA provides:
- Realistic elementary classroom scenario simulations
- Personalized feedback on teacher responses
- Alignment with Utah elementary education program objectives
- Research-based evaluation of classroom management strategies
- Progressive learning paths from basic to complex scenarios

## 📋 Table of Contents

- [Repository Structure](#-repository-structure)
- [For Each Role](#-for-each-role)
  - [Product Owner](#product-owner)
  - [Educational Content Specialist](#educational-content-specialist)
  - [AI/ML Developer](#aiml-developer)
  - [Frontend Developer](#frontend-developer)
  - [Project Manager](#project-manager)
  - [QA Specialist](#qa-specialist)
- [Getting Started](#-getting-started)
- [Development Workflow](#-development-workflow)
- [Key Features](#-key-features)
- [Documentation](#-documentation)
- [Contributing](#-contributing)
- [Configuration](#-configuration)
- [Getting Help](#-getting-help)
- [Monitoring](#-monitoring)
- [Security](#-security)
- [Regular Maintenance](#-regular-maintenance)

## 📁 Repository Structure

```
teacher-training-chatbot/
├── src/                              # Source code
│   ├── ai/                           # AI/ML components
│   │   ├── embedding.py              # Embedding generation
│   │   ├── rag_pipeline.py           # RAG implementation
│   │   ├── llm_config.py            # LLM configuration
│   │   └── evaluation.py            # Response evaluation
│   │
│   ├── database/                     # Database components
│   │   ├── models.py                # Database models
│   │   ├── vector_ops.py            # Vector operations
│   │   └── connection.py            # Database connection
│   │
│   ├── frontend/                     # Frontend components
│   │   ├── static/                  # Static assets
│   │   ├── templates/               # HTML templates
│   │   └── routes.py                # Frontend routes
│   │
│   └── api/                         # API endpoints
│       ├── routes.py                # API routes
│       └── middleware.py            # API middleware
│
├── data/                            # Data directory
│   ├── scenarios/                   # Teaching scenarios
│   │   ├── approved/               # Expert-approved scenarios
│   │   ├── drafts/                 # Scenario drafts
│   │   └── templates/              # Scenario templates
│   │
│   ├── personas/                    # Student personas
│   │   ├── templates/              # Persona templates
│   │   └── approved/               # Approved personas
│   │
│   └── evaluation/                  # Evaluation data
│       ├── criteria/               # Evaluation criteria
│       └── feedback/               # Feedback templates
│
├── docs/                            # Documentation
│   ├── roles/                       # Role-specific guides
│   ├── scenarios/                   # Scenario documentation
│   ├── technical/                   # Technical documentation
│   └── meetings/                    # Meeting notes
│
├── tests/                           # Test files
├── scripts/                         # Utility scripts
└── config/                          # Configuration files
```

## 🎯 For Each Role

### Product Owner
- **Primary Guide**: [`docs/roles/product_owner.md`](docs/roles/product_owner.md)
- **Key Resources**:
  - [Expert Meeting Notes](docs/meetings/expert_reviews/)
  - [Approved Scenarios](data/scenarios/approved/)
  - [Scenario Creation Guide](docs/scenarios/creation_guide.md)

### Educational Content Specialist
- **Primary Guide**: [`docs/roles/content_specialist.md`](docs/roles/content_specialist.md)
- **Key Resources**:
  - [Scenario Templates](data/scenarios/templates/)
  - [Student Personas](data/personas/templates/)
  - [Evaluation Criteria](data/evaluation/criteria/)

### AI/ML Developer
- **Primary Guide**: [`docs/roles/ai_developer.md`](docs/roles/ai_developer.md)
- **Key Resources**:
  - [AI Implementation](src/ai/)
  - [Model Configurations](config/)
  - [AI Component Tests](tests/ai/)

### Frontend Developer
- **Primary Guide**: [`docs/roles/frontend_developer.md`](docs/roles/frontend_developer.md)
- **Key Resources**:
  - [Frontend Code](src/frontend/)
  - [API Endpoints](src/api/)
  - [Frontend Tests](tests/frontend/)

### Project Manager
- **Primary Guide**: [`docs/roles/project_manager.md`](docs/roles/project_manager.md)
- **Key Resources**:
  - [Project Timeline](docs/project_pipeline.md)
  - [Meeting Notes](docs/meetings/)
  - [Repository Structure](docs/repository_structure.md)

### QA Specialist
- **Primary Guide**: [`docs/roles/qa_specialist.md`](docs/roles/qa_specialist.md)
- **Key Resources**:
  - [Test Files](tests/)
  - [Test Scenarios](data/scenarios/approved/)
  - [Technical Documentation](docs/technical/)

## 🚀 Getting Started

1. **Environment Setup**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # or `venv\Scripts\activate` on Windows
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Configuration**
   ```bash
   # Set environment
   export APP_ENV=development  # or production, testing
   
   # Set database credentials
   export DB_USER=your_username
   export DB_PASSWORD=your_password
   ```

3. **Database Setup**
   ```bash
   # Install PostgreSQL and pgvector
   python scripts/setup/init_database.py
   ```

4. **Running the Application**
   ```bash
   # Start the application
   python src/api/main.py
   ```

## 📝 Development Workflow

1. **Creating New Scenarios**
   - Draft in [`data/scenarios/drafts/`](data/scenarios/drafts/)
   - Get expert review
   - Move to [`data/scenarios/approved/`](data/scenarios/approved/)

2. **Making Changes**
   - Create feature branch
   - Update tests
   - Update documentation
   - Create pull request

3. **Running Tests**
   ```bash
   # Run all tests
   pytest tests/
   
   # Run specific component tests
   pytest tests/ai/
   pytest tests/frontend/
   ```

## 🔍 Key Features

1. **RAG Pipeline**
   - [Embedding Generation](src/ai/embedding.py)
   - [Vector Similarity Search](src/database/vector_ops.py)
   - [Context-aware Responses](src/ai/rag_pipeline.py)

2. **Scenario Management**
   - [Expert Validation Process](docs/scenarios/validation_process.md)
   - [Version Control Guide](docs/contributing/version_control.md)
   - [Performance Tracking](docs/technical/monitoring.md)

3. **Evaluation System**
   - [Response Scoring](src/ai/evaluation.py)
   - [Feedback Generation](data/evaluation/feedback/)
   - [Progress Tracking](docs/technical/progress_tracking.md)

## 📚 Documentation

- [Technical Guides](docs/technical/)
- [API Documentation](docs/technical/api/)
- [Database Schema](docs/technical/database/)
- [Architecture Overview](docs/technical/architecture/)

## 🤝 Contributing

1. Read [`docs/contributing/guidelines.md`](docs/contributing/guidelines.md)
2. Follow [role-specific guides](docs/roles/)
3. Use [templates](data/templates/)
4. Ensure test coverage
5. Update documentation

## 🔧 Configuration

- [Development Config](config/development.py)
- [Production Config](config/production.py)
- [Testing Config](config/testing.py)

## 🆘 Getting Help

1. Check [role-specific documentation](docs/roles/)
2. Review [technical guides](docs/technical/)
3. Contact team lead
4. [Create an issue](docs/contributing/creating_issues.md)

## 📊 Monitoring

- Application logs in [`logs/`](logs/)
- [Metrics Dashboard](docs/technical/metrics.md)
- [Performance Monitoring](docs/technical/performance.md)
- [Error Tracking](docs/technical/error_tracking.md)

## 🔐 Security

- [SSL/TLS Configuration](docs/technical/security/ssl_config.md)
- [Rate Limiting](docs/technical/security/rate_limiting.md)
- [Input Validation](docs/technical/security/input_validation.md)
- [Access Control](docs/technical/security/access_control.md)

## 📅 Regular Maintenance

1. [Database Backups](docs/technical/maintenance/backups.md)
2. [Log Rotation](docs/technical/maintenance/log_rotation.md)
3. [Performance Monitoring](docs/technical/maintenance/monitoring.md)
4. [Security Updates](docs/technical/maintenance/security_updates.md)
 