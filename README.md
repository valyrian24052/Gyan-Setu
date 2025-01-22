# Utah Elementary Teacher Training Assistant (UTAH-TTA)
## Second Grade Focus

A specialized LLM-powered educational chatbot designed for training elementary school teachers in Utah, with a specific focus on second-grade education. The system simulates authentic classroom scenarios to help teachers develop effective teaching strategies aligned with Utah's second-grade curriculum standards.

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Data Collection Focus](#-data-collection-focus)
- [Knowledge Base Structure](#knowledge-base-structure)
- [Repository Structure](#-repository-structure)
- [Team Roles](#-for-each-role)
- [Educational Content Specialist](#-educational-content-specialist)
- [Getting Started](#-getting-started)
- [Development Workflow](#-development-workflow)
- [Documentation](#-documentation)
- [Maintenance](#-maintenance)

## 🎯 Project Overview

UTAH-TTA Second Grade Edition provides:
- Second-grade specific classroom scenarios
- Alignment with Utah Core Standards for 2nd Grade
- Research-based teaching strategies for 7-8 year olds
- Age-appropriate classroom management techniques
- Progressive learning paths for second-grade teachers

## 📚 Data Collection Focus

## Knowledge Base Structure

Our data collection is organized into three main categories:

### 1. Educational Science (data/education_science/)
- **Utah Core Standards**
  - Second-grade specific standards
  - Learning objectives
  - Assessment criteria
- **Teaching Methodologies**
  - Evidence-based practices
  - Age-appropriate strategies
  - STEM integration approaches
- **Learning Psychology**
  - Child development principles
  - Cognitive development stages
  - Learning style adaptations

### 2. Teacher-Student Interactions (data/interactions/)
- **Classroom Dialogues**
  - Real-world examples
  - Best practice demonstrations
  - Common challenges
- **Behavior Management**
  - Positive reinforcement examples
  - Conflict resolution scenarios
  - Group dynamics management
- **Learning Support**
  - Differentiation strategies
  - Individual attention techniques
  - Progress monitoring methods

### 3. Teaching Scenarios (data/scenarios/)
- **Subject-Specific**
  - Mathematics teaching scenarios
  - Reading and writing activities
  - Science experiments
  - Social studies discussions
- **Classroom Management**
  - Transition periods
  - Group activities
  - Special events
- **Special Situations**
  - Learning difficulties
  - Behavioral challenges
  - Parent communication

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
│   ├── education_science/           # Educational foundations
│   │   ├── utah_core_standards/    # Utah 2nd grade standards
│   │   │   ├── mathematics/        # Math standards and objectives
│   │   │   ├── english_language_arts/ # ELA standards
│   │   │   ├── science/           # Science standards
│   │   │   └── social_studies/    # Social studies standards
│   │   │
│   │   ├── teaching_methods/      # Teaching methodologies
│   │   │   ├── stem_integration/  # STEM teaching approaches
│   │   │   ├── literacy_development/ # Reading/writing methods
│   │   │   └── differentiation/   # Learning adaptations
│   │   │
│   │   └── child_development/    # 7-8 year old development
│   │       ├── cognitive/        # Cognitive development
│   │       ├── social_emotional/ # Social-emotional growth
│   │       └── physical/         # Physical development
│   │
│   ├── interactions/             # Teacher-student interactions
│   │   ├── classroom_management/ # Management scenarios
│   │   │   ├── positive_reinforcement/ # Positive behavior examples
│   │   │   ├── conflict_resolution/ # Conflict handling
│   │   │   └── transitions/     # Activity transitions
│   │   │
│   │   ├── instructional_dialogs/ # Teaching conversations
│   │   │   ├── math_discussions/ # Math teaching dialogs
│   │   │   ├── reading_groups/   # Reading group interactions
│   │   │   └── science_experiments/ # Science lesson dialogs
│   │   │
│   │   └── support_strategies/  # Learning support
│   │       ├── struggling_learners/ # Support for challenges
│   │       ├── advanced_learners/ # Enrichment interactions
│   │       └── esl_support/     # Language support
│   │
│   └── scenarios/               # Teaching scenarios
│       ├── core_subjects/      # Subject-specific
│       │   ├── mathematics/    # Math teaching scenarios
│       │   ├── reading_writing/ # Literacy scenarios
│       │   ├── science/       # Science experiments
│       │   └── social_studies/ # Social studies activities
│       │
│       ├── classroom_situations/ # Management scenarios
│       │   ├── daily_routines/ # Regular procedures
│       │   ├── special_events/ # Special activities
│       │   └── challenges/    # Difficult situations
│       │
│       └── special_cases/     # Specific situations
│           ├── learning_support/ # Learning difficulties
│           ├── behavioral_support/ # Behavior management
│           └── parent_communication/ # Parent interactions
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
- **Primary Guide**: [`docs/product/README.md`](docs/product/README.md)
- **Key Responsibilities**:
  - Coordinate with Krista (UVU Education Chair) for content validation
  - Prioritize scenario development based on educational needs
  - Ensure alignment with Utah Core Standards
  - Manage feedback from education experts
- **Key Resources**:
  - [Expert Meeting Notes](docs/meetings/expert_reviews/)
  - [Approved Scenarios](data/scenarios/approved/)
  - [Content Validation Process](docs/validation/README.md)

### AI/ML Developer
- **Primary Guide**: [`docs/ai/README.md`](docs/ai/README.md)
- **Key Responsibilities**:
  - Implement AI models following educational guidelines
  - Consult with Krista on response appropriateness
  - Ensure age-appropriate language processing
  - Maintain educational accuracy in AI responses
- **Key Resources**:
  - [AI Implementation](src/ai/)
  - [Model Configurations](config/)
  - [Educational Guidelines](docs/education/guidelines.md)

### Frontend Developer
- **Primary Guide**: [`docs/frontend/README.md`](docs/frontend/README.md)
- **Key Responsibilities**:
  - Design teacher-friendly interfaces
  - Implement accessibility standards
  - Follow educational UX guidelines
  - Support content presentation requirements
- **Key Resources**:
  - [Frontend Code](src/frontend/)
  - [UI Guidelines](docs/frontend/ui_guidelines.md)
  - [Accessibility Standards](docs/frontend/accessibility.md)

### Project Manager
- **Primary Guide**: [`docs/project-management/README.md`](docs/project-management/README.md)
- **Key Responsibilities**:
  - Schedule reviews with Krista
  - Coordinate cross-team educational alignment
  - Track content validation progress
  - Manage educational feedback implementation
- **Key Resources**:
  - [Project Timeline](docs/project_pipeline.md)
  - [Meeting Notes](docs/meetings/)
  - [Educational Milestones](docs/milestones.md)

### QA Specialist
- **Primary Guide**: [`docs/qa/README.md`](docs/qa/README.md)
- **Key Responsibilities**:
  - Verify educational accuracy with Krista's guidance
  - Test age-appropriate interactions
  - Validate scenario authenticity
  - Ensure educational standard compliance
- **Key Resources**:
  - [Test Files](tests/)
  - [Educational Standards](docs/standards/)
  - [Validation Criteria](docs/qa/validation_criteria.md)

## 📋 Educational Content Specialist
### Dr. Krista Ruggles
**Associate Professor - Elementary Education STEM**
School of Education, Utah Valley University

#### Contact Information
- **Email**: kruggles@uvu.edu
- **Chat**: 10800383@uvu.edu
- **Phone**: 801-863-8057
- **Office**: ME-116B

#### Availability
- **Office Hours**: Schedule via email
- **Response Time**: 24-48 hours
- **Preferred Contact Method**: Email for initial contact

### Collaboration Guidelines
1. Schedule reviews through Project Manager
2. Submit content validation requests 48 hours in advance
3. Attend monthly cross-team alignment meetings
4. Follow educational feedback implementation process

### Review Schedule
- **Content Reviews**: Tuesdays and Thursdays
- **Team Meetings**: First Monday of each month
- **Emergency Reviews**: Contact via email with "URGENT" in subject

### Key Touchpoints
- Initial scenario validation
- Content accuracy review
- Age-appropriateness verification
- Educational standard alignment
- Response pattern validation
- Cross-team standardization

### Areas of Expertise
- Elementary Education STEM
- Second Grade Curriculum Development
- Teacher Training Methodologies
- Educational Technology Integration
- Student Assessment Strategies

## 🚀 Getting Started

### Windows Setup with WSL

1. **Install WSL**
   ```powershell
   # Open PowerShell as Administrator and run:
   wsl --install
   
   # After installation, restart your computer
   # WSL will finish Ubuntu setup on first launch
   ```

2. **Configure WSL**
   ```bash
   # Update package list
   sudo apt update && sudo apt upgrade
   
   # Install required system packages
   sudo apt install build-essential libpq-dev python3-dev
   ```

### Anaconda Environment Setup

1. **Install Anaconda in WSL**
   ```bash
   # Download Anaconda
   wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh
   
   # Install Anaconda
   bash Anaconda3-2024.02-1-Linux-x86_64.sh
   
   # Reload shell configuration
   source ~/.bashrc
   ```

2. **Create and Activate Environment**
   ```bash
   # Create new environment
   conda create -n utah-tta python=3.11
   
   # Activate environment
   conda activate utah-tta
   
   # Install required packages
   pip install -r requirements.txt
   ```

### Standard Environment Setup

1. **Environment Setup** (Alternative to Anaconda)
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
 