# Teacher Training Chatbot

A comprehensive chatbot system for teacher training, using LLMs to simulate classroom scenarios and provide feedback on teaching responses.

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
- **Primary Location**: `docs/roles/product_owner.md`
- **Key Files**:
  - `docs/meetings/expert_reviews/` - Expert meeting notes
  - `data/scenarios/approved/` - Approved scenarios
  - `docs/scenarios/creation_guide.md` - Scenario creation process

### Educational Content Specialist
- **Primary Location**: `docs/roles/content_specialist.md`
- **Key Files**:
  - `data/scenarios/templates/` - Scenario templates
  - `data/personas/templates/` - Student personas
  - `data/evaluation/criteria/` - Evaluation criteria

### AI/ML Developer
- **Primary Location**: `docs/roles/ai_developer.md`
- **Key Files**:
  - `src/ai/` - AI implementation
  - `config/` - Model configurations
  - `tests/ai/` - AI component tests

### Frontend Developer
- **Primary Location**: `docs/roles/frontend_developer.md`
- **Key Files**:
  - `src/frontend/` - Frontend code
  - `src/api/` - API endpoints
  - `tests/frontend/` - Frontend tests

### Project Manager
- **Primary Location**: `docs/roles/project_manager.md`
- **Key Files**:
  - `docs/project_pipeline.md` - Project timeline
  - `docs/meetings/` - Meeting notes
  - `docs/repository_structure.md` - Repo structure

### QA Specialist
- **Primary Location**: `docs/roles/qa_specialist.md`
- **Key Files**:
  - `tests/` - All test files
  - `data/scenarios/approved/` - Test scenarios
  - `docs/technical/` - Technical documentation

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
   - Draft in `data/scenarios/drafts/`
   - Get expert review
   - Move to `data/scenarios/approved/`

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
   - Embedding generation
   - Vector similarity search
   - Context-aware responses

2. **Scenario Management**
   - Expert validation
   - Version control
   - Performance tracking

3. **Evaluation System**
   - Response scoring
   - Feedback generation
   - Progress tracking

## 📚 Documentation

- **Technical Guides**: `docs/technical/`
- **API Documentation**: `docs/technical/api/`
- **Database Schema**: `docs/technical/database/`
- **Architecture**: `docs/technical/architecture/`

## 🤝 Contributing

1. Read `docs/contributing/guidelines.md`
2. Follow role-specific guides
3. Use templates from `data/templates/`
4. Ensure test coverage
5. Update documentation

## 🔧 Configuration

- **Development**: `config/development.py`
- **Production**: `config/production.py`
- **Testing**: `config/testing.py`

## 🆘 Getting Help

1. Check role-specific documentation
2. Review technical guides
3. Contact team lead
4. Create issue in repository

## 📊 Monitoring

- Application logs in `logs/`
- Metrics at `:8001/metrics`
- Performance dashboards
- Error tracking

## 🔐 Security

- SSL/TLS in production
- Rate limiting
- Input validation
- Access control

## 📅 Regular Maintenance

1. Database backups
2. Log rotation
3. Performance monitoring
4. Security updates
 