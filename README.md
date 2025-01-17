# Teacher Training Chatbot - Template Repository

## 🎯 Purpose of this Template

This repository serves as a comprehensive template for teams building an AI-powered teacher training chatbot. It provides a structured foundation with pre-defined roles, tasks, documentation templates, and development guidelines to help teams get started quickly and maintain consistent development practices.

## 📚 Repository Structure

```
teacher-training-chatbot/
├── src/                  # Source code directory
│   ├── database/        # Database models and operations
│   ├── ai/             # AI and LLM integration
│   ├── web/            # Web interface and API
│   └── config.py       # Configuration settings
├── docs/               # Comprehensive documentation
│   ├── architecture/   # System design and components
│   ├── api/           # API specifications
│   ├── database/      # Database guides
│   ├── ai/            # AI integration docs
│   ├── frontend/      # UI/UX guidelines
│   └── deployment/    # Deployment guides
├── templates/          # Role-specific templates
│   └── roles/         # Templates for each role
├── tests/             # Test suite
└── requirements.txt   # Project dependencies
```

## 🚀 Getting Started with this Template

1. **Fork the Repository**: Start by forking this template to your organization's GitHub account.

2. **Review Role Templates**: Check the `templates/roles/` directory for role-specific templates:
   - `database-developer.md`: For database setup and management
   - `ai-developer.md`: For AI model integration
   - `ui-developer.md`: For frontend development
   - Each template includes task checklists and progress tracking

3. **Setup Documentation**: The `docs/` directory contains comprehensive guides:
   - Start with `docs/README.md` for documentation overview
   - Each subdirectory contains role-specific technical documentation
   - Follow setup guides in `docs/getting-started.md`

4. **Project Structure**: Use the provided structure to organize your code:
   - `src/`: Main source code directory
   - `tests/`: Test files and test utilities
   - `docs/`: Project documentation
   - `templates/`: Progress tracking templates

## 👥 Role-Based Development

### For Product Owners
- Use `templates/roles/product-owner.md` to track requirements
- Review `docs/product-ownership/` for guidelines
- Manage stakeholder communication and product vision

### For Project Managers
- Use `templates/roles/project-manager.md` for task tracking
- Follow `docs/project-management/` for process guides
- Coordinate team activities and monitor progress

### For Database Developers
- Start with `templates/roles/database-developer.md`
- Follow setup guides in `docs/database/`
- Implement database models and vector search

### For AI Developers
- Use `templates/roles/ai-developer.md` for task tracking
- Check `docs/ai/` for implementation guides
- Integrate LLM models and develop response systems

### For UI/UX Developers
- Follow `templates/roles/ui-developer.md`
- Review `docs/frontend/` for guidelines
- Create responsive and accessible interfaces

## 📝 Using Templates

1. **Progress Tracking**:
   - Copy relevant role template from `templates/roles/`
   - Update progress in your copy
   - Commit updates regularly

2. **Documentation**:
   - Follow documentation structure in `docs/`
   - Update relevant sections as you develop
   - Keep documentation in sync with code

3. **Contributing**:
   - Follow Git workflow in `docs/contributing/git-workflow.md`
   - Use pull request template from `docs/contributing/templates/`
   - Review contribution guidelines

## 🛠️ Development Setup

1. **Clone your forked repository**
   ```bash
   git clone https://github.com/your-org/teacher-training-chatbot.git
   cd teacher-training-chatbot
   ```

2. **Set up Python environment**:

   Option A - Using Anaconda (Recommended):
   ```bash
   # Create a new conda environment
   conda create -n teacher-bot python=3.9
   
   # Activate the environment
   conda activate teacher-bot
   
   # Install key dependencies via conda
   conda install -c conda-forge ollama langchain-ollama transformers sentence-transformers pytorch flask python-dotenv
   
   # Install remaining dependencies via pip
   pip install -r requirements.txt
   ```

   Option B - Using pip venv:
   ```bash
   # Create a virtual environment
   python -m venv venv
   
   # Activate the environment
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Install and Setup Llama Model**:
   ```bash
   # Install Ollama (macOS/Linux)
   curl -fsSL https://ollama.com/install.sh | sh
   
   # Pull the Llama model
   ollama pull llama3.1
   
   # Start Ollama server
   ollama serve
   ```

4. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your configurations
   ```

5. **Initialize Educational Scenarios**:

   Create a Python script `src/ai/setup_scenarios.py`:
   ```python
   from langchain_ollama import OllamaLLM
   
   class TeacherTrainingBot:
       def __init__(self):
           self.llm = OllamaLLM(model="llama3.1")
           
           # Define elementary classroom scenarios
           self.scenarios = {
               "classroom_management": [
                   "Student disrupting class during lesson",
                   "Students not paying attention",
                   "Conflict between students"
               ],
               "learning_difficulties": [
                   "Student struggling with basic concepts",
                   "Student showing signs of learning disability",
                   "Student falling behind peers"
               ],
               "behavioral_issues": [
                   "Student showing aggressive behavior",
                   "Student refusing to participate",
                   "Student exhibiting anxiety"
               ]
           }
           
           # Define student personas
           self.student_personas = {
               "active": "Energetic student who struggles to stay focused",
               "shy": "Quiet student who rarely participates",
               "struggling": "Student having difficulty with subject matter",
               "disruptive": "Student who frequently interrupts class"
           }
   
       def evaluate_response(self, scenario, teacher_response):
           """
           Evaluate teacher's response to a given scenario
           """
           prompt = f"""
           As an educational expert, evaluate this teacher's response:
           
           Scenario: {scenario}
           Teacher's Response: {teacher_response}
           
           Evaluate based on:
           1. Professional appropriateness
           2. Educational effectiveness
           3. Student well-being consideration
           4. Classroom management impact
           
           Provide specific feedback and suggestions for improvement.
           """
           
           return self.llm.invoke(prompt)
   
       def generate_scenario(self, category, persona):
           """
           Generate a detailed classroom scenario
           """
           prompt = f"""
           Create a detailed elementary classroom scenario with:
           Category: {category}
           Student Persona: {self.student_personas[persona]}
           
           Include:
           1. Specific situation description
           2. Student's behavior
           3. Classroom context
           4. Immediate challenges
           5. Key considerations for teacher
           """
           
           return self.llm.invoke(prompt)
   ```

6. **Verify installation and run test scenario**:
   ```python
   # test_bot.py
   from src.ai.setup_scenarios import TeacherTrainingBot
   
   bot = TeacherTrainingBot()
   
   # Generate a test scenario
   scenario = bot.generate_scenario("classroom_management", "active")
   print("Generated Scenario:", scenario)
   
   # Test teacher response evaluation
   test_response = "I would calmly approach the student and quietly remind them of classroom expectations."
   feedback = bot.evaluate_response(scenario, test_response)
   print("\nFeedback:", feedback)
   ```

7. **Follow role-specific setup guides** in `docs/`

## 🤝 Best Practices

1. **Documentation**:
   - Keep documentation up to date
   - Follow the established directory structure
   - Include practical examples

2. **Code Organization**:
   - Follow the provided project structure
   - Use appropriate directories for different components
   - Maintain clean separation of concerns

3. **Collaboration**:
   - Use templates for consistency
   - Follow Git workflow guidelines
   - Regular progress updates

## 🆘 Need Help?

- Check `docs/faq.md` for common questions
- Review role-specific documentation
- Use issue templates for questions
- Contact team leads for clarification

## 📊 Progress Tracking

- Use GitHub Projects for task management
- Update role-specific templates regularly
- Track progress in sprint meetings
- Document decisions and changes 