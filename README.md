# Teacher Training Chatbot

A chatbot system that simulates student interactions for teacher training using smaller LLM models.

## Project Structure

```
teacher-bot/
├── src/
│   ├── database/      # Database models and operations
│   ├── ai/           # AI and LLM integration
│   ├── web/          # Web interface and API
│   ├── __init__.py
│   └── config.py
├── docs/            # Documentation
├── tests/           # Test suite
├── requirements.txt
└── README.md
```

## Features

- Student profile simulation with customizable scenarios and personalities
- Query generation using GPT or other LLM models
- Response evaluation using semantic similarity
- PostgreSQL database with vector support for efficient similarity search
- Web interface for interactions

## Development Roles

### Database Developer
- Implement and maintain PostgreSQL models with vector support
- Set up migrations and CRUD operations
- Optimize database performance
- Handle data validation and integrity

### AI Developer
- Implement LLM integration
- Fine-tune response generation
- Optimize model parameters
- Implement semantic similarity evaluation

### UX Developer
- Design and implement web interface
- Create responsive layouts
- Implement real-time feedback
- Ensure accessibility compliance

### Research and Documentation
- Document APIs and features
- Create user guides
- Research best practices
- Maintain project documentation

## Prerequisites

1. Install PostgreSQL and pgvector:
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo apt install postgresql-server-dev-all

# Install pgvector extension
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install
```

2. Create PostgreSQL databases:
```bash
sudo -u postgres psql

# In PostgreSQL shell:
CREATE DATABASE chatbot;
CREATE DATABASE chatbot_test;
\c chatbot
CREATE EXTENSION vector;
\c chatbot_test
CREATE EXTENSION vector;
\q
```

## Setup

1. Create a conda environment:
```bash
conda create -n teacher-bot python=3.10
conda activate teacher-bot
```

2. Install dependencies:
```bash
conda install -c conda-forge openai transformers sentence-transformers pytorch flask python-dotenv sqlalchemy
pip install psycopg2-binary alembic pgvector sqlalchemy-utils
```

3. Create a `.env` file:
```bash
echo "OPENAI_API_KEY=your-api-key-here" > .env
echo "FLASK_ENV=development" >> .env
echo "SECRET_KEY=your-secret-key" >> .env
echo "DATABASE_URL=postgresql://postgres:postgres@localhost:5432/chatbot" >> .env
```

4. Initialize the database:
```bash
python -m src.database.init_db
```

5. Run database migrations:
```bash
cd src/database/migrations
alembic upgrade head
```

6. Run the application:
```bash
python -m src.web.app
```

## Testing

Run the test suite:
```bash
pytest tests/
```

## Contributing

1. Choose your role (Database/AI/UX/Documentation)
2. Create a feature branch
3. Implement changes
4. Write tests
5. Submit a pull request

## License

MIT License 
4. Review feedback and similarity scores 