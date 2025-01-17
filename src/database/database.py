from sqlalchemy import create_engine, Column, Integer, String, Float, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class Interaction(Base):
    __tablename__ = 'interactions'
    
    id = Column(Integer, primary_key=True)
    scenario = Column(String(255))
    personality = Column(String(255))
    query = Column(Text)
    teacher_response = Column(Text)
    expected_response = Column(Text)
    similarity_score = Column(Float)

class Scenario(Base):
    __tablename__ = 'scenarios'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255))
    description = Column(Text)
    expected_response = Column(Text)

# Create database engine
engine = create_engine('sqlite:///chatbot.db')
Base.metadata.create_all(engine)

# Create session factory
Session = sessionmaker(bind=engine)

def log_interaction(scenario, personality, query, teacher_response, expected_response, similarity_score):
    """Log an interaction to the database."""
    session = Session()
    interaction = Interaction(
        scenario=scenario,
        personality=personality,
        query=query,
        teacher_response=teacher_response,
        expected_response=expected_response,
        similarity_score=similarity_score
    )
    session.add(interaction)
    session.commit()
    session.close()

def get_scenarios():
    """Get all available scenarios."""
    session = Session()
    scenarios = session.query(Scenario).all()
    session.close()
    return scenarios

def add_scenario(name, description, expected_response):
    """Add a new scenario to the database."""
    session = Session()
    scenario = Scenario(
        name=name,
        description=description,
        expected_response=expected_response
    )
    session.add(scenario)
    session.commit()
    session.close() 