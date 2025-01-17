"""Database models."""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import VECTOR
from src.config import active_config

Base = declarative_base()

class Scenario(Base):
    """Training scenario model."""
    __tablename__ = 'scenarios'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    expected_response = Column(Text, nullable=False)
    expected_response_embedding = Column(VECTOR(active_config.VECTOR_DIMENSION))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    interactions = relationship("Interaction", back_populates="scenario")

class Interaction(Base):
    """Student-teacher interaction model."""
    __tablename__ = 'interactions'
    
    id = Column(Integer, primary_key=True)
    scenario_id = Column(Integer, ForeignKey('scenarios.id'), nullable=False)
    personality = Column(String(255))
    tone = Column(String(255))
    query = Column(Text, nullable=False)
    query_embedding = Column(VECTOR(active_config.VECTOR_DIMENSION))
    teacher_response = Column(Text)
    teacher_response_embedding = Column(VECTOR(active_config.VECTOR_DIMENSION))
    similarity_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    scenario = relationship("Scenario", back_populates="interactions")

class TeacherProfile(Base):
    """Teacher profile model."""
    __tablename__ = 'teacher_profiles'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    subject_area = Column(String(255))
    experience_level = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Add vector embedding for teacher's style/preferences
    style_embedding = Column(VECTOR(active_config.VECTOR_DIMENSION))

class FeedbackTemplate(Base):
    """Feedback template model."""
    __tablename__ = 'feedback_templates'
    
    id = Column(Integer, primary_key=True)
    category = Column(String(255), nullable=False)
    template_text = Column(Text, nullable=False)
    min_similarity = Column(Float)
    max_similarity = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Vector embedding for semantic matching
    template_embedding = Column(VECTOR(active_config.VECTOR_DIMENSION)) 