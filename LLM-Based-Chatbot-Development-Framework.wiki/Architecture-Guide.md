# Architecture Guide

## Overview

UTTA (Universal Teacher Training Assistant) is a Streamlit-based educational application that leverages Large Language Models to provide interactive teacher training experiences.

### Core Flow

```mermaid
sequenceDiagram
    participant T as Teacher
    participant UI as Streamlit UI
    participant LLM as LLM Engine
    participant DB as ChromaDB
    
    T->>UI: Start Training Session
    UI->>DB: Load Teaching Materials
    DB-->>UI: Return Content
    UI->>LLM: Generate Scenario
    LLM-->>UI: Return Response
    UI-->>T: Present Scenario

    T->>UI: Submit Response
    UI->>LLM: Evaluate Response
    LLM-->>UI: Provide Feedback
    UI-->>T: Display Feedback
```

## Components

### 1. User Interface (Streamlit)
- Interactive training scenarios
- Real-time feedback display
- Progress tracking
- Session management

### 2. LLM Integration
- OpenAI API for scenario generation
- DSPy for response optimization
- Context-aware interactions

### 3. Knowledge Base
- ChromaDB for vector storage
- Educational content management
- Teaching strategy retrieval

## Data Flow

1. **Training Session**
   - Teacher initiates session
   - System loads relevant materials
   - LLM generates teaching scenario
   - Teacher receives interactive content

2. **Response Handling**
   - Teacher submits response
   - System evaluates using LLM
   - Feedback generated and displayed
   - Progress tracked and stored

## Technical Stack

- **Frontend**: Streamlit
- **LLM**: OpenAI + DSPy
- **Database**: ChromaDB
- **Storage**: Local filesystem
- **Caching**: Streamlit cache

## Security

- Environment-based API key management
- Secure session handling
- Input validation
- Rate limiting

## Performance

- Streamlit caching
- Optimized API calls
- Efficient resource management
- Error logging and monitoring 