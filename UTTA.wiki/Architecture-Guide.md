# UTTA System Architecture

## 🏗️ System Overview

UTTA is built as a Streamlit application that integrates LLMs with educational content to provide an interactive teacher training experience.

### Interaction Flow

```mermaid
sequenceDiagram
    autonumber
    participant T as Teacher
    participant ST as Streamlit UI
    participant LLM as LLM Engine
    participant DB as ChromaDB
    participant Cache as Streamlit Cache
    
    rect rgb(240, 248, 255)
        Note over T,Cache: Teaching Scenario Generation
        
        T->>ST: Request Teaching Scenario
        activate ST
        
        ST->>DB: Fetch Educational Content
        activate DB
        DB-->>ST: Return Context Documents
        deactivate DB
        
        ST->>LLM: Generate Scenario
        activate LLM
        LLM-->>ST: Return Scenario
        deactivate LLM
        
        ST->>Cache: Store Scenario State
        activate Cache
        Cache-->>ST: Confirm Storage
        deactivate Cache
        
        ST-->>T: Display Scenario
        deactivate ST
    end
    
    rect rgb(230, 255, 240)
        Note over T,Cache: Response Evaluation
        
        T->>ST: Submit Teaching Response
        activate ST
        
        ST->>Cache: Retrieve Scenario Context
        activate Cache
        Cache-->>ST: Return Context
        deactivate Cache
        
        ST->>DB: Get Teaching Strategies
        activate DB
        DB-->>ST: Return Relevant Strategies
        deactivate DB
        
        ST->>LLM: Evaluate Response
        activate LLM
        LLM-->>ST: Return Feedback
        deactivate LLM
        
        ST-->>T: Display Feedback
        deactivate ST
    end
    
    rect rgb(255, 240, 245)
        Note over T,Cache: Progress Review
        
        T->>ST: View Progress
        activate ST
        
        ST->>Cache: Fetch Session History
        activate Cache
        Cache-->>ST: Return History
        deactivate Cache
        
        ST->>LLM: Generate Progress Analysis
        activate LLM
        LLM-->>ST: Return Analysis
        deactivate LLM
        
        ST-->>T: Show Progress Report
        deactivate ST
    end
```

### Component Architecture

```mermaid
graph TB
    UI[Streamlit Interface] --> Core[Core Engine]
    Core --> Knowledge[Knowledge Base]
    Core --> LLM[LLM Service]
    
    subgraph Core
        State[Session State]
        Process[Text Processing]
        Context[Context Management]
    end

    subgraph Knowledge
        ChromaDB[Vector Store]
        Files[Document Store]
        Cache[Streamlit Cache]
    end

    subgraph LLM
        OpenAI[OpenAI API]
        SBERT[Sentence Transformers]
    end

    style UI fill:#f0f7ff,stroke:#1976d2
    style Core fill:#e8f5e9,stroke:#2e7d32
    style Knowledge fill:#fff3e0,stroke:#ef6c00
    style LLM fill:#f3e5f5,stroke:#6a1b9a
```

### Core Components

1. **Streamlit Interface**
   - Interactive web interface
   - Real-time response display
   - Session state management
   - File upload handling

2. **LLM Integration**
   - OpenAI API integration
   - Context management
   - Prompt optimization
   - Response generation

3. **Knowledge Management**
   - Vector embeddings storage
   - Semantic search functionality
   - Educational content processing
   - Context retrieval

## 🔌 Integration Points

### Component Communication
- Streamlit session state for data persistence
- Callback functions for user interactions
- State management through Streamlit primitives
- File handling through Streamlit's file uploader

### Data Storage
- ChromaDB for vector embeddings
- Local file system for document storage
- Streamlit cache for performance optimization

### External Services
- OpenAI API for LLM capabilities
- Sentence transformers for embeddings
- HuggingFace for model access

## 🔐 Security Considerations

1. **API Security**
   - Secure API key management
   - Environment variable configuration
   - Rate limiting implementation

2. **Data Protection**
   - Local data storage security
   - User input validation
   - Session data management

## 📈 Performance

1. **Optimization**
   - Streamlit caching mechanisms
   - Batch processing for embeddings
   - Efficient context window usage

2. **Resource Management**
   - Memory-efficient operations
   - Optimized API calls
   - Cache management

3. **Monitoring**
   - Error logging
   - Performance tracking
   - Usage analytics 