# Enhanced Teacher Training System

## Overview

This enhanced teacher training system is designed to provide realistic classroom management scenarios for elementary education teachers. The system features a sophisticated knowledge pipeline that processes educational books, extracts relevant knowledge, and uses this information to generate evidence-based teaching scenarios and feedback.

## Key Features

- **Educational Book Processing**: The system processes PDF, EPUB, TXT, and JSON files containing educational content.
- **Vector Knowledge Base**: Content is stored in a SQLite vector database with 17,980 knowledge chunks for semantic retrieval.
- **Enhanced Scenarios**: Teaching scenarios are generated using real educational knowledge.
- **Performance Tracking**: The system tracks which knowledge is most effective in different situations.
- **Evidence-Based Feedback**: Teacher responses are evaluated using educational best practices.
- **Knowledge Analysis**: Tools to analyze and visualize the content of the knowledge base.

## Why Educational Books?

Educational research literature offers several advantages as a knowledge source:

1. **Scientific Validity**: Educational books contain strategies and approaches backed by research and evidence.
2. **Domain Expertise**: Books written by education experts contain specialized knowledge refined through years of classroom experience.
3. **Structured Knowledge**: Educational texts often organize information in ways that align with teaching practices.
4. **Diverse Perspectives**: By incorporating multiple books, the system can present varied approaches to classroom challenges.
5. **Contextual Understanding**: Educational literature provides the context in which teaching strategies are most effective.

## How the Application Works

The application follows a comprehensive pipeline to transform educational literature into practical teaching scenarios:

### 1. Book Knowledge Extraction

The system extracts knowledge from educational books through several steps:

- **Text Extraction**: The `DocumentProcessor` class extracts raw text from various file formats (PDF, EPUB, TXT).
- **Chunking**: Text is divided into semantically meaningful chunks (paragraphs or sections of ~300-500 words).
- **Categorization**: Each chunk is automatically categorized into domains like "classroom management," "teaching strategies," etc.
- **Vectorization**: Text chunks are converted to numerical vector embeddings (384 dimensions) that capture semantic meaning.
- **Storage**: Processed knowledge is stored in a SQLite vector database for efficient retrieval.

You can test this process with any educational book using:
```
python test_book_extraction.py --book knowledge_base/books/your_book.pdf
```

### 2. Knowledge Retrieval

When generating scenarios or responding to queries:

- **Semantic Search**: The system searches for the most relevant knowledge chunks based on semantic similarity.
- **Context-Aware Results**: Results are filtered and ranked by relevance, category, and usage statistics.
- **Knowledge Integration**: Retrieved chunks provide evidence-based context for scenario generation.

### 3. Scenario Generation

The system generates realistic classroom scenarios using:

- **Knowledge-Grounded Creation**: The `ClassroomScenarioGenerator` class creates scenarios based on educational research.
- **Student Profiles**: Realistic student profiles provide context for the scenarios.
- **Subject-Specific Knowledge**: Different subject areas (math, reading, etc.) draw from specialized knowledge.
- **Evidence-Based Strategies**: Recommended approaches are grounded in educational research.

You can view demonstration scenarios with:
```
python demonstrate_book_knowledge.py --scenarios 3 --subject math
```

### 4. User Interaction

Users interact with the system through a terminal interface where they can:

- Respond to classroom management scenarios
- Receive evidence-based feedback on their responses
- Generate new scenarios for different subjects/grade levels
- View the knowledge sources behind recommendations

### 5. Performance Analytics

The system continuously improves through:

- **Usage Tracking**: Monitoring which knowledge chunks are retrieved most often
- **Effectiveness Scoring**: Rating how helpful each knowledge chunk is in practice
- **Knowledge Refinement**: Adjusting retrieval parameters based on performance data

## System Architecture

The Book-Based Knowledge System consists of the following components:

### 1. Document Processor (`document_processor.py`)

The Document Processor handles the extraction and processing of text content from educational books:

- **Text Extraction**: Extracts raw text from PDF, EPUB, and TXT files.
- **Content Chunking**: Divides raw text into semantically meaningful chunks of appropriate size.
- **Categorization**: Assigns categories to each chunk (e.g., "classroom management", "teaching strategies").
- **Vectorization**: Converts text chunks into numerical vector representations for efficient retrieval.

### 2. Vector Database (`vector_database.py`)

The Vector Database stores and retrieves knowledge chunks:

- **Efficient Storage**: Organizes vector representations for quick similarity search.
- **Semantic Search**: Enables retrieval of knowledge based on meaning rather than exact keyword matching.
- **Metadata Management**: Preserves source information, categories, and other metadata associated with knowledge chunks.
- **Statistical Analysis**: Provides insights into the knowledge base composition.

### 3. Scenario Generator (`scenario_generator.py`)

The Scenario Generator creates realistic classroom scenarios based on educational knowledge:

- **Knowledge Retrieval**: Queries the vector database for relevant educational content.
- **Scenario Construction**: Builds contextually appropriate classroom situations.
- **Strategy Integration**: Incorporates evidence-based teaching strategies from the knowledge base.
- **Educational Grounding**: Ensures scenarios are rooted in sound educational principles.

## How Knowledge Flows Through the System

1. **Knowledge Acquisition**: 
   - Educational books in PDF/EPUB/TXT format are placed in `knowledge_base/books/`.
   - The Document Processor extracts text content from these books.

2. **Knowledge Processing**:
   - Text is divided into meaningful chunks (paragraphs or sections).
   - Each chunk is categorized based on its content.
   - Chunks are converted to vector embeddings that capture their semantic meaning.

3. **Knowledge Storage**:
   - Processed chunks with their vectors and metadata are stored in the vector database.
   - The database organizes chunks by category and source for efficient retrieval.

4. **Knowledge Retrieval**:
   - When generating a scenario, the system queries the vector database with relevant criteria.
   - The database returns the most semantically similar chunks to the query.

5. **Knowledge Application**:
   - Retrieved knowledge is incorporated into classroom scenarios.
   - Teaching strategies from the literature are presented as recommended actions.
   - The system maintains attribution to the original source material.

## Knowledge Pipeline

The enhanced system follows these steps:

1. **Document Processing**: Educational books are processed and broken into chunks.
2. **Knowledge Categorization**: Content is categorized into:
   - Classroom Management: 5,829 chunks
   - Teaching Strategies: 2,466 chunks
   - Student Development: 2,028 chunks
   - General Education: 7,657 chunks
3. **Vector Storage**: Text chunks are converted to 384-dimensional vector embeddings for semantic search.
4. **Scenario Generation**: Relevant knowledge is retrieved to create realistic scenarios.
5. **Performance Analysis**: The system tracks which knowledge chunks are most effective.

## Configuration and Customization

The system is highly configurable through the `book_knowledge_config.py` file, which allows you to adjust:

- **Document Processing Parameters**: Chunk size, overlap, preprocessing options
- **Vector Database Settings**: Embedding model, database path, memory usage
- **Categorization Options**: Category definitions, keyword associations
- **Search Configuration**: Relevance thresholds, category boosting
- **Scenario Generation Parameters**: Knowledge diversity, reference inclusion

To customize the application for specific needs, modify these parameters to emphasize different types of educational knowledge or adjust the processing approach.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Required packages (install with `pip install -r requirements.txt`):
  - sentence-transformers
  - PyPDF2
  - EbookLib
  - SQLite3

### Basic Setup

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `knowledge_base/books` directory
4. Add educational books (PDF, EPUB, TXT) to the books directory
5. Build the knowledge base:
   ```
   python process_educational_books.py --clean
   ```
6. Run the enhanced system:
   ```
   python run_enhanced.py
   ```

### Testing the Book Knowledge System

To verify the book knowledge extraction system:

```
# Test extraction from a single book
python test_book_extraction.py --book knowledge_base/books/your_book.pdf

# Test the generated scenarios
python demonstrate_book_knowledge.py --scenarios 5 --subject reading

# Analyze the knowledge base
python knowledge_analyzer.py --db-path ./knowledge_base/vector_db.sqlite
```

### Large File Handling

The repository excludes large files from Git tracking:
- `knowledge_base/vector_db.sqlite` (58MB): Generated by running `process_educational_books.py`
- `knowledge_base_backup/`: Contains backup files including large educational textbooks

If you need the educational textbooks, please contact the repository maintainer for access.

### Usage

The system provides an interactive terminal interface where you can:

1. Respond to classroom management scenarios
2. View history of your interactions
3. Generate new scenarios
4. View knowledge performance insights
5. End the session

## Directory Structure

```
.
├── ai_agent.py                     # Main agent module
├── document_processor.py           # Educational book processing
├── vector_database.py              # Knowledge storage and retrieval
├── scenario_generator.py           # Enhanced scenario generation
├── book_knowledge_config.py        # Configuration for knowledge system
├── process_educational_books.py    # Script to process books into knowledge
├── test_book_extraction.py         # Tool to test knowledge extraction
├── demonstrate_book_knowledge.py   # Demo script for book-based scenarios
├── knowledge_analyzer.py           # Knowledge base analysis tools
├── knowledge_base/                 # Knowledge storage
│   ├── books/                      # Directory for educational books
│   ├── default_strategies/         # Default strategy files (CSV, TXT)
│   │   ├── behavior_management.csv # Behavior management strategies
│   │   ├── math_strategies.csv     # Math teaching strategies
│   │   ├── reading_strategies.csv  # Reading teaching strategies
│   │   └── teaching_strategies.txt # General teaching strategies
│   ├── vector_db.sqlite            # Main vector database (not in Git)
│   └── student_profiles.json       # Realistic student profiles
├── requirements.txt                # Required dependencies
└── run_enhanced.py                 # Script to run enhanced agent
```

## Knowledge Analysis

The system includes a knowledge analyzer tool to understand what information has been extracted from educational books:

```
python knowledge_analyzer.py --db-path ./knowledge_base/vector_db.sqlite --format html --visualize
```

This tool provides:
- Distribution analysis of knowledge categories
- Content analysis of key terms and concepts
- Effectiveness tracking of knowledge chunks
- Visualizations of the knowledge base
- Topic modeling to discover latent knowledge themes

## LLM Integration

The system integrates the knowledge base with large language models to create realistic classroom scenarios:

1. **Knowledge Retrieval**: When generating a scenario, the system searches for relevant knowledge chunks
2. **Context Augmentation**: The LLM receives the retrieved knowledge as context
3. **Knowledge-Grounded Generation**: Scenarios and evaluations are grounded in educational research
4. **Source Attribution**: All generated content includes references to the source material

## How Knowledge Improves Performance

Different types of knowledge improve the system's performance in various ways:

1. **Classroom Management Knowledge**: Improves the realism of behavioral scenarios and intervention strategies.
2. **Teaching Strategies Knowledge**: Enhances the quality of recommended approaches and alternative suggestions.
3. **Student Development Knowledge**: Provides age-appropriate context for student behaviors and needs.
4. **General Education Knowledge**: Offers broader pedagogical principles that apply across situations.

The system tracks effectiveness scores for each knowledge chunk to identify which sources provide the most valuable information for different scenarios.

## Understanding Knowledge Chunks and Parameters

### What are Knowledge Chunks?

Knowledge Chunks are small, self-contained units of educational information extracted from textbooks, research papers, and pedagogical materials. The system contains 17,980 knowledge chunks in total, distributed across four categories:

- **Classroom Management**: 5,829 chunks covering behavior management, discipline approaches, and classroom procedures
- **Teaching Strategies**: 2,466 chunks about instructional methods, lesson design, and pedagogical approaches
- **Student Development**: 2,028 chunks covering child development, learning styles, and psychology
- **General Education**: 7,657 chunks containing broader educational principles and theories

### How Knowledge Chunks are Created

1. **Text Extraction**: The system extracts raw text from various document formats (PDF, EPUB, TXT)
2. **Chunking Process**: Long documents are divided into smaller segments (chunks) of approximately 500 words each
3. **Overlap Mechanism**: Adjacent chunks have a small overlap (about 50 words) to preserve context across boundaries
4. **Metadata Attachment**: Each chunk retains information about its source, including:
   - Document title
   - Author
   - Section
   - Position in original text

### How Knowledge Chunks are Used

1. **Vector Conversion**: Each chunk is converted into a 384-dimensional vector embedding that captures its semantic meaning
2. **Semantic Search**: When generating scenarios, the system retrieves the most relevant chunks based on similarity
3. **Context Enhancement**: Selected chunks provide educational research context to the LLM
4. **Performance Tracking**: The system monitors which chunks are most useful through:
   - **Usage Count**: How often a chunk is retrieved
   - **Effectiveness Score**: How well the chunk improves generated content (0.0-1.0)

### Knowledge Chunk Storage

All chunks are stored in the SQLite database (`vector_db.sqlite`) with their:
- Full text content
- Source metadata as JSON
- Category classification
- Usage statistics
- Vector embeddings for quick semantic retrieval

This chunk-based approach allows the system to:
- Quickly find relevant educational knowledge
- Provide evidence-based context to generated scenarios
- Track which sources and content types are most valuable
- Continuously improve through usage analytics

### Key Parameters

The system uses several important parameters to process and utilize knowledge chunks:

#### 1. Chunking Parameters

- **Chunk Size**: Controls how many words (default: 500) are included in each knowledge segment
- **Chunk Overlap**: Determines how many words (default: 50) overlap between consecutive chunks to preserve context
- **Category Assignment**: Each chunk is classified into one of four categories (classroom management, teaching strategies, student development, or general education)

#### 2. Vector Embedding Parameters

- **Embedding Dimension**: Each chunk is converted to a 384-dimensional vector
- **Embedding Model**: Uses sentence-transformers with the all-MiniLM-L6-v2 model
- **Normalization**: Vectors are normalized for cosine similarity search

#### 3. Retrieval Parameters

- **Top-K**: Number of most relevant chunks to retrieve (default: 5) when generating scenarios
- **Similarity Threshold**: Minimum similarity score (usually 0.6+) for considering a chunk relevant
- **Category Filtering**: Option to filter retrieval by specific knowledge categories

#### 4. Performance Tracking Parameters

- **Usage Count**: Tracks how many times each chunk has been used in scenarios
- **Effectiveness Score**: A 0.0-1.0 metric indicating how helpful the knowledge has been in teacher evaluations
- **Feedback Loop**: The system adjusts retrieval preferences based on which chunks prove most effective

### Database Parameters

The knowledge chunks are stored in a SQLite database with the following structure:

```
knowledge_base/vector_db.sqlite:
- chunks table: Stores text content and metadata
- embeddings table: Stores the vector representation (384 dimensions) of each chunk
```

Optimized indices on category, usage count, and effectiveness score enable efficient querying and retrieval.

## Best Practices for Optimal Results

1. **Diverse Literature**: Include books covering different educational philosophies and approaches.
2. **Quality Sources**: Prioritize peer-reviewed, well-respected educational publications.
3. **Up-to-date Content**: Regularly update the knowledge base with current educational research.
4. **Domain Coverage**: Ensure coverage across all relevant educational domains and grade levels.
5. **Clean Text**: Pre-process books to remove indices, references, and other non-content sections.

## Fallback Strategies

In addition to book-based knowledge, the system includes default strategy files:

- **behavior_management.csv**: Default behavior management strategies
- **math_strategies.csv**: Subject-specific math teaching strategies
- **reading_strategies.csv**: Subject-specific reading teaching strategies
- **teaching_strategies.txt**: General teaching principles and approaches

These files provide a reliable foundation when book-based knowledge is unavailable or insufficient for a specific scenario.

## Contributing

To improve the knowledge base:

1. Add more educational books to the `knowledge_base/books` directory
2. Focus on books covering classroom management, teaching strategies, and child development
3. Ensure content is specifically relevant to elementary education
4. Run the knowledge analyzer to assess the impact of new content

## License

[Specify the license information here]