# Utah Teacher Training Assistant (UTTA)

A sophisticated chatbot framework for teacher training, leveraging advanced LLM capabilities with DSPy integration, knowledge base management, and automated evaluation metrics.

## Features

*   **Advanced LLM Integration:** Utilizes powerful language models for realistic teacher-student interaction simulation.
*   **DSPy Framework:** Leverages DSPy for optimizing language model prompts and logic, potentially improving interaction quality and goal alignment.
*   **Knowledge Base Management:** Incorporates a knowledge base, likely using Retrieval-Augmented Generation (RAG) with vector databases (`src/core/vector_database.py`, `src/core/document_processor.py`), to provide contextually relevant information.
*   **Automated Evaluation:** Includes mechanisms for evaluating the performance of the simulated teacher or student interactions using a comprehensive LLM-based feedback system that assesses teaching responses across multiple dimensions.
*   **Web Interface:** Provides a web application (`src/web/app.py`) for interacting with the chatbot.
*   **Configurable:** Settings managed through environment variables (see `.env.example`) and `config.py`.

## Project Structure

```
├── src/                      # Main source code
│   ├── core/                 # Core backend logic (document processing, vector DB)
│   ├── evaluation/           # Evaluation scripts and modules
│   ├── llm/                  # Language model interaction logic
│   ├── retrieval/            # Retrieval specific logic (likely RAG)
│   ├── utils/                # Utility functions
│   └── web/                  # Web application (Flask/FastAPI) and static files
├── knowledge_base/           # Directory for storing knowledge base documents
├── tests/                    # Unit and integration tests
├── examples/                 # Example scripts demonstrating usage
├── tools/                    # Utility tools or scripts
├── data/                     # Data files used by the application
├── docs/                     # Documentation files
├── .env.example              # Example environment variable file
├── config.py                 # Configuration settings
├── requirements.txt          # Python package dependencies
└── README.md                 # This file
```

## Getting Started

### Prerequisites

*   Python (version specified in requirements or assumed >= 3.8)
*   `pip` for package installation
*   A virtual environment tool (like `venv` or `conda`) is recommended.
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows use `venv\Scripts\activate`
    ```
    *Or using conda:*
    ```bash
    conda create -n utta python=3.9 # Or desired version
    conda activate utta
    ```
*   Required environment variables (see Configuration section).

### Installation

1.  **Clone the repository (if you haven't already):**
    ```bash
    git clone <repository-url>
    cd llm-chatbot-framework
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Configuration

1.  Copy the example environment file:
    ```bash
    cp .env.example .env
    ```
2.  Edit the `.env` file and fill in the required values, such as API keys for language models, database connection details, etc. Refer to `config.py` for details on how these variables are used.

## Usage

To run the web application:

```bash
# Set the Python path to include the src directory
PYTHONPATH=$PYTHONPATH:$(pwd)/src streamlit run src/web/app.py
```

Navigate to the URL provided by the application (usually `http://localhost:8501`) in your web browser.

## Evaluation Component

The UTTA includes a powerful evaluation system for assessing teaching responses. This feature helps teachers improve their instructional approaches by providing detailed feedback and metrics.

### How the Evaluation Works

1. **LLM-Based Analysis**: The system uses the integrated Language Model to analyze teaching responses within the context of the student profile and scenario.

2. **Multi-dimensional Assessment**: Each teaching response is evaluated across six key pedagogical dimensions:
   - **Clarity** (1-10): How clearly concepts are explained
   - **Engagement** (1-10): How effectively the response engages the student
   - **Pedagogical Approach** (1-10): Appropriateness of teaching strategies
   - **Emotional Support** (1-10): Degree of emotional encouragement and support
   - **Content Accuracy** (1-10): Accuracy of the subject matter
   - **Age Appropriateness** (1-10): Whether language and concepts match student's age/level

3. **Comprehensive Feedback**: For each evaluation, the system provides:
   - An overall effectiveness score
   - Specific strengths identified in the teaching approach
   - Areas for improvement
   - Actionable recommendations for enhancing teaching effectiveness

4. **Visual Representation**: Results are displayed as numerical scores and through an interactive bar chart for easy interpretation.

### Using the Evaluation Feature

1. Navigate to the application in your browser (http://localhost:8501)
2. Have a conversation with the simulated student in the Chat section
3. Switch to the "Evaluation" section using the sidebar navigation
4. Click the "Evaluate Last Response" button
5. Review the comprehensive feedback provided by the system

### Benefits of the Evaluation Component

- **Immediate Feedback**: Teachers receive instant feedback on their instructional approaches
- **Targeted Improvement**: Specific recommendations help focus professional development
- **Objective Assessment**: Standardized evaluation criteria ensure consistent feedback
- **Private Practice**: Teachers can experiment with different approaches in a safe environment

## Contributing

(Optional: Add guidelines for contributions if this is an open project).

## License

(Optional: Specify the license, e.g., This project is licensed under the MIT License - see the LICENSE file for details). 