# Utah Teacher Training Assistant (UTTA)

A sophisticated chatbot framework for teacher training, leveraging advanced LLM capabilities with DSPy integration, knowledge base management, and automated evaluation metrics.

## Features

*   **Advanced LLM Integration:** Utilizes powerful language models for realistic teacher-student interaction simulation.
*   **DSPy Framework:** Leverages DSPy for optimizing language model prompts and logic, potentially improving interaction quality and goal alignment.
*   **Knowledge Base Management:** Incorporates a knowledge base, likely using Retrieval-Augmented Generation (RAG) with vector databases (`src/core/vector_database.py`, `src/core/document_processor.py`), to provide contextually relevant information.
*   **Automated Evaluation:** Includes mechanisms for evaluating the performance of the simulated teacher or student interactions (details likely in `src/evaluation` or `examples`).
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
python src/web/app.py
```

Navigate to the URL provided by the application (usually `http://127.0.0.1:5000` or similar) in your web browser.

Refer to the `examples/` directory for other ways to interact with the framework or run specific tasks like evaluation.

## Evaluation

The framework includes capabilities for evaluating interactions. Check the `src/evaluation/` directory and potentially scripts within `examples/` for details on how to run and interpret evaluation results.

## Contributing

(Optional: Add guidelines for contributions if this is an open project).

## License

(Optional: Specify the license, e.g., This project is licensed under the MIT License - see the LICENSE file for details). 