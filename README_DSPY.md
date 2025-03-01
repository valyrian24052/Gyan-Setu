# Teacher Training Simulator with DSPy

This is an enhanced version of the Teacher Training Simulator that uses DSPy instead of LangChain for LLM interactions. DSPy provides several advantages over LangChain, including better prompt optimization, more reliable results, and improved error handling.

## Features

- **DSPy-powered LLM Interface**: Improved interaction with language models including Llama 3, GPT models, and Claude.
- **Enhanced Teaching Analysis**: Get detailed feedback on your teaching approach.
- **Student Simulation**: Practice with realistic student responses tailored to the scenario.
- **Scenario Creation**: Generate custom teaching scenarios based on subject, grade level, and learning styles.
- **Resource Recommendations**: Get tailored teaching resources for your scenarios.

## Installation

1. Ensure you have Python 3.10+ installed
2. Clone the repository
3. Install dependencies:

```bash
pip install -r requirements.txt
pip install dspy-ai torch
```

## Configuration

Before running the application, you need to set up your language model access:

### For OpenAI Models (GPT-4o-mini, GPT-3.5, GPT-4)
- Create a `.env` file in the project root
- Add your OpenAI API key:
  ```
  OPENAI_API_KEY=your_api_key_here
  ```
- The application defaults to the more cost-effective GPT-4o-mini model, which provides good performance at a lower price point than GPT-4.

### For Local Llama 3 Models
- Install Ollama: [https://ollama.ai/](https://ollama.ai/)
- Pull the Llama 3 model:
  ```bash
  ollama pull llama3:8b
  ```

## Running the Application

To launch the DSPy-based Teacher Training Simulator:

```bash
streamlit run dspy_web_app.py
```

## Usage

1. Select a language model from the sidebar (GPT-4o-mini is selected by default for cost-effective operation)
2. Click "Initialize LLM" to set up the LLM interface
3. Navigate to "Create Scenario" to set up a teaching scenario
4. Use the "Practice Teaching" interface to interact with the simulated student
5. View analysis of your teaching approach in the "View Analysis" section
6. Find relevant teaching resources in the "Resources" section

## Differences from the Original Version

This DSPy implementation includes several improvements over the original LangChain version:

1. **Optimized Prompting**: DSPy provides better prompt optimization for more consistent results.
2. **Error Handling**: Improved error recovery mechanisms for better reliability.
3. **Performance**: Faster response times, especially with local models.
4. **Customized Modules**: Specialized DSPy modules for teaching-specific tasks.
5. **Fallback Mechanisms**: If a model fails, the system gracefully falls back to alternative approaches.

## File Structure

- `dspy_web_app.py`: Main Streamlit web application
- `dspy_llm_handler.py`: DSPy implementation of the LLM interface
- `dspy_adapter.py`: Adapter for compatibility with existing code
- `test_dspy_chatbot.py`: Test script to verify functionality

## Upgrading From LangChain Version

If you've been using the original LangChain-based version, you can migrate to this DSPy version by:

1. Installing the additional dependencies: `pip install dspy-ai torch`
2. Using `dspy_web_app.py` instead of `web_app.py`
3. Your existing scenarios and settings should be compatible

## Troubleshooting

### Common Issues

1. **Model Initialization Fails**:
   - Check your API keys
   - Ensure Ollama is running if using local models

2. **Slow Responses**:
   - Local models may be slow on the first run as they load into memory
   - Consider using a smaller model or upgrading your GPU

3. **Error Messages**:
   - Check the application logs for detailed error information
   - Ensure your Python environment has all required dependencies

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- DSPy team for the powerful LLM framework
- Original Teacher Training Simulator developers
- Ollama for local model support 