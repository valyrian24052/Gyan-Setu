# Knowledge Analyzer Tool

The `knowledge_analyzer.py` tool helps you understand what information has been extracted from educational books in your knowledge base and how it's being used in scenario generation.

## Features

- **Content Analysis**: Identifies key topics and terms across educational content
- **Source Analysis**: Shows which books contributed the most valuable knowledge
- **Category Distribution**: Visualizes how knowledge is distributed across categories
- **Effectiveness Tracking**: Identifies which knowledge chunks perform best in training
- **Topic Modeling**: Discovers latent topics within the educational content
- **Visualization**: Creates charts, graphs, and word clouds of knowledge data

## Installation

The knowledge analyzer requires several dependencies:

```bash
# Core dependencies
pip install faiss-cpu matplotlib sqlite3

# Optional dependencies for enhanced functionality
pip install wordcloud pandas scikit-learn
```

## Basic Usage

Run the analyzer with default settings:

```bash
python knowledge_analyzer.py
```

This will:
1. Auto-detect the knowledge database in the current directory
2. Perform basic analysis of the content
3. Generate a text report in the `./analysis_results/` directory

## Command Line Options

```
usage: knowledge_analyzer.py [-h] [--db-path DB_PATH] [--output-dir OUTPUT_DIR]
                            [--format {text,html,markdown}] [--analyze-topics]
                            [--visualize] [--analyze-scenarios]
                            [--num-scenarios NUM_SCENARIOS]
                            [--num-topics NUM_TOPICS]

options:
  -h, --help            show this help message and exit
  --db-path DB_PATH     Path to the knowledge database
  --output-dir OUTPUT_DIR
                        Directory to save analysis results
  --format {text,html,markdown}
                        Output format for reports
  --analyze-topics      Perform topic modeling on the knowledge base
  --visualize           Generate visualizations
  --analyze-scenarios   Analyze scenario generation
  --num-scenarios NUM_SCENARIOS
                        Number of scenarios to generate for analysis
  --num-topics NUM_TOPICS
                        Number of topics for topic modeling
```

## Examples

### Generate HTML Report with Visualizations

```bash
python knowledge_analyzer.py --db-path ./knowledge_base/vector_db.sqlite --format html --visualize
```

This creates an interactive HTML report with charts showing knowledge distribution and content analysis.

### Perform Topic Modeling

```bash
python knowledge_analyzer.py --analyze-topics --num-topics 8
```

Discovers 8 latent topics in your knowledge base and the key terms associated with each.

### Analyze Scenario Generation

```bash
python knowledge_analyzer.py --analyze-scenarios --num-scenarios 20
```

Tests how knowledge is used in generating 20 sample scenarios and provides metrics on knowledge utilization.

## Understanding the Results

### Analysis Reports

The tool generates several reports in your specified output directory:

- `knowledge_analysis.json`: Raw analysis data
- `knowledge_report.txt/html/md`: Formatted human-readable report
- `topic_modeling.json`: Topic modeling results (if requested)
- `scenario_generation_analysis.json`: Scenario generation analysis (if requested)

### Visualizations

When `--visualize` is specified, the tool generates:

- `category_distribution.png`: Pie chart of knowledge categories
- `source_distribution.png`: Bar chart of top knowledge sources
- `word_cloud_all.png`: Word cloud of all knowledge content
- `word_cloud_{category}.png`: Word clouds for each category
- `effectiveness_histogram.png`: Distribution of effectiveness scores
- `effectiveness_by_category.png`: Comparison of category effectiveness

## Integration with the Training System

The analyzer works alongside the training system to help you:

1. Identify knowledge gaps that need additional educational materials
2. Understand which sources are most influential in scenario generation
3. Optimize the knowledge base for better training scenarios
4. Track which knowledge leads to the most effective teacher responses

Use this tool regularly as you update your educational materials to ensure your knowledge base continues to improve over time. 