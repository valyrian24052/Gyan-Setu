# Teaching Assistant Training Dataset

This dataset contains 1 human-rated teaching examples for fine-tuning language models to generate better teaching responses.

## Dataset Format

Each example in the JSONL file contains:

- `context`: Full context including scenario and student profile
- `student_message`: The specific student question/comment
- `responses`: Contains 4 different teaching responses (original + 3 alternatives)
- `rankings`: Order of preference from best (1st) to worst (4th)
- `quality_aspects`: Specific elements that make good teaching responses
- `explanation`: Human explanation of why certain responses are better
- `metadata`: Additional information about scenario type, subject, etc.

## Usage for Model Training

This dataset is formatted for:

1. Ranking-based preference learning
2. RLHF (Reinforcement Learning from Human Feedback)
3. Supervised fine-tuning with human preferences

## Latest Dataset

The most recent dataset is: `teaching_training_dataset_20250407_130223.jsonl`
Generated on: 2025-04-07 13:02:23
Contains: 1 examples
