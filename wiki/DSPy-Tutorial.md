# DSPy Optimization Tutorial

This tutorial provides a step-by-step guide to implementing DSPy Optimization for educational question-answering. You'll learn how to set up your environment, create a QA model, define evaluation metrics, optimize prompts, and test your optimized model.

## Prerequisites

- Python 3.8+
- OpenAI API key (or access to another LLM API)
- Basic understanding of language models and prompt engineering

## Step 1: Environment Setup

First, let's set up our environment with all necessary dependencies:

```bash
# Create a virtual environment
python -m venv dspy-env
source dspy-env/bin/activate  # On Windows: dspy-env\Scripts\activate

# Install required packages
pip install dspy-ai openai numpy pandas tqdm
```

## Step 2: Create a Simple Educational QA Dataset

Let's create a small dataset for educational question answering:

```python
# save_dataset.py
import json
import os

# Sample educational QA pairs
educational_qa = [
    {
        "question": "What is photosynthesis?",
        "answer": "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods with the help of chlorophyll. It converts carbon dioxide and water into glucose and oxygen using sunlight energy."
    },
    {
        "question": "How does the water cycle work?",
        "answer": "The water cycle is the continuous movement of water on, above, and below the Earth's surface. It involves processes like evaporation, condensation, precipitation, infiltration, and runoff, constantly recycling water throughout our environment."
    },
    {
        "question": "What causes seasons on Earth?",
        "answer": "Seasons are caused by Earth's tilted axis (about 23.5 degrees) as it revolves around the sun. This tilt means different parts of Earth receive different amounts of sunlight throughout the year, creating seasonal changes in temperature and daylight."
    },
    {
        "question": "What is the difference between a mixture and a compound?",
        "answer": "A mixture is a combination of two or more substances where no chemical reaction occurs. Each substance retains its chemical properties. A compound is formed when elements chemically combine in fixed ratios, creating substances with different properties than their components."
    },
    {
        "question": "Explain Newton's Third Law of Motion.",
        "answer": "Newton's Third Law states that for every action, there is an equal and opposite reaction. This means that when one object exerts a force on another object, the second object exerts an equal force in the opposite direction on the first object."
    },
    {
        "question": "What is the difference between weather and climate?",
        "answer": "Weather refers to short-term atmospheric conditions including temperature, humidity, precipitation, wind, etc. in a specific place at a specific time. Climate is the average weather pattern of a region over a long period, typically 30+ years."
    },
    {
        "question": "How do volcanoes form?",
        "answer": "Volcanoes form when magma from the Earth's upper mantle rises through the crust, creating pressure that eventually causes an eruption. This typically occurs at the boundaries of tectonic plates, either where plates are moving apart or where one plate is being forced under another."
    },
    {
        "question": "What are the main functions of the human skeletal system?",
        "answer": "The human skeletal system has several functions: providing structural support for the body, protecting vital organs, allowing movement through muscle attachment, producing blood cells in bone marrow, and storing minerals like calcium and phosphorus."
    },
    {
        "question": "How does a simple electrical circuit work?",
        "answer": "A simple electrical circuit works by providing a closed path for electrons to flow. It typically consists of a power source (like a battery), conducting wires, a load (like a light bulb), and often a switch. When complete, current flows from the negative terminal through the circuit to the positive terminal."
    },
    {
        "question": "What is cellular respiration?",
        "answer": "Cellular respiration is the process by which cells convert nutrients (primarily glucose) into ATP, the energy currency of cells. It involves three main stages: glycolysis, the Krebs cycle, and the electron transport chain, resulting in the production of energy with carbon dioxide and water as byproducts."
    }
]

# Create directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Save dataset as JSONL
with open("data/educational_qa.jsonl", "w") as f:
    for qa_pair in educational_qa:
        f.write(json.dumps(qa_pair) + "\n")

print(f"Educational QA dataset with {len(educational_qa)} examples saved to data/educational_qa.jsonl")
```

Run this script to save our dataset:

```bash
python save_dataset.py
```

## Step 3: Create a Simple DSPy Educational QA Model

Now, let's define our DSPy model for educational question answering:

```python
# educational_qa_model.py
import os
import dspy
import json
import random
from typing import List

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "your-api-key-here"  # Replace with your key

# Configure the language model
lm = dspy.OpenAI(model="gpt-3.5-turbo")
dspy.settings.configure(lm=lm)

# Define the Educational QA Signature
class EducationalQA(dspy.Signature):
    """Answer educational questions with accurate, grade-appropriate information."""
    
    question = dspy.InputField(desc="The educational question to answer")
    answer = dspy.OutputField(desc="A comprehensive, accurate, and educational answer")

# Create our model using Predict module
class EducationalQAModel(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(EducationalQA)
    
    def forward(self, question):
        return self.predictor(question=question)

# Load the dataset
def load_dataset(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

# Split into train and test sets
def split_dataset(data, test_size=0.3):
    shuffled = data.copy()
    random.shuffle(shuffled)
    split_point = int(len(shuffled) * (1 - test_size))
    return shuffled[:split_point], shuffled[split_point:]

# Test basic functionality
if __name__ == "__main__":
    model = EducationalQAModel()
    
    result = model("What is the process of photosynthesis?")
    print(f"Question: What is the process of photosynthesis?")
    print(f"Answer: {result.answer}")
    
    # Load and show dataset info
    dataset = load_dataset("data/educational_qa.jsonl")
    train_data, test_data = split_dataset(dataset)
    print(f"\nDataset loaded with {len(dataset)} examples")
    print(f"Train set: {len(train_data)} examples")
    print(f"Test set: {len(test_data)} examples")
```

Test the basic model to ensure it works:

```bash
python educational_qa_model.py
```

## Step 4: Implement DSPy Optimization

Now, let's create a script to optimize our model using DSPy:

```python
# optimize_qa_model.py
import os
import dspy
import json
import random
from educational_qa_model import EducationalQA, EducationalQAModel, load_dataset, split_dataset

# Set your OpenAI API key if not already set
os.environ["OPENAI_API_KEY"] = "your-api-key-here"  # Replace with your key

# Configure the language model
lm = dspy.OpenAI(model="gpt-3.5-turbo")
dspy.settings.configure(lm=lm)

# Define metrics for evaluation
class AnswerQuality(dspy.Metric):
    """Evaluates the quality, accuracy, and educational value of answers."""
    
    def __init__(self, lm=None):
        self.lm = lm or dspy.settings.lm
    
    def evaluate(self, example, pred, trace=None):
        # Create evaluation prompt
        eval_template = """
        You are evaluating an educational answer for accuracy, completeness, and grade-appropriateness.
        
        Question: {question}
        
        Reference Answer: {reference}
        
        Model Answer: {prediction}
        
        Rate the model answer on a scale of 0-10 where:
        - 0-3: Contains significant errors or misunderstandings
        - 4-6: Generally correct but incomplete or imprecise
        - 7-10: Accurate, comprehensive, and educational
        
        Provide only the numerical score.
        """
        
        prompt = eval_template.format(
            question=example.question,
            reference=example.answer,
            prediction=pred.answer
        )
        
        # Get score from language model
        score_response = self.lm(prompt)
        
        # Extract numerical score (handling potential formatting issues)
        try:
            score = float(score_response.strip())
            # Normalize to 0-1 range
            normalized_score = score / 10.0
            return normalized_score
        except ValueError:
            print(f"Warning: Could not parse score from: {score_response}")
            return 0.5  # Default mid-range score if parsing fails

# Load dataset
dataset = load_dataset("data/educational_qa.jsonl")
train_data, test_data = split_dataset(dataset)

# Convert to DSPy format
dspy_train = [dspy.Example(question=ex["question"], answer=ex["answer"]) for ex in train_data]
dspy_test = [dspy.Example(question=ex["question"], answer=ex["answer"]) for ex in test_data]

# Initialize the optimizer
teleprompter = dspy.teleprompt.BootstrapFewShot(
    metric=AnswerQuality(),
    max_bootstrapped_demos=3,
    num_candidate_programs=5
)

# Create the model to optimize
model = EducationalQAModel()

# Run optimization
optimized_model = teleprompter.compile(
    model=model,
    trainset=dspy_train,
    valset=dspy_test[:2],  # Using a small validation set for speed
    max_rounds=2  # Start with a small number for testing
)

# Save the optimized model's prompt
teleprompter.save("optimized_qa_model.json")

# Evaluate on test examples
def evaluate_model(model, test_examples):
    scores = []
    metric = AnswerQuality()
    
    for example in test_examples:
        # Generate prediction
        pred = model(example.question)
        
        # Evaluate prediction
        score = metric.evaluate(example, pred)
        scores.append(score)
        
        # Print example
        print(f"Question: {example.question}")
        print(f"Model Answer: {pred.answer}")
        print(f"Reference: {example.answer}")
        print(f"Score: {score:.2f}\n" + "-"*50 + "\n")
    
    # Calculate average score
    avg_score = sum(scores) / len(scores) if scores else 0
    print(f"Average Score: {avg_score:.2f} out of 1.0")
    return avg_score

# Evaluate base model vs optimized model
print("\n===== BASE MODEL EVALUATION =====\n")
base_model = EducationalQAModel()
base_score = evaluate_model(base_model, dspy_test[:3])  # Test on 3 examples

print("\n===== OPTIMIZED MODEL EVALUATION =====\n")
optimized_score = evaluate_model(optimized_model, dspy_test[:3])  # Test on same 3 examples

print(f"Improvement: {(optimized_score - base_score):.2f} points")
```

Run the optimization process:

```bash
python optimize_qa_model.py
```

## Step 5: Use the Optimized Model

Let's create a script to use our optimized model on new questions:

```python
# use_optimized_model.py
import os
import dspy
import json
from educational_qa_model import EducationalQAModel

# Set your OpenAI API key if not already set
os.environ["OPENAI_API_KEY"] = "your-api-key-here"  # Replace with your key

# Configure the language model
lm = dspy.OpenAI(model="gpt-3.5-turbo")
dspy.settings.configure(lm=lm)

# Load the optimized model
teleprompter = dspy.teleprompt.BootstrapFewShot()
teleprompter.load("optimized_qa_model.json")
base_model = EducationalQAModel()
optimized_model = teleprompter.apply(base_model)

def answer_question(question):
    """Generate an answer using the optimized model"""
    result = optimized_model(question)
    return result.answer

# Interactive loop
def main():
    print("Educational QA System (type 'exit' to quit)")
    print("="*50)
    
    while True:
        question = input("\nYour question: ")
        if question.lower() in ['exit', 'quit', 'q']:
            break
            
        try:
            answer = answer_question(question)
            print("\nAnswer:")
            print(answer)
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nThank you for using the Educational QA System!")

if __name__ == "__main__":
    main()
```

Use the optimized model interactively:

```bash
python use_optimized_model.py
```

## Exploring the Optimized Prompts

To understand what DSPy has learned, you can examine the optimized prompts:

```python
# examine_optimization.py
import os
import dspy
import json
from educational_qa_model import EducationalQAModel

# Load the optimized model
teleprompter = dspy.teleprompt.BootstrapFewShot()
teleprompter.load("optimized_qa_model.json")

# Print the optimized prompt
print("\n===== OPTIMIZED PROMPT =====\n")
print(json.dumps(teleprompter.compiled_prompt, indent=2))

# Show the examples it selected
print("\n===== SELECTED EXAMPLES =====\n")
for i, example in enumerate(teleprompter.compiled_demos):
    print(f"Example {i+1}:")
    print(f"Question: {example.question}")
    print(f"Answer: {example.answer}")
    print("-"*50)
```

Examine the optimized prompt:

```bash
python examine_optimization.py
```

## Advanced: Custom Teleprompters

For more advanced users, let's explore how to create a custom teleprompter:

```python
# custom_teleprompter.py
import os
import dspy
import json
import random
from educational_qa_model import EducationalQA, EducationalQAModel, load_dataset, split_dataset

# Configure the language model
os.environ["OPENAI_API_KEY"] = "your-api-key-here"  # Replace with your key
lm = dspy.OpenAI(model="gpt-3.5-turbo")
dspy.settings.configure(lm=lm)

# Create a custom teleprompter that uses both bootstrapping and field-level instructions
class CustomEducationalTeleprompter(dspy.teleprompt.Teleprompter):
    def __init__(self, metric, num_instructions=3, num_demos=3):
        super().__init__()
        self.metric = metric
        self.num_instructions = num_instructions
        self.num_demos = num_demos
        self.compiled_instructions = None
        self.compiled_demos = None
    
    def compile(self, model, trainset, valset=None, **config):
        # Generate candidate field-level instructions
        instructions_candidates = self._generate_field_instructions(trainset)
        
        # Select the best examples from the training set
        demo_candidates = trainset
        
        # Create and evaluate different combinations
        best_score = -float('inf')
        best_instructions = None
        best_demos = None
        
        # Try different combinations of instructions and demos
        for _ in range(5):  # Try 5 random combinations
            # Sample instructions and demos
            sampled_instructions = random.sample(instructions_candidates, min(self.num_instructions, len(instructions_candidates)))
            sampled_demos = random.sample(demo_candidates, min(self.num_demos, len(demo_candidates)))
            
            # Update model with these instructions and demos
            modified_model = self._modify_model(model, sampled_instructions, sampled_demos)
            
            # Evaluate
            score = self._evaluate_model(modified_model, valset or trainset)
            
            if score > best_score:
                best_score = score
                best_instructions = sampled_instructions
                best_demos = sampled_demos
        
        # Save the best configuration
        self.compiled_instructions = best_instructions
        self.compiled_demos = best_demos
        
        # Return the best model
        return self._modify_model(model, best_instructions, best_demos)
    
    def _generate_field_instructions(self, examples):
        # Generate specific instructions for each field in our signature
        question_instruction_prompt = """
        Generate 3 specific instructions for answering educational questions effectively.
        The instructions should help create comprehensive, accurate, and grade-appropriate responses.
        Provide each instruction on a separate line.
        """
        
        answer_instruction_prompt = """
        Generate 3 specific instructions for crafting educational answers that are:
        1. Factually accurate
        2. Comprehensive but concise
        3. Appropriately pitched for students
        Provide each instruction on a separate line.
        """
        
        # Get instruction candidates
        question_instructions = lm(question_instruction_prompt).strip().split('\n')
        answer_instructions = lm(answer_instruction_prompt).strip().split('\n')
        
        # Format as instruction dictionary
        instructions = []
        for q_inst in question_instructions:
            for a_inst in answer_instructions:
                instructions.append({
                    'question': q_inst.strip(),
                    'answer': a_inst.strip()
                })
        
        return instructions
    
    def _modify_model(self, model, instructions, demos):
        # Create a new model instance with the instructions and demos
        new_model = dspy.Module()
        
        # Store the base predictor
        predictor = model.predictor
        
        # Create a modified predictor with instructions and demonstrations
        new_model.predictor = dspy.FewShotWithInstruction(
            predictor.signature,
            demos=demos,
            instructions=instructions
        )
        
        # Copy the forward method
        new_model.forward = model.forward
        
        return new_model
    
    def _evaluate_model(self, model, examples):
        scores = []
        for example in examples:
            pred = model(example.question)
            score = self.metric.evaluate(example, pred)
            scores.append(score)
        return sum(scores) / len(scores) if scores else 0
    
    def apply(self, model):
        """Apply the compiled instructions and demos to a model."""
        if self.compiled_instructions is None or self.compiled_demos is None:
            raise ValueError("You must compile the teleprompter first")
        
        return self._modify_model(model, self.compiled_instructions, self.compiled_demos)
    
    def save(self, path):
        """Save the compiled instructions and demos."""
        data = {
            'instructions': [inst for inst in self.compiled_instructions],
            'demos': [{'question': d.question, 'answer': d.answer} for d in self.compiled_demos]
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, path):
        """Load compiled instructions and demos."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.compiled_instructions = data['instructions']
        self.compiled_demos = [dspy.Example(question=d['question'], answer=d['answer']) for d in data['demos']]

# Use the custom teleprompter
if __name__ == "__main__":
    # Load dataset
    dataset = load_dataset("data/educational_qa.jsonl")
    train_data, test_data = split_dataset(dataset)
    
    # Convert to DSPy format
    dspy_train = [dspy.Example(question=ex["question"], answer=ex["answer"]) for ex in train_data]
    dspy_test = [dspy.Example(question=ex["question"], answer=ex["answer"]) for ex in test_data]
    
    # Create metric
    from optimize_qa_model import AnswerQuality
    metric = AnswerQuality()
    
    # Initialize custom teleprompter
    custom_teleprompter = CustomEducationalTeleprompter(metric=metric)
    
    # Create and optimize model
    model = EducationalQAModel()
    optimized_model = custom_teleprompter.compile(
        model=model,
        trainset=dspy_train,
        valset=dspy_test[:2]
    )
    
    # Save the optimized config
    custom_teleprompter.save("custom_optimized_qa.json")
    
    # Test an example
    example_question = "What is the relationship between mass and weight?"
    prediction = optimized_model(example_question)
    
    print("\n===== CUSTOM TELEPROMPTER RESULT =====\n")
    print(f"Question: {example_question}")
    print(f"Answer: {prediction.answer}")
    
    # Print the optimized instructions
    print("\n===== OPTIMIZED INSTRUCTIONS =====\n")
    for inst in custom_teleprompter.compiled_instructions[:2]:  # Show first 2
        print(f"Question instruction: {inst['question']}")
        print(f"Answer instruction: {inst['answer']}")
        print("-"*50)
```

Run the custom teleprompter:

```bash
python custom_teleprompter.py
```

## Conclusion

In this tutorial, we've:

1. Set up a DSPy environment
2. Created an educational QA dataset
3. Built a basic DSPy model for question answering
4. Implemented prompt optimization using DSPy's teleprompter
5. Evaluated and compared base vs. optimized models
6. Created an interactive application using the optimized model
7. Explored the optimized prompts
8. Implemented a custom teleprompter for advanced optimization

DSPy offers a powerful middle ground between simple prompt engineering and full model fine-tuning. By systematically optimizing prompts, we can achieve better performance without the need for extensive datasets or computational resources.

For more advanced topics and applications, check out the [DSPy documentation](https://dspy-docs.vercel.app/) and the [Stanford NLP Group's GitHub repository](https://github.com/stanfordnlp/dspy). 