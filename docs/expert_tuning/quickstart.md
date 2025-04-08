# Expert Tuning Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Prerequisites
- Python 3.9+
- Conda environment
- Access credentials

### Installation

```bash
# Create and activate conda environment
conda create -n expert-tuning python=3.9
conda activate expert-tuning

# Install required packages
pip install expert-tuning dspy-ai

# Set up environment
export EXPERT_TUNING_API_KEY=your_key_here
```

### Basic Usage

```python
from expert_tuning import ExpertTuning
from expert_tuning.feedback import FeedbackCollector
from expert_tuning.optimization import ModelOptimizer

# Initialize
system = ExpertTuning()

# Collect expert feedback
feedback = FeedbackCollector()
rating = feedback.rate_response(
    response_id="example_1",
    ratings={
        "clarity": 8,
        "engagement": 7,
        "pedagogy": 9
    },
    improved_response="A clearer way to explain...",
    notes="Added concrete examples"
)

# Optimize model
optimizer = ModelOptimizer()
result = optimizer.improve(feedback=rating)
print(f"Improvement score: {result.score}")
```

## 🎯 Core Features

### 1. Expert Feedback
- Rate AI responses
- Provide improvements
- Share examples

### 2. Model Optimization
- Automatic enhancement
- Quality tracking
- Performance monitoring

### 3. Analytics
- View your impact
- Track improvements
- Earn recognition

## 📊 Example Use Cases

### Teaching Improvement
```python
# Improve a response
feedback.improve_response(
    original="The derivative represents rate of change",
    improved="Think of derivative as speed - it shows how fast something changes",
    context="Teaching calculus basics"
)
```

### Example Sharing
```python
# Share teaching example
feedback.add_example(
    subject="Math",
    topic="Derivatives",
    scenario="Student confused about rates of change",
    solution="Used car speed analogy"
)
```

## 🔍 Check Results

```python
# View your impact
stats = system.get_stats(expert_id="your_id")
print(f"Contributions: {stats.total}")
print(f"Quality score: {stats.quality}")
print(f"Impact score: {stats.impact}")
```

## 🎮 Achievement System

### Progress Tracking
```python
# Check your status
profile = system.get_profile(expert_id="your_id")
print(f"Level: {profile.level}")
print(f"Points: {profile.points}")
print(f"Next goal: {profile.next_achievement}")
```

### Activity Tracking
```python
# Check your activity
activity = system.get_activity(expert_id="your_id")
print(f"Active days: {activity.days}")
print(f"Best streak: {activity.max_streak}")
```

## 🆘 Common Issues

### Authentication
```python
# Check credentials
system.verify_access()
```

### Validation
```python
# Validate feedback
is_valid = feedback.validate(your_feedback)
if not is_valid:
    print(feedback.get_errors())
```

## 📈 Next Steps

1. Join Expert Community
2. Review First Response
3. Share Teaching Example
4. Track Your Impact

## 🔗 Links

- [Full Documentation](README.md)
- [Technical Guide](technical/README.md)
- [API Reference](technical/api.md)
- [Community](https://community.utta.ai)

## 💡 Tips

1. Be specific
2. Use examples
3. Stay consistent
4. Focus on quality
5. Share insights

## 🤝 Help

- Docs: [docs.utta.ai](https://docs.utta.ai)
- Email: support@utta.ai
- Forum: [forum.utta.ai](https://forum.utta.ai) 