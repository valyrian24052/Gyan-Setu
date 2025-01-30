# Data Directory Structure and Organization

This directory contains all educational content, training data, and scenarios for the UTAH-TTA project. The data is organized into three main categories to support the chatbot's training and operation.

## Directory Structure
```
data/
├── second_grade/                # Core educational content
│   ├── utah_core_standards/    # Utah 2nd grade standards
│   │   ├── mathematics/        # Math standards and objectives
│   │   ├── english_language_arts/ # ELA standards
│   │   ├── science/           # Science standards
│   │   └── social_studies/    # Social studies standards
│   │
│   ├── teaching_methods/       # Teaching methodologies
│   │   ├── stem_integration/  # STEM teaching approaches
│   │   ├── literacy_development/ # Reading/writing methods
│   │   └── differentiation/   # Learning adaptations
│   │
│   └── assessment_methods/     # Age-appropriate assessments
│
├── interactions/               # Teacher-student interactions
│   ├── classroom_management/   # Management scenarios
│   │   ├── positive_reinforcement/ # Positive behavior examples
│   │   ├── conflict_resolution/ # Conflict handling
│   │   └── transitions/       # Activity transitions
│   │
│   ├── instructional_dialogs/ # Teaching conversations
│   │   ├── math_discussions/  # Math teaching dialogs
│   │   ├── reading_groups/    # Reading group interactions
│   │   └── science_experiments/ # Science lesson dialogs
│   │
│   └── support_strategies/    # Learning support
│       ├── struggling_learners/ # Support for challenges
│       ├── advanced_learners/ # Enrichment interactions
│       └── esl_support/      # Language support
│
└── scenarios/                # Teaching scenarios
    ├── core_subjects/       # Subject-specific
    │   ├── mathematics/     # Math teaching scenarios
    │   ├── reading_writing/ # Literacy scenarios
    │   ├── science/        # Science experiments
    │   └── social_studies/ # Social studies activities
    │
    ├── classroom_situations/ # Management scenarios
    │   ├── daily_routines/  # Regular procedures
    │   ├── special_events/  # Special activities
    │   └── challenges/     # Difficult situations
    │
    └── special_cases/      # Specific situations
        ├── learning_support/ # Learning difficulties
        ├── behavioral_support/ # Behavior management
        └── parent_communication/ # Parent interactions
```

## Content Categories

### 1. Second Grade Core Content
#### Utah Core Standards
- **Math Standards**
  - Numbers and Operations
  - Basic Algebra Concepts
  - Measurement and Data
  - Geometry
- **Language Arts**
  - Reading Comprehension
  - Writing Skills
  - Phonics and Word Recognition
  - Speaking and Listening
- **Science**
  - Earth and Space Systems
  - Physical Science
  - Life Science
  - Engineering Design

#### Teaching Methodologies
- Second Grade Teaching Strategies
- Hands-on Learning Approaches
- Inquiry-based Teaching
- Collaborative Learning Strategies
- Visual and Kinesthetic Methods
- Age-Appropriate Assessment Methods
- Differentiated Instruction for 7-8 Year Olds

### 2. Teacher-Student Interactions
#### Classroom Dialogues
- Second Grade Communication Patterns
- Real-world examples
- Best practice demonstrations
- Common challenges

#### Behavior Management
- Age-Appropriate Management Techniques
- Positive reinforcement examples
- Conflict resolution scenarios
- Group dynamics management

### 3. Teaching Scenarios
#### Subject-Specific
- **Math Lessons**
  - Addition and Subtraction with Regrouping
  - Introduction to Multiplication
  - Basic Fractions
- **Reading and Writing**
  - Reading Comprehension Strategies
  - Writing Complete Sentences
  - Basic Paragraph Structure
- **Science Experiments**
  - Simple Machines
  - States of Matter
  - Plant Life Cycles

#### Classroom Management
- Transition periods
- Group activities
- Special events

#### Special Situations
- Learning difficulties
- Behavioral challenges
- Parent communication

## Data Management Guidelines

### File Formats
- Text content: Markdown (.md)
- Structured data: YAML or JSON
- Media content: MP4 (video), MP3 (audio), PNG/JPEG (images)

### Naming Conventions
- Use lowercase with hyphens
- Include category prefixes
- Add date stamps for versioned content
- Example: `math-addition-regrouping-v1.md`

### Content Requirements
1. **Educational Alignment**
   - Match Utah Core Standards
   - Age-appropriate language
   - Clear learning objectives

2. **Quality Standards**
   - Peer-reviewed content
   - Expert-validated scenarios
   - Regular updates and reviews

3. **Metadata Requirements**
   - Content category
   - Grade level indicators
   - Learning objectives
   - Related standards
   - Creation/update dates
   - Review status

## Contributing Guidelines

### Adding New Content
1. Use provided templates
2. Follow naming conventions
3. Include required metadata
4. Submit for review

### Updating Content
1. Maintain version history
2. Document changes
3. Update related content
4. Verify cross-references

### Review Process
1. Peer review
2. Expert validation
3. Content specialist approval
4. Integration testing

## Related Documentation
- [Content Creation Guide](../docs/content/README.md)
- [Data Quality Standards](../docs/data/quality_standards.md)
- [Review Process](../docs/content/review_process.md) 