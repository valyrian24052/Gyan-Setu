# Scenario Template Guide

This guide explains how to use the scenario template for creating new educational scenarios in the Teacher Training Chatbot project.

## Template Structure

The template is organized into several key sections:

### 1. Metadata
- `id`: Unique identifier for the scenario
- `version`: Version number of the scenario
- `created_date`: Initial creation date
- `last_modified`: Last modification date
- `approved_by`: Name of the Education Expert who approved the scenario
- `approval_date`: Date when the scenario was approved

### 2. Context
- `grade_level`: Target grade level (e.g., "3rd Grade", "Middle School")
- `subject`: Subject area (e.g., "Mathematics", "Science")
- `class_size`: Number of students in the class
- `duration`: Length of the lesson
- `setting`: Description of the classroom environment

### 3. Student Profile
- `behavior_type`: Category of student behavior (e.g., "Disruptive", "Withdrawn")
- `learning_style`: Student's learning preferences
- `academic_level`: Current academic performance
- `social_dynamics`: Pattern of social interactions
- `specific_needs`: List of specific learning or behavioral needs

### 4. Scenario
- `title`: Brief, descriptive title
- `description`: Detailed scenario description
- `trigger_event`: What initiates the situation
- `student_behavior`: Specific behavior description
- `class_impact`: How it affects class learning
- `immediate_concerns`: List of immediate issues to address

### 5. Evaluation Criteria
- `required_elements`: Essential components of an acceptable response
- `scoring_rubric`: Weighted criteria for evaluating responses:
  - Professional conduct (25%)
  - Pedagogical approach (25%)
  - Student support (25%)
  - Classroom management (25%)

### 6. Expected Response
- `immediate_action`: What should be done immediately
- `follow_up`: Subsequent actions needed
- `preventive_measures`: How to prevent similar situations
- `key_principles`: Core teaching principles demonstrated

### 7. Improvement Suggestions
- `common_mistakes`: Typical errors to avoid
- `advanced_strategies`: More sophisticated approaches

### 8. Expert Notes
- `key_considerations`: Important points to remember
- `research_basis`: Relevant educational theory/research
- `additional_resources`: Further reading/resources

## Usage Guidelines

1. **Creating a New Scenario**
   - Copy the template file
   - Fill in all fields with appropriate content
   - Ensure all placeholder text (e.g., "SCENARIO_ID") is replaced
   - Maintain the JSON structure

2. **Review Process**
   - Submit completed scenario to Product Owner
   - Product Owner reviews with Education Expert
   - Incorporate feedback and update as needed
   - Obtain final approval

3. **Best Practices**
   - Be specific and detailed in descriptions
   - Use clear, professional language
   - Include measurable criteria in evaluation rubrics
   - Provide concrete examples in improvement suggestions
   - Cite relevant research/theory in expert notes

4. **Validation**
   - Ensure all required fields are completed
   - Verify JSON format is valid
   - Check that weights in scoring rubric sum to 1.0
   - Validate that scenario aligns with project goals

## Example Usage

See `example_scenario.json` in this directory for a complete example of a filled-out scenario. 