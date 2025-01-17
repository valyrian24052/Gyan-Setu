# Data Directory Guide

This directory contains all the educational data used by the Utah Elementary Teacher Training Assistant (UTAH-TTA). This guide explains the directory structure, data creation process, and approval workflows.

## 📋 Table of Contents
- [Directory Structure](#directory-structure)
- [Scenarios](#scenarios)
- [Personas](#personas)
- [Evaluation Data](#evaluation-data)
- [Workflow](#workflow)
- [Role Responsibilities](#role-responsibilities)
- [Quality Standards](#quality-standards)

## Directory Structure

```
data/
├── scenarios/                   # Teaching scenarios
│   ├── approved/               # Expert-approved scenarios
│   ├── drafts/                # Scenario drafts in progress
│   └── templates/             # Scenario templates and guides
│
├── personas/                   # Student personas
│   ├── templates/             # Persona templates
│   └── approved/              # Approved persona profiles
│
└── evaluation/                # Evaluation data
    ├── criteria/              # Evaluation criteria
    └── feedback/              # Feedback templates
```

## Scenarios

### Creation Process

1. **Initial Draft**
   - Use the [scenario template](scenarios/templates/scenario_template.json)
   - Follow the [template guide](scenarios/templates/README.md)
   - Save draft in `scenarios/drafts/`

2. **Review Process**
   - Educational Content Specialist reviews for pedagogical accuracy
   - Product Owner schedules review with Education Expert
   - Feedback documented in `docs/meetings/expert_reviews/`

3. **Approval Process**
   - Education Expert reviews and provides feedback
   - Required revisions tracked in scenario metadata
   - Final approval moves file to `scenarios/approved/`

### Required Components
- Detailed context and student profile
- Clear trigger events and behaviors
- Specific evaluation criteria
- Research-backed expected responses
- Improvement suggestions
- Expert notes and references

## Personas

### Creation Process

1. **Research Phase**
   - Review Utah elementary student demographics
   - Consult with Education Expert on common profiles
   - Document behavioral patterns and learning needs

2. **Development Phase**
   - Use [persona template](personas/templates/persona_template.json)
   - Include learning styles, behaviors, and needs
   - Add specific examples and scenarios

3. **Validation Phase**
   - Review by Educational Content Specialist
   - Validation by Education Expert
   - Integration with scenarios

## Evaluation Data

### Criteria Development

1. **Standards Alignment**
   - Align with Utah teaching standards
   - Map to educational program objectives
   - Define measurable outcomes

2. **Rubric Creation**
   - Define scoring criteria
   - Set performance levels
   - Create feedback templates

3. **Validation**
   - Review by Education Expert
   - Test with sample responses
   - Calibrate scoring weights

## Workflow

1. **Initial Creation**
   ```mermaid
   graph TD
   A[Draft Creation] --> B[Internal Review]
   B --> C[Expert Review]
   C --> D[Revisions]
   D --> E[Final Approval]
   E --> F[Implementation]
   ```

2. **Review Cycles**
   - Weekly internal reviews
   - Bi-weekly expert consultations
   - Monthly content audits

3. **Version Control**
   - Use semantic versioning (v1.0.0)
   - Track changes in metadata
   - Document approval history

## Role Responsibilities

### Product Owner
- Schedule expert review sessions
- Track approval status
- Maintain communication with Education Expert
- Ensure alignment with project goals

### Educational Content Specialist
- Create initial drafts
- Review pedagogical accuracy
- Implement expert feedback
- Maintain templates
- Document best practices

### AI/ML Developer
- Implement approved scenarios
- Create embedding generation scripts
- Develop evaluation algorithms
- Test scenario effectiveness

### QA Specialist
- Validate scenario format
- Test scenario implementation
- Track scenario performance
- Report issues and inconsistencies

### Education Expert
- Review scenario accuracy
- Validate teaching approaches
- Approve final content
- Provide improvement suggestions

## Quality Standards

### Content Standards
- Clear and specific descriptions
- Research-based approaches
- Grade-level appropriate
- Culturally sensitive
- Aligned with Utah standards

### Technical Standards
- Valid JSON format
- Complete metadata
- Proper versioning
- Consistent formatting
- Comprehensive documentation

### Review Checklist
- [ ] Pedagogical accuracy
- [ ] Technical correctness
- [ ] Completeness
- [ ] Standards alignment
- [ ] Expert approval
- [ ] Implementation testing

## Getting Started

1. **For Content Creators**
   ```bash
   # Copy template
   cp scenarios/templates/scenario_template.json scenarios/drafts/new_scenario.json
   
   # Edit scenario
   # Submit for review
   ```

2. **For Reviewers**
   ```bash
   # Access review form
   docs/templates/review_template.md
   
   # Track changes
   docs/meetings/expert_reviews/YYYY-MM-DD_scenario_review.md
   ```

3. **For Implementers**
   ```bash
   # Validate JSON
   python scripts/validate_scenario.py data/scenarios/approved/scenario.json
   
   # Generate embeddings
   python scripts/generate_embeddings.py data/scenarios/approved/
   ```

## Additional Resources

- [Scenario Creation Guide](../docs/scenarios/creation_guide.md)
- [Review Process Documentation](../docs/scenarios/review_process.md)
- [Quality Assurance Guidelines](../docs/technical/qa_guidelines.md)
- [Implementation Guide](../docs/technical/implementation_guide.md)

## Support

For questions or issues:
1. Check role-specific documentation
2. Contact your team lead
3. Create an issue in the repository
4. Schedule a consultation with the Education Expert 