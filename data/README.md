# UTAH-TTA Data Directory

## Overview
This directory contains the structured educational data for the Utah Elementary Teacher Training Assistant (UTAH-TTA), specifically focused on second-grade education.

## Documentation
- For detailed data collection procedures, see [Data Collection Guide](../docs/data/collection_guide.md)
- For data governance standards, see [Data Governance](GOVERNANCE.md)

## Directory Structure
```
data/
├── second_grade/                        # Second grade content
│   ├── utah_core_standards/             # Utah 2nd grade standards
│   │   ├── mathematics/                 # Math standards and objectives
│   │   ├── english_language_arts/       # ELA standards
│   │   ├── science/                     # Science standards
│   │   └── social_studies/             # Social studies standards
│   │
│   ├── teaching_methods/               # Teaching methodologies
│   │   ├── stem_integration/           # STEM teaching approaches
│   │   ├── literacy_development/       # Reading/writing methods
│   │   └── differentiation/           # Learning adaptations
│   │
│   └── assessment_methods/            # Age-appropriate assessments
│
├── interactions/                      # Teacher-student interactions
│   ├── classroom_management/          # Management scenarios
│   ├── instructional_dialogs/        # Teaching conversations
│   └── support_strategies/           # Learning support
│
└── scenarios/                        # Teaching scenarios
    ├── core_subjects/               # Subject-specific
    ├── classroom_situations/        # Management scenarios
    └── special_cases/              # Specific situations
```

## Quick Links

### Content Guidelines
- [Utah Core Standards](second_grade/utah_core_standards/README.md)
- [Teaching Methods](second_grade/teaching_methods/README.md)
- [Assessment Guidelines](second_grade/assessment_methods/README.md)

### Templates
- [Scenario Templates](scenarios/templates/)
- [Interaction Templates](interactions/templates/)
- [Assessment Templates](second_grade/assessment_methods/templates/)

### Quality Control
- All content must be reviewed by Dr. Ruggles
- Follow naming conventions in [Data Governance](GOVERNANCE.md)
- Ensure metadata completeness
- Maintain privacy standards

## Getting Started
1. Review the [Data Collection Guide](../docs/data/collection_guide.md)
2. Check [Data Governance](GOVERNANCE.md) standards
3. Use appropriate templates
4. Follow review process

## Contact
For questions about data:
- Content validation: Dr. Ruggles (kruggles@uvu.edu)
- Technical issues: Technical Lead
- Process questions: Project Manager 