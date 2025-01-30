# Data Collection Guide - Utah Elementary Teacher Training Assistant

## Table of Contents
- [Data Collection Guide - Utah Elementary Teacher Training Assistant](#data-collection-guide---utah-elementary-teacher-training-assistant)
  - [Table of Contents](#table-of-contents)
  - [Server Access and Setup](#server-access-and-setup)
    - [VPN Requirements](#vpn-requirements)
    - [Server Information](#server-information)
    - [Team Accounts](#team-accounts)
    - [Connection Steps](#connection-steps)
  - [Data Collection Framework](#data-collection-framework)
    - [Team Assignments](#team-assignments)
    - [Collection Focus Areas](#collection-focus-areas)
    - [Data Types and Categories](#data-types-and-categories)
  - [Data Organization and Storage](#data-organization-and-storage)
    - [Directory Structure](#directory-structure)
    - [File Management Commands](#file-management-commands)
    - [File Type Guidelines](#file-type-guidelines)
  - [Documentation Requirements](#documentation-requirements)
    - [Metadata Template](#metadata-template)
  - [Team Collaboration Model](#team-collaboration-model)
    - [Independent Phase](#independent-phase)
    - [Future Integration](#future-integration)
    - [Best Practices](#best-practices)
  - [Support and Help](#support-and-help)
    - [Technical Issues](#technical-issues)
    - [Common Problems](#common-problems)

## Server Access and Setup

### VPN Requirements
- Access via [UVU VPN Service](https://www.uvu.edu/itservices/information-security/vpn_campus.html)
- Contact UVU IT: (801) 863-8888
- Semester-based access
- Renewal through myUVU required each semester

### Server Information
- Server: Ubuntu 24.04 LTS
- Hostname: d19559
- Purpose: Long-term data collection and storage
- Access: Requires UVU VPN connection

### Team Accounts
| Team | Username | Initial Password | Workspace Directory |
|------|----------|-----------------|---------------------|
| Team 1 | team1 | Team2ndGrade12024! | /mnt/shared_education_data/raw_data/team1_data |
| Team 2 | team2 | Team2ndGrade22024! | /mnt/shared_education_data/raw_data/team2_data |
| Team 3 | team3 | Team2ndGrade32024! | /mnt/shared_education_data/raw_data/team3_data |
| Team 4 | team4 | Team2ndGrade42024! | /mnt/shared_education_data/raw_data/team4_data |
| Team 5 | team5 | Team2ndGrade52024! | /mnt/shared_education_data/raw_data/team5_data |
| Team 6 | team6 | Team2ndGrade62024! | /mnt/shared_education_data/raw_data/team6_data |

⚠️ **IMPORTANT**: Password change required on first login!

### Connection Steps
1. Connect to UVU VPN using your UVU credentials
2. SSH command: `ssh teamX@d19559` (replace X with team number)
3. Enter initial password
4. Set new password following requirements:
   - Minimum 12 characters
   - Include uppercase and lowercase letters
   - Include numbers and special characters
   - Don't use parts of your username

## Data Collection Framework

### Team Assignments

1. **Team 1: Math Learning Dynamics**
   - Mathematical thinking processes
   - "Math talk" protocols
   - Problem-solving strategies
   - Manipulative effectiveness
   - Mathematical vocabulary

2. **Team 2: Literacy Development**
   - Reading comprehension
   - Writing process
   - Vocabulary acquisition
   - Phonics awareness
   - Guided reading

3. **Team 3: Science Inquiry**
   - Hands-on experiments
   - Scientific reasoning
   - Hypothesis formation
   - Predictions and results
   - Science vocabulary

4. **Team 4: Social Studies Engagement**
   - Cultural awareness
   - Historical concepts
   - Civic engagement
   - Geography skills
   - Current events

5. **Team 5: Cross-Disciplinary Integration**
   - STEAM integration
   - Project-based learning
   - Thematic units
   - Arts integration
   - Technology use

6. **Team 6: Student Support and Intervention**
   - Differentiation strategies
   - Intervention effectiveness
   - Support services
   - ELL progress
   - IEP implementation

### Collection Focus Areas

1. **Teaching Methods**
   - Innovative approaches
   - Differentiation strategies
   - Technology integration
   - Peer learning
   - Time management

2. **Student Learning**
   - Learning styles
   - Group interactions
   - Question patterns
   - Engagement levels
   - Problem-solving

3. **Assessment**
   - Formative techniques
   - Feedback methods
   - Self-assessment
   - Rubrics
   - Progress monitoring

### Data Types and Categories

1. **Classroom Observations**
   ```
   - Lesson delivery
   - Student engagement
   - Teacher-student interactions
   - Classroom management
   - Technology use
   ```

2. **Teacher Interviews**
   ```
   - Teaching methods
   - Challenges faced
   - Success stories
   - Resource needs
   - Professional development
   ```

3. **Student Work**
   ```
   - Daily assignments
   - Assessments
   - Projects
   - Progress records
   - Learning patterns
   ```

## Data Organization and Storage

### Directory Structure
```
/mnt/shared_education_data/
├── raw_data/                           # Original data
│   ├── team1_data/                     # Team 1 workspace
│   │   ├── classroom_observations/     
│   │   │   ├── math/                  
│   │   │   ├── reading/               
│   │   │   ├── science/               
│   │   │   └── social_studies/        
│   │   ├── teacher_interviews/        
│   │   ├── student_work/             
│   │   │   ├── assignments/          
│   │   │   ├── assessments/          
│   │   │   └── projects/             
│   │   └── metadata/                 
│   │
│   ├── team2_data/ ... team6_data/    # Similar structure for all teams
│
├── processed_data/                     
│   ├── by_subject/                    
│   ├── by_grade_level/                
│   └── cross_cutting/                 
│
└── documentation/                      
    ├── collection_protocols/          
    ├── metadata_templates/            
    └── best_practices/               
```

### File Management Commands
```bash
# List files
ls -l /mnt/shared_education_data/raw_data/teamX_data

# Check space usage
du -h /mnt/shared_education_data/raw_data/teamX_data

# Create directory
mkdir /mnt/shared_education_data/raw_data/teamX_data/new_folder

# Upload files (from local machine)
scp your_file.pdf teamX@d19559:/mnt/shared_education_data/raw_data/teamX_data/

# Download files (to local machine)
scp teamX@d19559:/mnt/shared_education_data/raw_data/teamX_data/file.pdf ./
```

### File Type Guidelines

1. **Observations**
   - Video: MP4
   - Audio: MP3
   - Notes: PDF/TXT
   - Photos: JPG

2. **Interviews**
   - Audio: MP3
   - Transcripts: PDF/TXT
   - Consent forms: PDF

3. **Student Work**
   - Scanned work: PDF
   - Digital work: Original + PDF
   - Photos: JPG

## Documentation Requirements

### Metadata Template
```
filename: teamX_YYYY-MM-DD_type_description_metadata.txt
content:
---
Collection Date: [YYYY-MM-DD]
Team Members: [Names]
Subject Area: [Math/Reading/Science/Social Studies]
Grade Level: [1st/2nd/3rd]
Data Type: [Observation/Interview/Student Work]
Description: [Brief description]
Duration: [For recordings]
Participants: [Number and roles]
Standards: [Educational standards]
Keywords: [3-5 keywords]
Notes: [Additional information]
Related Files: [Related files]
---
```

## Team Collaboration Model

### Independent Phase
- Teams work separately on assigned areas
- Develop specialized methods
- Build team expertise
- Establish workflows
- Maintain quality control

### Future Integration
1. **Data Integration**
   - Cross-subject insights
   - Pattern identification
   - Strategy sharing
   - Comprehensive analysis

2. **Knowledge Base Development**
   - Unified search
   - Cross-referencing
   - Analysis tools
   - Best practices

### Best Practices
1. **Documentation**
   - Use standard forms
   - Regular schedules
   - Complete metadata
   - Cross-reference materials

2. **Quality Control**
   - Peer review
   - Data validation
   - Format compliance
   - Regular backups

3. **Ethics**
   - Student privacy
   - Data security
   - Consent protocols
   - Cultural sensitivity

## Support and Help

### Technical Issues
- VPN problems: UVU IT (801) 863-8888
- Server access: System administrator [email]
- Data collection: Team lead

### Common Problems
1. **Connection Issues**
   - Check VPN status
   - Verify credentials
   - Contact IT support

2. **Login Problems**
   - Verify username/password
   - Check SSH connection
   - Request password reset

3. **Permission Errors**
   - Check directory permissions
   - Verify team workspace
   - Contact administrator 