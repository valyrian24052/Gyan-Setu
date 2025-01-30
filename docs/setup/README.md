# Data Collection Guide - Utah Elementary Teacher Training Assistant

This guide provides instructions for accessing the remote Ubuntu server and collecting data for the Utah Elementary Teacher Training Assistant project.

## Table of Contents
- [Data Collection Guide - Utah Elementary Teacher Training Assistant](#data-collection-guide---utah-elementary-teacher-training-assistant)
  - [Table of Contents](#table-of-contents)
  - [Server Access Setup](#server-access-setup)
    - [1. UVU VPN Access Required](#1-uvu-vpn-access-required)
    - [2. Server Information](#2-server-information)
    - [3. Team Account Details](#3-team-account-details)
    - [4. Connection Steps](#4-connection-steps)
  - [Data Collection Server](#data-collection-server)
    - [Directory Structure](#directory-structure)
    - [Data Collection Categories](#data-collection-categories)
    - [File Type Guidelines](#file-type-guidelines)
    - [Metadata Requirements](#metadata-requirements)
    - [File Management](#file-management)
    - [File Naming Convention](#file-naming-convention)
    - [Required Metadata](#required-metadata)
  - [Troubleshooting](#troubleshooting)
    - [Common Issues](#common-issues)
    - [Getting Help](#getting-help)

## Server Access Setup

### 1. UVU VPN Access Required
- Get VPN access: [UVU VPN Service](https://www.uvu.edu/itservices/information-security/vpn_campus.html)
- Contact UVU IT: (801) 863-8888
- VPN access is granted per semester
- Must renew through myUVU each semester

### 2. Server Information
- Server: Ubuntu 24.04 LTS
- Hostname: d19559
- Purpose: Long-term data collection and storage
- Access: Requires UVU VPN connection

### 3. Team Account Details

| Team | Username | Initial Password | Workspace Directory |
|------|----------|-----------------|---------------------|
| Team 1 | team1 | Team2ndGrade12024! | /mnt/shared_education_data/raw_data/team1_data |
| Team 2 | team2 | Team2ndGrade22024! | /mnt/shared_education_data/raw_data/team2_data |
| Team 3 | team3 | Team2ndGrade32024! | /mnt/shared_education_data/raw_data/team3_data |
| Team 4 | team4 | Team2ndGrade42024! | /mnt/shared_education_data/raw_data/team4_data |
| Team 5 | team5 | Team2ndGrade52024! | /mnt/shared_education_data/raw_data/team5_data |
| Team 6 | team6 | Team2ndGrade62024! | /mnt/shared_education_data/raw_data/team6_data |

⚠️ **IMPORTANT**: You MUST change your password on first login!

### 4. Connection Steps
1. Connect to UVU VPN using your UVU credentials
2. Use your team's SSH command:
   ```bash
   ssh teamX@d19559  # Replace X with your team number
   ```
3. Enter your initial password when prompted
4. You will be required to change your password immediately
5. Follow the password requirements:
   - Minimum 12 characters
   - Include uppercase and lowercase letters
   - Include numbers and special characters
   - Don't use parts of your username

## Data Collection Server

### Directory Structure
```
/mnt/shared_education_data/
├── raw_data/                           # Original, unmodified data
│   ├── team1_data/                     # Team 1's primary workspace
│   │   ├── classroom_observations/     # Direct classroom observations
│   │   │   ├── math/                  # Mathematics lessons
│   │   │   ├── reading/               # Reading and literacy
│   │   │   ├── science/               # Science lessons
│   │   │   └── social_studies/        # Social studies lessons
│   │   ├── teacher_interviews/        # Teacher interview recordings and transcripts
│   │   ├── student_work/             # Student work samples
│   │   │   ├── assignments/          # Daily assignments
│   │   │   ├── assessments/          # Tests and quizzes
│   │   │   └── projects/             # Student projects
│   │   └── metadata/                 # Metadata files for all collected data
│   │
│   ├── team2_data/                     # Team 2's primary workspace
│   │   ├── classroom_observations/     # Similar structure as team1
│   │   ├── teacher_interviews/
│   │   ├── student_work/
│   │   └── metadata/
│   │
│   ├── team3_data/                     # Team 3's primary workspace
│   │   ├── classroom_observations/
│   │   ├── teacher_interviews/
│   │   ├── student_work/
│   │   └── metadata/
│   │
│   ├── team4_data/                     # Team 4's primary workspace
│   │   ├── classroom_observations/
│   │   ├── teacher_interviews/
│   │   ├── student_work/
│   │   └── metadata/
│   │
│   ├── team5_data/                     # Team 5's primary workspace
│   │   ├── classroom_observations/
│   │   ├── teacher_interviews/
│   │   ├── student_work/
│   │   └── metadata/
│   │
│   └── team6_data/                     # Team 6's primary workspace
│       ├── classroom_observations/
│       ├── teacher_interviews/
│       ├── student_work/
│       └── metadata/
│
├── processed_data/                      # Cleaned and processed datasets
│   ├── by_subject/                     # Organized by subject area
│   │   ├── mathematics/
│   │   │   ├── teaching_strategies/
│   │   │   ├── common_challenges/
│   │   │   └── best_practices/
│   │   ├── reading_literacy/
│   │   ├── science/
│   │   └── social_studies/
│   │
│   ├── by_grade_level/                 # Organized by grade level
│   │   ├── first_grade/
│   │   ├── second_grade/
│   │   └── third_grade/
│   │
│   └── cross_cutting/                  # Cross-cutting themes
│       ├── classroom_management/
│       ├── student_engagement/
│       ├── differentiation/
│       └── assessment_strategies/
│
└── documentation/                       # Shared documentation
    ├── collection_protocols/           # Data collection protocols
    ├── metadata_templates/             # Templates for metadata
    ├── best_practices/                # Best practices guides
    └── analysis_guidelines/           # Guidelines for data analysis

```

### Data Collection Categories

1. **Classroom Observations**
   - Lesson delivery methods
   - Student engagement patterns
   - Teacher-student interactions
   - Classroom management techniques
   - Use of educational technology
   - Assessment strategies

2. **Teacher Interviews**
   - Teaching methodologies
   - Classroom challenges
   - Success stories
   - Resource needs
   - Professional development
   - Student support strategies

3. **Student Work**
   - Daily assignments
   - Assessment responses
   - Project outcomes
   - Progress tracking
   - Learning patterns
   - Areas of difficulty

### File Type Guidelines

1. **Observations**
   - Video recordings: MP4 format
   - Audio recordings: MP3 format
   - Field notes: PDF or TXT
   - Photos: JPG format

2. **Interviews**
   - Audio recordings: MP3 format
   - Transcripts: TXT or PDF
   - Consent forms: PDF

3. **Student Work**
   - Scanned documents: PDF
   - Digital submissions: Original format + PDF
   - Photos of physical work: JPG

### Metadata Requirements

Each file must have an accompanying metadata file following this format:
```
filename: teamX_YYYY-MM-DD_type_description_metadata.txt
content:
---
Collection Date: [YYYY-MM-DD]
Team Members: [Names of team members present]
Subject Area: [Mathematics/Reading/Science/Social Studies]
Grade Level: [1st/2nd/3rd]
Data Type: [Observation/Interview/Student Work]
Description: [Brief description of the content]
Duration: [For recordings]
Participants: [Number and roles of participants]
Standards Addressed: [Relevant educational standards]
Keywords: [3-5 relevant keywords]
Notes: [Any additional relevant information]
Related Files: [List of related files if any]
---
```

### File Management
```bash
# Common commands (replace X with your team number)
# List your files
ls -l /mnt/shared_education_data/raw_data/teamX_data

# Check your space usage
du -h /mnt/shared_education_data/raw_data/teamX_data

# Create a new directory
mkdir /mnt/shared_education_data/raw_data/teamX_data/new_folder

# Copy files to server (run this from your local machine)
scp your_local_file.pdf teamX@d19559:/mnt/shared_education_data/raw_data/teamX_data/

# Download files from server (run this from your local machine)
scp teamX@d19559:/mnt/shared_education_data/raw_data/teamX_data/file.pdf ./
```

### File Naming Convention
- Use format: `teamX_YYYY-MM-DD_type_description`
- Examples: 
  - `team1_2024-03-20_observation_math_lesson.pdf`
  - `team2_2024-03-21_interview_teacher_feedback.txt`
  - `team3_2024-03-22_survey_student_responses.csv`

### Required Metadata
Each dataset must include:
- Collection date
- Team member names
- Subject area
- Brief description
- Relevant standards addressed

Example metadata file (`team1_2024-03-20_metadata.txt`):
```
Collection Date: March 20, 2024
Team Members: Jane Doe, John Smith
Subject Area: Mathematics
Description: Classroom observation of 2nd grade addition and subtraction lesson
Standards: 
- 2.OA.A.1 (Use addition and subtraction within 100)
- 2.OA.B.2 (Fluently add and subtract within 20)
Notes: 30-minute observation, 23 students present
```

## Troubleshooting

### Common Issues

1. **Cannot connect to server**
   - Verify UVU VPN is connected
   - Check your VPN credentials
   - Ensure you're using the correct VPN client
   - Contact UVU IT for VPN issues

2. **Cannot log in**
   - Verify username and password
   - Check SSH connection to d19559
   - Ensure VPN is still connected
   - Contact system administrator if password reset needed

3. **Permission denied**
   - Verify you're in your team's directory
   - Check file permissions
   - Contact system administrator if persists

### Getting Help
1. For VPN issues: Contact UVU IT (801) 863-8888
2. For server access: Email system administrator [admin email]
3. For data collection questions: Contact your team lead 