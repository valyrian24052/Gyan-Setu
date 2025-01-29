# Data Collection Access Control

This document outlines the access control setup for the 2nd grade data collection project. It includes information about team access, directory structure, and data collection guidelines.

## Overview

- 6 teams collecting data
- Shared external hard drive mounted at `/mnt/shared_education_data`
- Individual team workspaces with specific permissions
- Standardized data collection procedures

## Directory Structure

```
/mnt/shared_education_data/
├── raw_data/               # Original, unmodified data
│   ├── team1_data/        # Team 1's primary workspace
│   ├── team2_data/        # Team 2's primary workspace
│   ├── team3_data/        # Team 3's primary workspace
│   ├── team4_data/        # Team 4's primary workspace
│   ├── team5_data/        # Team 5's primary workspace
│   └── team6_data/        # Team 6's primary workspace
├── processed_data/         # Cleaned and processed datasets
├── database/              # Database storage
├── documentation/         # Shared documentation
└── scripts/              # Shared scripts and tools
```

## Access Control Setup

### Team Access
- Each team has a dedicated account
- Teams can read all shared directories
- Write access restricted to team-specific directories
- Password change required on first login

### File Permissions
- Team directories: 775 (rwxrwxr-x)
- Shared directories: 755 (rwxr-xr-x)
- Documentation: 644 (rw-r--r--)

## Data Collection Guidelines

### File Organization
- Store raw data in team-specific directories
- Use consistent naming: `YYYY-MM-DD_type_description`
- Create organized subdirectories as needed

### File Naming Convention
- Lowercase letters and underscores
- Include team identifier: `team1_observation_001.pdf`
- Include date in YYYY-MM-DD format

### Required Metadata
Each dataset must include:
- Collection date
- Team member names
- Subject area
- Brief description
- Relevant standards addressed

## Security Protocols

### Password Policy
- Minimum 12 characters
- Must include uppercase, lowercase, numbers, and special characters
- Regular password changes required
- No password sharing between team members

### Data Protection
- Regular automated backups
- Data integrity verification
- Student privacy protection
- IRB compliance requirements

## Setup Instructions

For detailed setup instructions, see:
- [Initial Setup Guide](setup_guide.md)
- [Team Quick Reference](quick_reference.md)
- [Troubleshooting Guide](troubleshooting.md) 