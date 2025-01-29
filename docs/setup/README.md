# Development Environment Setup Guide

This guide provides detailed instructions for setting up your development environment and accessing the data collection server for the Utah Elementary Teacher Training Assistant project.

## Table of Contents
1. [Development Prerequisites](#development-prerequisites)
2. [Server Access Setup](#server-access-setup)
3. [Development Setup Options](#development-setup-options)
4. [Common Setup Steps](#common-setup-steps)
5. [Data Collection Server](#data-collection-server)
6. [Troubleshooting](#troubleshooting)

## Development Prerequisites
- Git
- Python 3.11+
- PostgreSQL 14+
- Node.js 18+

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

## Development Setup Options

Choose the setup guide that matches your development environment:

- [Windows with WSL](windows_wsl.md)
- [macOS](macos.md)
- [Linux](linux.md)

## Common Setup Steps

1. **Clone Repository**
   ```bash
   git clone https://github.com/your-org/utah-tta.git
   cd utah-tta
   ```

2. **Environment Setup**
   - [Anaconda Setup](environment.md#anaconda-setup)
   - [Standard Python Setup](environment.md#standard-setup)
   - [Database Configuration](database.md)

3. **Configuration**
   ```bash
   # Copy example config
   cp config/example.env .env
   
   # Edit configuration
   nano .env
   ```

## Data Collection Server

### Directory Structure
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

### File Management
```bash
# Common commands (replace X with your team number)
# List your files
ls -l /mnt/shared_education_data/raw_data/teamX_data

# Check your space usage
du -h /mnt/shared_education_data/raw_data/teamX_data

# Create a new directory
mkdir /mnt/shared_education_data/raw_data/teamX_data/new_folder
```

### File Naming Convention
- Use format: `teamX_YYYY-MM-DD_type_description`
- Example: `team1_2024-03-20_observation_math_lesson.pdf`

### Required Metadata
Each dataset must include:
- Collection date
- Team member names
- Subject area
- Brief description
- Relevant standards addressed

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
3. For development setup: Contact your team lead 