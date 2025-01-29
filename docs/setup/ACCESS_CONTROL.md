# Data Collection Access Control Guide

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Team Accounts](#team-accounts)
4. [Quick Start Guide](#quick-start-guide)
5. [Technical Setup](#technical-setup)
6. [Data Collection Guidelines](#data-collection-guidelines)
7. [Security Protocols](#security-protocols)
8. [Troubleshooting](#troubleshooting)

## Overview

- 6 teams collecting data, each with their own dedicated account
- Server hostname: d19559
- Shared external hard drive mounted at `/mnt/shared_education_data`
- Individual team workspaces with specific permissions
- Standardized data collection procedures
- Long-term server for the entire project duration

## Prerequisites

### UVU VPN Access
1. **Required for Remote Access**
   - All team members must have UVU VPN access
   - Server d19559 is only accessible through UVU's network
   - Get VPN access at: [UVU VPN Service](https://www.uvu.edu/itservices/information-security/vpn_campus.html)
   - Contact UVU IT at (801) 863-8888 for VPN support

2. **VPN Access Details**
   - Student VPN access is granted per semester
   - Must agree to Campus VPN policy in myUVU each semester
   - VPN access will automatically stop at semester end
   - Renew through myUVU at the start of each semester

3. **VPN Connection Steps**
   - Install UVU VPN client
   - Connect to VPN using your UVU credentials
   - Keep VPN connected while working with the server

## Team Accounts

### Initial Login Credentials (CHANGE ON FIRST LOGIN)

⚠️ **IMPORTANT**: You MUST change these passwords on your first login!

| Team | Username | Initial Password |
|------|----------|-----------------|
| Team 1 | team1 | Team2ndGrade12024! |
| Team 2 | team2 | Team2ndGrade22024! |
| Team 3 | team3 | Team2ndGrade32024! |
| Team 4 | team4 | Team2ndGrade42024! |
| Team 5 | team5 | Team2ndGrade52024! |
| Team 6 | team6 | Team2ndGrade62024! |

### Connection Information

| Team | Command to Connect | Workspace Directory |
|------|-------------------|---------------------|
| Team 1 | `ssh team1@d19559` | `/mnt/shared_education_data/raw_data/team1_data` |
| Team 2 | `ssh team2@d19559` | `/mnt/shared_education_data/raw_data/team2_data` |
| Team 3 | `ssh team3@d19559` | `/mnt/shared_education_data/raw_data/team3_data` |
| Team 4 | `ssh team4@d19559` | `/mnt/shared_education_data/raw_data/team4_data` |
| Team 5 | `ssh team5@d19559` | `/mnt/shared_education_data/raw_data/team5_data` |
| Team 6 | `ssh team6@d19559` | `/mnt/shared_education_data/raw_data/team6_data` |

### Account Structure
- Each account belongs to the `education_students` group
- Each team has a dedicated workspace
- Initial passwords MUST be changed on first login
- No account sharing between team members

## Quick Start Guide

### First Time Login
1. Connect to UVU VPN using your UVU credentials
2. Use your team's specific SSH command from the table above
3. Enter your team's initial password (see table above)
4. You will be required to change your password immediately
5. Follow the password requirements:
   - Minimum 12 characters
   - Include uppercase and lowercase letters
   - Include numbers and special characters
   - Don't use parts of your username
   - Don't reuse the initial password

### Connection Workflow
1. Start UVU VPN
2. Connect to server
3. Do your work
4. Log out when done
5. Disconnect VPN

### Common Commands

```bash
# Change your password
passwd

# List your files (replace X with your team number)
ls -l /mnt/shared_education_data/raw_data/teamX_data

# Check your space usage
du -h /mnt/shared_education_data/raw_data/teamX_data

# Create a new directory
mkdir /mnt/shared_education_data/raw_data/teamX_data/new_folder

# Copy files
cp source_file.txt /mnt/shared_education_data/raw_data/teamX_data/
```

## Technical Setup

### System Requirements
- Linux system with ext4 filesystem support
- External hard drive (18+ TB)
- SSH server (hostname: d19559)
- User management permissions

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

### File Permissions
- Team directories: 775 (rwxrwxr-x)
- Shared directories: 755 (rwxr-xr-x)
- Documentation: 644 (rw-r--r--)

### Setup Commands

1. Format and mount drive:
```bash
sudo mkfs.ext4 /dev/sda2
sudo mkdir -p /mnt/shared_education_data
echo "UUID=$(sudo blkid -s UUID -o value /dev/sda2) /mnt/shared_education_data ext4 defaults 0 2" | sudo tee -a /etc/fstab
sudo mount /mnt/shared_education_data
```

2. Create directory structure:
```bash
sudo mkdir -p /mnt/shared_education_data/{raw_data,processed_data,database,backup,documentation,scripts}
for i in {1..6}; do
    sudo mkdir -p /mnt/shared_education_data/raw_data/team${i}_data
done
```

3. Set up users and permissions:
```bash
sudo groupadd education_students
for i in {1..6}; do
    sudo useradd -m -g education_students team${i}
    sudo chage -d 0 team${i}
    sudo chown team${i}:education_students /mnt/shared_education_data/raw_data/team${i}_data
done
sudo chmod -R 2775 /mnt/shared_education_data
```

## Data Collection Guidelines

### File Organization
- Store raw data in team-specific directories
- Use consistent naming: `teamX_YYYY-MM-DD_type_description`
- Create organized subdirectories as needed

### File Naming Examples
```
team1_2024-03-20_observation_math_lesson.pdf
team1_2024-03-21_assessment_reading_test.csv
team1_2024-03-22_student_work_science_project.pdf
```

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
- Regular password changes required (every 90 days)
- No password sharing between team members

### Data Protection
- Regular automated backups
- Data integrity verification
- Student privacy protection
- IRB compliance requirements

### Backup Configuration
```bash
# Daily backups at 1 AM
0 1 * * * rsync -av --delete /mnt/shared_education_data/ /backup/shared_education_data_$(date +\%Y\%m\%d)/
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

4. **Disk space issues**
   - Run `du -h` to check usage
   - Clean up unnecessary files
   - Contact administrator if more space needed

5. **VPN Connection Issues**
   - Check internet connection
   - Verify UVU credentials
   - Try disconnecting and reconnecting VPN
   - Clear VPN client cache
   - Contact UVU IT support for persistent VPN issues

### Getting Help
1. Check this documentation
2. Contact your team leader
3. For VPN issues: Contact UVU IT Support
4. For server access: Email system administrator [admin email]

## Important Reminders
- Change your password every 90 days
- Always connect to VPN before accessing the server
- Always log out when done
- Keep your credentials secure
- Back up important data
- Follow naming conventions
- Include required metadata
- Never share your team account with others
- Only work in your team's directory 