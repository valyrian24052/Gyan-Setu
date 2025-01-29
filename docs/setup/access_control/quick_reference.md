# Quick Reference Guide

## First Time Login

1. SSH Command:
```bash
ssh teamX@hostname  # Replace X with your team number (1-6)
```

2. Enter your initial password when prompted
3. You will be required to change your password immediately
4. Follow the password requirements:
   - Minimum 12 characters
   - Include uppercase and lowercase letters
   - Include numbers and special characters
   - Don't use parts of your username

## Your Workspace

Your team's directory is at:
```bash
/mnt/shared_education_data/raw_data/teamX_data  # Replace X with your team number
```

## Common Commands

### Password Management
```bash
# Change your password
passwd
```

### File Management
```bash
# List your files
ls -l /mnt/shared_education_data/raw_data/teamX_data

# Check your space usage
du -h /mnt/shared_education_data/raw_data/teamX_data

# Create a new directory
mkdir /mnt/shared_education_data/raw_data/teamX_data/new_folder

# Copy files
cp source_file.txt /mnt/shared_education_data/raw_data/teamX_data/

# Create a new file
touch /mnt/shared_education_data/raw_data/teamX_data/filename.txt
```

### Viewing Documentation
```bash
# View README
less /mnt/shared_education_data/documentation/README.md

# List available documentation
ls /mnt/shared_education_data/documentation
```

## File Naming Examples

```
2024-03-20_observation_math_lesson.pdf
2024-03-21_assessment_reading_test.csv
2024-03-22_student_work_science_project.pdf
```

## Need Help?

1. Check the full documentation in `/mnt/shared_education_data/documentation`
2. Contact your team leader
3. Email system administrator: [admin email]

## Important Reminders

- Change your password every 90 days
- Always log out when done
- Keep your credentials secure
- Back up important data
- Follow naming conventions
- Include required metadata 