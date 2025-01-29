# Initial Setup Guide

This guide provides technical instructions for setting up team access to the data collection system.

## System Requirements

- Linux system with ext4 filesystem support
- External hard drive (18+ TB)
- SSH server
- User management permissions

## Hard Drive Setup

1. Format the drive with ext4:
```bash
sudo mkfs.ext4 /dev/sdX2  # Replace X with actual device letter
```

2. Create mount point:
```bash
sudo mkdir -p /mnt/shared_education_data
```

3. Add to fstab for automatic mounting:
```bash
echo "UUID=<drive-uuid> /mnt/shared_education_data ext4 defaults 0 2" | sudo tee -a /etc/fstab
```

## Directory Structure Setup

1. Create base directories:
```bash
sudo mkdir -p /mnt/shared_education_data/{raw_data,processed_data,database,backup,documentation,scripts}
```

2. Create team directories:
```bash
for i in {1..6}; do
    sudo mkdir -p /mnt/shared_education_data/raw_data/team${i}_data
done
```

## User and Group Setup

1. Create group:
```bash
sudo groupadd education_students
```

2. Create team accounts:
```bash
for i in {1..6}; do
    sudo useradd -m -g education_students team${i}
    sudo chage -d 0 team${i}  # Force password change on first login
done
```

## Permissions Setup

1. Set directory permissions:
```bash
sudo chown -R root:education_students /mnt/shared_education_data
sudo chmod -R 2775 /mnt/shared_education_data
```

2. Set team directory permissions:
```bash
for i in {1..6}; do
    sudo chown team${i}:education_students /mnt/shared_education_data/raw_data/team${i}_data
done
```

## Password Policy Setup

1. Install password quality tools:
```bash
sudo apt-get install libpam-pwquality
```

2. Configure password policy:
```bash
sudo cp /etc/pam.d/common-password /etc/pam.d/common-password.bak
```

Add to `/etc/pam.d/common-password`:
```
password requisite pam_pwquality.so retry=3 minlen=12 difok=3 ucredit=-1 lcredit=-1 dcredit=-1 ocredit=-1
password required pam_unix.so use_authtok sha512 shadow remember=5
```

## Backup Configuration

1. Create backup script:
```bash
sudo tee /usr/local/bin/backup_data.sh << 'EOF'
#!/bin/bash
DATE=$(date +%Y%m%d)
rsync -av --delete /mnt/shared_education_data/ /backup/shared_education_data_${DATE}/
find /backup -type d -name "shared_education_data_*" -mtime +30 -exec rm -rf {} \;
EOF
```

2. Make executable:
```bash
sudo chmod +x /usr/local/bin/backup_data.sh
```

3. Add to crontab:
```bash
sudo crontab -e
# Add line:
0 1 * * * /usr/local/bin/backup_data.sh
```

## Monitoring Setup

1. Install monitoring tools:
```bash
sudo apt-get install quota
```

2. Enable quota:
```bash
sudo quotacheck -ugm /mnt/shared_education_data
sudo quotaon -v /mnt/shared_education_data
```

## Verification Steps

1. Test mount point:
```bash
df -h /mnt/shared_education_data
```

2. Test permissions:
```bash
ls -la /mnt/shared_education_data
```

3. Test user access:
```bash
su - team1
touch /mnt/shared_education_data/raw_data/team1_data/test.txt
```

## Troubleshooting

For common issues and solutions, see [Troubleshooting Guide](troubleshooting.md). 