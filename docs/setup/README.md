# Setup and Access Guide

This guide covers all aspects of setting up your development environment and accessing project resources for the UTAH-TTA project.

## Prerequisites

### Development Environment
- Python 3.10 or higher
- Git
- Docker and Docker Compose
- Node.js 18+ (for frontend development)
- VSCode or PyCharm (recommended)

### Server Access Requirements
- UVU VPN access
- GitHub account with repository access
- Development server credentials

## Server Information
- **Server**: Ubuntu 24.04 LTS
- **Hostname**: d19559
- **Purpose**: Long-term data collection and storage
- **Access**: Requires UVU VPN connection

### VPN Access
1. Get VPN access: [UVU VPN Service](https://www.uvu.edu/itservices/information-security/vpn_campus.html)
2. Contact UVU IT: (801) 863-8888
3. VPN access is granted per semester
4. Must renew through myUVU each semester

### Team Accounts
Each team has a dedicated account with specific workspace:

| Team | Username | Initial Password | Workspace |
|------|----------|-----------------|-----------|
| Team 1 | team1 | Team2ndGrade12024! | team1_data |
| Team 2 | team2 | Team2ndGrade22024! | team2_data |
| Team 3 | team3 | Team2ndGrade32024! | team3_data |
| Team 4 | team4 | Team2ndGrade42024! | team4_data |
| Team 5 | team5 | Team2ndGrade52024! | team5_data |
| Team 6 | team6 | Team2ndGrade62024! | team6_data |

⚠️ **IMPORTANT**: Change password on first login!

## Initial Setup

### 1. Repository Setup
```bash
# Clone repository
git clone https://github.com/your-org/utah-tta.git
cd utah-tta
```

### 2. Python Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Unix/macOS:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration
```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your settings
# Required variables:
# - DATABASE_URL
# - AI_MODEL_PATH
# - API_KEY
```

### 4. Database Setup
```bash
# Start database container
docker-compose up -d db

# Run migrations
python manage.py migrate
```

### 5. Frontend Setup
```bash
cd src/frontend
npm install
```

## Development Server Access

### Connection Steps
```bash
# 1. Connect to UVU VPN
# 2. SSH to server
ssh teamX@d19559  # Replace X with your team number
# 3. Change password on first login
passwd
```

### Workspace Structure
```
/home/teamX/
├── data/              # Team data directory
├── models/            # AI model files
├── logs/             # Activity logs
└── backups/          # Local backups
```

### Storage Quotas
- 100GB per team
- Monitor usage with `df -h`
- Request increases if needed

## Security Guidelines

### Password Requirements
- Minimum 12 characters
- Mix of uppercase and lowercase letters
- Include numbers and special characters
- No dictionary words
- Change every 90 days

### Access Control
- Use only assigned team account
- No sharing of credentials
- Report suspicious activity
- Lock workstation when away

### Data Protection
- Keep sensitive data on server
- No unauthorized data transfers
- Follow backup procedures
- Encrypt sensitive files

## Development Tools

### Required Software
- Git (version control)
- Docker (containerization)
- Python 3.10+ (backend)
- Node.js 18+ (frontend)
- VSCode/PyCharm (IDE)

### Recommended Extensions
#### VSCode
- Python
- Docker
- ESLint
- Prettier
- GitLens

#### PyCharm
- Docker integration
- Python Scientific
- Database Tools

## Troubleshooting

### Common Issues
1. **VPN Connection Failed**
   - Check VPN credentials
   - Verify network connection
   - Contact UVU IT if persistent

2. **SSH Access Denied**
   - Verify username/password
   - Check VPN connection
   - Ensure within campus network

3. **Environment Setup**
   - Check Python version
   - Verify virtual environment
   - Confirm all dependencies

4. **Database Connection**
   - Check Docker status
   - Verify credentials
   - Confirm migrations

## Support Contacts

### Technical Support
- UVU IT Help Desk
  - Phone: (801) 863-8888
  - Email: helpdesk@uvu.edu
  - Hours: Mon-Fri 8am-5pm MST

### Project Support
- Project Manager
  - See [Project Management Contact](../roles/project_manager.md)
- System Administrator
  - See [Technical Contact](../roles/technical_lead.md)

## Additional Resources
- [Development Guide](../development/README.md)
- [Security Policies](../security/README.md)
- [Data Management](../data/README.md)
- [Contributing Guidelines](../contributing/README.md) 