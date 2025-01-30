# Getting Started with UTAH-TTA

This guide will help you set up and begin working with the Utah Elementary Teacher Training Assistant project.

## Prerequisites

### Development Environment
- Python 3.10 or higher
- Git
- Docker and Docker Compose
- Node.js 18+ (for frontend development)
- VSCode or PyCharm (recommended)

### Access Requirements
- UVU VPN access
- GitHub account with repository access
- Development server credentials

## Initial Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-org/utah-tta.git
   cd utah-tta
   ```

2. **Set Up Python Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

4. **Start Development Server**
   ```bash
   python manage.py runserver
   ```

## Next Steps

- Read the [Development Guide](../development/README.md)
- Review [Content Creation Guide](../content/README.md)
- Check [Architecture Overview](../ARCHITECTURE.md)

## Common Issues

See our [Troubleshooting Guide](../setup/troubleshooting.md) for solutions to common setup issues. 