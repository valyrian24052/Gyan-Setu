# GitHub Workflow Guide

## Project Manager's GitHub Workflow

### Project Board Management
- Create and maintain project boards for each sprint
- Use GitHub Projects with the following columns:
  - Backlog
  - Sprint Planning
  - In Progress
  - Review/QA
  - Done

### Issue Management
```yaml
# Example issue template for tasks
name: Task
description: Create a new task
labels: ["task"]
body:
  - type: dropdown
    id: role
    attributes:
      label: Assigned Role
      options:
        - Database Developer
        - AI Developer
        - UI/UX Developer
        - Research/Documentation
    validations:
      required: true
  - type: textarea
    id: description
    attributes:
      label: Task Description
      description: Detailed description of the task
    validations:
      required: true
  - type: input
    id: estimated-time
    attributes:
      label: Estimated Time
      description: Estimated time to complete this task
    validations:
      required: true
```

### Progress Tracking
- Monitor pull requests and code reviews
- Track milestone progress
- Generate weekly reports using GitHub's insights
- Use GitHub Actions for automated checks

### Team Management
- Assign reviewers for pull requests
- Monitor contribution graphs
- Track issue resolution time
- Use GitHub labels for task categorization

## Product Owner's GitHub Workflow

### Backlog Management
- Maintain user stories in GitHub Issues
- Prioritize issues using labels
- Link issues to milestones
- Track feature requests and feedback

### Release Management
- Create and maintain GitHub Releases
- Write release notes
- Tag versions appropriately
- Track feature completion

### Documentation Review
- Review documentation PRs
- Maintain wiki pages
- Update roadmap in GitHub Projects
- Track documentation coverage

## Role-Specific Guidelines

### Database Developer
- Branch naming: `db/feature-name`
- Required labels: `database`, `schema`, `migration`
- PR template focus: Schema changes, migrations
- Documentation: Update `docs/database/`

### AI Developer
- Branch naming: `ai/feature-name`
- Required labels: `ai`, `model`, `training`
- PR template focus: Model changes, training scripts
- Documentation: Update `docs/ai/`

### UI/UX Developer
- Branch naming: `ui/feature-name`
- Required labels: `frontend`, `ui`, `ux`
- PR template focus: UI changes, accessibility
- Documentation: Update `docs/frontend/`

### Research/Documentation
- Branch naming: `docs/topic-name`
- Required labels: `documentation`, `research`
- PR template focus: Documentation updates
- Documentation: Update relevant sections

## Contribution Tracking

### Metrics to Monitor
```markdown
1. Code Contributions
   - Number of commits
   - Lines of code changed
   - Pull requests merged
   - Code review participation

2. Issue Management
   - Issues created/resolved
   - Average resolution time
   - Comments and discussions
   - Documentation updates

3. Project Milestones
   - Tasks completed vs. assigned
   - Sprint completion rate
   - Documentation coverage
   - Test coverage
```

### GitHub Actions Workflow
```yaml
name: Contribution Metrics
on:
  schedule:
    - cron: '0 0 * * 0'  # Weekly report
jobs:
  generate-report:
    runs-on: ubuntu-latest
    steps:
      - name: Generate Contribution Report
        uses: actions/github-script@v6
        with:
          script: |
            // Generate weekly contribution metrics
            const metrics = await github.rest.repos.getContributorsStats({
              owner: context.repo.owner,
              repo: context.repo.repo
            });
            // Process and format metrics
            // Send to project manager
```

## Best Practices

### Branch Protection Rules
- Require pull request reviews
- Enforce status checks
- Require linear history
- Protect main branch

### Code Review Process
1. Create descriptive PR
2. Request relevant reviewers
3. Address feedback
4. Update documentation
5. Merge when approved

### Communication Guidelines
- Use issue comments for technical discussions
- Link PRs to issues
- Update project boards regularly
- Document decisions in PR descriptions 