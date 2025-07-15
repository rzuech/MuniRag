# Git Workflow Guide for MuniRag

## Local Git Ignore Setup (Complete!)

I've configured `.git/info/exclude` to ignore Claude-specific files. This file:
- ✅ Stays local to your machine (never pushed)
- ✅ Works like .gitignore but remains private
- ✅ Already excludes all Claude documentation files

## Working with Local Git (No Cloud Push)

### 1. Check What's Staged
```bash
git status
```

### 2. Commit Locally Only
```bash
# Add specific files (avoid using 'git add .')
git add src/*.py
git add requirements.txt
git add docker-compose.yml

# Commit locally
git commit -m "Add parallel PDF processing"
```

### 3. Prevent Accidental Push
```bash
# Remove remote temporarily
git remote rename origin origin-backup

# Or set push URL to nowhere
git remote set-url --push origin no_push
```

### 4. View Local Commits
```bash
# See commits not pushed
git log origin/main..HEAD

# See what would be pushed
git log origin/main..HEAD --oneline
```

## Best Practices for Claude + Git

### Files to NEVER Commit
- `CLAUDE.md` - Project context (excluded ✓)
- `*_SESSION_*.md` - Session files (excluded ✓)
- `*_RESEARCH.md` - Research docs (excluded ✓)
- `.env` - Secrets (check it's in .gitignore)

### Files to COMMIT
- `src/*.py` - Source code
- `requirements.txt` - Dependencies
- `docker-compose.yml` - Config
- `README.md` - Public docs

### Workflow Commands
```bash
# 1. Before starting work
git status                    # Check clean state
git pull origin main         # Get latest (if connected)

# 2. During work with Claude
# (Claude creates various .md files - all auto-ignored)

# 3. After work - commit locally
git add src/ requirements.txt docker-compose.yml
git commit -m "Your message"

# 4. Check what would be pushed
git diff origin/main --name-only

# 5. When ready to push (later)
git remote rename origin-backup origin  # Restore remote
git push origin main
```

## Quick Status Check
```bash
# See ignored files
git status --ignored

# Verify Claude files are ignored
git check-ignore CLAUDE.md
# Should output: CLAUDE.md

# See all excluded patterns
cat .git/info/exclude
```

## Emergency: If You Accidentally Committed Claude Files

```bash
# Remove from history (before pushing)
git rm --cached CLAUDE.md SESSION_RESUME_PROMPT.md
git commit --amend

# If already in history
git filter-branch --force --index-filter \
  'git rm --cached --ignore-unmatch *CLAUDE*.md' \
  --prune-empty --tag-name-filter cat -- --all
```

## Recommended .gitignore Additions

Make sure your `.gitignore` includes:
```
# Environment
.env
.env.local

# Data
/data/
/logs/
/models/
/qdrant_storage/
/chroma_data/

# Python
__pycache__/
*.pyc
.pytest_cache/

# IDE
.vscode/
.idea/
```

This setup gives you complete control over what goes to Git cloud vs stays local!