# Git Commands to Merge Feature Branch to Main

## Option 1: Merge via GitHub Web Interface (Recommended)

1. **Create PR manually:**
   - Visit: https://github.com/Hawksight-AI/semantica/pull/new/feature/utils-implementation
   - Click "Create pull request"
   - Fill in title and description
   - Click "Create pull request"

2. **Merge PR on GitHub:**
   - Go to the PR page
   - Click "Merge pull request"
   - Click "Confirm merge"
   - Delete branch if prompted

## Option 2: Merge Locally (Direct to Main)

### Commands to merge locally:

```powershell
# 1. Switch to main branch
git checkout main

# 2. Pull latest changes from remote
git pull origin main

# 3. Merge your feature branch
git merge feature/utils-implementation

# 4. Push merged changes to remote
git push origin main

# 5. Delete local feature branch
git branch -d feature/utils-implementation

# 6. Delete remote feature branch (optional)
git push origin --delete feature/utils-implementation
```

## Option 3: Install GitHub CLI (For Future Use)

If you want to use `gh` commands in the future:

```powershell
# Install via winget
winget install --id GitHub.cli

# Or download from: https://cli.github.com/

# Then authenticate
gh auth login
```

## Quick Merge Sequence (Copy-Paste)

```powershell
git checkout main
git pull origin main
git merge feature/utils-implementation
git push origin main
git branch -d feature/utils-implementation
```
