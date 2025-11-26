# Semantica Release Script
# Usage: .\release.ps1 -Version "0.0.5"

param(
    [Parameter(Mandatory=$true)]
    [string]$Version
)

Write-Host "🚀 Releasing Semantica v$Version" -ForegroundColor Cyan
Write-Host ""

# Step 1: Update version in pyproject.toml
Write-Host "📝 Updating pyproject.toml..." -ForegroundColor Yellow
(Get-Content pyproject.toml) -replace 'version = "[^"]*"', "version = `"$Version`"" | Set-Content pyproject.toml

# Step 2: Update version in __init__.py
Write-Host "📝 Updating semantica/__init__.py..." -ForegroundColor Yellow
(Get-Content semantica/__init__.py) -replace '__version__ = "[^"]*"', "__version__ = `"$Version`"" | Set-Content semantica/__init__.py

# Step 3: Update CHANGELOG.md
Write-Host "📝 Updating CHANGELOG.md..." -ForegroundColor Yellow
$date = Get-Date -Format "yyyy-MM-dd"
$changelog = Get-Content CHANGELOG.md -Raw
$newEntry = "## [Unreleased]`n`n## [$Version] - $date`n`n### Changed`n- Version $Version release`n"
$changelog = $changelog -replace "## \[Unreleased\]", $newEntry
Set-Content CHANGELOG.md $changelog

# Step 4: Commit changes
Write-Host "📦 Committing changes..." -ForegroundColor Yellow
git add pyproject.toml semantica/__init__.py CHANGELOG.md
git commit -m "Release v$Version"

# Step 5: Create and push tag
Write-Host "🏷️  Creating tag v$Version..." -ForegroundColor Yellow
git tag "v$Version"

# Step 6: Push to GitHub
Write-Host "⬆️  Pushing to GitHub..." -ForegroundColor Yellow
git push origin main
git push origin "v$Version"

Write-Host ""
Write-Host "✅ Release v$Version triggered!" -ForegroundColor Green
Write-Host "📍 Monitor deployment at: https://github.com/Hawksight-AI/semantica/actions" -ForegroundColor Cyan
Write-Host "📦 Package will be at: https://pypi.org/project/semantica/$Version/" -ForegroundColor Cyan

