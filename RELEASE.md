# Release Process for Semantica

This document outlines the steps to release a new version of the Semantica framework.

## 1. Versioning Policy

Semantica follows [Semantic Versioning (SemVer)](https://semver.org/).
- **MAJOR** version for incompatible API changes.
- **MINOR** version for functionality added in a backwards compatible manner.
- **PATCH** version for backwards compatible bug fixes.

## 2. Pre-release Checklist

Before releasing, ensure:
- [ ] All tests pass: `pytest`
- [ ] Documentation is up to date in `docs/` and `MkDocs` config.
- [ ] `CHANGELOG.md` is updated with the latest changes.
- [ ] Version is updated in:
  - `semantica/__init__.py`
  - `pyproject.toml`
  - `docs/citation.md` (BibTeX entry)

## 3. Release Steps

### Automated Release (Recommended)

The project uses GitHub Actions for automated releases to PyPI.

1. **Tag the commit**: Create a new git tag for the version (e.g., `v0.1.1`).
   ```bash
   git tag -a v0.1.1 -m "Release v0.1.1"
   git push origin v0.1.1
   ```
2. **GitHub Action**: The `Release` workflow will automatically trigger, build the package, create a GitHub Release, and publish to PyPI using Trusted Publishing.

### Manual Release

If you need to release manually:

1. **Build the package**:
   ```bash
   python -m build
   ```
2. **Verify the build**:
   ```bash
   twine check dist/*
   ```
3. **Upload to PyPI**:
   ```bash
   twine upload dist/*
   ```

## 4. Post-release

- Verify the new version is available on [PyPI](https://pypi.org/project/semantica/).
- Check the [GitHub Releases](https://github.com/your-org/semantica/releases) page for the new release notes.
