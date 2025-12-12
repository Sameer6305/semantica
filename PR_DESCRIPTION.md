# PR: Enhance SeedDataManager with Robust CSV/JSON Support

## Summary
This PR significantly enhances the `SeedDataManager` class to provide more robust handling of CSV and JSON seed data files. It introduces delimiter auto-detection for CSV files and expands support for various JSON structural patterns. Additionally, the documentation has been updated to reflect these new capabilities.

## Key Changes

### 1. Robust CSV Loading (`load_from_csv`)
- **Custom Delimiter Support**: Added a `delimiter` argument to explicitly specify the CSV delimiter (e.g., `|`, `;`).
- **Auto-Detection**: Implemented `csv.Sniffer` to automatically detect delimiters when not provided, falling back to a comma (`,`) if detection fails.
- **Improved Parsing**: Ensures consistent parsing across different CSV formats.

### 2. Flexible JSON Loading (`load_from_json`)
- **Expanded Structure Support**: Now supports multiple top-level keys for list wrapping:
    - `records`
    - `data`
    - `entities`
- **Better Error Handling**: Added warning logs when an unsupported JSON structure is encountered (which is then loaded as a single record), aiding in debugging.

### 3. Documentation Updates
- Updated `semantica/seed/seed_usage.md` to include:
    - Examples of loading CSVs with custom delimiters.
    - Explanation of the new auto-detection algorithm.
    - Clarification on supported JSON structures and associated warnings.

## Testing Verification
- **CSV Tests**: Verified loading with comma, semicolon, and pipe delimiters.
- **JSON Tests**: Verified loading of lists, and dicts wrapped in `data`, `entities`, and `records`.
- **Edge Cases**: Verified behavior with empty files and malformed inputs.
- **Existing Tests**: All existing tests in `tests/test_seed_manager.py` passed successfully.

## Checklist
- [x] Code follows the project's coding standards.
- [x] Documentation has been updated.
- [x] All new and existing tests pass.
- [x] No breaking changes introduced.
