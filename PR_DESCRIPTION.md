# Enhanced Export Module Testing, Bug Fixes & Notebook Updates

## Summary
This PR significantly hardens the `semantica.export` module by adding comprehensive unit tests, fixing critical bugs in export wrappers and logic, and updating documentation and cookbooks to match current API signatures.

## Key Changes

### 1. Bug Fixes & Logic Improvements
- **`semantica/export/methods.py`**: 
  - Fixed `export_yaml(method="schema")` to correctly call `export_ontology_schema` and handle file writing (previously failed as the underlying method returns a string).
  - Added safeguards to all convenience functions (`export_rdf`, `export_json`, etc.) to prevent infinite recursion if the registry returns the wrapper function itself.
- **`semantica/kg/graph_builder.py`**: Fixed a critical bug where `ConflictDetector` was receiving the entire graph dictionary instead of the entity list.
- **`semantica/export/rdf_exporter.py`**: Fixed `export_to_rdf` to correctly return serialized data for all formats.

### 2. Comprehensive Testing (`tests/`)
- **`tests/test_export_module.py`**: A full suite of unit tests covering all 11 export classes (`JSON`, `CSV`, `RDF`, `GraphML`, `YAML`, `OWL`, `Vector`, `LPG`, etc.).
- **`tests/test_export_methods_wrapper.py`**: Added specific tests for convenience wrapper functions in `methods.py`, verifying the fix for schema export.
- **`tests/test_notebook_15_export.py`** & **`tests/test_notebooks_simulation.py`**: Simulation tests that replicate cookbook logic to ensure end-to-end functionality.

### 3. Documentation & Notebook Updates
- **`docs/reference/export.md`** & **`semantica/export/export_usage.md`**: Updated to correctly document `YAMLSchemaExporter.export_ontology_schema` instead of the deprecated `export` method.
- **Cookbooks** (`15_Export.ipynb`, `05_Multi_Format_Export.ipynb`):
  - Updated `GraphBuilder.build()` calls to pass combined lists (fixing API mismatch).
  - Corrected `YAMLSchemaExporter` usage.
  - Fixed `VectorExporter` data preparation.
  - Adjusted `CSVExporter` paths.

## Verification
All tests passed successfully:
```bash
$ pytest tests/test_export_module.py tests/test_notebooks_simulation.py tests/test_notebook_15_export.py tests/test_export_methods_wrapper.py
...
13 passed in 3.82s
```
