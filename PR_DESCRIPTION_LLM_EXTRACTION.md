# PR: Robust LLM Extraction - Auto-Chunking, Retries, and Diagnostics

## Description
This PR addresses several reliability issues in the `semantic_extract` module by introducing robust error handling, automatic text chunking, and enhanced LLM provider diagnostics.

## Related Issue
Fixes #149

## Changes

### 🧠 Semantic Extract Improvements
- **Auto-Chunking**: Added recursive text splitting for `extract_entities_llm`, `extract_relations_llm`, and `extract_triplets_llm`.
- **Deduplication**: Implemented intelligent merging of results across multiple text chunks.
- **Observability**: Switched to "Raise by Default" for processing errors (API keys, connectivity) with a `silent_fail` parameter for backward compatibility.

### 🤖 LLM Provider Enhancements
- **Robust JSON Parsing**: `BaseProvider` now handles markdown code blocks and inconsistent LLM formatting.
- **Built-in Retries**: `generate_structured` now features automatic 3-attempt retry logic with exponential backoff.
- **Groq Diagnostics**: Improved connection testing and error reporting in `GroqProvider`.

### 🐛 Bug Fixes
- **TripletExtractor**: Fixed shadowing of `validate_triplets` method by an internal attribute.
- **Imports**: Fixed incorrect `TextSplitter` import paths in helper functions.

## Verification Results
- **Unit Tests**: All tests in `tests/test_llm_extraction_fixes.py` passed (6/6).
- **Integration Tests**: Successfully verified with live Groq API using `llama-3.3-70b-versatile`, including successful chunked extraction of 90+ entities from long text.

## Documentation
- Updated `README.md` and `docs/modules.md`.
- Updated technical references in `docs/reference/`.
- Updated `CHANGELOG.md` with detailed entries.
- Added comprehensive examples in `semantic_extract_usage.md`.

---
*Verified and tested in a production-mirror environment.*
