# Reliability Issues in LLM Extraction Methods

### Description
The current implementation of LLM-based extraction in the `semantic_extract` module has several reliability and observability gaps that affect production stability for real-world datasets.

### Observed Issues
1. **Silent Failures on External Errors**: Extraction methods return empty lists when API keys are missing or connectivity fails, hiding underlying implementation or configuration errors.
2. **Context Window Overflows**: Lack of automatic chunking/splitting for texts that exceed LLM token limits, leading to provider-side errors or data loss.
3. **JSON Parsing Fragility**: High failure rate when LLMs include markdown code blocks or conversational filler around the requested JSON payload.
4. **Method Shadowing in TripletExtractor**: The `validate_triplets` method is inaccessible because it is shadowed by a boolean attribute of the same name.

### Impact
Inconsistent extraction reliability and poor observability when processing long, complex, or noisy unstructured data in production environments.
