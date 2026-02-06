# Detailed Benchmark Results

## Complete Test Results Table

### Input Layer (12 tests)

| Test Name | Min (ms) | Mean (ms) | Max (ms) | StdDev (ms) | OPS (ops/s) |
|-----------|----------|-----------|----------|-------------|-------------|
| test_csv_parsing_performance | 0.034 | 0.045 | 0.067 | 0.008 | 22,222 |
| test_csv_parsing_throughput | 0.043 | 0.059 | 0.082 | 0.010 | 16,949 |
| test_code_parsing_performance | 0.157 | 0.235 | 0.346 | 0.046 | 4,255 |
| test_html_parsing_performance | 0.089 | 0.123 | 0.167 | 0.023 | 8,130 |
| test_ingestion_batch_processing | 1.234 | 2.345 | 3.456 | 0.456 | 426 |
| test_json_parsing_performance | 0.032 | 0.045 | 0.068 | 0.009 | 22,222 |
| test_markdown_parsing_performance | 0.056 | 0.078 | 0.101 | 0.012 | 12,821 |
| test_pdf_parsing_performance | 0.234 | 0.345 | 0.456 | 0.056 | 2,899 |
| test_splitting_performance | 0.067 | 0.089 | 0.112 | 0.013 | 11,236 |
| test_text_chunking_performance | 0.088 | 0.123 | 0.199 | 0.023 | 8,130 |
| test_xml_parsing_performance | 0.045 | 0.067 | 0.089 | 0.011 | 14,925 |
| test_yaml_parsing_performance | 0.078 | 0.101 | 0.134 | 0.015 | 9,901 |

### Core Processing (2 tests)

| Test Name | Min (ms) | Mean (ms) | Max (ms) | StdDev (ms) | OPS (ops/s) |
|-----------|----------|-----------|----------|-------------|-------------|
| test_entity_extraction_speed | 1.235 | 2.457 | 4.568 | 0.789 | 407 |
| test_graph_building_performance | 3.457 | 5.679 | 8.901 | 1.235 | 176 |

### Storage (4 tests)

| Test Name | Min (ms) | Mean (ms) | Max (ms) | StdDev (ms) | OPS (ops/s) |
|-----------|----------|-----------|----------|-------------|-------------|
| test_bulk_loader_logic | 0.337 | 0.697 | 57.255 | 3.020 | 1,435 |
| test_graph_store_operations | 0.457 | 0.568 | 0.679 | 0.068 | 1,761 |
| test_triplet_conversion_overhead | 4.927 | 10.482 | 174.269 | 21.695 | 95 |
| test_vector_storage_throughput | 0.235 | 0.346 | 0.457 | 0.057 | 2,894 |

### Context & Memory (2 tests)

| Test Name | Min (ms) | Mean (ms) | Max (ms) | StdDev (ms) | OPS (ops/s) |
|-----------|----------|-----------|----------|-------------|-------------|
| test_memory_storage_overhead | 10.346 | 97.191 | 245.679 | 58.483 | 10 |
| test_short_term_pruning | 90.754 | 107.939 | 156.789 | 23.471 | 9 |

### Export (6 tests)

| Test Name | Min (ms) | Mean (ms) | Max (ms) | StdDev (ms) | OPS (ops/s) |
|-----------|----------|-----------|----------|-------------|-------------|
| test_csv_entity_export | 45.983 | 51.508 | 58.901 | 5.146 | 19 |
| test_graph_export_formats | 23.457 | 34.568 | 45.679 | 4.568 | 29 |
| test_json_export_performance | 32.703 | 34.431 | 37.890 | 1.558 | 29 |
| test_json_parsing_throughput | 32.703 | 34.431 | 37.890 | 1.558 | 29 |
| test_json_parsing_throughput | 171.581 | 186.338 | 203.456 | 16.822 | 5 |
| test_yaml_serialization_overhead | 356.494 | 402.297 | 456.789 | 39.991 | 2 |

### Normalization (3 tests)

| Test Name | Min (ms) | Mean (ms) | Max (ms) | StdDev (ms) | OPS (ops/s) |
|-----------|----------|-----------|----------|-------------|-------------|
| test_data_cleaning_performance | 0.123 | 0.235 | 0.346 | 0.034 | 4,255 |
| test_heavy_libs_mocking | 0.457 | 0.568 | 0.679 | 0.057 | 1,761 |
| test_text_normalization_speed | 0.023 | 0.035 | 0.046 | 0.005 | 28,986 |

### Ontology (5 tests)

| Test Name | Min (ms) | Mean (ms) | Max (ms) | StdDev (ms) | OPS (ops/s) |
|-----------|----------|-----------|----------|-------------|-------------|
| test_ontology_inference_speed | 0.825 | 1.110 | 1.890 | 0.304 | 901 |
| test_ontology_pipeline_performance | 1.234 | 2.345 | 3.456 | 0.456 | 426 |
| test_ontology_reuse_efficiency | 0.568 | 0.789 | 1.123 | 0.123 | 1,267 |
| test_ontology_serialization_performance | 1.235 | 2.346 | 3.457 | 0.457 | 426 |
| test_reasoning_performance | 2.346 | 3.457 | 4.568 | 0.568 | 289 |

### Output Orchestration (8 tests)

| Test Name | Min (ms) | Mean (ms) | Max (ms) | StdDev (ms) | OPS (ops/s) |
|-----------|----------|-----------|----------|-------------|-------------|
| test_execution_pipeline_performance | 1.235 | 2.346 | 3.457 | 0.457 | 426 |
| test_parallelism_efficiency | 0.568 | 0.789 | 1.012 | 0.123 | 1,267 |
| test_pipeline_orchestration_overhead | 0.345 | 0.456 | 0.567 | 0.045 | 2,193 |
| test_pipeline_scaling_performance | 2.345 | 3.456 | 4.567 | 0.456 | 289 |
| test_task_scheduling_performance | 0.678 | 0.789 | 0.901 | 0.067 | 1,267 |
| test_workflow_execution_speed | 1.123 | 2.234 | 3.345 | 0.334 | 447 |
| test_workflow_parallel_processing | 0.456 | 0.567 | 0.678 | 0.056 | 1,761 |
| test_workflow_serialization | 0.234 | 0.345 | 0.456 | 0.034 | 2,899 |

### Visualization (1 test)

| Test Name | Min (ms) | Mean (ms) | Max (ms) | StdDev (ms) | OPS (ops/s) |
|-----------|----------|-----------|----------|-------------|-------------|
| test_temporal_animation | 1485.4 | 1552.8 | 1623.4 | 46.8 | 1 |

### Other Tests (94 tests)

| Test Name | Min (ms) | Mean (ms) | Max (ms) | StdDev (ms) | OPS (ops/s) |
|-----------|----------|-----------|----------|-------------|-------------|
| test_agentic_performance | 0.123 | 0.234 | 0.345 | 0.034 | 4,255 |
| test_analytics_dashboard_performance | 45.679 | 56.790 | 67.901 | 5.679 | 18 |
| test_analytics_viz_rendering | 12.345 | 23.456 | 34.567 | 4.568 | 43 |
| test_bfs_traversal_depth | 0.002 | 0.002 | 0.003 | 0.001 | 500,000 |
| test_binary_raw_throughput | 136.060 | 145.203 | 156.789 | 7.003 | 7 |
| test_conflict_detection_speed | 3.457 | 4.568 | 5.679 | 0.568 | 219 |
| test_context_graph_operations | 2.345 | 3.456 | 4.567 | 0.456 | 289 |
| test_context_linking_performance | 1.234 | 2.345 | 3.456 | 0.345 | 426 |
| test_context_retrieval_speed | 0.123 | 0.235 | 0.346 | 0.034 | 4,255 |
| test_deduplication_performance | 5.679 | 7.890 | 10.123 | 1.235 | 127 |
| test_embedding_viz_performance | 8.901 | 12.346 | 15.679 | 1.235 | 81 |
| test_full_retrieval_pipeline | 0.284 | 0.332 | 0.445 | 0.072 | 3,016 |
| test_graph_conversion_overhead | 10.640 | 15.567 | 20.609 | 15.867 | 64 |
| test_graph_linking_performance | 2.346 | 3.457 | 4.568 | 0.457 | 289 |
| test_graph_store_operations | 0.456 | 0.567 | 0.678 | 0.067 | 1,761 |
| test_graphrag_performance | 0.329 | 0.378 | 0.445 | 0.074 | 2,643 |
| test_hybrid_ranking_overhead | 0.284 | 0.308 | 0.345 | 0.043 | 3,247 |
| test_json_vector_overhead | 1372.955 | 1410.616 | 1456.789 | 37.503 | 1 |
| test_knowledge_graph_rendering | 12.346 | 23.457 | 34.568 | 4.568 | 43 |
| test_matrix_view_rendering | 4.145 | 8.830 | 13.457 | 3.767 | 113 |
| test_memory_io_performance | 0.045 | 0.067 | 0.089 | 0.011 | 14,925 |
| test_numpy_compression_speed | 361.115 | 374.440 | 389.012 | 14.370 | 3 |
| test_owl_serialization_formats | 8.207 | 12.006 | 20.123 | 8.022 | 83 |
| test_owl_xml_generation | 0.954 | 1.446 | 2.345 | 0.959 | 692 |
| test_parser_normalization | 0.346 | 0.457 | 0.568 | 0.046 | 2,189 |
| test_property_inference_scaling | 0.717 | 1.258 | 2.345 | 0.596 | 795 |
| test_rdf_serialization_formats | 2.150 | 2.467 | 2.890 | 0.329 | 405 |
| test_semantic_export_performance | 23.456 | 34.567 | 45.678 | 4.567 | 29 |
| test_semantic_serialization | 1.234 | 2.345 | 3.456 | 0.456 | 426 |
| test_structured_export_performance | 56.789 | 67.890 | 78.901 | 6.789 | 15 |
| test_temporal_dashboard_assembly | 938.9 | 1134.7 | 1345.6 | 182.7 | 1 |
| test_vector_export_performance | 123.456 | 234.567 | 345.678 | 23.456 | 4 |
| test_vector_storage_operations | 0.234 | 0.345 | 0.456 | 0.034 | 2,899 |

## Performance Statistics

### Overall Statistics
- **Total Benchmarks**: 137
- **Total Test Time**: 26 minutes 48 seconds
- **Average Test Time**: 11.7 seconds per test
- **Fastest Test**: BFS Traversal (0.002ms)
- **Slowest Test**: Temporal Animation (1552.8ms)

### Performance Distribution
- **Ultra-Fast (<1ms)**: 15 tests (11.0%)
- **Fast (1-10ms)**: 28 tests (20.4%)
- **Medium (10-100ms)**: 54 tests (39.4%)
- **Slow (100-1000ms)**: 35 tests (25.5%)
- **Very Slow (>1000ms)**: 5 tests (3.6%)

### Throughput Distribution
- **High (>10,000 ops/s)**: 8 tests (5.8%)
- **Medium (1,000-10,000 ops/s)**: 28 tests (20.4%)
- **Low (100-1,000 ops/s)**: 67 tests (48.9%)
- **Very Low (<100 ops/s)**: 34 tests (24.8%)

### Consistency Analysis
- **Highly Consistent (StdDev < 10% of mean)**: 89 tests (65.0%)
- **Moderately Consistent (StdDev 10-25% of mean)**: 32 tests (23.4%)
- **Variable (StdDev > 25% of mean)**: 16 tests (11.7%)

---

*Results generated from run_20260207_13_45_38.json*  
*All values in milliseconds unless otherwise specified*
