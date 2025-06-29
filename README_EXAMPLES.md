# Polyglot Type System - Examples

This directory contains comprehensive examples demonstrating the capabilities of the Polyglot Type System.

## Quick Start

Run all examples at once:
```bash
./run_all_examples.sh
```

## Available Scripts

### Main Runner Scripts
- **`run_all_examples.sh`** - Comprehensive runner with colored output, timeouts, and detailed reporting
- **`run_examples_simple.sh`** - Simple runner with basic output
- **`run_examples.sh`** - Advanced runner (may have terminal compatibility issues)

### Usage
```bash
# Run all examples
./run_all_examples.sh

# Show help
./run_all_examples.sh --help

# Show verbose output with log file details
./run_all_examples.sh --verbose

# Clean up log files
./run_all_examples.sh --clean
```

## Individual Examples

### 1. Basic Type Extraction
- **`extract_types.py`** - Basic C++ type extraction and RAG storage demo

### 2. Advanced Type Analysis
- **`advanced_type_extraction.py`** - Advanced C++ features analysis (templates, concepts, SFINAE)
- **`type_analysis_tools.py`** - Comprehensive type analysis tools (complexity, dependencies, ABI compatibility)

### 3. Cross-Language Integration
- **`cross_language_mapping.py`** - Cross-language type mapping (C++ ↔ Python/TypeScript/Java)
- **`binding_generator.py`** - Language binding generation (Python/Node.js bindings)
- **`serialization_generator.py`** - Serialization code generation (JSON, Protocol Buffers, MessagePack)

### 4. Advanced Features
- **`rag_search_examples.py`** - RAG-based semantic search capabilities
- **`function_chain_examples.py`** - Function chain synthesis and analysis
- **`batch_processing.py`** - Batch processing with incremental updates

### 5. Development Workflow
- **`cicd_integration.py`** - CI/CD integration examples (pre-commit hooks, compatibility checking)

## Expected Output

All examples should run successfully. Some may show warnings related to C++ parsing diagnostics, which is normal behavior for the clang parser when encountering:
- Missing header files
- C++20 features (concepts, requires clauses)
- Platform-specific code

## Log Files

When using `run_all_examples.sh`, detailed logs are saved to:
```
/tmp/polyglot_example_*.log
```

These logs contain the full output of each example and can be useful for debugging.

## Requirements

- Python 3.7+
- All dependencies from the main project requirements
- C++ compiler toolchain (for parsing C++ code)

## Troubleshooting

If an example fails:
1. Check the log file in `/tmp/polyglot_example_*.log`
2. Run the specific example individually: `python examples/example_name.py`
3. Ensure all dependencies are installed
4. Check that the clang library is properly configured

## Example Features Demonstrated

- **Type Extraction**: Parse C++ code and extract type information
- **RAG Storage**: Store and search types using semantic similarity
- **Cross-Language Mapping**: Map types between different programming languages
- **Code Generation**: Generate bindings and serialization code
- **Analysis Tools**: Analyze code complexity, dependencies, and compatibility
- **Batch Processing**: Process multiple files efficiently
- **CI/CD Integration**: Integrate type checking into development workflows