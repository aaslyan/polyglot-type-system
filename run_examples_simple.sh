#!/bin/bash

# Polyglot Type System - Simple Example Runner
# This script runs all example scripts and reports their status

echo "================================================================================"
echo "               Polyglot Type System - Example Runner"
echo "================================================================================"
echo ""

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="$SCRIPT_DIR/examples"

# Check if examples directory exists
if [ ! -d "$EXAMPLES_DIR" ]; then
    echo "❌ Examples directory not found: $EXAMPLES_DIR"
    exit 1
fi

# List of example scripts to run
examples=(
    "extract_types.py"
    "advanced_type_extraction.py"
    "cross_language_mapping.py"
    "batch_processing.py"
    "binding_generator.py"
    "cicd_integration.py"
    "serialization_generator.py"
    "type_analysis_tools.py"
    "function_chain_examples.py"
    "rag_search_examples.py"
)

# Counters
total=0
passed=0
failed=0

# Arrays to store results
passed_examples=()
failed_examples=()

echo "🚀 Running ${#examples[@]} example scripts..."
echo ""

# Function to run a single example
run_example() {
    local example="$1"
    local example_path="$EXAMPLES_DIR/$example"
    
    if [ ! -f "$example_path" ]; then
        echo "❌ $example (not found)"
        failed_examples+=("$example (not found)")
        ((failed++))
        return
    fi
    
    echo -n "⏳ Running $example... "
    
    # Run the example and capture output
    if python "$example_path" > /dev/null 2>&1; then
        echo "✅"
        passed_examples+=("$example")
        ((passed++))
    else
        echo "❌"
        failed_examples+=("$example")
        ((failed++))
    fi
    
    ((total++))
}

# Run all examples
for example in "${examples[@]}"; do
    run_example "$example"
done

echo ""
echo "================================================================================"
echo "                              SUMMARY"
echo "================================================================================"
echo ""

# Calculate success rate
if [ $total -gt 0 ]; then
    success_rate=$(( (passed * 100) / total ))
else
    success_rate=0
fi

echo "📊 Results:"
echo "   Total examples: $total"
echo "   ✅ Passed: $passed"
echo "   ❌ Failed: $failed"
echo "   Success rate: ${success_rate}%"
echo ""

# Show detailed results
if [ ${#passed_examples[@]} -gt 0 ]; then
    echo "✅ Successful examples:"
    for example in "${passed_examples[@]}"; do
        echo "   • $example"
    done
    echo ""
fi

if [ ${#failed_examples[@]} -gt 0 ]; then
    echo "❌ Failed examples:"
    for example in "${failed_examples[@]}"; do
        echo "   • $example"
    done
    echo ""
fi

# Exit with appropriate code
if [ $failed -eq 0 ]; then
    echo "🎉 All examples completed successfully!"
    exit 0
else
    echo "💥 Some examples failed."
    exit 1
fi