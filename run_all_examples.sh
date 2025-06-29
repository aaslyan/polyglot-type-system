#!/bin/bash

# Polyglot Type System - Example Runner
# This script runs all example scripts and reports their status

# Handle help argument first
if [ "$1" = "--help" ]; then
    echo "================================================================================"
    echo "               Polyglot Type System - Example Runner"
    echo "================================================================================"
    echo ""
    echo "This script runs all Polyglot Type System examples and reports results."
    echo "Examples are run in a logical order with timeouts to prevent hanging."
    echo ""
    echo "Usage:"
    echo "   ./run_all_examples.sh              # Run all examples"
    echo "   ./run_all_examples.sh --help       # Show this help"
    echo "   ./run_all_examples.sh --verbose    # Show detailed output"
    echo "   ./run_all_examples.sh --clean      # Clean up log files"
    echo ""
    echo "Notes:"
    echo "• Warnings are typically C++ parsing diagnostics and are expected"
    echo "• Each example has a 90-second timeout to prevent hanging"
    echo "• Log files are saved to /tmp/polyglot_example_*.log for debugging"
    echo "• The script will exit with code 0 if all examples pass, 1 if any fail"
    exit 0
fi

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

# List of example scripts to run (in recommended order)
examples=(
    "extract_types.py"
    "advanced_type_extraction.py"
    "cross_language_mapping.py"
    "binding_generator.py"
    "serialization_generator.py"
    "type_analysis_tools.py"
    "rag_search_examples.py"
    "function_chain_examples.py"
    "batch_processing.py"
    "cicd_integration.py"
)

# Counters
total=0
passed=0
failed=0
warnings=0

# Arrays to store results
passed_examples=()
failed_examples=()
warning_examples=()

echo "🚀 Running ${#examples[@]} example scripts..."
echo ""

# Function to run a single example
run_example() {
    local example="$1"
    local example_path="$EXAMPLES_DIR/$example"
    local log_file="/tmp/polyglot_example_$(basename "$example" .py).log"
    
    if [ ! -f "$example_path" ]; then
        printf "❌ %-35s (not found)\n" "$example"
        failed_examples+=("$example (not found)")
        ((failed++))
        return
    fi
    
    printf "⏳ Running %-35s" "$example..."
    
    # Run the example and capture output (with timeout to prevent hanging)
    if timeout 90 python "$example_path" > "$log_file" 2>&1; then
        # Check if there were warnings in the output
        if grep -q "WARNING\|warning" "$log_file" 2>/dev/null; then
            printf "\r⚠️  %-35s (warnings)\n" "$example"
            warning_examples+=("$example")
            ((warnings++))
        else
            printf "\r✅ %-35s\n" "$example"
            passed_examples+=("$example")
        fi
        ((passed++))
    else
        local exit_code=$?
        if [ $exit_code -eq 124 ]; then
            printf "\r⏰ %-35s (timeout)\n" "$example"
            failed_examples+=("$example (timeout)")
        else
            printf "\r❌ %-35s (error)\n" "$example"
            failed_examples+=("$example")
            
            # Show last few lines of error
            if [ -f "$log_file" ]; then
                echo "   Last error lines:"
                tail -3 "$log_file" | sed 's/^/   | /'
            fi
        fi
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

echo "📊 Execution Results:"
echo "   Total examples:     $total"
echo "   ✅ Passed:          $passed"
echo "   ⚠️  With warnings:   $warnings"
echo "   ❌ Failed:          $failed"
echo "   📈 Success rate:    ${success_rate}%"
echo ""

# Show categories
if [ ${#passed_examples[@]} -gt 0 ]; then
    echo "✅ Examples that passed cleanly:"
    for example in "${passed_examples[@]}"; do
        echo "   • $example"
    done
    echo ""
fi

if [ ${#warning_examples[@]} -gt 0 ]; then
    echo "⚠️  Examples that passed with warnings:"
    for example in "${warning_examples[@]}"; do
        echo "   • $example"
    done
    echo ""
fi

if [ ${#failed_examples[@]} -gt 0 ]; then
    echo "❌ Examples that failed:"
    for example in "${failed_examples[@]}"; do
        echo "   • $example"
    done
    echo ""
fi


# Usage instructions
echo "💡 Usage:"
echo "   ./run_all_examples.sh              # Run all examples"
echo "   ./run_all_examples.sh --help       # Show this help"
echo "   ./run_all_examples.sh --verbose    # Show detailed output"
echo ""
echo "📁 Log files are saved to /tmp/polyglot_example_*.log"
echo ""

if [ "$1" = "--verbose" ]; then
    echo "📄 Showing recent log files:"
    ls -la /tmp/polyglot_example_*.log 2>/dev/null || echo "   No log files found"
fi

# Clean up option
if [ "$1" = "--clean" ]; then
    echo "🧹 Cleaning up log files..."
    rm -f /tmp/polyglot_example_*.log
    echo "✅ Log files cleaned"
fi

# Final status
if [ $failed -eq 0 ]; then
    echo "🎉 All examples completed successfully!"
    if [ $warnings -gt 0 ]; then
        echo "   (Some examples had warnings, which is normal for C++ parsing)"
    fi
    exit 0
else
    echo "💥 $failed example(s) failed. Check logs for details."
    exit 1
fi