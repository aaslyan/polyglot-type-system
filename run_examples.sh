#!/bin/bash

# Polyglot Type System - Example Runner
# This script runs all example scripts and reports their status

set -e  # Exit on any error (but we'll handle errors gracefully)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="$SCRIPT_DIR/examples"

echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}               Polyglot Type System - Example Runner${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""

# Check if examples directory exists
if [ ! -d "$EXAMPLES_DIR" ]; then
    echo -e "${RED}❌ Examples directory not found: $EXAMPLES_DIR${NC}"
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
warnings=0

# Arrays to store results
passed_examples=()
failed_examples=()
warning_examples=()

echo -e "${CYAN}🚀 Running ${#examples[@]} example scripts...${NC}"
echo ""

# Function to run a single example
run_example() {
    local example="$1"
    local example_path="$EXAMPLES_DIR/$example"
    local log_file="/tmp/polyglot_example_$(basename "$example" .py).log"
    
    if [ ! -f "$example_path" ]; then
        echo -e "${RED}❌ $example (not found)${NC}"
        failed_examples+=("$example (not found)")
        ((failed++))
        return
    fi
    
    printf "${YELLOW}⏳ Running %-30s${NC}" "$example..."
    
    # Run the example and capture output
    if timeout 60 python "$example_path" > "$log_file" 2>&1; then
        # Check if there were warnings in the output
        if grep -q "WARNING\|warning" "$log_file" 2>/dev/null; then
            printf "\r${YELLOW}⚠️  %-30s (passed with warnings)${NC}\n" "$example"
            warning_examples+=("$example")
            ((warnings++))
        else
            printf "\r${GREEN}✅ %-30s${NC}\n" "$example"
            passed_examples+=("$example")
        fi
        ((passed++))
    else
        printf "\r${RED}❌ %-30s (failed)${NC}\n" "$example"
        failed_examples+=("$example")
        ((failed++))
        
        # Show error details
        echo -e "${RED}   Error details:${NC}"
        if [ -f "$log_file" ]; then
            tail -5 "$log_file" | sed 's/^/   /'
        else
            echo "   No log file generated"
        fi
        echo ""
    fi
    
    ((total++))
}

# Run all examples
for example in "${examples[@]}"; do
    run_example "$example"
done

echo ""
echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}                              SUMMARY${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""

# Calculate success rate
if [ $total -gt 0 ]; then
    success_rate=$(( (passed * 100) / total ))
else
    success_rate=0
fi

echo -e "${CYAN}📊 Results:${NC}"
echo -e "   Total examples: $total"
echo -e "   ${GREEN}✅ Passed: $passed${NC}"
echo -e "   ${YELLOW}⚠️  Passed with warnings: $warnings${NC}"
echo -e "   ${RED}❌ Failed: $failed${NC}"
echo -e "   Success rate: ${success_rate}%"
echo ""

# Show detailed results
if [ ${#passed_examples[@]} -gt 0 ]; then
    echo -e "${GREEN}✅ Successful examples:${NC}"
    for example in "${passed_examples[@]}"; do
        echo -e "   • $example"
    done
    echo ""
fi

if [ ${#warning_examples[@]} -gt 0 ]; then
    echo -e "${YELLOW}⚠️  Examples with warnings:${NC}"
    for example in "${warning_examples[@]}"; do
        echo -e "   • $example"
    done
    echo ""
fi

if [ ${#failed_examples[@]} -gt 0 ]; then
    echo -e "${RED}❌ Failed examples:${NC}"
    for example in "${failed_examples[@]}"; do
        echo -e "   • $example"
    done
    echo ""
fi

# Additional information
echo -e "${PURPLE}📝 Notes:${NC}"
echo -e "   • Log files are stored in /tmp/polyglot_example_*.log"
echo -e "   • Warnings are typically C++ parsing diagnostics and are expected"
echo -e "   • For detailed output, check individual log files"
echo ""

# Clean up old log files (optional)
if [ "$1" = "--clean-logs" ]; then
    echo -e "${CYAN}🧹 Cleaning up log files...${NC}"
    rm -f /tmp/polyglot_example_*.log
    echo -e "${GREEN}✅ Log files cleaned${NC}"
    echo ""
fi

# Exit with appropriate code
if [ $failed -eq 0 ]; then
    echo -e "${GREEN}🎉 All examples completed successfully!${NC}"
    exit 0
else
    echo -e "${RED}💥 Some examples failed. Check the logs for details.${NC}"
    exit 1
fi