#!/usr/bin/env python3
"""
Advanced C++ Type Extraction Example

This example demonstrates extracting complex C++ types including:
- Variadic templates
- SFINAE patterns
- C++20 concepts
- Multiple inheritance
- Function pointers and std::function
- std::variant and std::optional
"""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.extractors.cpp_extractor import CppTypeExtractor
from src.converters.cpp_to_polyglot import CppToPolyglotConverter
from src.storage.rag_store import PolyglotRAGStore
from src.types.polyglot_types import PolyglotType

def main():
    # Initialize extractor and storage
    extractor = CppTypeExtractor()
    storage = PolyglotRAGStore("advanced_cpp_types_db")
    
    # Extract types from advanced C++ header
    cpp_file = Path(__file__).parent / "advanced_cpp_types.hpp"
    extracted_types = extractor.extract_from_file(str(cpp_file))
    
    # Convert C++ types to PolyglotType objects
    converter = CppToPolyglotConverter()
    polyglot_types = {}
    for name, cpp_type in extracted_types.items():
        poly_type = converter.convert(cpp_type)
        polyglot_types[name] = poly_type
    
    print("=" * 80)
    print(f"Extracted {len(polyglot_types)} types from {cpp_file.name}")
    print("=" * 80)
    
    # Categorize and display types
    variadic_templates = []
    concept_constrained = []
    multiple_inheritance = []
    function_types = []
    advanced_features = []
    
    for polyglot_type in polyglot_types.values():
        # Store in RAG
        storage.store_type(polyglot_type)
        
        # Categorize for display
        if "..." in str(polyglot_type.metadata.get("template_params", "")):
            variadic_templates.append(polyglot_type)
        elif polyglot_type.metadata.get("concepts"):
            concept_constrained.append(polyglot_type)
        elif len(polyglot_type.metadata.get("base_classes", [])) > 1:
            multiple_inheritance.append(polyglot_type)
        elif "function" in polyglot_type.kind.value.lower():
            function_types.append(polyglot_type)
        elif any(feature in str(polyglot_type.metadata) for feature in ["variant", "optional", "reference_wrapper"]):
            advanced_features.append(polyglot_type)
    
    # Display categorized results
    print("\n📦 Variadic Templates:")
    for t in variadic_templates:
        print(f"  - {t.canonical_name}: {t.metadata.get('template_params', 'N/A')}")
    
    print("\n🔧 Concept-Constrained Types:")
    for t in concept_constrained:
        print(f"  - {t.canonical_name}: requires {t.metadata.get('concepts', 'N/A')}")
    
    print("\n🔗 Multiple Inheritance:")
    for t in multiple_inheritance:
        bases = ", ".join(t.metadata.get("base_classes", []))
        print(f"  - {t.canonical_name}: inherits from {bases}")
    
    print("\n🎯 Function Types:")
    for t in function_types:
        print(f"  - {t.canonical_name}: {t.metadata.get('signature', 'N/A')}")
    
    print("\n✨ Advanced Features (variant, optional, etc.):")
    for t in advanced_features:
        print(f"  - {t.canonical_name}")
    
    # Demonstrate SFINAE pattern detection
    print("\n🔍 SFINAE Pattern Detection:")
    sfinae_types = storage.search_by_metadata({"pattern": "SFINAE"})
    for t in sfinae_types:
        print(f"  - {t.canonical_name}: {t.metadata.get('description', 'N/A')}")
    
    # Search for types with specific requirements
    print("\n🔎 Types with Size Constraints:")
    size_constrained = storage.search_by_content("size()")
    for t in size_constrained:
        print(f"  - {t.canonical_name}")

if __name__ == "__main__":
    main()