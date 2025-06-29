#!/usr/bin/env python3
"""
RAG Search Examples

This example demonstrates advanced search capabilities of the RAG storage system:
- Semantic similarity searches
- Metadata-based filtering
- Complex queries
- Type compatibility searches
- Finding types by usage patterns
"""

from pathlib import Path
import sys
from typing import List, Dict, Any
import json

sys.path.append(str(Path(__file__).parent.parent))

from src.polyglot_type_system.extractors.cpp_extractor import CppTypeExtractor
from src.polyglot_type_system.storage.rag_storage import RagTypeStorage
from src.polyglot_type_system.models.type_models import PolyglotType

class AdvancedTypeSearcher:
    """Advanced search operations on the RAG storage"""
    
    def __init__(self, storage: RagTypeStorage):
        self.storage = storage
    
    def find_similar_types(self, query_type: PolyglotType, similarity_threshold: float = 0.7) -> List[PolyglotType]:
        """Find types similar to the query type"""
        # This would use vector similarity in a real implementation
        similar_types = []
        all_types = self.storage.get_all_types()
        
        for stored_type in all_types:
            similarity = self._calculate_structural_similarity(query_type, stored_type)
            if similarity >= similarity_threshold:
                similar_types.append((stored_type, similarity))
        
        # Sort by similarity (highest first)
        similar_types.sort(key=lambda x: x[1], reverse=True)
        return [t[0] for t in similar_types]
    
    def find_types_with_interface(self, interface_methods: List[str]) -> List[PolyglotType]:
        """Find types that implement a specific interface (have specific methods)"""
        matching_types = []
        all_types = self.storage.get_all_types()
        
        for stored_type in all_types:
            type_methods = [m.get('name', '') for m in stored_type.metadata.get('methods', [])]
            if all(method in type_methods for method in interface_methods):
                matching_types.append(stored_type)
        
        return matching_types
    
    def find_container_types(self) -> List[PolyglotType]:
        """Find all container-like types"""
        container_indicators = ['size', 'begin', 'end', 'iterator', 'empty', 'clear']
        
        container_types = []
        all_types = self.storage.get_all_types()
        
        for stored_type in all_types:
            type_methods = [m.get('name', '') for m in stored_type.metadata.get('methods', [])]
            if any(indicator in type_methods for indicator in container_indicators):
                container_types.append(stored_type)
        
        return container_types
    
    def find_template_specializations(self, base_template: str) -> List[PolyglotType]:
        """Find all specializations of a template"""
        specializations = []
        all_types = self.storage.get_all_types()
        
        for stored_type in all_types:
            if stored_type.metadata.get('template_base') == base_template:
                specializations.append(stored_type)
        
        return specializations
    
    def find_types_by_complexity(self, min_complexity: int = 0, max_complexity: int = 100) -> List[PolyglotType]:
        """Find types based on complexity score"""
        complex_types = []
        all_types = self.storage.get_all_types()
        
        for stored_type in all_types:
            complexity = self._calculate_complexity(stored_type)
            if min_complexity <= complexity <= max_complexity:
                complex_types.append((stored_type, complexity))
        
        # Sort by complexity
        complex_types.sort(key=lambda x: x[1], reverse=True)
        return [t[0] for t in complex_types]
    
    def find_types_with_dependencies(self, dependency_type: str) -> List[PolyglotType]:
        """Find types that depend on a specific type"""
        dependent_types = []
        all_types = self.storage.get_all_types()
        
        for stored_type in all_types:
            dependencies = stored_type.metadata.get('dependencies', [])
            if dependency_type in dependencies:
                dependent_types.append(stored_type)
        
        return dependent_types
    
    def search_by_usage_pattern(self, pattern: str) -> List[PolyglotType]:
        """Search for types matching a usage pattern"""
        pattern_map = {
            'RAII': ['constructor', 'destructor', 'resource'],
            'Factory': ['create', 'make', 'build'],
            'Observer': ['subscribe', 'notify', 'update'],
            'Singleton': ['instance', 'get_instance'],
            'Iterator': ['next', 'has_next', 'current'],
            'Builder': ['build', 'with_', 'set_']
        }
        
        if pattern not in pattern_map:
            return []
        
        keywords = pattern_map[pattern]
        matching_types = []
        all_types = self.storage.get_all_types()
        
        for stored_type in all_types:
            type_methods = [m.get('name', '').lower() for m in stored_type.metadata.get('methods', [])]
            type_fields = [f.get('name', '').lower() for f in stored_type.metadata.get('fields', [])]
            all_names = type_methods + type_fields + [stored_type.name.lower()]
            
            if any(keyword in ' '.join(all_names) for keyword in keywords):
                matching_types.append(stored_type)
        
        return matching_types
    
    def find_polymorphic_types(self) -> List[PolyglotType]:
        """Find types that use polymorphism (virtual functions)"""
        polymorphic_types = []
        all_types = self.storage.get_all_types()
        
        for stored_type in all_types:
            methods = stored_type.metadata.get('methods', [])
            if any(m.get('is_virtual', False) for m in methods):
                polymorphic_types.append(stored_type)
        
        return polymorphic_types
    
    def _calculate_structural_similarity(self, type1: PolyglotType, type2: PolyglotType) -> float:
        """Calculate structural similarity between two types"""
        if type1.name == type2.name:
            return 1.0
        
        score = 0.0
        
        # Compare type kinds
        if type1.type_kind == type2.type_kind:
            score += 0.3
        
        # Compare number of methods
        methods1 = len(type1.metadata.get('methods', []))
        methods2 = len(type2.metadata.get('methods', []))
        if methods1 > 0 and methods2 > 0:
            method_similarity = min(methods1, methods2) / max(methods1, methods2)
            score += 0.3 * method_similarity
        
        # Compare number of fields
        fields1 = len(type1.metadata.get('fields', []))
        fields2 = len(type2.metadata.get('fields', []))
        if fields1 > 0 and fields2 > 0:
            field_similarity = min(fields1, fields2) / max(fields1, fields2)
            score += 0.2 * field_similarity
        
        # Compare base classes
        bases1 = set(type1.metadata.get('base_classes', []))
        bases2 = set(type2.metadata.get('base_classes', []))
        if bases1 or bases2:
            base_similarity = len(bases1 & bases2) / len(bases1 | bases2)
            score += 0.2 * base_similarity
        
        return min(score, 1.0)
    
    def _calculate_complexity(self, polyglot_type: PolyglotType) -> int:
        """Calculate complexity score for a type"""
        complexity = 0
        
        # Base complexity
        complexity += 1
        
        # Template parameters add complexity
        template_params = polyglot_type.metadata.get('template_params', [])
        complexity += len(template_params) * 2
        
        # Methods add complexity
        methods = polyglot_type.metadata.get('methods', [])
        complexity += len(methods)
        
        # Virtual methods add extra complexity
        virtual_methods = [m for m in methods if m.get('is_virtual', False)]
        complexity += len(virtual_methods)
        
        # Inheritance adds complexity
        base_classes = polyglot_type.metadata.get('base_classes', [])
        complexity += len(base_classes) * 2
        
        # Nested types add complexity
        nested_types = polyglot_type.metadata.get('nested_types', [])
        complexity += len(nested_types) * 3
        
        return complexity

def create_sample_types() -> List[PolyglotType]:
    """Create sample types for demonstration"""
    sample_types = [
        PolyglotType(
            name="Logger",
            type_kind="class",
            metadata={
                "methods": [
                    {"name": "log", "parameters": [{"name": "message", "type": "std::string"}]},
                    {"name": "set_level", "parameters": [{"name": "level", "type": "int"}]},
                    {"name": "get_instance", "return_type": "Logger*", "is_static": True}
                ],
                "pattern": "Singleton"
            }
        ),
        PolyglotType(
            name="EventHandler",
            type_kind="class",
            metadata={
                "methods": [
                    {"name": "subscribe", "parameters": [{"name": "callback", "type": "std::function<void()>"}]},
                    {"name": "notify", "parameters": []},
                    {"name": "unsubscribe", "parameters": [{"name": "id", "type": "int"}]}
                ],
                "pattern": "Observer"
            }
        ),
        PolyglotType(
            name="VectorIterator",
            type_kind="class",
            metadata={
                "methods": [
                    {"name": "next", "return_type": "T&"},
                    {"name": "has_next", "return_type": "bool"},
                    {"name": "current", "return_type": "T&"}
                ],
                "template_params": ["T"],
                "pattern": "Iterator"
            }
        ),
        PolyglotType(
            name="Shape",
            type_kind="class",
            metadata={
                "methods": [
                    {"name": "area", "return_type": "double", "is_virtual": True},
                    {"name": "perimeter", "return_type": "double", "is_virtual": True}
                ]
            }
        ),
        PolyglotType(
            name="Circle",
            type_kind="class",
            metadata={
                "base_classes": ["Shape"],
                "methods": [
                    {"name": "area", "return_type": "double", "is_virtual": True},
                    {"name": "perimeter", "return_type": "double", "is_virtual": True}
                ],
                "fields": [
                    {"name": "radius", "type": "double"}
                ]
            }
        )
    ]
    return sample_types

def main():
    """Demonstrate RAG search capabilities"""
    
    print("=" * 80)
    print("RAG Storage Advanced Search Examples")
    print("=" * 80)
    
    # Initialize storage and searcher
    storage = RagTypeStorage("rag_search_demo")
    searcher = AdvancedTypeSearcher(storage)
    
    # Create and add sample types
    sample_types = create_sample_types()
    for sample_type in sample_types:
        storage.add_type(sample_type)
    
    print(f"\n📊 Added {len(sample_types)} sample types to storage")
    
    # Demonstrate different search capabilities
    print("\n🔍 1. Finding Types with Specific Interface:")
    iterable_types = searcher.find_types_with_interface(["next", "has_next"])
    for t in iterable_types:
        print(f"   - {t.name}: implements Iterator pattern")
    
    print("\n🔍 2. Finding Types by Usage Pattern:")
    patterns = ["Singleton", "Observer", "Iterator"]
    for pattern in patterns:
        pattern_types = searcher.search_by_usage_pattern(pattern)
        if pattern_types:
            print(f"   {pattern} Pattern:")
            for t in pattern_types:
                print(f"     - {t.name}")
    
    print("\n🔍 3. Finding Polymorphic Types:")
    polymorphic_types = searcher.find_polymorphic_types()
    for t in polymorphic_types:
        virtual_methods = [m['name'] for m in t.metadata.get('methods', []) if m.get('is_virtual')]
        print(f"   - {t.name}: virtual methods = {virtual_methods}")
    
    print("\n🔍 4. Types by Complexity:")
    complex_types = searcher.find_types_by_complexity(min_complexity=3)
    for t in complex_types:
        complexity = searcher._calculate_complexity(t)
        print(f"   - {t.name}: complexity score = {complexity}")
    
    print("\n🔍 5. Finding Similar Types:")
    # Find types similar to Circle
    circle_type = next((t for t in sample_types if t.name == "Circle"), None)
    if circle_type:
        similar_types = searcher.find_similar_types(circle_type, similarity_threshold=0.3)
        print(f"   Types similar to {circle_type.name}:")
        for t in similar_types:
            if t.name != circle_type.name:  # Don't show the type itself
                similarity = searcher._calculate_structural_similarity(circle_type, t)
                print(f"     - {t.name}: similarity = {similarity:.2f}")
    
    # Demonstrate semantic search with extracted C++ types
    print("\n" + "=" * 80)
    print("Semantic Search with Extracted C++ Types")
    print("=" * 80)
    
    # Extract types from existing example
    extractor = CppTypeExtractor()
    cpp_file = Path(__file__).parent / "vector_utils.hpp"
    
    if cpp_file.exists():
        extracted_types = extractor.extract_from_file(str(cpp_file))
        
        # Add extracted types to storage
        for extracted_type in extracted_types:
            storage.add_type(extracted_type)
        
        print(f"\n📄 Added {len(extracted_types)} extracted C++ types")
        
        # Search for container types
        container_types = searcher.find_container_types()
        print(f"\n📦 Container Types Found:")
        for t in container_types:
            container_methods = [m['name'] for m in t.metadata.get('methods', []) 
                               if any(indicator in m['name'].lower() 
                                     for indicator in ['size', 'begin', 'end', 'empty'])]
            if container_methods:
                print(f"   - {t.name}: {container_methods}")
    
    # Demonstrate metadata-based search
    print("\n🔍 Metadata-Based Search Examples:")
    
    # Search by template parameters
    template_types = [t for t in storage.get_all_types() 
                     if t.metadata.get('template_params')]
    print(f"\n📋 Template Types ({len(template_types)} found):")
    for t in template_types:
        params = t.metadata.get('template_params', [])
        print(f"   - {t.name}<{', '.join(params)}>")
    
    # Search by inheritance
    derived_types = [t for t in storage.get_all_types() 
                    if t.metadata.get('base_classes')]
    print(f"\n🌳 Types with Inheritance ({len(derived_types)} found):")
    for t in derived_types:
        bases = t.metadata.get('base_classes', [])
        print(f"   - {t.name} extends {', '.join(bases)}")

if __name__ == "__main__":
    main()