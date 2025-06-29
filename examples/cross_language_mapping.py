#!/usr/bin/env python3
"""
Cross-Language Type Mapping Examples

This example demonstrates mapping C++ types to equivalent types in other languages:
- C++ to Python mappings
- C++ to JavaScript/TypeScript mappings
- C++ to Java mappings
- Handling language-specific features
"""

from pathlib import Path
import sys
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

sys.path.append(str(Path(__file__).parent.parent))

from src.polyglot_type_system.extractors.cpp_extractor import CppTypeExtractor
from src.polyglot_type_system.models.type_models import PolyglotType

@dataclass
class LanguageMapping:
    """Represents a type mapping to a specific language"""
    language: str
    type_name: str
    import_statement: Optional[str] = None
    notes: Optional[str] = None

class CrossLanguageMapper:
    """Maps C++ types to equivalent types in other languages"""
    
    def __init__(self):
        # C++ to Python mappings
        self.cpp_to_python = {
            "int": LanguageMapping("python", "int"),
            "double": LanguageMapping("python", "float"),
            "float": LanguageMapping("python", "float"),
            "bool": LanguageMapping("python", "bool"),
            "std::string": LanguageMapping("python", "str"),
            "std::vector": LanguageMapping("python", "List", "from typing import List"),
            "std::unordered_map": LanguageMapping("python", "Dict", "from typing import Dict"),
            "std::map": LanguageMapping("python", "Dict", "from typing import Dict", "Ordered dict in Python 3.7+"),
            "std::set": LanguageMapping("python", "Set", "from typing import Set"),
            "std::unordered_set": LanguageMapping("python", "Set", "from typing import Set"),
            "std::optional": LanguageMapping("python", "Optional", "from typing import Optional"),
            "std::variant": LanguageMapping("python", "Union", "from typing import Union"),
            "std::shared_ptr": LanguageMapping("python", "object", notes="Python uses garbage collection"),
            "std::unique_ptr": LanguageMapping("python", "object", notes="Python uses garbage collection"),
            "std::function": LanguageMapping("python", "Callable", "from typing import Callable"),
        }
        
        # C++ to TypeScript mappings
        self.cpp_to_typescript = {
            "int": LanguageMapping("typescript", "number"),
            "double": LanguageMapping("typescript", "number"),
            "float": LanguageMapping("typescript", "number"),
            "bool": LanguageMapping("typescript", "boolean"),
            "std::string": LanguageMapping("typescript", "string"),
            "std::vector": LanguageMapping("typescript", "Array<T>"),
            "std::unordered_map": LanguageMapping("typescript", "Map<K, V>"),
            "std::map": LanguageMapping("typescript", "Map<K, V>"),
            "std::set": LanguageMapping("typescript", "Set<T>"),
            "std::unordered_set": LanguageMapping("typescript", "Set<T>"),
            "std::optional": LanguageMapping("typescript", "T | undefined"),
            "std::variant": LanguageMapping("typescript", "T1 | T2 | T3"),
            "std::shared_ptr": LanguageMapping("typescript", "T", notes="JavaScript uses garbage collection"),
            "std::unique_ptr": LanguageMapping("typescript", "T", notes="JavaScript uses garbage collection"),
            "std::function": LanguageMapping("typescript", "(...args: any[]) => ReturnType"),
        }
        
        # C++ to Java mappings
        self.cpp_to_java = {
            "int": LanguageMapping("java", "int"),
            "double": LanguageMapping("java", "double"),
            "float": LanguageMapping("java", "float"),
            "bool": LanguageMapping("java", "boolean"),
            "std::string": LanguageMapping("java", "String"),
            "std::vector": LanguageMapping("java", "ArrayList<T>", "import java.util.ArrayList;"),
            "std::unordered_map": LanguageMapping("java", "HashMap<K, V>", "import java.util.HashMap;"),
            "std::map": LanguageMapping("java", "TreeMap<K, V>", "import java.util.TreeMap;"),
            "std::set": LanguageMapping("java", "HashSet<T>", "import java.util.HashSet;"),
            "std::unordered_set": LanguageMapping("java", "HashSet<T>", "import java.util.HashSet;"),
            "std::optional": LanguageMapping("java", "Optional<T>", "import java.util.Optional;"),
            "std::variant": LanguageMapping("java", "Object", notes="Use sealed classes or union types"),
            "std::shared_ptr": LanguageMapping("java", "T", notes="Java uses garbage collection"),
            "std::unique_ptr": LanguageMapping("java", "T", notes="Java uses garbage collection"),
            "std::function": LanguageMapping("java", "Function<T, R>", "import java.util.function.Function;"),
        }
    
    def map_type(self, cpp_type: str, target_language: str) -> Optional[LanguageMapping]:
        """Map a C++ type to the target language"""
        mapping_dict = getattr(self, f"cpp_to_{target_language.lower()}", {})
        return mapping_dict.get(cpp_type)
    
    def map_template_type(self, base_type: str, template_args: List[str], target_language: str) -> Optional[str]:
        """Map a templated C++ type to the target language"""
        base_mapping = self.map_type(base_type, target_language)
        if not base_mapping:
            return None
        
        # Handle template parameter substitution
        mapped_type = base_mapping.type_name
        if "<T>" in mapped_type:
            if len(template_args) == 1:
                arg_mapping = self.map_type(template_args[0], target_language)
                arg_type = arg_mapping.type_name if arg_mapping else template_args[0]
                mapped_type = mapped_type.replace("<T>", f"<{arg_type}>")
        elif "<K, V>" in mapped_type:
            if len(template_args) == 2:
                key_mapping = self.map_type(template_args[0], target_language)
                val_mapping = self.map_type(template_args[1], target_language)
                key_type = key_mapping.type_name if key_mapping else template_args[0]
                val_type = val_mapping.type_name if val_mapping else template_args[1]
                mapped_type = mapped_type.replace("<K, V>", f"<{key_type}, {val_type}>")
        
        return mapped_type
    
    def generate_bindings(self, polyglot_type: PolyglotType, target_language: str) -> Dict[str, Any]:
        """Generate language bindings for a polyglot type"""
        bindings = {
            "original_type": polyglot_type.name,
            "target_language": target_language,
            "mapped_type": None,
            "imports": [],
            "notes": []
        }
        
        # Try direct mapping first
        mapping = self.map_type(polyglot_type.name, target_language)
        if mapping:
            bindings["mapped_type"] = mapping.type_name
            if mapping.import_statement:
                bindings["imports"].append(mapping.import_statement)
            if mapping.notes:
                bindings["notes"].append(mapping.notes)
        
        # Handle class/struct mappings
        if polyglot_type.type_kind == "class" or polyglot_type.type_kind == "struct":
            bindings["mapped_type"] = self._generate_class_binding(polyglot_type, target_language)
        
        return bindings
    
    def _generate_class_binding(self, polyglot_type: PolyglotType, target_language: str) -> str:
        """Generate class binding for different languages"""
        if target_language == "python":
            return self._generate_python_class(polyglot_type)
        elif target_language == "typescript":
            return self._generate_typescript_interface(polyglot_type)
        elif target_language == "java":
            return self._generate_java_class(polyglot_type)
        return polyglot_type.name
    
    def _generate_python_class(self, polyglot_type: PolyglotType) -> str:
        """Generate Python class definition"""
        methods = polyglot_type.metadata.get("methods", [])
        fields = polyglot_type.metadata.get("fields", [])
        
        class_def = f"class {polyglot_type.name}:\n"
        
        if fields:
            class_def += "    def __init__(self):\n"
            for field in fields:
                class_def += f"        self.{field['name']}: {self._map_field_type(field['type'], 'python')} = None\n"
        
        for method in methods:
            return_type = self._map_field_type(method.get("return_type", "void"), "python")
            params = method.get("parameters", [])
            param_str = ", ".join([f"{p['name']}: {self._map_field_type(p['type'], 'python')}" for p in params])
            class_def += f"    \n    def {method['name']}(self, {param_str}) -> {return_type}:\n        pass\n"
        
        return class_def
    
    def _generate_typescript_interface(self, polyglot_type: PolyglotType) -> str:
        """Generate TypeScript interface definition"""
        methods = polyglot_type.metadata.get("methods", [])
        fields = polyglot_type.metadata.get("fields", [])
        
        interface_def = f"interface {polyglot_type.name} {{\n"
        
        for field in fields:
            field_type = self._map_field_type(field['type'], 'typescript')
            interface_def += f"  {field['name']}: {field_type};\n"
        
        for method in methods:
            return_type = self._map_field_type(method.get("return_type", "void"), "typescript")
            params = method.get("parameters", [])
            param_str = ", ".join([f"{p['name']}: {self._map_field_type(p['type'], 'typescript')}" for p in params])
            interface_def += f"  {method['name']}({param_str}): {return_type};\n"
        
        interface_def += "}"
        return interface_def
    
    def _generate_java_class(self, polyglot_type: PolyglotType) -> str:
        """Generate Java class definition"""
        methods = polyglot_type.metadata.get("methods", [])
        fields = polyglot_type.metadata.get("fields", [])
        
        class_def = f"public class {polyglot_type.name} {{\n"
        
        for field in fields:
            field_type = self._map_field_type(field['type'], 'java')
            class_def += f"    private {field_type} {field['name']};\n"
        
        for method in methods:
            return_type = self._map_field_type(method.get("return_type", "void"), "java")
            params = method.get("parameters", [])
            param_str = ", ".join([f"{self._map_field_type(p['type'], 'java')} {p['name']}" for p in params])
            class_def += f"    \n    public {return_type} {method['name']}({param_str}) {{\n        // TODO: Implement\n    }}\n"
        
        class_def += "}"
        return class_def
    
    def _map_field_type(self, cpp_type: str, target_language: str) -> str:
        """Map a field type to target language"""
        mapping = self.map_type(cpp_type, target_language)
        return mapping.type_name if mapping else cpp_type

def main():
    """Demonstrate cross-language type mapping"""
    
    # Initialize components
    extractor = CppTypeExtractor()
    mapper = CrossLanguageMapper()
    
    # Extract types from vector_utils.hpp
    cpp_file = Path(__file__).parent / "vector_utils.hpp"
    if not cpp_file.exists():
        print(f"Creating sample C++ file: {cpp_file}")
        with open(cpp_file, 'w') as f:
            f.write("""
#pragma once
#include <vector>
#include <string>
#include <unordered_map>
#include <optional>

template<typename T>
class VectorWrapper {
    std::vector<T> data;
public:
    void add(const T& item);
    std::optional<T> get(size_t index) const;
    size_t size() const;
};

struct UserProfile {
    std::string username;
    int age;
    std::vector<std::string> interests;
    std::unordered_map<std::string, std::string> metadata;
};
""")
    
    extracted_types = extractor.extract_from_file(str(cpp_file))
    
    print("=" * 80)
    print("Cross-Language Type Mapping Examples")
    print("=" * 80)
    
    for polyglot_type in extracted_types:
        print(f"\n📄 Original C++ Type: {polyglot_type.name}")
        print("-" * 50)
        
        # Generate mappings for each target language
        for language in ["python", "typescript", "java"]:
            bindings = mapper.generate_bindings(polyglot_type, language)
            
            print(f"\n🔄 {language.title()} Mapping:")
            if bindings["imports"]:
                for imp in bindings["imports"]:
                    print(f"   Import: {imp}")
            
            if bindings["mapped_type"]:
                if "\n" in bindings["mapped_type"]:  # Multi-line class definition
                    print("   Generated Code:")
                    for line in bindings["mapped_type"].split("\n"):
                        print(f"   {line}")
                else:
                    print(f"   Type: {bindings['mapped_type']}")
            
            if bindings["notes"]:
                for note in bindings["notes"]:
                    print(f"   Note: {note}")
    
    # Demonstrate specific type mappings
    print("\n" + "=" * 80)
    print("Common C++ to Other Language Mappings")
    print("=" * 80)
    
    common_types = [
        "std::vector<int>",
        "std::unordered_map<std::string, int>",
        "std::optional<std::string>",
        "std::function<int(double)>",
        "std::shared_ptr<UserProfile>"
    ]
    
    for cpp_type in common_types:
        print(f"\n📋 {cpp_type}:")
        for language in ["python", "typescript", "java"]:
            # For complex types, we'd need better parsing, but this shows the concept
            base_type = cpp_type.split('<')[0] if '<' in cpp_type else cpp_type
            mapping = mapper.map_type(base_type, language)
            if mapping:
                print(f"   {language.title()}: {mapping.type_name}")
                if mapping.notes:
                    print(f"     Note: {mapping.notes}")

if __name__ == "__main__":
    main()