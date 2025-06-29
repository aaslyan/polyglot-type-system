#!/usr/bin/env python3
"""
Language Binding Generator Example

This example demonstrates generating language bindings automatically:
- Python bindings (pybind11 style)
- Node.js bindings (N-API)
- JSON schema generation
- OpenAPI specifications
- Protocol buffer definitions
"""

from pathlib import Path
import sys
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from textwrap import indent

sys.path.append(str(Path(__file__).parent.parent))

from src.extractors.cpp_extractor import CppTypeExtractor
from src.converters.cpp_to_polyglot import CppToPolyglotConverter
from src.storage.rag_store import PolyglotRAGStore
from src.types.polyglot_types import PolyglotType

@dataclass
class BindingConfig:
    """Configuration for binding generation"""
    module_name: str
    target_language: str
    include_private: bool = False
    generate_docs: bool = True
    output_dir: Optional[Path] = None

class PythonBindingGenerator:
    """Generates Python bindings using pybind11"""
    
    def __init__(self, config: BindingConfig):
        self.config = config
    
    def generate_bindings(self, types: List[PolyglotType]) -> str:
        """Generate pybind11 bindings"""
        lines = [
            "#include <pybind11/pybind11.h>",
            "#include <pybind11/stl.h>",
            "#include <pybind11/functional.h>",
            "",
            "// Include original headers",
        ]
        
        # Add includes for all source files
        source_files = set()
        for t in types:
            source_file = t.metadata.get('source_file')
            if source_file:
                source_files.add(Path(source_file).name)
        
        for source_file in sorted(source_files):
            lines.append(f'#include "{source_file}"')
        
        lines.extend([
            "",
            "namespace py = pybind11;",
            "",
            f"PYBIND11_MODULE({self.config.module_name}, m) {{",
            f'    m.doc() = "{self.config.module_name} bindings";',
            ""
        ])
        
        # Generate bindings for each type
        for type_obj in types:
            if type_obj.kind in ['class', 'struct']:
                class_binding = self._generate_class_binding(type_obj)
                lines.append(indent(class_binding, "    "))
                lines.append("")
        
        lines.append("}")
        
        return "\n".join(lines)
    
    def _generate_class_binding(self, type_obj: PolyglotType) -> str:
        """Generate binding for a single class"""
        class_name = type_obj.canonical_name
        binding_name = class_name.replace("::", "_")
        
        lines = [
            f'py::class_<{class_name}>(m, "{binding_name}")'
        ]
        
        if self.config.generate_docs and type_obj.metadata.get('description'):
            lines[0] += f', "{type_obj.metadata["description"]}"'
        
        # Add constructor
        has_default_constructor = True  # Assume for simplicity
        if has_default_constructor:
            lines.append('    .def(py::init<>())')
        
        # Add methods
        methods = type_obj.metadata.get('methods', [])
        for method in methods:
            if method['name'] in ['constructor', 'destructor'] or method['name'].startswith('~'):
                continue
            
            method_binding = self._generate_method_binding(method)
            lines.append(f'    .def("{method["name"]}", {method_binding})')
        
        # Add fields (properties)
        fields = type_obj.metadata.get('fields', [])
        for field in fields:
            if not self.config.include_private and field.get('access') == 'private':
                continue
            
            field_name = field['name']
            lines.append(f'    .def_readwrite("{field_name}", &{class_name}::{field_name})')
        
        lines[-1] += ";"
        return "\n".join(lines)
    
    def _generate_method_binding(self, method: Dict[str, Any]) -> str:
        """Generate method binding"""
        class_name = method.get('class_name', '')
        method_name = method['name']
        
        # Simple method binding
        if class_name:
            return f"&{class_name}::{method_name}"
        else:
            return f"&{method_name}"

class NodeJSBindingGenerator:
    """Generates Node.js bindings using N-API"""
    
    def __init__(self, config: BindingConfig):
        self.config = config
    
    def generate_bindings(self, types: List[PolyglotType]) -> str:
        """Generate N-API bindings"""
        lines = [
            "#include <napi.h>",
            "",
            "// Include original headers",
        ]
        
        # Add includes
        source_files = set()
        for t in types:
            source_file = t.metadata.get('source_file')
            if source_file:
                source_files.add(Path(source_file).name)
        
        for source_file in sorted(source_files):
            lines.append(f'#include "{source_file}"')
        
        lines.extend([
            "",
            "using namespace Napi;",
            ""
        ])
        
        # Generate wrapper classes
        for type_obj in types:
            if type_obj.kind in ['class', 'struct']:
                wrapper_class = self._generate_wrapper_class(type_obj)
                lines.append(wrapper_class)
                lines.append("")
        
        # Generate module initialization
        lines.extend([
            "Object Init(Env env, Object exports) {",
        ])
        
        for type_obj in types:
            if type_obj.kind in ['class', 'struct']:
                wrapper_name = f"{type_obj.canonical_name.replace('::', '')}Wrapper"
                export_name = type_obj.canonical_name.split("::")[-1]
                lines.append(f'    exports.Set("{export_name}", {wrapper_name}::GetClass(env));')
        
        lines.extend([
            "    return exports;",
            "}",
            "",
            f'NODE_API_MODULE({self.config.module_name}, Init)'
        ])
        
        return "\n".join(lines)
    
    def _generate_wrapper_class(self, type_obj: PolyglotType) -> str:
        """Generate N-API wrapper class"""
        class_name = type_obj.canonical_name
        wrapper_name = f"{class_name.replace('::', '')}Wrapper"
        
        lines = [
            f"class {wrapper_name} : public ObjectWrap<{wrapper_name}> {{",
            "public:",
            f"    static Object Init(Env env, Object exports);",
            f"    static Function GetClass(Env env);",
            f"    {wrapper_name}(const CallbackInfo& info);",
            "",
            "private:",
            f"    {class_name} instance_;",
        ]
        
        # Add method declarations
        methods = type_obj.metadata.get('methods', [])
        for method in methods:
            if method['name'] not in ['constructor', 'destructor']:
                lines.append(f"    Value {method['name']}(const CallbackInfo& info);")
        
        lines.extend([
            "};",
            "",
            # Implementation
            f"Object {wrapper_name}::Init(Env env, Object exports) {{",
            f'    Function func = DefineClass(env, "{class_name}", {{',
        ])
        
        # Add method bindings
        for method in methods:
            if method['name'] not in ['constructor', 'destructor']:
                lines.append(f'        InstanceMethod("{method["name"]}", &{wrapper_name}::{method["name"]}),')
        
        lines.extend([
            "    });",
            "    return func;",
            "}",
            "",
            f"Function {wrapper_name}::GetClass(Env env) {{",
            "    return Init(env, Object::New(env));",
            "}"
        ])
        
        return "\n".join(lines)

class JSONSchemaGenerator:
    """Generates JSON schemas from C++ types"""
    
    def __init__(self, config: BindingConfig):
        self.config = config
    
    def generate_schema(self, types: List[PolyglotType]) -> Dict[str, Any]:
        """Generate JSON schema"""
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": f"{self.config.module_name} Types",
            "type": "object",
            "definitions": {}
        }
        
        for type_obj in types:
            if type_obj.kind in ['class', 'struct']:
                type_schema = self._generate_type_schema(type_obj)
                schema["definitions"][type_obj.canonical_name] = type_schema
        
        return schema
    
    def _generate_type_schema(self, type_obj: PolyglotType) -> Dict[str, Any]:
        """Generate schema for a single type"""
        schema = {
            "type": "object",
            "properties": {},
            "required": []
        }
        
        if type_obj.metadata.get('description'):
            schema["description"] = type_obj.metadata['description']
        
        # Add fields as properties
        fields = type_obj.metadata.get('fields', [])
        for field in fields:
            field_name = field['name']
            field_type = field['type']
            
            property_schema = self._cpp_type_to_json_schema(field_type)
            schema["properties"][field_name] = property_schema
            
            # Assume all fields are required for now
            schema["required"].append(field_name)
        
        return schema
    
    def _cpp_type_to_json_schema(self, cpp_type: str) -> Dict[str, Any]:
        """Convert C++ type to JSON schema type"""
        type_mapping = {
            "int": {"type": "integer"},
            "double": {"type": "number"},
            "float": {"type": "number"},
            "bool": {"type": "boolean"},
            "std::string": {"type": "string"},
            "string": {"type": "string"},
        }
        
        if cpp_type in type_mapping:
            return type_mapping[cpp_type]
        elif cpp_type.startswith("std::vector<"):
            # Extract element type
            element_type = cpp_type[12:-1]  # Remove "std::vector<" and ">"
            return {
                "type": "array",
                "items": self._cpp_type_to_json_schema(element_type)
            }
        elif cpp_type.startswith("std::optional<"):
            # Extract element type
            element_type = cpp_type[14:-1]  # Remove "std::optional<" and ">"
            return {
                "anyOf": [
                    self._cpp_type_to_json_schema(element_type),
                    {"type": "null"}
                ]
            }
        else:
            # Assume it's a reference to another type
            return {"$ref": f"#/definitions/{cpp_type}"}

class OpenAPIGenerator:
    """Generates OpenAPI specifications from C++ types"""
    
    def __init__(self, config: BindingConfig):
        self.config = config
    
    def generate_spec(self, types: List[PolyglotType]) -> Dict[str, Any]:
        """Generate OpenAPI specification"""
        spec = {
            "openapi": "3.0.0",
            "info": {
                "title": f"{self.config.module_name} API",
                "version": "1.0.0",
                "description": f"API generated from {self.config.module_name} C++ types"
            },
            "components": {
                "schemas": {}
            },
            "paths": {}
        }
        
        # Generate schemas from types
        for type_obj in types:
            if type_obj.kind in ['class', 'struct']:
                schema = self._generate_openapi_schema(type_obj)
                spec["components"]["schemas"][type_obj.canonical_name] = schema
                
                # Generate basic CRUD paths
                paths = self._generate_crud_paths(type_obj)
                spec["paths"].update(paths)
        
        return spec
    
    def _generate_openapi_schema(self, type_obj: PolyglotType) -> Dict[str, Any]:
        """Generate OpenAPI schema for a type"""
        schema = {
            "type": "object",
            "properties": {}
        }
        
        if type_obj.metadata.get('description'):
            schema["description"] = type_obj.metadata['description']
        
        # Add fields
        fields = type_obj.metadata.get('fields', [])
        for field in fields:
            field_name = field['name']
            field_type = field['type']
            
            property_schema = self._cpp_type_to_openapi_type(field_type)
            schema["properties"][field_name] = property_schema
        
        return schema
    
    def _cpp_type_to_openapi_type(self, cpp_type: str) -> Dict[str, Any]:
        """Convert C++ type to OpenAPI type"""
        type_mapping = {
            "int": {"type": "integer", "format": "int32"},
            "long": {"type": "integer", "format": "int64"},
            "double": {"type": "number", "format": "double"},
            "float": {"type": "number", "format": "float"},
            "bool": {"type": "boolean"},
            "std::string": {"type": "string"},
            "string": {"type": "string"},
        }
        
        if cpp_type in type_mapping:
            return type_mapping[cpp_type]
        elif cpp_type.startswith("std::vector<"):
            element_type = cpp_type[12:-1]
            return {
                "type": "array",
                "items": self._cpp_type_to_openapi_type(element_type)
            }
        else:
            return {"$ref": f"#/components/schemas/{cpp_type}"}
    
    def _generate_crud_paths(self, type_obj: PolyglotType) -> Dict[str, Any]:
        """Generate basic CRUD paths for a type"""
        type_name = type_obj.canonical_name.lower()
        schema_ref = f"#/components/schemas/{type_obj.canonical_name}"
        
        return {
            f"/{type_name}": {
                "post": {
                    "summary": f"Create {type_obj.canonical_name}",
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {"$ref": schema_ref}
                            }
                        }
                    },
                    "responses": {
                        "201": {
                            "description": f"{type_obj.canonical_name} created",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": schema_ref}
                                }
                            }
                        }
                    }
                },
                "get": {
                    "summary": f"List {type_obj.canonical_name} objects",
                    "responses": {
                        "200": {
                            "description": f"List of {type_obj.canonical_name} objects",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "array",
                                        "items": {"$ref": schema_ref}
                                    }
                                }
                            }
                        }
                    }
                }
            },
            f"/{type_name}/{{id}}": {
                "get": {
                    "summary": f"Get {type_obj.canonical_name} by ID",
                    "parameters": [
                        {
                            "name": "id",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "integer"}
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": f"{type_obj.canonical_name} object",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": schema_ref}
                                }
                            }
                        }
                    }
                }
            }
        }

class BindingOrchestrator:
    """Orchestrates the generation of multiple binding types"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_all_bindings(self, types: List[PolyglotType], module_name: str):
        """Generate all types of bindings"""
        print(f"🔧 Generating bindings for {len(types)} types...")
        
        # Python bindings
        python_config = BindingConfig(module_name, "python")
        python_gen = PythonBindingGenerator(python_config)
        python_bindings = python_gen.generate_bindings(types)
        
        python_file = self.output_dir / f"{module_name}_python.cpp"
        with open(python_file, 'w') as f:
            f.write(python_bindings)
        print(f"✅ Python bindings: {python_file}")
        
        # Node.js bindings
        nodejs_config = BindingConfig(module_name, "nodejs")
        nodejs_gen = NodeJSBindingGenerator(nodejs_config)
        nodejs_bindings = nodejs_gen.generate_bindings(types)
        
        nodejs_file = self.output_dir / f"{module_name}_nodejs.cpp"
        with open(nodejs_file, 'w') as f:
            f.write(nodejs_bindings)
        print(f"✅ Node.js bindings: {nodejs_file}")
        
        # JSON Schema
        json_config = BindingConfig(module_name, "json")
        json_gen = JSONSchemaGenerator(json_config)
        json_schema = json_gen.generate_schema(types)
        
        json_file = self.output_dir / f"{module_name}_schema.json"
        with open(json_file, 'w') as f:
            import json
            json.dump(json_schema, f, indent=2)
        print(f"✅ JSON Schema: {json_file}")
        
        # OpenAPI
        openapi_config = BindingConfig(module_name, "openapi")
        openapi_gen = OpenAPIGenerator(openapi_config)
        openapi_spec = openapi_gen.generate_spec(types)
        
        openapi_file = self.output_dir / f"{module_name}_openapi.yaml"
        with open(openapi_file, 'w') as f:
            import yaml
            yaml.dump(openapi_spec, f, default_flow_style=False)
        print(f"✅ OpenAPI spec: {openapi_file}")

def create_sample_cpp_for_bindings():
    """Create sample C++ code for binding generation"""
    sample_code = '''
#pragma once
#include <string>
#include <vector>
#include <memory>

/// A simple geometric point
struct Point {
    double x;  ///< X coordinate
    double y;  ///< Y coordinate
    
    Point(double x = 0, double y = 0);
    double distance_to(const Point& other) const;
};

/// User profile data
class UserProfile {
private:
    std::string username_;
    int age_;
    std::vector<std::string> interests_;

public:
    UserProfile(const std::string& username, int age);
    
    // Getters
    const std::string& get_username() const { return username_; }
    int get_age() const { return age_; }
    const std::vector<std::string>& get_interests() const { return interests_; }
    
    // Setters
    void set_age(int age) { age_ = age; }
    void add_interest(const std::string& interest);
    
    // Utility methods
    bool has_interest(const std::string& interest) const;
    size_t interest_count() const { return interests_.size(); }
};

/// Simple calculator interface
class Calculator {
public:
    virtual ~Calculator() = default;
    virtual double add(double a, double b) = 0;
    virtual double subtract(double a, double b) = 0;
    virtual double multiply(double a, double b) = 0;
    virtual double divide(double a, double b) = 0;
};

/// Basic calculator implementation
class BasicCalculator : public Calculator {
public:
    double add(double a, double b) override { return a + b; }
    double subtract(double a, double b) override { return a - b; }
    double multiply(double a, double b) override { return a * b; }
    double divide(double a, double b) override { return a / b; }
};
'''
    
    bindings_header = Path(__file__).parent / "bindings_sample.hpp"
    with open(bindings_header, 'w') as f:
        f.write(sample_code)
    
    return bindings_header

def main():
    """Demonstrate binding generation"""
    
    print("=" * 80)
    print("Language Binding Generator Examples")
    print("=" * 80)
    
    # Create sample C++ code
    print("📝 Creating sample C++ code...")
    sample_file = create_sample_cpp_for_bindings()
    
    # Extract types
    print("🔍 Extracting types...")
    extractor = CppTypeExtractor()
    extracted_types = extractor.extract_from_file(str(sample_file))
    
    # Convert C++ types to PolyglotType objects
    converter = CppToPolyglotConverter()
    polyglot_types = []
    for name, cpp_type in extracted_types.items():
        poly_type = converter.convert(cpp_type)
        polyglot_types.append(poly_type)
    
    print(f"📊 Extracted {len(polyglot_types)} types:")
    for t in polyglot_types:
        print(f"  - {t.canonical_name} ({t.kind})")
    
    # Generate bindings
    output_dir = Path(__file__).parent / "generated_bindings"
    orchestrator = BindingOrchestrator(output_dir)
    
    print(f"\n🔧 Generating bindings in {output_dir}...")
    orchestrator.generate_all_bindings(polyglot_types, "sample_module")
    
    print(f"\n📁 Generated files:")
    for file_path in output_dir.iterdir():
        if file_path.is_file():
            print(f"  - {file_path.name} ({file_path.stat().st_size} bytes)")
    
    print("\n✅ Binding generation complete!")

if __name__ == "__main__":
    main()