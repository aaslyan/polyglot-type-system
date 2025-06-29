#!/usr/bin/env python3
"""
Serialization Code Generator Example

This example demonstrates generating serialization code for C++ types:
- JSON serialization (nlohmann/json)
- Protocol Buffers
- MessagePack
- Custom binary format
- XML serialization
"""

from pathlib import Path
import sys
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from textwrap import indent

sys.path.append(str(Path(__file__).parent.parent))

from src.polyglot_type_system.extractors.cpp_extractor import CppTypeExtractor
from src.polyglot_type_system.models.type_models import PolyglotType

@dataclass
class SerializationConfig:
    """Configuration for serialization generation"""
    format_type: str  # json, protobuf, messagepack, binary, xml
    namespace: str = "serialization"
    include_versioning: bool = True
    generate_validation: bool = True

class JSONSerializationGenerator:
    """Generates JSON serialization using nlohmann/json"""
    
    def __init__(self, config: SerializationConfig):
        self.config = config
    
    def generate_serialization(self, types: List[PolyglotType]) -> str:
        """Generate JSON serialization code"""
        lines = [
            "#pragma once",
            "#include <nlohmann/json.hpp>",
            "",
            "// Include original headers",
        ]
        
        # Add includes for source files
        source_files = set()
        for t in types:
            source_file = t.metadata.get('source_file')
            if source_file:
                source_files.add(Path(source_file).name)
        
        for source_file in sorted(source_files):
            lines.append(f'#include "{source_file}"')
        
        lines.extend([
            "",
            "using json = nlohmann::json;",
            "",
            f"namespace {self.config.namespace} {{",
            ""
        ])
        
        # Generate serialization functions for each type
        for type_obj in types:
            if type_obj.type_kind in ['class', 'struct']:
                serialization_code = self._generate_type_serialization(type_obj)
                lines.append(indent(serialization_code, "    "))
                lines.append("")
        
        lines.append("}")
        
        return "\n".join(lines)
    
    def _generate_type_serialization(self, type_obj: PolyglotType) -> str:
        """Generate serialization for a single type"""
        class_name = type_obj.name
        
        lines = [
            f"// Serialization for {class_name}",
            f"void to_json(json& j, const {class_name}& obj) {{",
        ]
        
        if self.config.include_versioning:
            lines.append('    j["_version"] = 1;')
        
        # Add fields
        fields = type_obj.metadata.get('fields', [])
        for field in fields:
            field_name = field['name']
            # Use getter method if available, otherwise direct field access
            getter_method = f"get_{field_name}"
            methods = type_obj.metadata.get('methods', [])
            method_names = [m['name'] for m in methods]
            
            if getter_method in method_names:
                lines.append(f'    j["{field_name}"] = obj.{getter_method}();')
            else:
                lines.append(f'    j["{field_name}"] = obj.{field_name};')
        
        lines.extend([
            "}",
            "",
            f"void from_json(const json& j, {class_name}& obj) {{",
        ])
        
        if self.config.generate_validation:
            lines.extend([
                "    // Version check",
                '    if (j.contains("_version") && j["_version"].get<int>() != 1) {',
                '        throw std::runtime_error("Unsupported version");',
                "    }",
                ""
            ])
        
        # Add deserialization for fields
        for field in fields:
            field_name = field['name']
            field_type = field['type']
            
            setter_method = f"set_{field_name}"
            methods = type_obj.metadata.get('methods', [])
            method_names = [m['name'] for m in methods]
            
            if self.config.generate_validation:
                lines.append(f'    if (j.contains("{field_name}")) {{')
                if setter_method in method_names:
                    lines.append(f'        obj.{setter_method}(j["{field_name}"].get<{self._get_json_type(field_type)}>());')
                else:
                    lines.append(f'        obj.{field_name} = j["{field_name}"].get<{self._get_json_type(field_type)}>();')
                lines.append("    }")
            else:
                if setter_method in method_names:
                    lines.append(f'    obj.{setter_method}(j["{field_name}"].get<{self._get_json_type(field_type)}>());')
                else:
                    lines.append(f'    obj.{field_name} = j["{field_name}"].get<{self._get_json_type(field_type)}>();')
        
        lines.append("}")
        
        return "\n".join(lines)
    
    def _get_json_type(self, cpp_type: str) -> str:
        """Get appropriate type for JSON deserialization"""
        return cpp_type  # nlohmann/json handles type conversion automatically

class ProtobufGenerator:
    """Generates Protocol Buffer definitions"""
    
    def __init__(self, config: SerializationConfig):
        self.config = config
    
    def generate_proto(self, types: List[PolyglotType]) -> str:
        """Generate .proto file"""
        lines = [
            'syntax = "proto3";',
            "",
            f'package {self.config.namespace};',
            "",
            "// Generated from C++ types",
            ""
        ]
        
        field_counter = 1
        
        for type_obj in types:
            if type_obj.type_kind in ['class', 'struct']:
                message_def = self._generate_message(type_obj, field_counter)
                lines.append(message_def)
                lines.append("")
                field_counter += 100  # Leave space between messages
        
        return "\n".join(lines)
    
    def _generate_message(self, type_obj: PolyglotType, start_field_num: int) -> str:
        """Generate protobuf message for a type"""
        class_name = type_obj.name.replace("::", "_")
        
        lines = [
            f"message {class_name} {{",
        ]
        
        fields = type_obj.metadata.get('fields', [])
        field_num = start_field_num
        
        for field in fields:
            field_name = field['name']
            field_type = field['type']
            proto_type = self._cpp_type_to_proto_type(field_type)
            
            lines.append(f"  {proto_type} {field_name} = {field_num};")
            field_num += 1
        
        lines.append("}")
        
        return "\n".join(lines)
    
    def _cpp_type_to_proto_type(self, cpp_type: str) -> str:
        """Convert C++ type to Protocol Buffer type"""
        type_mapping = {
            "int": "int32",
            "long": "int64",
            "double": "double",
            "float": "float",
            "bool": "bool",
            "std::string": "string",
            "string": "string",
        }
        
        if cpp_type in type_mapping:
            return type_mapping[cpp_type]
        elif cpp_type.startswith("std::vector<"):
            element_type = cpp_type[12:-1]
            proto_element_type = self._cpp_type_to_proto_type(element_type)
            return f"repeated {proto_element_type}"
        else:
            # Assume it's a custom type
            return cpp_type.replace("::", "_")

class MessagePackGenerator:
    """Generates MessagePack serialization"""
    
    def __init__(self, config: SerializationConfig):
        self.config = config
    
    def generate_serialization(self, types: List[PolyglotType]) -> str:
        """Generate MessagePack serialization code"""
        lines = [
            "#pragma once",
            "#include <msgpack.hpp>",
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
            f"namespace {self.config.namespace} {{",
            ""
        ])
        
        # Generate MSGPACK_DEFINE macros for each type
        for type_obj in types:
            if type_obj.type_kind in ['class', 'struct']:
                msgpack_code = self._generate_msgpack_macro(type_obj)
                lines.append(indent(msgpack_code, "    "))
                lines.append("")
        
        lines.append("}")
        
        return "\n".join(lines)
    
    def _generate_msgpack_macro(self, type_obj: PolyglotType) -> str:
        """Generate MSGPACK_DEFINE macro for a type"""
        class_name = type_obj.name
        fields = type_obj.metadata.get('fields', [])
        
        if not fields:
            return f"// {class_name} has no serializable fields"
        
        field_names = [f['name'] for f in fields]
        field_list = ", ".join(field_names)
        
        lines = [
            f"// MessagePack serialization for {class_name}",
            f"template<>",
            f"struct msgpack::define<{class_name}> {{",
            f"    template<typename Packer>",
            f"    void msgpack_pack(Packer& pk, const {class_name}& obj) const {{",
            f"        pk.pack_array({len(fields)});",
        ]
        
        for field in fields:
            field_name = field['name']
            lines.append(f"        pk.pack(obj.{field_name});")
        
        lines.extend([
            "    }",
            "",
            f"    void msgpack_unpack(const msgpack::object& obj, {class_name}& dst) const {{",
            f"        if (obj.type != msgpack::type::ARRAY) {{",
            f"            throw msgpack::type_error();",
            f"        }}",
            f"        if (obj.via.array.size != {len(fields)}) {{",
            f"            throw msgpack::type_error();",
            f"        }}",
        ])
        
        for i, field in enumerate(fields):
            field_name = field['name']
            lines.append(f"        obj.via.array.ptr[{i}].convert(dst.{field_name});")
        
        lines.extend([
            "    }",
            "};"
        ])
        
        return "\n".join(lines)

class BinaryFormatGenerator:
    """Generates custom binary serialization"""
    
    def __init__(self, config: SerializationConfig):
        self.config = config
    
    def generate_serialization(self, types: List[PolyglotType]) -> str:
        """Generate binary serialization code"""
        lines = [
            "#pragma once",
            "#include <vector>",
            "#include <cstring>",
            "#include <stdexcept>",
            "",
            f"namespace {self.config.namespace} {{",
            "",
            "class BinarySerializer {",
            "private:",
            "    std::vector<uint8_t> buffer_;",
            "    size_t pos_ = 0;",
            "",
            "    template<typename T>",
            "    void write_primitive(const T& value) {",
            "        size_t size = sizeof(T);",
            "        buffer_.resize(buffer_.size() + size);",
            "        std::memcpy(buffer_.data() + buffer_.size() - size, &value, size);",
            "    }",
            "",
            "    template<typename T>",
            "    T read_primitive() {",
            "        if (pos_ + sizeof(T) > buffer_.size()) {",
            "            throw std::runtime_error(\"Buffer underflow\");",
            "        }",
            "        T value;",
            "        std::memcpy(&value, buffer_.data() + pos_, sizeof(T));",
            "        pos_ += sizeof(T);",
            "        return value;",
            "    }",
            "",
            "    void write_string(const std::string& str) {",
            "        write_primitive<uint32_t>(static_cast<uint32_t>(str.size()));",
            "        buffer_.insert(buffer_.end(), str.begin(), str.end());",
            "    }",
            "",
            "    std::string read_string() {",
            "        uint32_t size = read_primitive<uint32_t>();",
            "        if (pos_ + size > buffer_.size()) {",
            "            throw std::runtime_error(\"Buffer underflow\");",
            "        }",
            "        std::string str(buffer_.begin() + pos_, buffer_.begin() + pos_ + size);",
            "        pos_ += size;",
            "        return str;",
            "    }",
            "",
            "public:",
            "    void clear() { buffer_.clear(); pos_ = 0; }",
            "    const std::vector<uint8_t>& data() const { return buffer_; }",
            "    void set_data(const std::vector<uint8_t>& data) { buffer_ = data; pos_ = 0; }",
            ""
        ]
        
        # Generate serialization methods for each type
        for type_obj in types:
            if type_obj.type_kind in ['class', 'struct']:
                serialization_methods = self._generate_binary_methods(type_obj)
                lines.append(indent(serialization_methods, "    "))
                lines.append("")
        
        lines.extend([
            "};",
            "",
            "}"
        ])
        
        return "\n".join(lines)
    
    def _generate_binary_methods(self, type_obj: PolyglotType) -> str:
        """Generate binary serialization methods for a type"""
        class_name = type_obj.name
        method_name = class_name.replace("::", "_").lower()
        
        lines = [
            f"// Binary serialization for {class_name}",
            f"void serialize_{method_name}(const {class_name}& obj) {{",
        ]
        
        if self.config.include_versioning:
            lines.append("    write_primitive<uint8_t>(1); // version")
        
        fields = type_obj.metadata.get('fields', [])
        for field in fields:
            field_name = field['name']
            field_type = field['type']
            
            if field_type in ['int', 'double', 'float', 'bool']:
                lines.append(f"    write_primitive(obj.{field_name});")
            elif field_type in ['std::string', 'string']:
                lines.append(f"    write_string(obj.{field_name});")
            elif field_type.startswith('std::vector<'):
                element_type = field_type[12:-1]
                lines.extend([
                    f"    write_primitive<uint32_t>(static_cast<uint32_t>(obj.{field_name}.size()));",
                    f"    for (const auto& item : obj.{field_name}) {{",
                ])
                if element_type in ['int', 'double', 'float', 'bool']:
                    lines.append("        write_primitive(item);")
                elif element_type in ['std::string', 'string']:
                    lines.append("        write_string(item);")
                lines.append("    }")
        
        lines.extend([
            "}",
            "",
            f"{class_name} deserialize_{method_name}() {{",
            f"    {class_name} obj;",
        ])
        
        if self.config.include_versioning:
            lines.extend([
                "    uint8_t version = read_primitive<uint8_t>();",
                "    if (version != 1) {",
                "        throw std::runtime_error(\"Unsupported version\");",
                "    }",
            ])
        
        for field in fields:
            field_name = field['name']
            field_type = field['type']
            
            if field_type in ['int', 'double', 'float', 'bool']:
                lines.append(f"    obj.{field_name} = read_primitive<{field_type}>();")
            elif field_type in ['std::string', 'string']:
                lines.append(f"    obj.{field_name} = read_string();")
            elif field_type.startswith('std::vector<'):
                element_type = field_type[12:-1]
                lines.extend([
                    f"    uint32_t {field_name}_size = read_primitive<uint32_t>();",
                    f"    obj.{field_name}.reserve({field_name}_size);",
                    f"    for (uint32_t i = 0; i < {field_name}_size; ++i) {{",
                ])
                if element_type in ['int', 'double', 'float', 'bool']:
                    lines.append(f"        obj.{field_name}.push_back(read_primitive<{element_type}>());")
                elif element_type in ['std::string', 'string']:
                    lines.append(f"        obj.{field_name}.push_back(read_string());")
                lines.append("    }")
        
        lines.extend([
            "    return obj;",
            "}"
        ])
        
        return "\n".join(lines)

class SerializationOrchestrator:
    """Orchestrates generation of multiple serialization formats"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_all_formats(self, types: List[PolyglotType], namespace: str = "serialization"):
        """Generate all serialization formats"""
        print(f"🔧 Generating serialization for {len(types)} types...")
        
        # JSON serialization
        json_config = SerializationConfig("json", namespace)
        json_gen = JSONSerializationGenerator(json_config)
        json_code = json_gen.generate_serialization(types)
        
        json_file = self.output_dir / f"{namespace}_json.hpp"
        with open(json_file, 'w') as f:
            f.write(json_code)
        print(f"✅ JSON serialization: {json_file}")
        
        # Protocol Buffers
        protobuf_config = SerializationConfig("protobuf", namespace)
        protobuf_gen = ProtobufGenerator(protobuf_config)
        proto_code = protobuf_gen.generate_proto(types)
        
        proto_file = self.output_dir / f"{namespace}.proto"
        with open(proto_file, 'w') as f:
            f.write(proto_code)
        print(f"✅ Protocol Buffers: {proto_file}")
        
        # MessagePack
        msgpack_config = SerializationConfig("messagepack", namespace)
        msgpack_gen = MessagePackGenerator(msgpack_config)
        msgpack_code = msgpack_gen.generate_serialization(types)
        
        msgpack_file = self.output_dir / f"{namespace}_msgpack.hpp"
        with open(msgpack_file, 'w') as f:
            f.write(msgpack_code)
        print(f"✅ MessagePack serialization: {msgpack_file}")
        
        # Binary format
        binary_config = SerializationConfig("binary", namespace)
        binary_gen = BinaryFormatGenerator(binary_config)
        binary_code = binary_gen.generate_serialization(types)
        
        binary_file = self.output_dir / f"{namespace}_binary.hpp"
        with open(binary_file, 'w') as f:
            f.write(binary_code)
        print(f"✅ Binary serialization: {binary_file}")

def create_sample_types_for_serialization():
    """Create sample C++ types for serialization"""
    sample_code = '''
#pragma once
#include <string>
#include <vector>

struct Person {
    std::string name;
    int age;
    std::string email;
    
    // Getters
    const std::string& get_name() const { return name; }
    int get_age() const { return age; }
    const std::string& get_email() const { return email; }
    
    // Setters
    void set_name(const std::string& n) { name = n; }
    void set_age(int a) { age = a; }
    void set_email(const std::string& e) { email = e; }
};

struct Company {
    std::string name;
    std::vector<Person> employees;
    std::vector<std::string> departments;
    double revenue;
    bool is_public;
    
    const std::string& get_name() const { return name; }
    const std::vector<Person>& get_employees() const { return employees; }
    void set_name(const std::string& n) { name = n; }
    void add_employee(const Person& p) { employees.push_back(p); }
};
'''
    
    serialization_header = Path(__file__).parent / "serialization_sample.hpp"
    with open(serialization_header, 'w') as f:
        f.write(sample_code)
    
    return serialization_header

def main():
    """Demonstrate serialization generation"""
    
    print("=" * 80)
    print("Serialization Code Generator Examples")
    print("=" * 80)
    
    # Create sample C++ code
    print("📝 Creating sample C++ code...")
    sample_file = create_sample_types_for_serialization()
    
    # Extract types
    print("🔍 Extracting types...")
    extractor = CppTypeExtractor()
    extracted_types = extractor.extract_from_file(str(sample_file))
    
    print(f"📊 Extracted {len(extracted_types)} types:")
    for t in extracted_types:
        fields = t.metadata.get('fields', [])
        print(f"  - {t.name} ({t.type_kind}) - {len(fields)} fields")
    
    # Generate serialization code
    output_dir = Path(__file__).parent / "generated_serialization"
    orchestrator = SerializationOrchestrator(output_dir)
    
    print(f"\n🔧 Generating serialization code in {output_dir}...")
    orchestrator.generate_all_formats(extracted_types, "sample_types")
    
    print(f"\n📁 Generated files:")
    for file_path in output_dir.iterdir():
        if file_path.is_file():
            print(f"  - {file_path.name} ({file_path.stat().st_size} bytes)")
    
    print("\n✅ Serialization generation complete!")

if __name__ == "__main__":
    main()