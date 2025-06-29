from typing import Dict, List, Optional
import re
from ..types.polyglot_types import *

class CppToPolyglotConverter:
    """Convert C++ types to polyglot representation"""

    def __init__(self):
        self.type_mappings = {
            # Primitive mappings
            "int": "Integer32",
            "unsigned int": "UInteger32",
            "long": "Integer64",
            "unsigned long": "UInteger64",
            "short": "Integer16",
            "unsigned short": "UInteger16",
            "char": "Character",
            "unsigned char": "UInteger8",
            "float": "Float32",
            "double": "Float64",
            "long double": "Float128",
            "bool": "Boolean",
            "void": "Void",

            # STL mappings
            "std::string": "String",
            "std::wstring": "WideString",
            "std::vector": "DynamicArray",
            "std::array": "FixedArray",
            "std::map": "OrderedMap",
            "std::unordered_map": "HashMap",
            "std::set": "OrderedSet",
            "std::unordered_set": "HashSet",
            "std::unique_ptr": "UniquePointer",
            "std::shared_ptr": "SharedPointer",
            "std::weak_ptr": "WeakPointer",
        }

    def convert(self, cpp_type: PolyglotType) -> PolyglotType:
        """Convert C++ type to polyglot canonical form"""
        # Clone the type
        poly_type = self._clone_type(cpp_type)

        # Convert canonical name
        poly_type.canonical_name = self._convert_name(cpp_type.canonical_name)

        # Add language mapping
        poly_type.metadata["cpp_original"] = cpp_type.canonical_name

        # Handle specific type kinds
        if isinstance(cpp_type, TemplateType):
            poly_type = self._convert_template(cpp_type)
        elif isinstance(cpp_type, ObjectType):
            poly_type = self._convert_object(cpp_type)

        return poly_type

    def _convert_name(self, cpp_name: str) -> str:
        """Convert C++ type name to polyglot canonical name"""
        # Check direct mappings
        if cpp_name in self.type_mappings:
            return self.type_mappings[cpp_name]

        # Handle templates
        template_match = re.match(r"(\w+(?:::\w+)*)<(.+)>", cpp_name)
        if template_match:
            base_name = template_match.group(1)
            template_args = template_match.group(2)

            if base_name in self.type_mappings:
                return f"{self.type_mappings[base_name]}<{self._convert_template_args(template_args)}>"

        # Default: use the C++ name
        return cpp_name

    def _convert_template_args(self, args: str) -> str:
        """Convert template arguments"""
        # Simple parsing - would need more sophisticated approach for complex templates
        parts = []
        current = ""
        depth = 0

        for char in args:
            if char == '<':
                depth += 1
            elif char == '>':
                depth -= 1
            elif char == ',' and depth == 0:
                parts.append(self._convert_name(current.strip()))
                current = ""
                continue

            current += char

        if current:
            parts.append(self._convert_name(current.strip()))

        return ", ".join(parts)

    def _convert_template(self, template: TemplateType) -> TemplateType:
        """Convert template type"""
        # Handle STL containers specially
        if template.template_name.startswith("std::"):
            return self._convert_stl_template(template)

        # Convert template arguments
        converted_args = []
        for arg in template.template_arguments:
            converted_args.append(self.convert(arg))

        template.template_arguments = converted_args
        return template

    def _convert_stl_template(self, template: TemplateType) -> TemplateType:
        """Convert STL template types"""
        stl_map = {
            "std::vector": "DynamicArray",
            "std::array": "FixedArray",
            "std::map": "OrderedMap",
            "std::unordered_map": "HashMap",
            "std::set": "OrderedSet",
            "std::unordered_set": "HashSet",
            "std::pair": "Pair",
            "std::tuple": "Tuple",
            "std::optional": "Optional",
            "std::variant": "Variant",
        }

        if template.template_name in stl_map:
            template.canonical_name = stl_map[template.template_name]
            template.metadata["stl_original"] = template.template_name

        return template

    def _convert_object(self, obj: ObjectType) -> ObjectType:
        """Convert object type"""
        # Convert member types
        converted_members = {}
        for name, member_type in obj.members.items():
            converted_members[name] = self.convert(member_type)
        obj.members = converted_members

        # Convert method types
        converted_methods = {}
        for name, method_type in obj.methods.items():
            converted_methods[name] = self.convert(method_type)
        obj.methods = converted_methods

        # Convert base types
        converted_bases = []
        for base_type in obj.base_types:
            converted_bases.append(self.convert(base_type))
        obj.base_types = converted_bases

        return obj

    def _clone_type(self, cpp_type: PolyglotType) -> PolyglotType:
        """Create a deep copy of a type"""
        # This is a simplified clone - in production, use proper deep copy
        if isinstance(cpp_type, PrimitiveType):
            return PrimitiveType(
                canonical_name=cpp_type.canonical_name,
                kind=cpp_type.kind,
                bit_width=cpp_type.bit_width,
                is_signed=cpp_type.is_signed,
                qualifiers=cpp_type.qualifiers.copy(),
                metadata=cpp_type.metadata.copy()
            )
        elif isinstance(cpp_type, PointerType):
            return PointerType(
                canonical_name=cpp_type.canonical_name,
                kind=cpp_type.kind,
                pointee_type=cpp_type.pointee_type,
                is_const_pointer=cpp_type.is_const_pointer,
                qualifiers=cpp_type.qualifiers.copy(),
                metadata=cpp_type.metadata.copy()
            )
        elif isinstance(cpp_type, ObjectType):
            return ObjectType(
                canonical_name=cpp_type.canonical_name,
                kind=cpp_type.kind,
                members=cpp_type.members.copy(),
                methods=cpp_type.methods.copy(),
                base_types=cpp_type.base_types.copy(),
                is_abstract=cpp_type.is_abstract,
                qualifiers=cpp_type.qualifiers.copy(),
                metadata=cpp_type.metadata.copy()
            )
        # Add other type cloning as needed

        return cpp_type
