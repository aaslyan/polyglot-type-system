import clang.cindex
from clang.cindex import CursorKind, TypeKind as ClangTypeKind, AccessSpecifier
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
import logging
from dataclasses import dataclass, field

from ..types.polyglot_types import *
from .clang_utils import setup_clang, get_type_spelling

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ExtractionContext:
    """Context for type extraction"""
    visited_types: Set[str] = field(default_factory=set)
    type_map: Dict[str, PolyglotType] = field(default_factory=dict)
    current_namespace: List[str] = field(default_factory=list)
    include_paths: List[str] = field(default_factory=list)

class CppTypeExtractor:
    """Extract type information from C++ code using Clang"""

    def __init__(self, clang_lib_path: Optional[str] = None):
        setup_clang(clang_lib_path)
        self.index = clang.cindex.Index.create()
        self.context = ExtractionContext()

    def extract_from_file(self, file_path: str, include_paths: List[str] = None) -> Dict[str, PolyglotType]:
        """Extract all types from a C++ file"""
        self.context = ExtractionContext(include_paths=include_paths or [])

        # Parse the file
        args = self._build_parse_args()
        tu = self.index.parse(
            file_path,
            args=args,
            options=clang.cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
        )

        # Check for parse errors
        if tu.diagnostics:
            for diag in tu.diagnostics:
                logger.warning(f"Parse diagnostic: {diag}")

        # Walk the AST
        self._walk_cursor(tu.cursor)

        return self.context.type_map

    def _build_parse_args(self) -> List[str]:
        """Build clang parse arguments"""
        args = [
            '-std=c++17',
            '-xc++',
        ]
        
        # Add platform-specific include paths
        import platform
        if platform.system() == "Darwin":  # macOS
            args.extend([
                '-I/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1',
                '-I/Library/Developer/CommandLineTools/usr/include/c++/v1',
                '-I/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include',
                '-I/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/c++/v1',
            ])
        else:  # Linux
            args.extend([
                '-I/usr/include/c++/11',
                '-I/usr/include/x86_64-linux-gnu/c++/11',
            ])

        # Add custom include paths
        for path in self.context.include_paths:
            args.append(f'-I{path}')

        return args

    def _walk_cursor(self, cursor: clang.cindex.Cursor, depth: int = 0):
        """Recursively walk the AST"""
        # Skip system headers
        if cursor.location.file and 'usr/include' in cursor.location.file.name:
            return

        # Handle namespaces
        if cursor.kind == CursorKind.NAMESPACE:
            self.context.current_namespace.append(cursor.spelling)
            for child in cursor.get_children():
                self._walk_cursor(child, depth + 1)
            self.context.current_namespace.pop()
            return

        # Extract different kinds of declarations
        if cursor.kind in [CursorKind.CLASS_DECL, CursorKind.STRUCT_DECL]:
            if cursor.is_definition():
                self._extract_class(cursor)

        elif cursor.kind == CursorKind.TYPEDEF_DECL:
            self._extract_typedef(cursor)

        elif cursor.kind == CursorKind.FUNCTION_DECL:
            self._extract_function(cursor)

        elif cursor.kind == CursorKind.ENUM_DECL:
            self._extract_enum(cursor)

        elif cursor.kind == CursorKind.CLASS_TEMPLATE:
            self._extract_class_template(cursor)

        # Recurse into children
        for child in cursor.get_children():
            self._walk_cursor(child, depth + 1)

    def _extract_class(self, cursor: clang.cindex.Cursor) -> Optional[ObjectType]:
        """Extract class/struct information"""
        full_name = self._get_full_name(cursor.spelling)

        # Avoid re-processing
        if full_name in self.context.visited_types:
            return self.context.type_map.get(full_name)

        self.context.visited_types.add(full_name)

        # Create object type
        obj_type = ObjectType(
            canonical_name=full_name,
            kind=TypeKind.OBJECT,
            source_language="cpp",
            metadata={
                "kind": "class" if cursor.kind == CursorKind.CLASS_DECL else "struct",
                "location": self._get_location(cursor)
            }
        )

        # Extract members and methods
        for child in cursor.get_children():
            if child.kind == CursorKind.FIELD_DECL:
                member_type = self._extract_type(child.type)
                if member_type:
                    obj_type.members[child.spelling] = member_type

            elif child.kind in [CursorKind.CXX_METHOD, CursorKind.CONSTRUCTOR, CursorKind.DESTRUCTOR]:
                method_type = self._extract_method(child)
                if method_type:
                    obj_type.methods[child.spelling] = method_type

            elif child.kind == CursorKind.CXX_BASE_SPECIFIER:
                base_type = self._extract_type(child.type)
                if base_type:
                    obj_type.base_types.append(base_type)

        # Check if abstract
        obj_type.is_abstract = any(
            method.metadata.get("is_pure_virtual", False) 
            for method in obj_type.methods.values()
        )

        self.context.type_map[full_name] = obj_type
        return obj_type

    def _extract_type(self, clang_type: clang.cindex.Type) -> Optional[PolyglotType]:
        """Extract type information from clang type"""
        canonical = clang_type.get_canonical()

        # Handle primitive types
        if clang_type.kind in [ClangTypeKind.INT, ClangTypeKind.UINT, 
                               ClangTypeKind.LONG, ClangTypeKind.ULONG,
                               ClangTypeKind.SHORT, ClangTypeKind.USHORT]:
            return self._create_primitive_type(clang_type)

        elif clang_type.kind in [ClangTypeKind.FLOAT, ClangTypeKind.DOUBLE, 
                                ClangTypeKind.LONGDOUBLE]:
            return self._create_primitive_type(clang_type)

        elif clang_type.kind == ClangTypeKind.BOOL:
            return PrimitiveType(canonical_name="bool", kind=TypeKind.PRIMITIVE, bit_width=1)

        elif clang_type.kind == ClangTypeKind.VOID:
            return PrimitiveType(canonical_name="void", kind=TypeKind.PRIMITIVE)

        # Handle pointer types
        elif clang_type.kind == ClangTypeKind.POINTER:
            return self._create_pointer_type(clang_type)

        # Handle reference types
        elif clang_type.kind in [ClangTypeKind.LVALUEREFERENCE, ClangTypeKind.RVALUEREFERENCE]:
            return self._create_reference_type(clang_type)

        # Handle array types
        elif clang_type.kind in [ClangTypeKind.CONSTANTARRAY, ClangTypeKind.INCOMPLETEARRAY]:
            return self._create_array_type(clang_type)

        # Handle function types
        elif clang_type.kind == ClangTypeKind.FUNCTIONPROTO:
            return self._create_function_type(clang_type)

        # Handle record types (class/struct)
        elif clang_type.kind == ClangTypeKind.RECORD:
            decl = clang_type.get_declaration()
            if decl.kind in [CursorKind.CLASS_DECL, CursorKind.STRUCT_DECL]:
                return self._extract_class(decl)

        # Handle elaborated types (e.g., "class Foo")
        elif clang_type.kind == ClangTypeKind.ELABORATED:
            return self._extract_type(clang_type.get_named_type())

        # Handle typedef types
        elif clang_type.kind == ClangTypeKind.TYPEDEF:
            return self._extract_type(canonical)

        # Default case - create a generic type
        return PolyglotType(
            canonical_name=get_type_spelling(clang_type),
            kind=TypeKind.PRIMITIVE,
            metadata={"clang_kind": str(clang_type.kind)}
        )

    def _create_primitive_type(self, clang_type: clang.cindex.Type) -> PrimitiveType:
        """Create primitive type from clang type"""
        type_map = {
            ClangTypeKind.INT: ("int", 32, True),
            ClangTypeKind.UINT: ("unsigned int", 32, False),
            ClangTypeKind.LONG: ("long", 64, True),
            ClangTypeKind.ULONG: ("unsigned long", 64, False),
            ClangTypeKind.SHORT: ("short", 16, True),
            ClangTypeKind.USHORT: ("unsigned short", 16, False),
            ClangTypeKind.FLOAT: ("float", 32, True),
            ClangTypeKind.DOUBLE: ("double", 64, True),
            ClangTypeKind.LONGDOUBLE: ("long double", 128, True),
        }

        name, bits, signed = type_map.get(clang_type.kind, ("unknown", None, True))

        prim_type = PrimitiveType(
            canonical_name=name,
            kind=TypeKind.PRIMITIVE,
            bit_width=bits,
            is_signed=signed
        )

        # Add qualifiers
        if clang_type.is_const_qualified():
            prim_type.qualifiers.append(TypeQualifier.CONST)
        if clang_type.is_volatile_qualified():
            prim_type.qualifiers.append(TypeQualifier.VOLATILE)

        return prim_type

    def _create_pointer_type(self, clang_type: clang.cindex.Type) -> Optional[PointerType]:
        """Create pointer type"""
        pointee = self._extract_type(clang_type.get_pointee())
        
        if not pointee:
            return None

        ptr_type = PointerType(
            canonical_name=f"{pointee.canonical_name}*",
            kind=TypeKind.POINTER,
            pointee_type=pointee,
            is_const_pointer=clang_type.is_const_qualified()
        )

        return ptr_type

    def _create_reference_type(self, clang_type: clang.cindex.Type) -> Optional[ReferenceType]:
        """Create reference type"""
        referred = self._extract_type(clang_type.get_pointee())
        
        if not referred:
            return None

        ref_type = ReferenceType(
            canonical_name=f"{referred.canonical_name}&",
            kind=TypeKind.REFERENCE,
            referred_type=referred,
            is_rvalue=(clang_type.kind == ClangTypeKind.RVALUEREFERENCE)
        )

        return ref_type

    def _create_array_type(self, clang_type: clang.cindex.Type) -> Optional[ArrayType]:
        """Create array type"""
        element = self._extract_type(clang_type.get_array_element_type())
        
        if not element:
            return None

        size = None
        if clang_type.kind == ClangTypeKind.CONSTANTARRAY:
            size = clang_type.get_array_size()

        return ArrayType(
            canonical_name=f"{element.canonical_name}[{size if size else ''}]",
            kind=TypeKind.ARRAY,
            element_type=element,
            size=size
        )

    def _create_function_type(self, clang_type: clang.cindex.Type) -> FunctionType:
        """Create function type"""
        return_type = self._extract_type(clang_type.get_result())
        
        # Default to void if return type extraction fails
        if not return_type:
            return_type = PrimitiveType(canonical_name="void", kind=TypeKind.PRIMITIVE)

        param_types = []
        for arg_type in clang_type.argument_types():
            param_type = self._extract_type(arg_type)
            if param_type:
                param_types.append(param_type)

        func_type = FunctionType(
            canonical_name=self._build_function_signature(return_type, param_types),
            kind=TypeKind.FUNCTION,
            return_type=return_type,
            parameter_types=param_types,
            is_variadic=clang_type.is_function_variadic()
        )

        return func_type

    def _extract_method(self, cursor: clang.cindex.Cursor) -> Optional[FunctionType]:
        """Extract method information"""
        func_type = self._create_function_type(cursor.type)

        if func_type:
            # Add method-specific metadata
            func_type.metadata.update({
                "is_const": cursor.is_const_method(),
                "is_static": cursor.is_static_method(),
                "is_virtual": cursor.is_virtual_method(),
                "is_pure_virtual": cursor.is_pure_virtual_method(),
                "access": self._get_access_specifier(cursor)
            })

            # Check for noexcept
            if "noexcept" in cursor.type.spelling:
                func_type.is_noexcept = True

        return func_type

    def _extract_class_template(self, cursor: clang.cindex.Cursor) -> Optional[TemplateType]:
        """Extract template class information"""
        full_name = self._get_full_name(cursor.spelling)

        # Extract template parameters
        template_params = []
        for child in cursor.get_children():
            if child.kind in [CursorKind.TEMPLATE_TYPE_PARAMETER, 
                            CursorKind.TEMPLATE_NON_TYPE_PARAMETER]:
                template_params.append(child.spelling)

        template_type = TemplateType(
            canonical_name=full_name,
            template_name=cursor.spelling,
            kind=TypeKind.TEMPLATE,
            metadata={
                "template_parameters": template_params,
                "location": self._get_location(cursor)
            }
        )

        self.context.type_map[full_name] = template_type
        return template_type

    def _extract_typedef(self, cursor: clang.cindex.Cursor):
        """Extract typedef information"""
        # For now, just skip typedefs
        pass

    def _extract_function(self, cursor: clang.cindex.Cursor):
        """Extract free function information"""
        # For now, just skip free functions
        pass

    def _extract_enum(self, cursor: clang.cindex.Cursor):
        """Extract enum information"""
        # For now, just skip enums
        pass

    def _get_full_name(self, name: str) -> str:
        """Get fully qualified name including namespace"""
        if self.context.current_namespace:
            return "::".join(self.context.current_namespace + [name])
        return name

    def _get_location(self, cursor: clang.cindex.Cursor) -> Dict[str, Any]:
        """Get source location information"""
        loc = cursor.location
        return {
            "file": loc.file.name if loc.file else None,
            "line": loc.line,
            "column": loc.column
        }

    def _get_access_specifier(self, cursor: clang.cindex.Cursor) -> str:
        """Get access specifier as string"""
        access = cursor.access_specifier
        if access == AccessSpecifier.PUBLIC:
            return "public"
        elif access == AccessSpecifier.PRIVATE:
            return "private"
        elif access == AccessSpecifier.PROTECTED:
            return "protected"
        return "unknown"

    def _build_function_signature(self, return_type: PolyglotType, 
                                 param_types: List[PolyglotType]) -> str:
        """Build function signature string"""
        return_name = return_type.canonical_name if return_type else "void"
        params = ", ".join(p.canonical_name for p in param_types if p)
        return f"{return_name}({params})"
