#!/usr/bin/env python3
"""
generate_polyglot_project.py

This script generates the complete Polyglot Type System project.
Run it in an empty directory where you want to create the project.

Usage: python generate_polyglot_project.py
"""

import os
import textwrap
from pathlib import Path

def create_file(path, content):
    """Create a file with the given content."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Dedent the content to remove common leading whitespace
    content = textwrap.dedent(content).strip()
    
    with open(path, 'w') as f:
        f.write(content + '\n')
    
    print(f"Created: {path}")

def main():
    print("Generating Polyglot Type System project...")
    print("=" * 50)
    
    # Define all project files and their contents
    files = {
        # Project configuration files
        "README.md": '''
        # Polyglot Type System

        A system for extracting type information from C++ code and storing it in a language-agnostic format with RAG capabilities.

        ## Features

        - C++ type extraction using Clang AST
        - Polyglot type representation  
        - RAG storage with semantic search
        - Cross-language type compatibility checking

        ## Installation

        ```bash
        # Create virtual environment
        python -m venv venv
        source venv/bin/activate  # On Windows: venv\\Scripts\\activate

        # Install dependencies
        pip install -r requirements.txt
        pip install -e .
        ```

        ## Quick Start

        ```bash
        # Extract types from C++ code
        python examples/extract_types.py
        ```

        ## Project Structure

        ```
        polyglot-type-system/
        ├── src/
        │   ├── extractors/     # Language-specific extractors
        │   ├── types/          # Polyglot type definitions
        │   ├── converters/     # Type converters
        │   └── storage/        # RAG storage implementation
        ├── examples/           # Example usage
        └── tests/             # Unit tests
        ```

        ## Usage Example

        ```python
        from src.extractors.cpp_extractor import CppTypeExtractor
        from src.storage.rag_store import PolyglotRAGStore

        # Extract types from C++ file
        extractor = CppTypeExtractor()
        types = extractor.extract_from_file("your_code.cpp")

        # Store in RAG
        rag_store = PolyglotRAGStore()
        for type_obj in types.values():
            rag_store.store_type(type_obj)

        # Search for types
        results = rag_store.search_types("vector container")
        ```
        ''',

        "setup.py": '''
        from setuptools import setup, find_packages

        with open("README.md", "r", encoding="utf-8") as fh:
            long_description = fh.read()

        setup(
            name="polyglot-cpp",
            version="0.1.0",
            author="Your Name",
            author_email="your.email@example.com",
            description="A polyglot type system for cross-language compatibility",
            long_description=long_description,
            long_description_content_type="text/markdown",
            url="https://github.com/yourusername/polyglot-type-system",
            packages=find_packages(where="src"),
            package_dir={"": "src"},
            classifiers=[
                "Development Status :: 3 - Alpha",
                "Intended Audience :: Developers",
                "Topic :: Software Development :: Libraries",
                "License :: OSI Approved :: MIT License",
                "Programming Language :: Python :: 3",
                "Programming Language :: Python :: 3.8",
                "Programming Language :: Python :: 3.9",
                "Programming Language :: Python :: 3.10",
                "Programming Language :: Python :: 3.11",
            ],
            python_requires=">=3.8",
            install_requires=[
                "clang>=14.0",
                "libclang>=14.0.6",
                "dataclasses-json>=0.6.1",
                "llama-index>=0.9.48",
                "qdrant-client>=1.7.0",
                "pydantic>=2.5.3",
                "chromadb>=0.4.22",
                "click>=8.1.7",
                "rich>=13.7.0",
            ],
            extras_require={
                "dev": [
                    "pytest>=7.0",
                    "pytest-cov>=4.0",
                    "black>=23.0",
                    "flake8>=6.0",
                    "mypy>=1.0",
                ],
            },
            entry_points={
                "console_scripts": [
                    "extract-cpp-types=examples.extract_types:main",
                ],
            },
        )
        ''',

        "requirements.txt": '''
        clang==14.0
        libclang==14.0.6
        dataclasses-json==0.6.1
        llama-index==0.9.48
        qdrant-client==1.7.0
        chromadb==0.4.22
        pydantic==2.5.3
        numpy==1.24.3
        click==8.1.7
        rich==13.7.0
        pyarrow==14.0.2
        ''',

        ".gitignore": '''
        # Byte-compiled / optimized / DLL files
        __pycache__/
        *.py[cod]
        *$py.class

        # C extensions
        *.so

        # Distribution / packaging
        .Python
        build/
        develop-eggs/
        dist/
        downloads/
        eggs/
        .eggs/
        lib/
        lib64/
        parts/
        sdist/
        var/
        wheels/
        share/python-wheels/
        *.egg-info/
        .installed.cfg
        *.egg
        MANIFEST

        # Virtual environments
        venv/
        env/
        ENV/
        env.bak/
        venv.bak/

        # IDE
        .idea/
        .vscode/
        *.swp
        *.swo
        *~
        .DS_Store

        # Project specific
        polyglot_storage/
        *.db
        *.sqlite
        type_index.json
        chroma/

        # Testing
        .tox/
        .nox/
        .coverage
        .coverage.*
        .cache
        nosetests.xml
        coverage.xml
        *.cover
        *.py,cover
        .hypothesis/
        .pytest_cache/
        cover/
        htmlcov/

        # Clang
        compile_commands.json
        .clangd/
        .clang-format

        # Documentation
        docs/_build/
        docs/_static/
        docs/_templates/
        ''',

        # Source files - Type definitions
        "src/__init__.py": '',
        "src/types/__init__.py": '''
        from .polyglot_types import *
        ''',

        "src/types/polyglot_types.py": '''
        from dataclasses import dataclass, field
        from typing import Dict, List, Optional, Any, Set
        from enum import Enum
        import hashlib
        import json

        class TypeKind(Enum):
            PRIMITIVE = "primitive"
            OBJECT = "object"
            POINTER = "pointer"
            REFERENCE = "reference"
            ARRAY = "array"
            FUNCTION = "function"
            TEMPLATE = "template"
            ENUM = "enum"
            UNION = "union"

        class TypeQualifier(Enum):
            CONST = "const"
            VOLATILE = "volatile"
            MUTABLE = "mutable"
            RESTRICT = "restrict"

        @dataclass
        class PolyglotType:
            """Base class for all polyglot types"""
            canonical_name: str
            kind: TypeKind
            source_language: str = "cpp"
            qualifiers: List[TypeQualifier] = field(default_factory=list)
            metadata: Dict[str, Any] = field(default_factory=dict)
            
            @property
            def id(self) -> str:
                """Generate unique ID for the type"""
                content = f"{self.canonical_name}:{self.kind.value}:{self.source_language}"
                return hashlib.md5(content.encode()).hexdigest()
            
            def to_json(self) -> str:
                """Convert to JSON representation"""
                return json.dumps(self.to_dict(), indent=2)
            
            def to_dict(self) -> Dict[str, Any]:
                """Convert to dictionary"""
                return {
                    "id": self.id,
                    "canonical_name": self.canonical_name,
                    "kind": self.kind.value,
                    "source_language": self.source_language,
                    "qualifiers": [q.value for q in self.qualifiers],
                    "metadata": self.metadata
                }

        @dataclass
        class PrimitiveType(PolyglotType):
            """Primitive types like int, float, bool"""
            bit_width: Optional[int] = None
            is_signed: bool = True
            
            def __post_init__(self):
                self.kind = TypeKind.PRIMITIVE

        @dataclass
        class PointerType(PolyglotType):
            """Pointer types"""
            pointee_type: PolyglotType = None
            is_const_pointer: bool = False
            
            def __post_init__(self):
                self.kind = TypeKind.POINTER

        @dataclass
        class ReferenceType(PolyglotType):
            """Reference types (lvalue and rvalue)"""
            referred_type: PolyglotType = None
            is_rvalue: bool = False
            
            def __post_init__(self):
                self.kind = TypeKind.REFERENCE

        @dataclass
        class ArrayType(PolyglotType):
            """Array types"""
            element_type: PolyglotType = None
            size: Optional[int] = None  # None for dynamic arrays
            
            def __post_init__(self):
                self.kind = TypeKind.ARRAY

        @dataclass
        class FunctionType(PolyglotType):
            """Function types"""
            return_type: PolyglotType = None
            parameter_types: List[PolyglotType] = field(default_factory=list)
            is_variadic: bool = False
            is_noexcept: bool = False
            
            def __post_init__(self):
                self.kind = TypeKind.FUNCTION

        @dataclass 
        class TemplateType(PolyglotType):
            """Template types"""
            template_name: str = ""
            template_arguments: List[PolyglotType] = field(default_factory=list)
            
            def __post_init__(self):
                self.kind = TypeKind.TEMPLATE

        @dataclass
        class ObjectType(PolyglotType):
            """Class/Struct types"""
            members: Dict[str, PolyglotType] = field(default_factory=dict)
            methods: Dict[str, FunctionType] = field(default_factory=dict)
            base_types: List[PolyglotType] = field(default_factory=list)
            is_abstract: bool = False
            
            def __post_init__(self):
                self.kind = TypeKind.OBJECT
        ''',

        # Extractors
        "src/extractors/__init__.py": '',
        
        "src/extractors/clang_utils.py": '''
        import clang.cindex
        import platform
        import os
        from typing import Optional

        def setup_clang(lib_path: Optional[str] = None):
            """Setup clang library path"""
            if lib_path:
                clang.cindex.Config.set_library_file(lib_path)
            else:
                # Try to find clang library automatically
                system = platform.system()
                
                if system == "Linux":
                    possible_paths = [
                        "/usr/lib/llvm-14/lib/libclang.so.1",
                        "/usr/lib/llvm-13/lib/libclang.so.1",
                        "/usr/lib/llvm-12/lib/libclang.so.1",
                        "/usr/lib/x86_64-linux-gnu/libclang-14.so.1",
                        "/usr/lib/libclang.so",
                    ]
                elif system == "Darwin":  # macOS
                    possible_paths = [
                        "/Library/Developer/CommandLineTools/usr/lib/libclang.dylib",
                        "/usr/local/opt/llvm/lib/libclang.dylib",
                        "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib/libclang.dylib",
                    ]
                elif system == "Windows":
                    possible_paths = [
                        r"C:\\Program Files\\LLVM\\bin\\libclang.dll",
                        r"C:\\Program Files (x86)\\LLVM\\bin\\libclang.dll",
                    ]
                else:
                    possible_paths = []
                
                for path in possible_paths:
                    if os.path.exists(path):
                        clang.cindex.Config.set_library_file(path)
                        break

        def get_type_spelling(clang_type: clang.cindex.Type) -> str:
            """Get clean type spelling"""
            spelling = clang_type.spelling
            
            # Clean up some common patterns
            spelling = spelling.replace("struct ", "")
            spelling = spelling.replace("class ", "")
            spelling = spelling.replace("enum ", "")
            
            return spelling
        ''',

        "src/extractors/cpp_extractor.py": '''
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
                    '-I/usr/include/c++/11',
                    '-I/usr/include/x86_64-linux-gnu/c++/11',
                ]
                
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
                    return PrimitiveType(canonical_name="bool", bit_width=1)
                
                elif clang_type.kind == ClangTypeKind.VOID:
                    return PrimitiveType(canonical_name="void")
                
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
                    bit_width=bits,
                    is_signed=signed
                )
                
                # Add qualifiers
                if clang_type.is_const_qualified():
                    prim_type.qualifiers.append(TypeQualifier.CONST)
                if clang_type.is_volatile_qualified():
                    prim_type.qualifiers.append(TypeQualifier.VOLATILE)
                
                return prim_type
            
            def _create_pointer_type(self, clang_type: clang.cindex.Type) -> PointerType:
                """Create pointer type"""
                pointee = self._extract_type(clang_type.get_pointee())
                
                ptr_type = PointerType(
                    canonical_name=f"{pointee.canonical_name}*",
                    pointee_type=pointee,
                    is_const_pointer=clang_type.is_const_qualified()
                )
                
                return ptr_type
            
            def _create_reference_type(self, clang_type: clang.cindex.Type) -> ReferenceType:
                """Create reference type"""
                referred = self._extract_type(clang_type.get_pointee())
                
                ref_type = ReferenceType(
                    canonical_name=f"{referred.canonical_name}&",
                    referred_type=referred,
                    is_rvalue=(clang_type.kind == ClangTypeKind.RVALUEREFERENCE)
                )
                
                return ref_type
            
            def _create_array_type(self, clang_type: clang.cindex.Type) -> ArrayType:
                """Create array type"""
                element = self._extract_type(clang_type.get_array_element_type())
                
                size = None
                if clang_type.kind == ClangTypeKind.CONSTANTARRAY:
                    size = clang_type.get_array_size()
                
                return ArrayType(
                    canonical_name=f"{element.canonical_name}[{size if size else ''}]",
                    element_type=element,
                    size=size
                )
            
            def _create_function_type(self, clang_type: clang.cindex.Type) -> FunctionType:
                """Create function type"""
                return_type = self._extract_type(clang_type.get_result())
                
                param_types = []
                for arg_type in clang_type.argument_types():
                    param_type = self._extract_type(arg_type)
                    if param_type:
                        param_types.append(param_type)
                
                func_type = FunctionType(
                    canonical_name=self._build_function_signature(return_type, param_types),
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
                params = ", ".join(p.canonical_name for p in param_types)
                return f"{return_type.canonical_name}({params})"
        ''',

        # Converters
        "src/converters/__init__.py": '',
        
        "src/converters/cpp_to_polyglot.py": '''
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
                template_match = re.match(r"(\\w+(?:::\\w+)*)<(.+)>", cpp_name)
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
                        bit_width=cpp_type.bit_width,
                        is_signed=cpp_type.is_signed,
                        qualifiers=cpp_type.qualifiers.copy(),
                        metadata=cpp_type.metadata.copy()
                    )
                elif isinstance(cpp_type, PointerType):
                    return PointerType(
                        canonical_name=cpp_type.canonical_name,
                        pointee_type=cpp_type.pointee_type,
                        is_const_pointer=cpp_type.is_const_pointer,
                        qualifiers=cpp_type.qualifiers.copy(),
                        metadata=cpp_type.metadata.copy()
                    )
                elif isinstance(cpp_type, ObjectType):
                    return ObjectType(
                        canonical_name=cpp_type.canonical_name,
                        members=cpp_type.members.copy(),
                        methods=cpp_type.methods.copy(),
                        base_types=cpp_type.base_types.copy(),
                        is_abstract=cpp_type.is_abstract,
                        qualifiers=cpp_type.qualifiers.copy(),
                        metadata=cpp_type.metadata.copy()
                    )
                # Add other type cloning as needed
                
                return cpp_type
        ''',

        # Storage
        "src/storage/__init__.py": '',
        
        "src/storage/rag_store.py": '''
        from typing import List, Dict, Any, Optional
        import json
        from pathlib import Path
        import chromadb
        from chromadb.config import Settings
        from llama_index import Document, VectorStoreIndex, ServiceContext
        from llama_index.embeddings import HuggingFaceEmbedding

        from ..types.polyglot_types import PolyglotType

        class PolyglotRAGStore:
            """RAG storage for polyglot types"""
            
            def __init__(self, storage_path: str = "./polyglot_storage"):
                self.storage_path = Path(storage_path)
                self.storage_path.mkdir(exist_ok=True)
                
                # Initialize ChromaDB for vector storage
                self.chroma_client = chromadb.Client(Settings(
                    chroma_db_impl="duckdb+parquet",
                    persist_directory=str(self.storage_path / "chroma")
                ))
                
                # Create or get collection
                self.collection = self.chroma_client.get_or_create_collection(
                    name="polyglot_types",
                    metadata={"description": "Polyglot type storage"}
                )
                
                # Initialize embedding model
                self.embed_model = HuggingFaceEmbedding(
                    model_name="microsoft/codebert-base"
                )
                
                # Type index
                self.type_index = {}
                self._load_index()
            
            def store_type(self, poly_type: PolyglotType):
                """Store a polyglot type in RAG"""
                type_id = poly_type.id
                
                # Create searchable document
                doc_text = self._create_document_text(poly_type)
                
                # Create metadata
                metadata = {
                    "type_id": type_id,
                    "canonical_name": poly_type.canonical_name,
                    "kind": poly_type.kind.value,
                    "source_language": poly_type.source_language,
                    "qualifiers": json.dumps([q.value for q in poly_type.qualifiers]),
                }
                
                # Add to ChromaDB
                self.collection.add(
                    documents=[doc_text],
                    metadatas=[metadata],
                    ids=[type_id]
                )
                
                # Save full type data
                type_file = self.storage_path / "types" / f"{type_id}.json"
                type_file.parent.mkdir(exist_ok=True)
                
                with open(type_file, 'w') as f:
                    json.dump(poly_type.to_dict(), f, indent=2)
                
                # Update index
                self.type_index[type_id] = {
                    "canonical_name": poly_type.canonical_name,
                    "file": str(type_file)
                }
                self._save_index()
            
            def search_types(self, query: str, n_results: int = 10) -> List[Dict[str, Any]]:
                """Search for types using semantic search"""
                results = self.collection.query(
                    query_texts=[query],
                    n_results=n_results
                )
                
                # Load full type data for results
                types = []
                for i, type_id in enumerate(results['ids'][0]):
                    type_data = self.load_type(type_id)
                    if type_data:
                        types.append({
                            "type": type_data,
                            "score": results['distances'][0][i] if 'distances' in results else 1.0,
                            "metadata": results['metadatas'][0][i]
                        })
                
                return types
            
            def load_type(self, type_id: str) -> Optional[Dict[str, Any]]:
                """Load a type by ID"""
                if type_id in self.type_index:
                    type_file = Path(self.type_index[type_id]["file"])
                    if type_file.exists():
                        with open(type_file, 'r') as f:
                            return json.load(f)
                return None
            
            def _create_document_text(self, poly_type: PolyglotType) -> str:
                """Create searchable document text from type"""
                parts = [
                    f"Type: {poly_type.canonical_name}",
                    f"Kind: {poly_type.kind.value}",
                    f"Language: {poly_type.source_language}",
                ]
                
                if poly_type.qualifiers:
                    parts.append(f"Qualifiers: {', '.join(q.value for q in poly_type.qualifiers)}")
                
                # Add type-specific information
                if isinstance(poly_type, ObjectType):
                    if poly_type.members:
                        parts.append(f"Members: {', '.join(poly_type.members.keys())}")
                    if poly_type.methods:
                        parts.append(f"Methods: {', '.join(poly_type.methods.keys())}")
                    if poly_type.base_types:
                        parts.append(f"Inherits from: {', '.join(b.canonical_name for b in poly_type.base_types)}")
                
                elif isinstance(poly_type, FunctionType):
                    parts.append(f"Returns: {poly_type.return_type.canonical_name if poly_type.return_type else 'void'}")
                    if poly_type.parameter_types:
                        params = ", ".join(p.canonical_name for p in poly_type.parameter_types)
                        parts.append(f"Parameters: {params}")
                
                elif isinstance(poly_type, TemplateType):
                    parts.append(f"Template: {poly_type.template_name}")
                    if poly_type.template_arguments:
                        args = ", ".join(a.canonical_name for a in poly_type.template_arguments)
                        parts.append(f"Arguments: {args}")
                
                return "\\n".join(parts)
            
            def _load_index(self):
                """Load type index from disk"""
                index_file = self.storage_path / "type_index.json"
                if index_file.exists():
                    with open(index_file, 'r') as f:
                        self.type_index = json.load(f)
            
            def _save_index(self):
                """Save type index to disk"""
                index_file = self.storage_path / "type_index.json"
                with open(index_file, 'w') as f:
                    json.dump(self.type_index, f, indent=2)
        ''',

        # Utils
        "src/utils/__init__.py": '',

        # Example files
        "examples/__init__.py": '',
        
        "examples/sample_cpp_code/vector_utils.hpp": '''
        #pragma once

        #include <vector>
        #include <string>
        #include <memory>
        #include <map>

        namespace utils {

        template<typename T>
        class VectorWrapper {
        private:
            std::vector<T> data_;
            mutable size_t access_count_;

        public:
            VectorWrapper() : access_count_(0) {}
            explicit VectorWrapper(size_t size) : data_(size), access_count_(0) {}
            
            void push_back(const T& value) {
                data_.push_back(value);
            }
            
            T& operator[](size_t index) {
                ++access_count_;
                return data_[index];
            }
            
            const T& operator[](size_t index) const {
                ++access_count_;
                return data_[index];
            }
            
            size_t size() const noexcept {
                return data_.size();
            }
            
            bool empty() const noexcept {
                return data_.empty();
            }
            
            size_t get_access_count() const noexcept {
                return access_count_;
            }
        };

        struct Point3D {
            double x, y, z;
            
            Point3D() : x(0), y(0), z(0) {}
            Point3D(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}
            
            double magnitude() const noexcept;
            Point3D normalize() const;
            
            Point3D operator+(const Point3D& other) const {
                return Point3D(x + other.x, y + other.y, z + other.z);
            }
        };

        class DataProcessor {
        public:
            using ResultMap = std::map<std::string, std::vector<double>>;
            
            virtual ~DataProcessor() = default;
            
            virtual ResultMap process(const std::vector<Point3D>& points) = 0;
            virtual void reset() noexcept = 0;
            
        protected:
            std::unique_ptr<VectorWrapper<double>> cache_;
        };

        // Function declarations
        std::vector<int> parse_integers(const std::string& input);
        std::unique_ptr<DataProcessor> create_processor(const std::string& type);

        template<typename T, typename U>
        std::vector<std::pair<T, U>> zip_vectors(const std::vector<T>& v1, 
                                                 const std::vector<U>& v2);

        } // namespace utils
        ''',

        "examples/extract_types.py": '''
        #!/usr/bin/env python3
        import sys
        from pathlib import Path
        import json
        from rich.console import Console
        from rich.table import Table
        from rich.tree import Tree

        # Add src to path
        sys.path.insert(0, str(Path(__file__).parent.parent))

        from src.extractors.cpp_extractor import CppTypeExtractor
        from src.converters.cpp_to_polyglot import CppToPolyglotConverter
        from src.storage.rag_store import PolyglotRAGStore

        console = Console()

        def display_type_tree(poly_type, tree=None):
            """Display type information as a tree"""
            if tree is None:
                tree = Tree(f"[bold blue]{poly_type.canonical_name}[/bold blue]")
            
            tree.add(f"Kind: [yellow]{poly_type.kind.value}[/yellow]")
            tree.add(f"ID: [dim]{poly_type.id}[/dim]")
            
            if poly_type.qualifiers:
                quals = tree.add("Qualifiers:")
                for q in poly_type.qualifiers:
                    quals.add(f"[red]{q.value}[/red]")
            
            if isinstance(poly_type, ObjectType):
                if poly_type.members:
                    members = tree.add(f"Members ({len(poly_type.members)}):")
                    for name, member_type in poly_type.members.items():
                        members.add(f"{name}: [cyan]{member_type.canonical_name}[/cyan]")
                
                if poly_type.methods:
                    methods = tree.add(f"Methods ({len(poly_type.methods)}):")
                    for name, method_type in poly_type.methods.items():
                        methods.add(f"{name}: [cyan]{method_type.canonical_name}[/cyan]")
            
            return tree

        def main():
            # Setup paths
            cpp_file = Path("examples/sample_cpp_code/vector_utils.hpp")
            include_paths = [str(cpp_file.parent)]
            
            console.print("[bold green]C++ Type Extraction Demo[/bold green]\\n")
            
            # Extract types
            console.print(f"Extracting types from: [blue]{cpp_file}[/blue]")
            extractor = CppTypeExtractor()
            cpp_types = extractor.extract_from_file(str(cpp_file), include_paths)
            
            console.print(f"Found [bold]{len(cpp_types)}[/bold] types\\n")
            
            # Convert to polyglot types
            converter = CppToPolyglotConverter()
            polyglot_types = {}
            
            for name, cpp_type in cpp_types.items():
                poly_type = converter.convert(cpp_type)
                polyglot_types[name] = poly_type
            
            # Display types
            for name, poly_type in polyglot_types.items():
                tree = display_type_tree(poly_type)
                console.print(tree)
                console.print()
            
            # Store in RAG
            console.print("[bold]Storing in RAG system...[/bold]")
            rag_store = PolyglotRAGStore()
            
            for poly_type in polyglot_types.values():
                rag_store.store_type(poly_type)
            
            # Test search
            console.print("\\n[bold]Testing search functionality:[/bold]")
            
            test_queries = [
                "vector container class",
                "3D point structure",
                "data processing interface",
                "template class with size method"
            ]
            
            for query in test_queries:
                console.print(f"\\nSearching for: [yellow]{query}[/yellow]")
                results = rag_store.search_types(query, n_results=3)
                
                if results:
                    table = Table(title="Search Results")
                    table.add_column("Type", style="cyan")
                    table.add_column("Kind", style="green")
                    table.add_column("Score", style="yellow")
                    
                    for result in results:
                        table.add_row(
                            result["type"]["canonical_name"],
                            result["type"]["kind"],
                            f"{result.get('score', 0):.3f}"
                        )
                    
                    console.print(table)
                else:
                    console.print("[red]No results found[/red]")

        if __name__ == "__main__":
            main()
        ''',

        # Test files
        "tests/__init__.py": '',
        
        "tests/test_cpp_extraction.py": '''
        import unittest
        import tempfile
        from pathlib import Path
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent))

        from src.extractors.cpp_extractor import CppTypeExtractor
        from src.types.polyglot_types import *

        class TestCppExtraction(unittest.TestCase):
            def setUp(self):
                self.extractor = CppTypeExtractor()
                self.temp_dir = tempfile.mkdtemp()
            
            def test_primitive_types(self):
                code = """
                int global_int;
                const double PI = 3.14159;
                unsigned long long big_number;
                """
                
                cpp_file = Path(self.temp_dir) / "test.cpp"
                cpp_file.write_text(code)
                
                types = self.extractor.extract_from_file(str(cpp_file))
                
                # We should find function types for the implicit constructors
                # but the primitive types themselves are built-in
            
            def test_class_extraction(self):
                code = """
                class TestClass {
                public:
                    int public_member;
                    void public_method();
                    
                private:
                    double private_member;
                    void private_method() const;
                };
                """
                
                cpp_file = Path(self.temp_dir) / "test.cpp"
                cpp_file.write_text(code)
                
                types = self.extractor.extract_from_file(str(cpp_file))
                
                # Find TestClass
                test_class = None
                for name, type_obj in types.items():
                    if name == "TestClass" and isinstance(type_obj, ObjectType):
                        test_class = type_obj
                        break
                
                self.assertIsNotNone(test_class)
                self.assertIn("public_member", test_class.members)
                self.assertIn("public_method", test_class.methods)
            
            def test_template_extraction(self):
                code = """
                template<typename T>
                class Container {
                    T* data;
                    size_t size;
                public:
                    T& operator[](size_t index);
                };
                """
                
                cpp_file = Path(self.temp_dir) / "test.cpp"
                cpp_file.write_text(code)
                
                types = self.extractor.extract_from_file(str(cpp_file))
                
                # Find Container template
                container = None
                for name, type_obj in types.items():
                    if name == "Container" and isinstance(type_obj, TemplateType):
                        container = type_obj
                        break
                
                self.assertIsNotNone(container)

        if __name__ == "__main__":
            unittest.main()
        ''',

        # Additional configuration files
        "pyproject.toml": '''
        [build-system]
        requires = ["setuptools>=61.0", "wheel"]
        build-backend = "setuptools.build_meta"

        [project]
        name = "polyglot-cpp"
        version = "0.1.0"
        description = "A polyglot type system for cross-language compatibility"
        readme = "README.md"
        requires-python = ">=3.8"
        classifiers = [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Developers",
            "Topic :: Software Development :: Libraries",
            "Programming Language :: Python :: 3",
        ]

        [tool.pytest.ini_options]
        testpaths = ["tests"]
        python_files = "test_*.py"

        [tool.black]
        line-length = 100
        target-version = ['py38']

        [tool.mypy]
        python_version = "3.8"
        warn_return_any = true
        warn_unused_configs = true
        ignore_missing_imports = true
        ''',
    }

    # Create all files
    for filepath, content in files.items():
        create_file(filepath, content)
    
    print("\n" + "=" * 50)
    print("Project generation complete!")
    print("\nNext steps:")
    print("1. Create a virtual environment: python -m venv venv")
    print("2. Activate it: source venv/bin/activate  # On Windows: venv\\Scripts\\activate")
    print("3. Install dependencies: pip install -r requirements.txt")
    print("4. Install package: pip install -e .")
    print("5. Run the example: python examples/extract_types.py")
    print("\nTo upload to GitHub:")
    print("1. git init")
    print("2. git add .")
    print("3. git commit -m 'Initial commit'")
    print("4. git remote add origin https://github.com/YOUR_USERNAME/polyglot-type-system.git")
    print("5. git push -u origin main")

if __name__ == "__main__":
    main()
