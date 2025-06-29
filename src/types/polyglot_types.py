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
