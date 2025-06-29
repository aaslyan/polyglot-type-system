#!/usr/bin/env python3
"""
Function Chain Synthesis Examples

This module demonstrates function chain synthesis using the RAG storage system
and the Function Chain System Specification. It includes examples with:
- Basic type chains
- Object-oriented chains with member functions
- Gap-filling and suggestions
- Multiple solution paths
- Complex type relationships
"""

from pathlib import Path
import sys
import json
import argparse
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import networkx as nx
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from src.storage.rag_store import PolyglotRAGStore
from src.types.polyglot_types import PolyglotType, TypeKind

# Global verbosity level
VERBOSITY = 0

def log(message: str, level: int = 1, indent: int = 0):
    """Log message if verbosity level is high enough"""
    if VERBOSITY >= level:
        prefix = "  " * indent
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3] if VERBOSITY >= 3 else ""
        if timestamp:
            print(f"[{timestamp}] {prefix}{message}")
        else:
            print(f"{prefix}{message}")

# ============================================================================
# Type System Definition
# ============================================================================

@dataclass
class TypeDefinition:
    """Represents a type in our type system"""
    name: str
    kind: str  # "basic", "object", "generic"
    properties: Dict[str, str] = field(default_factory=dict)
    methods: List[Dict[str, Any]] = field(default_factory=list)
    constructor: Optional[Dict[str, Any]] = None
    supertype: Optional[str] = None

@dataclass
class FunctionDefinition:
    """Represents a function in our library"""
    name: str
    args: List[str]
    return_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    owner: Optional[str] = None  # For member functions

@dataclass
class ChainResult:
    """Result of function chain synthesis"""
    chain: List[str]
    success: bool
    output_type: str
    gaps: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    confidence: float = 1.0

class TypeSystem:
    """Dynamic type system for function chain synthesis"""
    
    def __init__(self, definition: Dict[str, Any]):
        self.types = {}
        self.hierarchy = {}
        self.coercions = []
        self.matching_mode = "nominal"
        self.parse_definition(definition)
    
    def parse_definition(self, definition: Dict[str, Any]):
        """Parse type system definition from JSON-like structure"""
        # Parse types
        for type_def in definition.get("types", []):
            td = TypeDefinition(
                name=type_def["name"],
                kind=type_def["kind"],
                properties=type_def.get("properties", {}),
                methods=type_def.get("methods", []),
                constructor=type_def.get("constructor")
            )
            self.types[td.name] = td
        
        # Parse hierarchy
        self.hierarchy = definition.get("hierarchy", {})
        
        # Parse coercions
        self.coercions = definition.get("coercions", [])
        
        # Parse matching mode
        self.matching_mode = definition.get("matching", "nominal")
    
    def is_compatible(self, type1: str, type2: str) -> bool:
        """Check if type1 is compatible with type2"""
        log(f"Checking compatibility: {type1} -> {type2}", level=3, indent=4)
        
        if type1 == type2:
            log(f"✓ Exact match: {type1} == {type2}", level=3, indent=5)
            return True
        
        # Check hierarchy
        if type1 in self.hierarchy:
            supertype = self.hierarchy[type1].get("supertype")
            if supertype == type2:
                log(f"✓ Hierarchy match: {type1} <: {type2}", level=3, indent=5)
                return True
            # Recursive check
            if supertype and self.is_compatible(supertype, type2):
                log(f"✓ Transitive hierarchy: {type1} <: ... <: {type2}", level=3, indent=5)
                return True
        
        # Check coercions
        for coercion in self.coercions:
            if coercion["from"] == type1 and coercion["to"] == type2:
                log(f"✓ Coercion found: {type1} -> {type2} via {coercion.get('condition', 'implicit')}", 
                    level=3, indent=5)
                return True
        
        # Structural matching for objects
        if self.matching_mode == "structural":
            if type1 in self.types and type2 in self.types:
                t1 = self.types[type1]
                t2 = self.types[type2]
                if t1.kind == "object" and t2.kind == "object":
                    log(f"Checking structural compatibility for objects", level=3, indent=5)
                    # Check if t1 has all properties of t2
                    for prop, ptype in t2.properties.items():
                        if prop not in t1.properties:
                            log(f"✗ Missing property: {prop}", level=3, indent=6)
                            return False
                        if not self.is_compatible(t1.properties[prop], ptype):
                            log(f"✗ Incompatible property type: {prop}", level=3, indent=6)
                            return False
                    log(f"✓ Structural match: {type1} ~ {type2}", level=3, indent=5)
                    return True
        
        log(f"✗ No compatibility found: {type1} -> {type2}", level=3, indent=5)
        return False
    
    def get_type_distance(self, type1: str, type2: str) -> float:
        """Calculate distance between two types (0 = same, higher = more different)"""
        if type1 == type2:
            return 0.0
        
        if self.is_compatible(type1, type2):
            return 0.5  # Compatible but not same
        
        # Check if they share a common ancestor
        ancestors1 = self.get_ancestors(type1)
        ancestors2 = self.get_ancestors(type2)
        common = ancestors1.intersection(ancestors2)
        if common:
            return 1.0 + min(len(ancestors1 - common), len(ancestors2 - common))
        
        return float('inf')  # No relationship
    
    def get_ancestors(self, type_name: str) -> Set[str]:
        """Get all ancestor types"""
        ancestors = set()
        current = type_name
        while current in self.hierarchy:
            supertype = self.hierarchy[current].get("supertype")
            if supertype:
                ancestors.add(supertype)
                current = supertype
            else:
                break
        return ancestors

class FunctionChainSynthesizer:
    """Synthesizes function chains to produce desired output types"""
    
    def __init__(self, type_system: TypeSystem, library: List[FunctionDefinition], 
                 rag_storage: Optional[PolyglotRAGStore] = None):
        self.type_system = type_system
        self.library = library
        self.rag_storage = rag_storage
        self.graph = self._build_dependency_graph()
        self.search_tree = []  # For visualization
    
    def _build_dependency_graph(self) -> nx.DiGraph:
        """Build dependency graph from functions and types"""
        G = nx.DiGraph()
        
        # Add type nodes
        for type_name in self.type_system.types:
            G.add_node(type_name, node_type="type")
        
        # Add function nodes and edges
        for i, func in enumerate(self.library):
            func_node = f"func_{i}_{func.name}"
            G.add_node(func_node, node_type="function", function=func)
            
            # Edge from function to return type
            G.add_edge(func_node, func.return_type, edge_type="produces")
            
            # Edges from argument types to function
            for arg_type in func.args:
                G.add_edge(arg_type, func_node, edge_type="requires")
        
        # Log graph statistics if verbose
        if VERBOSITY >= 3:
            log(f"\n=== Dependency Graph Statistics ===", level=3)
            log(f"Nodes: {G.number_of_nodes()} ({len(self.type_system.types)} types, {len(self.library)} functions)", level=3)
            log(f"Edges: {G.number_of_edges()}", level=3)
            
            # Show some connections
            log(f"\nSample connections:", level=3)
            for node in list(G.nodes())[:5]:
                in_edges = list(G.in_edges(node))
                out_edges = list(G.out_edges(node))
                if in_edges or out_edges:
                    log(f"  {node}:", level=3)
                    if in_edges:
                        log(f"    ← {[e[0] for e in in_edges[:3]]}", level=3)
                    if out_edges:
                        log(f"    → {[e[1] for e in out_edges[:3]]}", level=3)
        
        return G
    
    def _visualize_search_tree(self):
        """Print visual representation of search tree"""
        if VERBOSITY >= 2 and self.search_tree:
            log("\n=== Search Tree Visualization ===", level=2)
            for entry in self.search_tree:
                indent = "  " * entry["depth"]
                symbol = "✓" if entry["success"] else "✗" if entry["failed"] else "→"
                log(f"{indent}{symbol} {entry['target']} via {entry['function']}", level=2)
    
    def synthesize_chain(self, target_type: str, available_resources: List[str], 
                        constraints: Optional[Dict[str, Any]] = None) -> ChainResult:
        """Synthesize a function chain to produce the target type"""
        log(f"\n{'='*60}", level=1)
        log(f"Starting chain synthesis", level=1)
        log(f"Target type: {target_type}", level=1, indent=1)
        log(f"Available resources: {available_resources}", level=1, indent=1)
        log(f"Constraints: {constraints}", level=2, indent=1)
        
        # Reset search tree for visualization
        self.search_tree = []
        
        # Initialize search state
        state = {
            "resources": set(available_resources),
            "chain": [],
            "visited": set(),
            "max_depth": constraints.get("max_calls", 10) if constraints else 10
        }
        
        log(f"\nSearching for complete chain...", level=2)
        
        # Try to find complete chain
        result = self._search_chain(target_type, state, 0)
        
        # Visualize search tree if verbose
        self._visualize_search_tree()
        
        if result:
            log(f"\n✓ Found complete chain: {' -> '.join(result)}", level=1)
            return ChainResult(
                chain=result,
                success=True,
                output_type=target_type,
                confidence=1.0
            )
        
        log(f"\n✗ No complete chain found, attempting gap-filling...", level=1)
        
        # If no complete chain, try gap-filling
        partial_result = self._find_partial_chain(target_type, state)
        return self._generate_gap_filling_suggestions(partial_result, target_type)
    
    def _search_chain(self, target: str, state: Dict, depth: int) -> Optional[List[str]]:
        """Recursive search for function chain"""
        log(f"{'  '*depth}→ Searching for: {target} (depth={depth})", level=2)
        
        if depth > state["max_depth"]:
            log(f"{'  '*depth}✗ Max depth reached ({depth})", level=2)
            return None
        
        # Check if target is already available
        if target in state["resources"]:
            log(f"{'  '*depth}✓ Target already in resources: {target}", level=2)
            self.search_tree.append({
                "depth": depth,
                "target": target,
                "function": "already_available",
                "success": True,
                "failed": False
            })
            return []
        
        # Find functions that produce the target
        log(f"{'  '*depth}Finding candidates for {target}...", level=2)
        candidates = []
        for i, func in enumerate(self.library):
            if self.type_system.is_compatible(func.return_type, target):
                candidates.append((i, func))
                log(f"{'  '*depth}  + Candidate: {func.name} (returns {func.return_type})", level=2)
        
        if not candidates:
            log(f"{'  '*depth}✗ No candidates found for {target}", level=2)
            self.search_tree.append({
                "depth": depth,
                "target": target,
                "function": "no_candidates",
                "success": False,
                "failed": True
            })
            return None
        
        # Try each candidate
        for idx, func in candidates:
            func_id = f"{func.name}_{idx}"
            if func_id in state["visited"]:
                log(f"{'  '*depth}  - Skipping {func.name} (already visited)", level=2)
                continue
            
            log(f"{'  '*depth}Trying: {func.name} ({func.args} -> {func.return_type})", level=2)
            
            # Add to search tree
            tree_entry = {
                "depth": depth,
                "target": target,
                "function": func.name,
                "success": False,
                "failed": False
            }
            self.search_tree.append(tree_entry)
            
            # Check if all arguments are satisfiable
            sub_chains = []
            all_satisfied = True
            
            for j, arg_type in enumerate(func.args):
                log(f"{'  '*depth}  Checking arg[{j}]: {arg_type}", level=2)
                
                if arg_type in state["resources"]:
                    log(f"{'  '*depth}    ✓ Available in resources", level=2)
                    sub_chains.append([])
                else:
                    # Recursively search for argument
                    log(f"{'  '*depth}    → Recursively searching for {arg_type}", level=2)
                    state["visited"].add(func_id)
                    sub_chain = self._search_chain(arg_type, state, depth + 1)
                    state["visited"].remove(func_id)
                    
                    if sub_chain is None:
                        log(f"{'  '*depth}    ✗ Cannot satisfy argument {arg_type}", level=2)
                        all_satisfied = False
                        break
                    else:
                        log(f"{'  '*depth}    ✓ Found sub-chain for {arg_type}: {sub_chain}", level=2)
                        sub_chains.append(sub_chain)
            
            if all_satisfied:
                # Build complete chain
                chain = []
                for sub_chain in sub_chains:
                    chain.extend(sub_chain)
                chain.append(func.name)
                
                # Update resources
                state["resources"].add(func.return_type)
                
                # Mark success in tree
                tree_entry["success"] = True
                
                log(f"{'  '*depth}✓ Success! Chain for {target}: {chain}", level=2)
                return chain
            else:
                # Mark failure in tree
                tree_entry["failed"] = True
                log(f"{'  '*depth}✗ Cannot use {func.name} (missing arguments)", level=2)
        
        log(f"{'  '*depth}✗ No valid chain found for {target}", level=2)
        return None
    
    def _find_partial_chain(self, target: str, state: Dict) -> ChainResult:
        """Find best partial chain when complete chain is not possible"""
        log(f"\nSearching for partial chains to approximate {target}", level=2)
        
        best_partial = None
        best_distance = float('inf')
        
        # Try each function that produces something related to target
        for i, func in enumerate(self.library):
            distance = self.type_system.get_type_distance(func.return_type, target)
            log(f"  Evaluating {func.name}: returns {func.return_type}, distance={distance}", level=2)
            
            if distance < best_distance:
                # Check how many arguments we can satisfy
                satisfied_args = []
                missing_args = []
                
                for arg_type in func.args:
                    if arg_type in state["resources"]:
                        satisfied_args.append(arg_type)
                        log(f"    ✓ Can satisfy arg: {arg_type}", level=3)
                    else:
                        # Try to find a chain for this argument
                        arg_chain = self._search_chain(arg_type, state, 0)
                        if arg_chain is not None:
                            satisfied_args.append(arg_type)
                            log(f"    ✓ Can produce arg: {arg_type} via chain", level=3)
                        else:
                            missing_args.append(arg_type)
                            log(f"    ✗ Cannot satisfy arg: {arg_type}", level=3)
                
                if len(satisfied_args) > 0:  # At least some progress
                    confidence = len(satisfied_args) / len(func.args) if func.args else 1.0
                    log(f"  → Partial candidate: {func.name} (confidence={confidence:.2f})", level=2)
                    
                    best_partial = ChainResult(
                        chain=[func.name],
                        success=False,
                        output_type=func.return_type,
                        gaps=missing_args,
                        confidence=confidence
                    )
                    best_distance = distance
        
        if best_partial:
            log(f"\nBest partial chain: {best_partial.chain[0]} -> {best_partial.output_type}", level=1)
            log(f"Missing arguments: {best_partial.gaps}", level=1)
        else:
            log(f"\nNo partial chain found", level=1)
        
        return best_partial or ChainResult(
            chain=[],
            success=False,
            output_type="",
            gaps=[target],
            confidence=0.0
        )
    
    def _generate_gap_filling_suggestions(self, partial: ChainResult, 
                                        target: str) -> ChainResult:
        """Generate suggestions for completing partial chains"""
        log(f"\nGenerating gap-filling suggestions", level=2)
        log(f"Target: {target}, Partial output: {partial.output_type}", level=2)
        log(f"Gaps: {partial.gaps}", level=2)
        
        suggestions = []
        
        # Suggest type conversions
        if partial.output_type:
            log(f"\nChecking for type conversions from {partial.output_type}...", level=3)
            for coercion in self.type_system.coercions:
                if coercion["from"] == partial.output_type:
                    suggestion = (f"Convert {partial.output_type} to {coercion['to']} "
                                f"using {coercion.get('condition', 'coercion')}")
                    suggestions.append(suggestion)
                    log(f"  + {suggestion}", level=3)
        
        # Suggest missing functions
        log(f"\nAnalyzing missing functions for gaps...", level=3)
        for gap in partial.gaps:
            suggestion = f"Need function that produces {gap}"
            suggestions.append(suggestion)
            log(f"  + {suggestion}", level=3)
            
            # Look for similar types
            similar_types = []
            for type_name in self.type_system.types:
                distance = self.type_system.get_type_distance(type_name, gap)
                if 0 < distance < 2:
                    similar_types.append(type_name)
            
            if similar_types:
                suggestion = f"Consider using similar types: {', '.join(similar_types)}"
                suggestions.append(suggestion)
                log(f"  + {suggestion}", level=3)
        
        # Suggest constructors for object types
        if target in self.type_system.types:
            type_def = self.type_system.types[target]
            if type_def.kind == "object" and type_def.constructor:
                args = type_def.constructor.get("args", [])
                suggestion = f"Create {target} using constructor: new {target}({', '.join(args)})"
                suggestions.append(suggestion)
                log(f"  + {suggestion}", level=3)
        
        partial.suggestions = suggestions
        log(f"\nGenerated {len(suggestions)} suggestions", level=2)
        return partial

# ============================================================================
# Example Scenarios
# ============================================================================

def show_example_summary(example_name: str, start_time: float = None):
    """Show summary statistics for an example"""
    if VERBOSITY >= 1:
        log(f"\n{'─'*60}", level=1)
        if start_time:
            elapsed = datetime.now().timestamp() - start_time
            log(f"Example completed in {elapsed:.3f} seconds", level=1)

def example_1_basic_type_chain():
    """Example 1: Basic type chain synthesis"""
    start_time = datetime.now().timestamp() if VERBOSITY >= 1 else None
    
    print("\n" + "="*80)
    print("Example 1: Basic Type Chain Synthesis")
    print("="*80)
    
    # Define type system
    type_system_def = {
        "types": [
            {"name": "int", "kind": "basic"},
            {"name": "float", "kind": "basic"},
            {"name": "string", "kind": "basic"},
            {"name": "bool", "kind": "basic"}
        ],
        "hierarchy": {
            "int": {"supertype": "number"},
            "float": {"supertype": "number"}
        },
        "coercions": [
            {"from": "int", "to": "float", "condition": "implicit_cast"},
            {"from": "float", "to": "string", "condition": "to_string"},
            {"from": "bool", "to": "int", "condition": "bool_to_int"}
        ],
        "matching": "nominal"
    }
    
    # Define function library
    functions = [
        FunctionDefinition("parse_int", ["string"], "int", {"cost": "low"}),
        FunctionDefinition("add", ["int", "int"], "int", {"cost": "low"}),
        FunctionDefinition("multiply", ["float", "float"], "float", {"cost": "low"}),
        FunctionDefinition("format_number", ["float"], "string", {"cost": "medium"}),
        FunctionDefinition("is_positive", ["int"], "bool", {"cost": "low"}),
        FunctionDefinition("bool_to_int", ["bool"], "int", {"cost": "low"})
    ]
    
    log("\n=== Type System Definition ===", level=2)
    log(f"Types: {[t['name'] for t in type_system_def['types']]}", level=2)
    log(f"Hierarchy: {type_system_def['hierarchy']}", level=2)
    log(f"Coercions: {[(c['from'], c['to']) for c in type_system_def['coercions']]}", level=2)
    
    log("\n=== Function Library ===", level=2)
    for func in functions:
        log(f"  {func.name}: {func.args} -> {func.return_type}", level=2)
    
    # Create synthesizer
    type_system = TypeSystem(type_system_def)
    synthesizer = FunctionChainSynthesizer(type_system, functions)
    
    # Test case 1: Simple chain
    print("\nTest 1: Produce 'string' from available 'int'")
    result = synthesizer.synthesize_chain(
        target_type="string",
        available_resources=["int"],
        constraints={"max_calls": 3}
    )
    print(f"Success: {result.success}")
    print(f"Chain: {' -> '.join(result.chain) if result.chain else 'None'}")
    print(f"Output type: {result.output_type}")
    
    # Test case 2: Complex chain
    print("\nTest 2: Produce 'float' from available 'bool'")
    result = synthesizer.synthesize_chain(
        target_type="float",
        available_resources=["bool"],
        constraints={"max_calls": 5}
    )
    print(f"Success: {result.success}")
    print(f"Chain: {' -> '.join(result.chain) if result.chain else 'None'}")
    
    # Test case 3: Impossible chain (gap-filling)
    print("\nTest 3: Produce 'complex' from available 'string' (should fail)")
    result = synthesizer.synthesize_chain(
        target_type="complex",
        available_resources=["string"],
        constraints={"max_calls": 5}
    )
    print(f"Success: {result.success}")
    print(f"Partial chain: {' -> '.join(result.chain) if result.chain else 'None'}")
    print(f"Gaps: {result.gaps}")
    if result.suggestions:
        print(f"Suggestions:")
        for suggestion in result.suggestions:
            print(f"  - {suggestion}")
    
    show_example_summary("Example 1", start_time)

def example_2_object_oriented_chains():
    """Example 2: Object-oriented function chains with member functions"""
    print("\n" + "="*80)
    print("Example 2: Object-Oriented Function Chains")
    print("="*80)
    
    # Define type system with objects
    type_system_def = {
        "types": [
            {"name": "string", "kind": "basic"},
            {"name": "int", "kind": "basic"},
            {"name": "float", "kind": "basic"},
            {
                "name": "User",
                "kind": "object",
                "properties": {
                    "name": "string",
                    "age": "int"
                },
                "methods": [
                    {"name": "getName", "args": [], "return": "string"},
                    {"name": "getAge", "args": [], "return": "int"},
                    {"name": "setAge", "args": ["int"], "return": "void"}
                ],
                "constructor": {"args": ["string", "int"]}
            },
            {
                "name": "Account",
                "kind": "object",
                "properties": {
                    "owner": "User",
                    "balance": "float"
                },
                "methods": [
                    {"name": "getBalance", "args": [], "return": "float"},
                    {"name": "getOwnerName", "args": [], "return": "string"},
                    {"name": "deposit", "args": ["float"], "return": "void"}
                ],
                "constructor": {"args": ["User", "float"]}
            }
        ],
        "hierarchy": {},
        "coercions": [
            {"from": "int", "to": "float", "condition": "implicit_cast"}
        ],
        "matching": "nominal"
    }
    
    # Define function library with member functions
    functions = [
        # Constructors
        FunctionDefinition("new_User", ["string", "int"], "User", 
                         {"cost": "low"}, owner=None),
        FunctionDefinition("new_Account", ["User", "float"], "Account", 
                         {"cost": "medium"}, owner=None),
        # Member functions
        FunctionDefinition("User.getName", ["User"], "string", 
                         {"cost": "low"}, owner="User"),
        FunctionDefinition("User.getAge", ["User"], "int", 
                         {"cost": "low"}, owner="User"),
        FunctionDefinition("Account.getBalance", ["Account"], "float", 
                         {"cost": "low"}, owner="Account"),
        FunctionDefinition("Account.getOwnerName", ["Account"], "string", 
                         {"cost": "medium"}, owner="Account"),
        # Free functions
        FunctionDefinition("format_currency", ["float"], "string", 
                         {"cost": "low"}, owner=None),
        FunctionDefinition("parse_age", ["string"], "int", 
                         {"cost": "low"}, owner=None)
    ]
    
    # Create synthesizer
    type_system = TypeSystem(type_system_def)
    synthesizer = FunctionChainSynthesizer(type_system, functions)
    
    # Test case 1: Create object and call method
    print("\nTest 1: Produce 'string' (user name) from 'string' and 'int'")
    result = synthesizer.synthesize_chain(
        target_type="string",
        available_resources=["string", "int"],
        constraints={"max_calls": 4}
    )
    print(f"Success: {result.success}")
    print(f"Chain: {' -> '.join(result.chain) if result.chain else 'None'}")
    print("Interpretation: Create User object, then call getName()")
    
    # Test case 2: Complex object chain
    print("\nTest 2: Get account balance as string from user data")
    result = synthesizer.synthesize_chain(
        target_type="string",
        available_resources=["string", "int", "float"],
        constraints={"max_calls": 6}
    )
    print(f"Success: {result.success}")
    print(f"Chain: {' -> '.join(result.chain) if result.chain else 'None'}")
    
    # Test case 3: Missing object constructor
    print("\nTest 3: Try to use Account without enough resources")
    result = synthesizer.synthesize_chain(
        target_type="float",
        available_resources=["string"],  # Missing User object
        constraints={"max_calls": 5}
    )
    print(f"Success: {result.success}")
    print(f"Gaps: {result.gaps}")
    print(f"Suggestions:")
    for suggestion in result.suggestions[:3]:
        print(f"  - {suggestion}")

def example_3_real_world_data_processing():
    """Example 3: Real-world data processing pipeline"""
    print("\n" + "="*80)
    print("Example 3: Real-World Data Processing Pipeline")
    print("="*80)
    
    # Define type system for data processing
    type_system_def = {
        "types": [
            {"name": "string", "kind": "basic"},
            {"name": "int", "kind": "basic"},
            {"name": "float", "kind": "basic"},
            {"name": "bool", "kind": "basic"},
            {
                "name": "DataFrame",
                "kind": "object",
                "properties": {"columns": "List[string]", "rows": "int"},
                "methods": [
                    {"name": "filter", "args": ["string"], "return": "DataFrame"},
                    {"name": "aggregate", "args": ["string"], "return": "float"},
                    {"name": "to_csv", "args": [], "return": "string"}
                ]
            },
            {
                "name": "Model",
                "kind": "object",
                "properties": {"name": "string", "accuracy": "float"},
                "methods": [
                    {"name": "predict", "args": ["DataFrame"], "return": "DataFrame"},
                    {"name": "evaluate", "args": ["DataFrame"], "return": "float"}
                ]
            },
            {"name": "List[T]", "kind": "generic", "params": ["T"]}
        ],
        "hierarchy": {},
        "coercions": [
            {"from": "DataFrame", "to": "string", "condition": "to_csv"}
        ],
        "matching": "structural"
    }
    
    # Define data processing functions
    functions = [
        # Data loading
        FunctionDefinition("load_csv", ["string"], "DataFrame", 
                         {"cost": "high", "description": "Load CSV file"}),
        FunctionDefinition("load_json", ["string"], "DataFrame", 
                         {"cost": "high", "description": "Load JSON file"}),
        # Data processing
        FunctionDefinition("clean_data", ["DataFrame"], "DataFrame", 
                         {"cost": "medium", "description": "Clean missing values"}),
        FunctionDefinition("normalize", ["DataFrame"], "DataFrame", 
                         {"cost": "medium", "description": "Normalize numeric columns"}),
        # Model operations
        FunctionDefinition("train_model", ["DataFrame"], "Model", 
                         {"cost": "very_high", "description": "Train ML model"}),
        FunctionDefinition("load_model", ["string"], "Model", 
                         {"cost": "low", "description": "Load pre-trained model"}),
        # Output operations
        FunctionDefinition("generate_report", ["DataFrame", "float"], "string", 
                         {"cost": "medium", "description": "Generate analysis report"}),
        # Member functions
        FunctionDefinition("DataFrame.filter", ["DataFrame", "string"], "DataFrame", 
                         {"cost": "low"}, owner="DataFrame"),
        FunctionDefinition("DataFrame.aggregate", ["DataFrame", "string"], "float", 
                         {"cost": "medium"}, owner="DataFrame"),
        FunctionDefinition("Model.predict", ["Model", "DataFrame"], "DataFrame", 
                         {"cost": "medium"}, owner="Model"),
        FunctionDefinition("Model.evaluate", ["Model", "DataFrame"], "float", 
                         {"cost": "medium"}, owner="Model")
    ]
    
    # Create synthesizer with RAG storage
    type_system = TypeSystem(type_system_def)
    rag_storage = PolyglotRAGStore("data_processing_functions")
    
    # Store functions in RAG for semantic search
    for func in functions:
        poly_type = PolyglotType(
            canonical_name=func.name,
            kind=TypeKind.FUNCTION,
            metadata={
                "args": func.args,
                "return": func.return_type,
                "description": func.metadata.get("description", ""),
                "cost": func.metadata.get("cost", "unknown")
            }
        )
        rag_storage.store_type(poly_type)
    
    synthesizer = FunctionChainSynthesizer(type_system, functions, rag_storage)
    
    # Test case 1: Build data processing pipeline
    print("\nTest 1: Build pipeline from CSV to prediction results")
    result = synthesizer.synthesize_chain(
        target_type="DataFrame",  # Predicted DataFrame
        available_resources=["string"],  # CSV filename
        constraints={"max_calls": 6, "prefer_low_cost": True}
    )
    print(f"Success: {result.success}")
    print(f"Chain: {' -> '.join(result.chain) if result.chain else 'None'}")
    print("\nPipeline interpretation:")
    print("1. Load CSV file -> DataFrame")
    print("2. Clean data -> DataFrame")
    print("3. Either train model or load pre-trained model -> Model")
    print("4. Make predictions -> DataFrame")
    
    # Test case 2: Generate analysis report
    print("\nTest 2: Generate analysis report from raw data")
    result = synthesizer.synthesize_chain(
        target_type="string",  # Report
        available_resources=["string", "string"],  # CSV file and model file
        constraints={"max_calls": 8}
    )
    print(f"Success: {result.success}")
    print(f"Chain: {' -> '.join(result.chain) if result.chain else 'None'}")
    
    # Test case 3: Evaluate model performance
    print("\nTest 3: Evaluate model accuracy (missing test data)")
    result = synthesizer.synthesize_chain(
        target_type="float",  # Accuracy score
        available_resources=["Model"],  # Only have model, no test data
        constraints={"max_calls": 5}
    )
    print(f"Success: {result.success}")
    print(f"Gaps: {result.gaps}")
    print(f"Suggestions:")
    for suggestion in result.suggestions:
        print(f"  - {suggestion}")

def example_4_multi_path_synthesis():
    """Example 4: Multiple solution paths and optimization"""
    print("\n" + "="*80)
    print("Example 4: Multi-Path Synthesis with Optimization")
    print("="*80)
    
    # Enhanced synthesizer that finds multiple paths
    class MultiPathSynthesizer(FunctionChainSynthesizer):
        def find_all_chains(self, target_type: str, available_resources: List[str], 
                           max_solutions: int = 5) -> List[ChainResult]:
            """Find multiple solution paths"""
            solutions = []
            state = {
                "resources": set(available_resources),
                "chain": [],
                "visited": set(),
                "max_depth": 10
            }
            
            # Use modified search that doesn't stop at first solution
            self._find_all_chains_recursive(
                target_type, state, 0, solutions, max_solutions
            )
            
            # Sort by quality metrics
            solutions.sort(key=lambda x: (
                len(x.chain),  # Shorter is better
                -x.confidence,  # Higher confidence is better
                sum(self._get_function_cost(f) for f in x.chain)  # Lower cost
            ))
            
            return solutions
        
        def _find_all_chains_recursive(self, target: str, state: Dict, depth: int,
                                     solutions: List[ChainResult], max_solutions: int):
            """Recursive search for all possible chains"""
            if len(solutions) >= max_solutions:
                return
            
            if depth > state["max_depth"]:
                return
            
            if target in state["resources"]:
                # Found a solution
                solutions.append(ChainResult(
                    chain=state["chain"].copy(),
                    success=True,
                    output_type=target,
                    confidence=1.0
                ))
                return
            
            # Try all functions that produce the target
            for i, func in enumerate(self.library):
                if self.type_system.is_compatible(func.return_type, target):
                    func_id = f"{func.name}_{i}"
                    if func_id not in state["visited"]:
                        # Try this function
                        state["visited"].add(func_id)
                        state["chain"].append(func.name)
                        
                        # Continue search
                        self._find_all_chains_recursive(
                            target, state, depth + 1, solutions, max_solutions
                        )
                        
                        # Backtrack
                        state["chain"].pop()
                        state["visited"].remove(func_id)
        
        def _get_function_cost(self, func_name: str) -> float:
            """Get cost of a function"""
            for func in self.library:
                if func.name == func_name:
                    cost_map = {
                        "low": 1.0,
                        "medium": 2.0,
                        "high": 5.0,
                        "very_high": 10.0
                    }
                    return cost_map.get(func.metadata.get("cost", "medium"), 2.0)
            return 2.0
    
    # Define type system
    type_system_def = {
        "types": [
            {"name": "string", "kind": "basic"},
            {"name": "int", "kind": "basic"},
            {"name": "float", "kind": "basic"},
            {"name": "json", "kind": "basic"},
            {"name": "xml", "kind": "basic"}
        ],
        "hierarchy": {},
        "coercions": [
            {"from": "int", "to": "string"},
            {"from": "float", "to": "string"},
            {"from": "json", "to": "string"},
            {"from": "xml", "to": "string"}
        ],
        "matching": "nominal"
    }
    
    # Define multiple paths to same output
    functions = [
        # Path 1: Direct conversion
        FunctionDefinition("to_string_direct", ["int"], "string", 
                         {"cost": "low", "quality": "basic"}),
        # Path 2: Through float
        FunctionDefinition("to_float", ["int"], "float", 
                         {"cost": "low"}),
        FunctionDefinition("format_float", ["float"], "string", 
                         {"cost": "medium", "quality": "formatted"}),
        # Path 3: Through JSON
        FunctionDefinition("to_json", ["int"], "json", 
                         {"cost": "medium"}),
        FunctionDefinition("json_to_string", ["json"], "string", 
                         {"cost": "low", "quality": "structured"}),
        # Path 4: Through XML
        FunctionDefinition("to_xml", ["int"], "xml", 
                         {"cost": "high"}),
        FunctionDefinition("xml_to_string", ["xml"], "string", 
                         {"cost": "low", "quality": "structured"}),
        # Additional formatters
        FunctionDefinition("pretty_format", ["string"], "string", 
                         {"cost": "low", "quality": "enhanced"})
    ]
    
    # Create multi-path synthesizer
    type_system = TypeSystem(type_system_def)
    synthesizer = MultiPathSynthesizer(type_system, functions)
    
    # Find all paths from int to string
    print("\nFinding all paths from 'int' to 'string':")
    solutions = synthesizer.find_all_chains(
        target_type="string",
        available_resources=["int"],
        max_solutions=10
    )
    
    print(f"\nFound {len(solutions)} solution paths:")
    for i, solution in enumerate(solutions):
        total_cost = sum(synthesizer._get_function_cost(f) for f in solution.chain)
        print(f"\nPath {i + 1}:")
        print(f"  Chain: {' -> '.join(solution.chain)}")
        print(f"  Length: {len(solution.chain)}")
        print(f"  Total cost: {total_cost}")
        print(f"  Confidence: {solution.confidence}")

def example_5_gap_filling_advanced():
    """Example 5: Advanced gap-filling with semantic search"""
    print("\n" + "="*80)
    print("Example 5: Advanced Gap-Filling with Semantic Search")
    print("="*80)
    
    # Enhanced gap-filling synthesizer
    class SemanticGapFillingSynthesizer(FunctionChainSynthesizer):
        def _generate_gap_filling_suggestions(self, partial: ChainResult, 
                                            target: str) -> ChainResult:
            """Enhanced gap-filling with semantic search"""
            suggestions = []
            
            # Basic suggestions from parent class
            partial = super()._generate_gap_filling_suggestions(partial, target)
            suggestions.extend(partial.suggestions)
            
            # Semantic search for similar functions
            if self.rag_storage:
                for gap in partial.gaps:
                    # Search for functions that might help
                    query = f"function that produces {gap} or similar type"
                    similar_funcs = self.rag_storage.search_by_content(query, n_results=3)
                    
                    if similar_funcs:
                        suggestions.append(f"\nSemantic search results for '{gap}':")
                        for func in similar_funcs:
                            func_name = func.name
                            func_return = func.metadata.get("return", "unknown")
                            suggestions.append(
                                f"  - Consider: {func_name} (returns {func_return})"
                            )
            
            # Suggest function compositions
            if len(partial.gaps) > 1:
                suggestions.append("\nFunction composition suggestions:")
                suggestions.append(
                    f"  - Combine functions to bridge gaps: {' + '.join(partial.gaps)}"
                )
            
            # Suggest design patterns
            if target in self.type_system.types:
                type_def = self.type_system.types[target]
                if type_def.kind == "object":
                    suggestions.append("\nDesign pattern suggestions:")
                    suggestions.append(f"  - Use Builder pattern for {target}")
                    suggestions.append(f"  - Use Factory pattern for {target}")
            
            partial.suggestions = suggestions
            return partial
    
    # Complex type system
    type_system_def = {
        "types": [
            {"name": "string", "kind": "basic"},
            {"name": "int", "kind": "basic"},
            {"name": "float", "kind": "basic"},
            {
                "name": "Image",
                "kind": "object",
                "properties": {"width": "int", "height": "int", "data": "bytes"},
                "methods": [
                    {"name": "resize", "args": ["int", "int"], "return": "Image"},
                    {"name": "toGrayscale", "args": [], "return": "Image"}
                ]
            },
            {
                "name": "Video",
                "kind": "object",
                "properties": {"frames": "List[Image]", "fps": "float"},
                "methods": [
                    {"name": "extractFrame", "args": ["int"], "return": "Image"},
                    {"name": "getDuration", "args": [], "return": "float"}
                ]
            },
            {"name": "bytes", "kind": "basic"},
            {"name": "List[T]", "kind": "generic"}
        ],
        "hierarchy": {
            "Image": {"supertype": "Media"},
            "Video": {"supertype": "Media"}
        },
        "coercions": [],
        "matching": "structural"
    }
    
    # Limited function library (creating gaps)
    functions = [
        FunctionDefinition("load_image", ["string"], "Image", {"cost": "medium"}),
        FunctionDefinition("Image.resize", ["Image", "int", "int"], "Image", 
                         {"cost": "low"}, owner="Image"),
        FunctionDefinition("Image.toGrayscale", ["Image"], "Image", 
                         {"cost": "low"}, owner="Image"),
        # Missing: Video creation, frame extraction, etc.
    ]
    
    # Create synthesizer with RAG
    type_system = TypeSystem(type_system_def)
    rag_storage = PolyglotRAGStore("media_processing")
    
    # Add more functions to RAG (simulating larger library)
    additional_funcs = [
        ("create_video", ["List[Image]", "float"], "Video", "Create video from frames"),
        ("video_to_gif", ["Video"], "string", "Convert video to GIF"),
        ("extract_frames", ["Video"], "List[Image]", "Extract all frames"),
        ("apply_filter", ["Image", "string"], "Image", "Apply image filter")
    ]
    
    for name, args, ret, desc in additional_funcs:
        poly_type = PolyglotType(
            canonical_name=name,
            kind=TypeKind.FUNCTION,
            metadata={
                "args": args,
                "return": ret,
                "description": desc
            }
        )
        rag_storage.store_type(poly_type)
    
    synthesizer = SemanticGapFillingSynthesizer(type_system, functions, rag_storage)
    
    # Test: Try to create Video (will fail due to missing functions)
    print("\nTest: Create 'Video' from 'string' (image path)")
    result = synthesizer.synthesize_chain(
        target_type="Video",
        available_resources=["string", "float"],  # image path and fps
        constraints={"max_calls": 5}
    )
    
    print(f"Success: {result.success}")
    print(f"Partial chain: {' -> '.join(result.chain) if result.chain else 'None'}")
    print(f"Output type: {result.output_type}")
    print(f"Gaps: {result.gaps}")
    print(f"\nGap-filling suggestions:")
    for suggestion in result.suggestions:
        print(f"{suggestion}")

# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Run all examples with configurable verbosity"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Function Chain Synthesis Examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Verbosity levels:
  0 - Quiet: Only show final results
  1 - Normal: Show main operations and results (default)
  2 - Verbose: Show search process and detailed operations
  3 - Debug: Show all operations including type compatibility checks

Examples:
  python %(prog)s                    # Run with normal output
  python %(prog)s -v                 # Run with verbose output
  python %(prog)s -vv                # Run with very verbose output
  python %(prog)s --verbosity 3      # Run with debug output
  python %(prog)s --example 1 -vv    # Run only example 1 with very verbose output
        """
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='count',
        default=0,
        help='Increase verbosity level (can be repeated: -v, -vv, -vvv)'
    )
    
    parser.add_argument(
        '--verbosity',
        type=int,
        choices=[0, 1, 2, 3],
        help='Set specific verbosity level'
    )
    
    parser.add_argument(
        '--example',
        type=int,
        choices=[1, 2, 3, 4, 5],
        help='Run specific example only'
    )
    
    args = parser.parse_args()
    
    # Set verbosity level
    global VERBOSITY
    if args.verbosity is not None:
        VERBOSITY = args.verbosity
    else:
        VERBOSITY = min(args.verbose, 3)  # Cap at 3
    
    # Print header
    print("Function Chain Synthesis Examples")
    print("================================")
    
    if VERBOSITY > 0:
        print(f"Verbosity level: {VERBOSITY}")
        print("Demonstrating function chain synthesis with:")
        print("- Type system support")
        print("- Object-oriented programming")
        print("- Gap-filling suggestions")
        print("- Multiple solution paths")
        print("- RAG integration")
    
    # Run examples
    if args.example:
        # Run specific example
        example_functions = {
            1: example_1_basic_type_chain,
            2: example_2_object_oriented_chains,
            3: example_3_real_world_data_processing,
            4: example_4_multi_path_synthesis,
            5: example_5_gap_filling_advanced
        }
        
        if args.example in example_functions:
            log(f"\nRunning example {args.example} only", level=1)
            example_functions[args.example]()
    else:
        # Run all examples
        example_1_basic_type_chain()
        example_2_object_oriented_chains()
        example_3_real_world_data_processing()
        example_4_multi_path_synthesis()
        example_5_gap_filling_advanced()
    
    print("\n" + "="*80)
    print("Examples completed!")
    if VERBOSITY >= 2:
        print(f"Verbosity level was: {VERBOSITY}")
    print("="*80)

if __name__ == "__main__":
    main()