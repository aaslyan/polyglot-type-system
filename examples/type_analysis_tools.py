#!/usr/bin/env python3
"""
Type Analysis Tools Examples

This example demonstrates various type analysis tools:
- Circular dependency detection
- Type complexity analysis
- Unused type detection
- ABI compatibility checking
- Performance impact analysis
- Code metrics generation
"""

from pathlib import Path
import sys
from typing import List, Dict, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import time

sys.path.append(str(Path(__file__).parent.parent))

from src.extractors.cpp_extractor import CppTypeExtractor
from src.converters.cpp_to_polyglot import CppToPolyglotConverter
from src.storage.rag_store import PolyglotRAGStore
from src.types.polyglot_types import PolyglotType

@dataclass
class DependencyInfo:
    """Information about type dependencies"""
    depends_on: Set[str] = field(default_factory=set)
    used_by: Set[str] = field(default_factory=set)
    depth: int = 0

@dataclass
class ComplexityMetrics:
    """Complexity metrics for a type"""
    cyclomatic_complexity: int = 0
    inheritance_depth: int = 0
    coupling: int = 0
    cohesion: float = 0.0
    lines_of_code: int = 0
    method_count: int = 0
    field_count: int = 0

@dataclass
class ABIChangeInfo:
    """Information about ABI changes"""
    change_type: str  # added, removed, modified
    severity: str     # breaking, compatible, warning
    description: str
    affected_type: str
    details: Dict[str, Any] = field(default_factory=dict)

class CircularDependencyDetector:
    """Detects circular dependencies in type hierarchies"""
    
    def __init__(self):
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        self.dependency_info: Dict[str, DependencyInfo] = {}
    
    def build_dependency_graph(self, types: List[PolyglotType]):
        """Build dependency graph from types"""
        self.dependency_graph.clear()
        self.dependency_info.clear()
        
        # Initialize dependency info for all types
        for t in types:
            self.dependency_info[t.canonical_name] = DependencyInfo()
        
        # Build dependencies
        for t in types:
            dependencies = self._extract_dependencies(t)
            self.dependency_graph[t.canonical_name] = dependencies
            self.dependency_info[t.canonical_name].depends_on = dependencies
            
            # Update reverse dependencies
            for dep in dependencies:
                if dep in self.dependency_info:
                    self.dependency_info[dep].used_by.add(t.canonical_name)
    
    def _extract_dependencies(self, type_obj: PolyglotType) -> Set[str]:
        """Extract dependencies from a type"""
        dependencies = set()
        
        # Base classes
        base_classes = type_obj.metadata.get('base_classes', [])
        dependencies.update(base_classes)
        
        # Field types
        fields = type_obj.metadata.get('fields', [])
        for field in fields:
            field_type = field['type']
            # Extract type name (simplified - would need better parsing in practice)
            clean_type = self._clean_type_name(field_type)
            if clean_type:
                dependencies.add(clean_type)
        
        # Method parameter and return types
        methods = type_obj.metadata.get('methods', [])
        for method in methods:
            # Return type
            return_type = method.get('return_type', '')
            clean_return_type = self._clean_type_name(return_type)
            if clean_return_type:
                dependencies.add(clean_return_type)
            
            # Parameter types
            parameters = method.get('parameters', [])
            for param in parameters:
                param_type = param.get('type', '')
                clean_param_type = self._clean_type_name(param_type)
                if clean_param_type:
                    dependencies.add(clean_param_type)
        
        # Filter out built-in types
        filtered_dependencies = set()
        builtin_types = {'int', 'double', 'float', 'bool', 'void', 'char', 'std::string', 'string'}
        
        for dep in dependencies:
            if dep not in builtin_types and not dep.startswith('std::'):
                filtered_dependencies.add(dep)
        
        return filtered_dependencies
    
    def _clean_type_name(self, type_name: str) -> Optional[str]:
        """Clean type name to extract base type"""
        if not type_name or type_name == 'void':
            return None
        
        # Remove const, &, *, std::shared_ptr, etc.
        cleaned = type_name.replace('const ', '').replace('&', '').replace('*', '').strip()
        
        # Handle templated types
        if '<' in cleaned:
            cleaned = cleaned.split('<')[0]
        
        # Handle std::shared_ptr, std::unique_ptr, etc.
        if cleaned.startswith('std::'):
            return None
        
        return cleaned if cleaned else None
    
    def detect_circular_dependencies(self) -> List[List[str]]:
        """Detect circular dependencies using DFS"""
        visited = set()
        rec_stack = set()
        cycles = []
        
        def dfs(node: str, path: List[str]) -> bool:
            if node in rec_stack:
                # Found cycle
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                cycles.append(cycle)
                return True
            
            if node in visited:
                return False
            
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in self.dependency_graph.get(node, set()):
                if neighbor in self.dependency_graph:  # Only consider known types
                    dfs(neighbor, path.copy())
            
            rec_stack.remove(node)
            return False
        
        for node in self.dependency_graph:
            if node not in visited:
                dfs(node, [])
        
        return cycles
    
    def calculate_dependency_depth(self):
        """Calculate dependency depth for each type"""
        def calculate_depth(type_name: str, visited: Set[str]) -> int:
            if type_name in visited:
                return 0  # Circular dependency, return 0
            
            visited.add(type_name)
            dependencies = self.dependency_graph.get(type_name, set())
            
            if not dependencies:
                return 0
            
            max_depth = 0
            for dep in dependencies:
                if dep in self.dependency_graph:
                    depth = calculate_depth(dep, visited.copy())
                    max_depth = max(max_depth, depth + 1)
            
            return max_depth
        
        for type_name in self.dependency_graph:
            depth = calculate_depth(type_name, set())
            self.dependency_info[type_name].depth = depth

class ComplexityAnalyzer:
    """Analyzes type complexity"""
    
    def analyze_type_complexity(self, type_obj: PolyglotType) -> ComplexityMetrics:
        """Analyze complexity of a single type"""
        metrics = ComplexityMetrics()
        
        # Basic counts
        methods = type_obj.metadata.get('methods', [])
        fields = type_obj.metadata.get('fields', [])
        
        metrics.method_count = len(methods)
        metrics.field_count = len(fields)
        
        # Cyclomatic complexity (simplified)
        metrics.cyclomatic_complexity = self._calculate_cyclomatic_complexity(methods)
        
        # Inheritance depth
        metrics.inheritance_depth = self._calculate_inheritance_depth(type_obj)
        
        # Coupling (number of dependencies)
        dependencies = type_obj.metadata.get('dependencies', [])
        metrics.coupling = len(dependencies)
        
        # Cohesion (simplified)
        metrics.cohesion = self._calculate_cohesion(type_obj)
        
        # Estimated lines of code
        metrics.lines_of_code = self._estimate_lines_of_code(type_obj)
        
        return metrics
    
    def _calculate_cyclomatic_complexity(self, methods: List[Dict[str, Any]]) -> int:
        """Calculate cyclomatic complexity (simplified estimate)"""
        complexity = 1  # Base complexity
        
        for method in methods:
            # Add 1 for each method
            complexity += 1
            
            # Add complexity for virtual methods (higher complexity)
            if method.get('is_virtual', False):
                complexity += 1
            
            # Add complexity based on parameter count (more parameters = more paths)
            param_count = len(method.get('parameters', []))
            complexity += max(0, param_count - 2)  # Penalty for many parameters
        
        return complexity
    
    def _calculate_inheritance_depth(self, type_obj: PolyglotType) -> int:
        """Calculate inheritance depth"""
        base_classes = type_obj.metadata.get('base_classes', [])
        return len(base_classes)  # Simplified - would need full hierarchy analysis
    
    def _calculate_cohesion(self, type_obj: PolyglotType) -> float:
        """Calculate cohesion (simplified LCOM metric)"""
        methods = type_obj.metadata.get('methods', [])
        fields = type_obj.metadata.get('fields', [])
        
        if not methods or not fields:
            return 1.0
        
        # Simplified cohesion: assume methods that share similar names are related
        field_names = {f['name'] for f in fields}
        method_field_usage = 0
        
        for method in methods:
            method_name = method['name'].lower()
            for field_name in field_names:
                if field_name.lower() in method_name or method_name in field_name.lower():
                    method_field_usage += 1
                    break
        
        return method_field_usage / len(methods) if methods else 0.0
    
    def _estimate_lines_of_code(self, type_obj: PolyglotType) -> int:
        """Estimate lines of code for a type"""
        methods = type_obj.metadata.get('methods', [])
        fields = type_obj.metadata.get('fields', [])
        
        # Rough estimation
        lines = 5  # Class declaration overhead
        lines += len(fields) * 2  # Field declarations
        lines += len(methods) * 8  # Average method size
        
        # Add complexity for virtual methods
        virtual_methods = [m for m in methods if m.get('is_virtual', False)]
        lines += len(virtual_methods) * 3
        
        return lines

class UnusedTypeDetector:
    """Detects unused types in the codebase"""
    
    def __init__(self):
        self.type_usage: Dict[str, Set[str]] = defaultdict(set)
        self.all_types: Set[str] = set()
    
    def analyze_usage(self, types: List[PolyglotType]) -> Dict[str, List[str]]:
        """Analyze type usage and find unused types"""
        self.all_types = {t.canonical_name for t in types}
        self.type_usage.clear()
        
        # Build usage map
        for t in types:
            self._analyze_type_usage(t)
        
        # Find unused types
        used_types = set()
        for usages in self.type_usage.values():
            used_types.update(usages)
        
        unused_types = self.all_types - used_types
        
        # Categorize results
        results = {
            'unused': list(unused_types),
            'rarely_used': [],
            'heavily_used': []
        }
        
        # Find rarely and heavily used types
        usage_counts = defaultdict(int)
        for usages in self.type_usage.values():
            for used_type in usages:
                usage_counts[used_type] += 1
        
        for type_name, count in usage_counts.items():
            if count == 1:
                results['rarely_used'].append(type_name)
            elif count >= 5:
                results['heavily_used'].append(type_name)
        
        return results
    
    def _analyze_type_usage(self, type_obj: PolyglotType):
        """Analyze how a type uses other types"""
        type_name = type_obj.canonical_name
        
        # Base classes
        base_classes = type_obj.metadata.get('base_classes', [])
        for base in base_classes:
            if base in self.all_types:
                self.type_usage[type_name].add(base)
        
        # Field types
        fields = type_obj.metadata.get('fields', [])
        for field in fields:
            field_type = self._extract_type_name(field['type'])
            if field_type and field_type in self.all_types:
                self.type_usage[type_name].add(field_type)
        
        # Method types
        methods = type_obj.metadata.get('methods', [])
        for method in methods:
            # Return type
            return_type = self._extract_type_name(method.get('return_type', ''))
            if return_type and return_type in self.all_types:
                self.type_usage[type_name].add(return_type)
            
            # Parameter types
            parameters = method.get('parameters', [])
            for param in parameters:
                param_type = self._extract_type_name(param.get('type', ''))
                if param_type and param_type in self.all_types:
                    self.type_usage[type_name].add(param_type)
    
    def _extract_type_name(self, type_str: str) -> Optional[str]:
        """Extract clean type name from type string"""
        if not type_str or type_str == 'void':
            return None
        
        # Remove modifiers
        cleaned = type_str.replace('const ', '').replace('&', '').replace('*', '').strip()
        
        # Handle templates
        if '<' in cleaned:
            cleaned = cleaned.split('<')[0]
        
        return cleaned if cleaned else None

class ABICompatibilityChecker:
    """Checks ABI compatibility between type versions"""
    
    def check_compatibility(self, old_types: List[PolyglotType], 
                          new_types: List[PolyglotType]) -> List[ABIChangeInfo]:
        """Check ABI compatibility between two versions"""
        old_types_map = {t.canonical_name: t for t in old_types}
        new_types_map = {t.canonical_name: t for t in new_types}
        
        changes = []
        
        # Check for removed types
        for old_name in old_types_map:
            if old_name not in new_types_map:
                changes.append(ABIChangeInfo(
                    change_type="removed",
                    severity="breaking",
                    description=f"Type {old_name} was removed",
                    affected_type=old_name
                ))
        
        # Check for added types
        for new_name in new_types_map:
            if new_name not in old_types_map:
                changes.append(ABIChangeInfo(
                    change_type="added",
                    severity="compatible",
                    description=f"Type {new_name} was added",
                    affected_type=new_name
                ))
        
        # Check for modified types
        for name in old_types_map:
            if name in new_types_map:
                old_type = old_types_map[name]
                new_type = new_types_map[name]
                type_changes = self._check_type_changes(old_type, new_type)
                changes.extend(type_changes)
        
        return changes
    
    def _check_type_changes(self, old_type: PolyglotType, 
                          new_type: PolyglotType) -> List[ABIChangeInfo]:
        """Check changes in a specific type"""
        changes = []
        type_name = old_type.canonical_name
        
        # Check field changes
        old_fields = {f['name']: f for f in old_type.metadata.get('fields', [])}
        new_fields = {f['name']: f for f in new_type.metadata.get('fields', [])}
        
        # Removed fields
        for field_name in old_fields:
            if field_name not in new_fields:
                changes.append(ABIChangeInfo(
                    change_type="removed",
                    severity="breaking",
                    description=f"Field {field_name} was removed",
                    affected_type=type_name,
                    details={"field": field_name}
                ))
        
        # Added fields
        for field_name in new_fields:
            if field_name not in old_fields:
                changes.append(ABIChangeInfo(
                    change_type="added",
                    severity="warning",
                    description=f"Field {field_name} was added",
                    affected_type=type_name,
                    details={"field": field_name}
                ))
        
        # Modified fields
        for field_name in old_fields:
            if field_name in new_fields:
                old_field = old_fields[field_name]
                new_field = new_fields[field_name]
                
                if old_field['type'] != new_field['type']:
                    changes.append(ABIChangeInfo(
                        change_type="modified",
                        severity="breaking",
                        description=f"Field {field_name} type changed from {old_field['type']} to {new_field['type']}",
                        affected_type=type_name,
                        details={"field": field_name, "old_type": old_field['type'], "new_type": new_field['type']}
                    ))
        
        # Check method changes
        old_methods = {m['name']: m for m in old_type.metadata.get('methods', [])}
        new_methods = {m['name']: m for m in new_type.metadata.get('methods', [])}
        
        # Removed methods
        for method_name in old_methods:
            if method_name not in new_methods:
                changes.append(ABIChangeInfo(
                    change_type="removed",
                    severity="breaking",
                    description=f"Method {method_name} was removed",
                    affected_type=type_name,
                    details={"method": method_name}
                ))
        
        # Check method signature changes
        for method_name in old_methods:
            if method_name in new_methods:
                old_method = old_methods[method_name]
                new_method = new_methods[method_name]
                
                # Check return type
                if old_method.get('return_type') != new_method.get('return_type'):
                    changes.append(ABIChangeInfo(
                        change_type="modified",
                        severity="breaking",
                        description=f"Method {method_name} return type changed",
                        affected_type=type_name,
                        details={"method": method_name, "change": "return_type"}
                    ))
                
                # Check parameters
                old_params = old_method.get('parameters', [])
                new_params = new_method.get('parameters', [])
                
                if len(old_params) != len(new_params):
                    changes.append(ABIChangeInfo(
                        change_type="modified",
                        severity="breaking",
                        description=f"Method {method_name} parameter count changed",
                        affected_type=type_name,
                        details={"method": method_name, "change": "parameter_count"}
                    ))
        
        return changes

class TypeAnalysisOrchestrator:
    """Orchestrates all type analysis tools"""
    
    def __init__(self, storage: PolyglotRAGStore):
        self.storage = storage
        self.dependency_detector = CircularDependencyDetector()
        self.complexity_analyzer = ComplexityAnalyzer()
        self.unused_detector = UnusedTypeDetector()
        self.abi_checker = ABICompatibilityChecker()
    
    def run_full_analysis(self, types: List[PolyglotType]) -> Dict[str, Any]:
        """Run comprehensive type analysis"""
        print("🔍 Running comprehensive type analysis...")
        start_time = time.time()
        
        results = {
            'summary': {
                'total_types': len(types),
                'analysis_time': 0.0
            },
            'circular_dependencies': [],
            'complexity_metrics': {},
            'unused_types': {},
            'high_complexity_types': [],
            'dependency_depths': {}
        }
        
        # Circular dependency analysis
        print("  🔄 Analyzing circular dependencies...")
        self.dependency_detector.build_dependency_graph(types)
        circles = self.dependency_detector.detect_circular_dependencies()
        self.dependency_detector.calculate_dependency_depth()
        
        results['circular_dependencies'] = circles
        results['dependency_depths'] = {
            name: info.depth 
            for name, info in self.dependency_detector.dependency_info.items()
        }
        
        # Complexity analysis
        print("  📊 Analyzing type complexity...")
        complexity_results = {}
        high_complexity_types = []
        
        for t in types:
            metrics = self.complexity_analyzer.analyze_type_complexity(t)
            complexity_results[t.canonical_name] = {
                'cyclomatic_complexity': metrics.cyclomatic_complexity,
                'inheritance_depth': metrics.inheritance_depth,
                'coupling': metrics.coupling,
                'cohesion': metrics.cohesion,
                'method_count': metrics.method_count,
                'field_count': metrics.field_count,
                'estimated_loc': metrics.lines_of_code
            }
            
            # Flag high complexity types
            if (metrics.cyclomatic_complexity > 10 or 
                metrics.coupling > 8 or 
                metrics.method_count > 15):
                high_complexity_types.append({
                    'name': t.canonical_name,
                    'complexity': metrics.cyclomatic_complexity,
                    'coupling': metrics.coupling,
                    'methods': metrics.method_count
                })
        
        results['complexity_metrics'] = complexity_results
        results['high_complexity_types'] = high_complexity_types
        
        # Unused type analysis
        print("  🗑️  Analyzing type usage...")
        usage_analysis = self.unused_detector.analyze_usage(types)
        results['unused_types'] = usage_analysis
        
        # Performance summary
        analysis_time = time.time() - start_time
        results['summary']['analysis_time'] = analysis_time
        
        return results
    
    def generate_analysis_report(self, results: Dict[str, Any], output_file: Optional[Path] = None) -> str:
        """Generate human-readable analysis report"""
        lines = [
            "=" * 80,
            "TYPE ANALYSIS REPORT",
            "=" * 80,
            f"📊 Total Types Analyzed: {results['summary']['total_types']}",
            f"⏱️  Analysis Time: {results['summary']['analysis_time']:.2f} seconds",
            ""
        ]
        
        # Circular dependencies
        circles = results['circular_dependencies']
        lines.extend([
            f"🔄 Circular Dependencies: {len(circles)} found",
            "-" * 40
        ])
        
        if circles:
            for i, circle in enumerate(circles, 1):
                circle_str = " -> ".join(circle)
                lines.append(f"  {i}. {circle_str}")
        else:
            lines.append("  ✅ No circular dependencies detected")
        lines.append("")
        
        # High complexity types
        high_complexity = results['high_complexity_types']
        lines.extend([
            f"📈 High Complexity Types: {len(high_complexity)} found",
            "-" * 40
        ])
        
        if high_complexity:
            for hc_type in high_complexity[:10]:  # Show top 10
                lines.append(f"  - {hc_type['name']}: complexity={hc_type['complexity']}, coupling={hc_type['coupling']}, methods={hc_type['methods']}")
        else:
            lines.append("  ✅ No high complexity types detected")
        lines.append("")
        
        # Unused types
        unused = results['unused_types']
        lines.extend([
            f"🗑️  Unused Types: {len(unused['unused'])} found",
            "-" * 40
        ])
        
        if unused['unused']:
            for unused_type in unused['unused'][:10]:  # Show first 10
                lines.append(f"  - {unused_type}")
            if len(unused['unused']) > 10:
                lines.append(f"  ... and {len(unused['unused']) - 10} more")
        else:
            lines.append("  ✅ All types appear to be used")
        lines.append("")
        
        # Usage statistics
        lines.extend([
            "📊 Usage Statistics:",
            "-" * 40,
            f"  Rarely Used Types: {len(unused.get('rarely_used', []))}",
            f"  Heavily Used Types: {len(unused.get('heavily_used', []))}",
            ""
        ])
        
        # Dependency depths
        depths = results['dependency_depths']
        if depths:
            max_depth = max(depths.values())
            avg_depth = sum(depths.values()) / len(depths)
            lines.extend([
                "🌳 Dependency Analysis:",
                "-" * 40,
                f"  Maximum Dependency Depth: {max_depth}",
                f"  Average Dependency Depth: {avg_depth:.1f}",
                ""
            ])
        
        lines.append("=" * 80)
        
        report = "\n".join(lines)
        
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    f.write(report)
                print(f"📄 Analysis report saved to {output_file}")
            except IOError as e:
                print(f"Warning: Could not save report: {e}")
        
        return report

def create_sample_types_for_analysis():
    """Create sample C++ types with various issues for analysis"""
    sample_code = '''
#pragma once
#include <string>
#include <vector>
#include <memory>

// Forward declarations to create dependencies
class Database;
class Logger;

// Circular dependency example
class UserManager;
class SessionManager {
    std::shared_ptr<UserManager> user_mgr_;
public:
    void create_session(int user_id);
    void end_session(int session_id);
};

class UserManager {
    std::shared_ptr<SessionManager> session_mgr_;
    std::shared_ptr<Database> db_;
public:
    void create_user(const std::string& name);
    void delete_user(int user_id);
    bool validate_user(int user_id);
};

// High complexity class
class ComplexProcessor {
private:
    std::vector<std::string> data_;
    std::vector<int> indices_;
    std::vector<double> weights_;
    std::shared_ptr<Logger> logger_;
    std::shared_ptr<Database> db_;
    bool is_initialized_;
    int processing_mode_;
    double threshold_;

public:
    ComplexProcessor();
    ~ComplexProcessor();
    
    // Many methods indicating high complexity
    bool initialize(const std::string& config);
    void set_processing_mode(int mode);
    void set_threshold(double threshold);
    bool process_data(const std::vector<std::string>& input);
    bool process_batch(const std::vector<std::vector<std::string>>& batches);
    std::vector<double> calculate_weights(const std::vector<std::string>& data);
    std::vector<int> find_patterns(const std::vector<std::string>& data);
    bool validate_input(const std::string& input);
    bool validate_batch(const std::vector<std::string>& batch);
    void log_processing_stats();
    void save_results_to_db();
    void clear_cache();
    void reset_state();
    std::string get_status_report();
    bool is_processing_complete();
    double get_completion_percentage();
    
    // Virtual methods for inheritance
    virtual void on_processing_start();
    virtual void on_processing_complete();
    virtual void on_error(const std::string& error);
};

// Potentially unused class
class UnusedUtility {
private:
    std::string name_;
public:
    void do_something();
    std::string get_name() const { return name_; }
};

// Simple class for comparison
class SimplePoint {
public:
    double x, y;
    
    SimplePoint(double x = 0, double y = 0);
    double distance_to(const SimplePoint& other) const;
};

// Database interface (creates dependency)
class Database {
public:
    virtual ~Database() = default;
    virtual bool connect(const std::string& connection_string) = 0;
    virtual bool execute_query(const std::string& query) = 0;
    virtual void disconnect() = 0;
};

// Logger interface (creates dependency)
class Logger {
public:
    virtual ~Logger() = default;
    virtual void log(const std::string& message) = 0;
    virtual void set_level(int level) = 0;
};
'''
    
    analysis_header = Path(__file__).parent / "analysis_sample.hpp"
    with open(analysis_header, 'w') as f:
        f.write(sample_code)
    
    return analysis_header

def main():
    """Demonstrate type analysis tools"""
    
    print("=" * 80)
    print("Type Analysis Tools Examples")
    print("=" * 80)
    
    # Create sample C++ code with various issues
    print("📝 Creating sample C++ code with analysis targets...")
    sample_file = create_sample_types_for_analysis()
    
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
        methods = len(t.metadata.get('methods', []))
        fields = len(t.metadata.get('fields', []))
        print(f"  - {t.canonical_name} ({t.kind}): {methods} methods, {fields} fields")
    
    # Initialize storage and orchestrator
    storage = PolyglotRAGStore("analysis_demo")
    for t in polyglot_types:
        storage.store_type(t)
    
    orchestrator = TypeAnalysisOrchestrator(storage)
    
    # Run comprehensive analysis
    print(f"\n🔬 Running comprehensive analysis...")
    analysis_results = orchestrator.run_full_analysis(polyglot_types)
    
    # Generate and display report
    report_file = Path(__file__).parent / "type_analysis_report.txt"
    report = orchestrator.generate_analysis_report(analysis_results, report_file)
    print("\n" + report)
    
    # Save detailed results as JSON
    json_file = Path(__file__).parent / "analysis_results.json"
    with open(json_file, 'w') as f:
        json.dump(analysis_results, f, indent=2, default=str)
    print(f"📄 Detailed results saved to {json_file}")
    
    # Demonstrate ABI compatibility checking
    print("\n🔍 Demonstrating ABI compatibility checking...")
    
    # Create a modified version of one type for comparison
    modified_types = polyglot_types.copy()
    for t in modified_types:
        if t.canonical_name == "SimplePoint":
            # Add a field to simulate ABI change
            t.metadata.setdefault('fields', []).append({
                'name': 'z',
                'type': 'double',
                'access': 'public'
            })
            break
    
    abi_changes = orchestrator.abi_checker.check_compatibility(polyglot_types, modified_types)
    
    if abi_changes:
        print("⚠️  ABI Changes Detected:")
        for change in abi_changes:
            print(f"  - {change.severity.upper()}: {change.description}")
    else:
        print("✅ No ABI changes detected")
    
    print("\n✅ Type analysis complete!")

if __name__ == "__main__":
    main()