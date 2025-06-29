#!/usr/bin/env python3
"""
CI/CD Integration Example

This example demonstrates integrating the polyglot type system into CI/CD pipelines:
- Git hook integration
- Type compatibility checking
- ABI change detection
- Automated documentation generation
- Performance monitoring
"""

from pathlib import Path
import sys
import json
import subprocess
import time
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from src.polyglot_type_system.extractors.cpp_extractor import CppTypeExtractor
from src.polyglot_type_system.storage.rag_storage import RagTypeStorage
from src.polyglot_type_system.models.type_models import PolyglotType

@dataclass
class CompatibilityReport:
    """Report of compatibility changes between versions"""
    breaking_changes: List[str]
    new_types: List[str]
    removed_types: List[str]
    modified_types: List[str]
    is_compatible: bool
    version_from: str
    version_to: str
    timestamp: datetime

class GitIntegration:
    """Git integration utilities"""
    
    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
    
    def get_current_commit(self) -> str:
        """Get current git commit hash"""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return "unknown"
    
    def get_changed_files(self, base_commit: str = "HEAD~1", target_commit: str = "HEAD") -> List[Path]:
        """Get list of changed files between commits"""
        try:
            result = subprocess.run(
                ['git', 'diff', '--name-only', f"{base_commit}..{target_commit}"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            changed_files = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    file_path = self.repo_path / line
                    if file_path.exists() and file_path.suffix in {'.hpp', '.h', '.cpp', '.cc', '.cxx'}:
                        changed_files.append(file_path)
            
            return changed_files
        except subprocess.CalledProcessError:
            return []
    
    def get_commit_message(self, commit: str = "HEAD") -> str:
        """Get commit message"""
        try:
            result = subprocess.run(
                ['git', 'log', '-1', '--pretty=format:%s', commit],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return "Unknown commit"

class TypeCompatibilityChecker:
    """Checks type compatibility between versions"""
    
    def __init__(self):
        self.extractor = CppTypeExtractor()
    
    def compare_versions(self, old_types: List[PolyglotType], new_types: List[PolyglotType]) -> CompatibilityReport:
        """Compare two sets of types for compatibility"""
        
        # Create lookup maps
        old_types_map = {t.name: t for t in old_types}
        new_types_map = {t.name: t for t in new_types}
        
        # Find changes
        breaking_changes = []
        new_types_list = []
        removed_types_list = []
        modified_types_list = []
        
        # Check for removed types
        for old_name in old_types_map:
            if old_name not in new_types_map:
                removed_types_list.append(old_name)
                breaking_changes.append(f"Removed type: {old_name}")
        
        # Check for new types
        for new_name in new_types_map:
            if new_name not in old_types_map:
                new_types_list.append(new_name)
        
        # Check for modified types
        for name in old_types_map:
            if name in new_types_map:
                old_type = old_types_map[name]
                new_type = new_types_map[name]
                
                changes = self._compare_types(old_type, new_type)
                if changes:
                    modified_types_list.append(name)
                    for change in changes:
                        if change['breaking']:
                            breaking_changes.append(f"{name}: {change['description']}")
        
        is_compatible = len(breaking_changes) == 0
        
        return CompatibilityReport(
            breaking_changes=breaking_changes,
            new_types=new_types_list,
            removed_types=removed_types_list,
            modified_types=modified_types_list,
            is_compatible=is_compatible,
            version_from="old",
            version_to="new",
            timestamp=datetime.now()
        )
    
    def _compare_types(self, old_type: PolyglotType, new_type: PolyglotType) -> List[Dict]:
        """Compare two types and return list of changes"""
        changes = []
        
        # Check method signature changes
        old_methods = {m['name']: m for m in old_type.metadata.get('methods', [])}
        new_methods = {m['name']: m for m in new_type.metadata.get('methods', [])}
        
        # Removed methods
        for method_name in old_methods:
            if method_name not in new_methods:
                changes.append({
                    'type': 'method_removed',
                    'description': f"Method '{method_name}' was removed",
                    'breaking': True
                })
        
        # Modified method signatures
        for method_name in old_methods:
            if method_name in new_methods:
                old_method = old_methods[method_name]
                new_method = new_methods[method_name]
                
                # Check return type
                if old_method.get('return_type') != new_method.get('return_type'):
                    changes.append({
                        'type': 'return_type_changed',
                        'description': f"Method '{method_name}' return type changed",
                        'breaking': True
                    })
                
                # Check parameters
                old_params = old_method.get('parameters', [])
                new_params = new_method.get('parameters', [])
                
                if len(old_params) != len(new_params):
                    changes.append({
                        'type': 'parameter_count_changed',
                        'description': f"Method '{method_name}' parameter count changed",
                        'breaking': True
                    })
        
        # Check field changes
        old_fields = {f['name']: f for f in old_type.metadata.get('fields', [])}
        new_fields = {f['name']: f for f in new_type.metadata.get('fields', [])}
        
        # Removed fields
        for field_name in old_fields:
            if field_name not in new_fields:
                changes.append({
                    'type': 'field_removed',
                    'description': f"Field '{field_name}' was removed",
                    'breaking': True
                })
        
        # Field type changes
        for field_name in old_fields:
            if field_name in new_fields:
                if old_fields[field_name]['type'] != new_fields[field_name]['type']:
                    changes.append({
                        'type': 'field_type_changed',
                        'description': f"Field '{field_name}' type changed",
                        'breaking': True
                    })
        
        return changes

class CICDIntegrator:
    """Main CI/CD integration class"""
    
    def __init__(self, repo_path: Path, storage_name: str = "cicd_types"):
        self.repo_path = repo_path
        self.git = GitIntegration(repo_path)
        self.extractor = CppTypeExtractor()
        self.storage = RagTypeStorage(storage_name)
        self.compatibility_checker = TypeCompatibilityChecker()
    
    def run_pre_commit_check(self) -> bool:
        """Run type extraction and compatibility check before commit"""
        print("🔍 Running pre-commit type compatibility check...")
        
        # Get changed files
        changed_files = self.git.get_changed_files()
        
        if not changed_files:
            print("✅ No C++ files changed, skipping type check")
            return True
        
        print(f"📄 Checking {len(changed_files)} changed files:")
        for file in changed_files:
            print(f"  - {file.relative_to(self.repo_path)}")
        
        # Extract types from changed files
        new_types = []
        for file_path in changed_files:
            try:
                extracted_types = self.extractor.extract_from_file(str(file_path))
                new_types.extend(extracted_types)
            except Exception as e:
                print(f"❌ Error extracting types from {file_path}: {e}")
                return False
        
        # Get existing types for comparison
        old_types = self.storage.get_all_types()
        
        # Check compatibility
        report = self.compatibility_checker.compare_versions(old_types, new_types)
        
        # Display results
        self._display_compatibility_report(report)
        
        # Save new types if compatible
        if report.is_compatible:
            for new_type in new_types:
                new_type.metadata['commit'] = self.git.get_current_commit()
                new_type.metadata['timestamp'] = time.time()
                self.storage.add_type(new_type)
            print("✅ Types updated in storage")
        
        return report.is_compatible
    
    def run_post_commit_analysis(self):
        """Run analysis after commit"""
        print("📊 Running post-commit type analysis...")
        
        commit = self.git.get_current_commit()
        commit_message = self.git.get_commit_message()
        
        print(f"🔖 Commit: {commit[:8]} - {commit_message}")
        
        # Extract all types from current state
        cpp_files = self._find_all_cpp_files()
        all_types = []
        
        for file_path in cpp_files:
            try:
                extracted_types = self.extractor.extract_from_file(str(file_path))
                for t in extracted_types:
                    t.metadata['commit'] = commit
                    t.metadata['source_file'] = str(file_path.relative_to(self.repo_path))
                all_types.extend(extracted_types)
            except Exception as e:
                print(f"⚠️  Error processing {file_path}: {e}")
        
        # Generate metrics
        self._generate_metrics(all_types, commit)
        
        # Update storage
        for type_obj in all_types:
            self.storage.add_type(type_obj)
        
        print(f"✅ Analysis complete: {len(all_types)} types processed")
    
    def generate_documentation(self, output_dir: Path):
        """Generate documentation from extracted types"""
        print("📚 Generating type documentation...")
        
        output_dir.mkdir(exist_ok=True)
        all_types = self.storage.get_all_types()
        
        # Group types by file
        files_to_types = {}
        for t in all_types:
            source_file = t.metadata.get('source_file', 'unknown')
            if source_file not in files_to_types:
                files_to_types[source_file] = []
            files_to_types[source_file].append(t)
        
        # Generate index
        index_content = ["# Type Documentation Index\n"]
        
        for file_path, types in files_to_types.items():
            if file_path != 'unknown':
                index_content.append(f"## {file_path}")
                index_content.append(f"**{len(types)} types**\n")
                
                for t in types:
                    doc_file = f"{t.name.replace('::', '_')}.md"
                    index_content.append(f"- [{t.name}]({doc_file}) ({t.type_kind})")
                    
                    # Generate individual type documentation
                    self._generate_type_doc(t, output_dir / doc_file)
                
                index_content.append("")
        
        # Write index
        with open(output_dir / "index.md", 'w') as f:
            f.write("\n".join(index_content))
        
        print(f"📄 Documentation generated in {output_dir}")
    
    def _find_all_cpp_files(self) -> List[Path]:
        """Find all C++ files in repository"""
        cpp_files = []
        extensions = {'.hpp', '.h', '.cpp', '.cc', '.cxx'}
        
        for file_path in self.repo_path.rglob('*'):
            if (file_path.is_file() and 
                file_path.suffix.lower() in extensions and
                not any(skip_dir in file_path.parts for skip_dir in ['.git', 'build', 'dist'])):
                cpp_files.append(file_path)
        
        return cpp_files
    
    def _display_compatibility_report(self, report: CompatibilityReport):
        """Display compatibility report"""
        print("\n" + "=" * 60)
        print("COMPATIBILITY REPORT")
        print("=" * 60)
        
        if report.is_compatible:
            print("✅ COMPATIBLE - No breaking changes detected")
        else:
            print("❌ INCOMPATIBLE - Breaking changes detected")
        
        if report.breaking_changes:
            print(f"\n💥 Breaking Changes ({len(report.breaking_changes)}):")
            for change in report.breaking_changes:
                print(f"  - {change}")
        
        if report.new_types:
            print(f"\n➕ New Types ({len(report.new_types)}):")
            for new_type in report.new_types[:5]:  # Show first 5
                print(f"  - {new_type}")
            if len(report.new_types) > 5:
                print(f"  ... and {len(report.new_types) - 5} more")
        
        if report.removed_types:
            print(f"\n➖ Removed Types ({len(report.removed_types)}):")
            for removed_type in report.removed_types:
                print(f"  - {removed_type}")
        
        if report.modified_types:
            print(f"\n🔄 Modified Types ({len(report.modified_types)}):")
            for modified_type in report.modified_types[:5]:
                print(f"  - {modified_type}")
            if len(report.modified_types) > 5:
                print(f"  ... and {len(report.modified_types) - 5} more")
        
        print("=" * 60)
    
    def _generate_metrics(self, types: List[PolyglotType], commit: str):
        """Generate and save metrics"""
        metrics = {
            'commit': commit,
            'timestamp': datetime.now().isoformat(),
            'total_types': len(types),
            'type_breakdown': {},
            'complexity_stats': {},
            'file_stats': {}
        }
        
        # Type breakdown
        for t in types:
            kind = t.type_kind
            metrics['type_breakdown'][kind] = metrics['type_breakdown'].get(kind, 0) + 1
        
        # File statistics
        files_to_types = {}
        for t in types:
            source_file = t.metadata.get('source_file', 'unknown')
            files_to_types[source_file] = files_to_types.get(source_file, 0) + 1
        
        metrics['file_stats'] = {
            'total_files': len(files_to_types),
            'types_per_file': {k: v for k, v in files_to_types.items() if k != 'unknown'}
        }
        
        # Save metrics
        metrics_file = Path("type_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"📊 Metrics saved to {metrics_file}")
    
    def _generate_type_doc(self, type_obj: PolyglotType, output_file: Path):
        """Generate documentation for a single type"""
        doc_lines = [
            f"# {type_obj.name}",
            "",
            f"**Type:** {type_obj.type_kind}",
            f"**Source:** {type_obj.metadata.get('source_file', 'unknown')}",
            ""
        ]
        
        if type_obj.metadata.get('description'):
            doc_lines.extend([
                "## Description",
                type_obj.metadata['description'],
                ""
            ])
        
        # Methods
        methods = type_obj.metadata.get('methods', [])
        if methods:
            doc_lines.extend([
                "## Methods",
                ""
            ])
            for method in methods:
                params = method.get('parameters', [])
                param_str = ', '.join([f"{p.get('type', 'auto')} {p.get('name', '')}" for p in params])
                return_type = method.get('return_type', 'void')
                doc_lines.append(f"- `{return_type} {method['name']}({param_str})`")
            doc_lines.append("")
        
        # Fields
        fields = type_obj.metadata.get('fields', [])
        if fields:
            doc_lines.extend([
                "## Fields",
                ""
            ])
            for field in fields:
                doc_lines.append(f"- `{field.get('type', 'auto')} {field['name']}`")
            doc_lines.append("")
        
        with open(output_file, 'w') as f:
            f.write("\n".join(doc_lines))

def main():
    """Demonstrate CI/CD integration"""
    
    print("=" * 80)
    print("CI/CD Integration Examples")
    print("=" * 80)
    
    # Use current directory as mock repository
    repo_path = Path(__file__).parent
    integrator = CICDIntegrator(repo_path)
    
    print("\n🔄 Simulating CI/CD workflow...")
    
    # Simulate pre-commit check
    print("\n1. Pre-commit Hook:")
    try:
        is_compatible = integrator.run_pre_commit_check()
        if is_compatible:
            print("✅ Pre-commit check passed")
        else:
            print("❌ Pre-commit check failed - breaking changes detected")
    except Exception as e:
        print(f"⚠️  Pre-commit check error: {e}")
    
    # Simulate post-commit analysis
    print("\n2. Post-commit Analysis:")
    try:
        integrator.run_post_commit_analysis()
    except Exception as e:
        print(f"⚠️  Post-commit analysis error: {e}")
    
    # Generate documentation
    print("\n3. Documentation Generation:")
    docs_dir = Path(__file__).parent / "generated_docs"
    try:
        integrator.generate_documentation(docs_dir)
    except Exception as e:
        print(f"⚠️  Documentation generation error: {e}")
    
    print("\n✅ CI/CD workflow simulation complete!")

if __name__ == "__main__":
    main()