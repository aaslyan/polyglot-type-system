#!/usr/bin/env python3
"""
Batch Processing Example

This example demonstrates processing multiple C++ files at once:
- Scanning entire project directories
- Incremental type extraction
- Progress tracking and error handling
- Parallel processing
- CI/CD integration patterns
"""

from pathlib import Path
import sys
import json
import time
from typing import List, Dict, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
import hashlib

sys.path.append(str(Path(__file__).parent.parent))

from src.polyglot_type_system.extractors.cpp_extractor import CppTypeExtractor
from src.polyglot_type_system.storage.rag_storage import RagTypeStorage
from src.polyglot_type_system.models.type_models import PolyglotType

@dataclass
class ProcessingStats:
    """Statistics for batch processing"""
    total_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    total_types_extracted: int = 0
    processing_time: float = 0.0
    failed_file_paths: List[str] = None
    
    def __post_init__(self):
        if self.failed_file_paths is None:
            self.failed_file_paths = []

@dataclass
class FileProcessingResult:
    """Result of processing a single file"""
    file_path: str
    success: bool
    types_extracted: int = 0
    processing_time: float = 0.0
    error_message: Optional[str] = None
    file_hash: Optional[str] = None

class IncrementalProcessor:
    """Handles incremental type extraction with caching"""
    
    def __init__(self, cache_file: str = "processing_cache.json"):
        self.cache_file = Path(cache_file)
        self.cache = self._load_cache()
    
    def _load_cache(self) -> Dict[str, str]:
        """Load processing cache from file"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}
    
    def _save_cache(self):
        """Save processing cache to file"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except IOError as e:
            print(f"Warning: Could not save cache: {e}")
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate hash of file contents"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except IOError:
            return ""
    
    def needs_processing(self, file_path: Path) -> bool:
        """Check if file needs to be processed (changed since last run)"""
        file_str = str(file_path)
        current_hash = self._calculate_file_hash(file_path)
        
        if file_str not in self.cache:
            return True
        
        return self.cache[file_str] != current_hash
    
    def mark_processed(self, file_path: Path):
        """Mark file as processed in cache"""
        file_str = str(file_path)
        current_hash = self._calculate_file_hash(file_path)
        self.cache[file_str] = current_hash
        self._save_cache()

class BatchTypeProcessor:
    """Batch processor for C++ type extraction"""
    
    def __init__(self, storage_name: str = "batch_processing_db", max_workers: int = 4):
        self.extractor = CppTypeExtractor()
        self.storage = RagTypeStorage(storage_name)
        self.max_workers = max_workers
        self.incremental_processor = IncrementalProcessor()
    
    def find_cpp_files(self, root_dir: Path, extensions: Set[str] = {'.hpp', '.h', '.cpp', '.cc', '.cxx'}) -> List[Path]:
        """Find all C++ files in directory tree"""
        cpp_files = []
        
        if not root_dir.exists():
            print(f"Warning: Directory {root_dir} does not exist")
            return cpp_files
        
        for file_path in root_dir.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                # Skip common build and dependency directories
                if any(skip_dir in file_path.parts for skip_dir in ['build', 'dist', '.git', 'node_modules', '__pycache__']):
                    continue
                cpp_files.append(file_path)
        
        return sorted(cpp_files)
    
    def process_single_file(self, file_path: Path, force_reprocess: bool = False) -> FileProcessingResult:
        """Process a single C++ file"""
        start_time = time.time()
        
        try:
            # Check if incremental processing is needed
            if not force_reprocess and not self.incremental_processor.needs_processing(file_path):
                return FileProcessingResult(
                    file_path=str(file_path),
                    success=True,
                    types_extracted=0,
                    processing_time=time.time() - start_time,
                    error_message="Skipped - no changes detected"
                )
            
            # Extract types
            extracted_types = self.extractor.extract_from_file(str(file_path))
            
            # Store types in RAG
            for polyglot_type in extracted_types:
                # Add file source metadata
                polyglot_type.metadata['source_file'] = str(file_path)
                polyglot_type.metadata['extraction_timestamp'] = time.time()
                self.storage.add_type(polyglot_type)
            
            # Mark as processed
            self.incremental_processor.mark_processed(file_path)
            
            processing_time = time.time() - start_time
            return FileProcessingResult(
                file_path=str(file_path),
                success=True,
                types_extracted=len(extracted_types),
                processing_time=processing_time,
                file_hash=self.incremental_processor._calculate_file_hash(file_path)
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return FileProcessingResult(
                file_path=str(file_path),
                success=False,
                processing_time=processing_time,
                error_message=str(e)
            )
    
    def process_files_parallel(self, file_paths: List[Path], force_reprocess: bool = False) -> ProcessingStats:
        """Process multiple files in parallel"""
        stats = ProcessingStats(total_files=len(file_paths))
        start_time = time.time()
        
        print(f"🚀 Starting batch processing of {len(file_paths)} files with {self.max_workers} workers...")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self.process_single_file, file_path, force_reprocess): file_path
                for file_path in file_paths
            }
            
            # Process completed tasks
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    
                    if result.success:
                        stats.processed_files += 1
                        stats.total_types_extracted += result.types_extracted
                        if result.types_extracted > 0:
                            print(f"✅ {file_path.name}: {result.types_extracted} types ({result.processing_time:.2f}s)")
                        elif "no changes" in (result.error_message or ""):
                            print(f"⏭️  {file_path.name}: skipped (no changes)")
                        else:
                            print(f"📄 {file_path.name}: processed ({result.processing_time:.2f}s)")
                    else:
                        stats.failed_files += 1
                        stats.failed_file_paths.append(str(file_path))
                        print(f"❌ {file_path.name}: {result.error_message}")
                        
                except Exception as e:
                    stats.failed_files += 1
                    stats.failed_file_paths.append(str(file_path))
                    print(f"💥 {file_path.name}: Unexpected error: {e}")
        
        stats.processing_time = time.time() - start_time
        return stats
    
    def process_directory(self, directory: Path, force_reprocess: bool = False) -> ProcessingStats:
        """Process all C++ files in a directory"""
        cpp_files = self.find_cpp_files(directory)
        
        if not cpp_files:
            print(f"No C++ files found in {directory}")
            return ProcessingStats()
        
        print(f"📁 Found {len(cpp_files)} C++ files in {directory}")
        return self.process_files_parallel(cpp_files, force_reprocess)
    
    def generate_report(self, stats: ProcessingStats, output_file: Optional[Path] = None) -> str:
        """Generate processing report"""
        report_lines = [
            "=" * 80,
            "BATCH PROCESSING REPORT",
            "=" * 80,
            f"📊 Total Files: {stats.total_files}",
            f"✅ Successfully Processed: {stats.processed_files}",
            f"❌ Failed: {stats.failed_files}",
            f"🏷️  Total Types Extracted: {stats.total_types_extracted}",
            f"⏱️  Total Processing Time: {stats.processing_time:.2f} seconds",
            f"📈 Average Time per File: {stats.processing_time / max(stats.total_files, 1):.2f} seconds",
            ""
        ]
        
        if stats.failed_files > 0:
            report_lines.extend([
                "Failed Files:",
                "-" * 40
            ])
            for failed_file in stats.failed_file_paths:
                report_lines.append(f"  - {failed_file}")
            report_lines.append("")
        
        # Add efficiency metrics
        success_rate = (stats.processed_files / max(stats.total_files, 1)) * 100
        types_per_second = stats.total_types_extracted / max(stats.processing_time, 1)
        
        report_lines.extend([
            "Performance Metrics:",
            "-" * 40,
            f"Success Rate: {success_rate:.1f}%",
            f"Types/Second: {types_per_second:.1f}",
            f"Files/Second: {stats.processed_files / max(stats.processing_time, 1):.1f}",
            "=" * 80
        ])
        
        report = "\n".join(report_lines)
        
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    f.write(report)
                print(f"📄 Report saved to {output_file}")
            except IOError as e:
                print(f"Warning: Could not save report: {e}")
        
        return report

def create_sample_project(project_dir: Path):
    """Create a sample C++ project for demonstration"""
    project_dir.mkdir(exist_ok=True)
    
    # Create include directory with headers
    include_dir = project_dir / "include"
    include_dir.mkdir(exist_ok=True)
    
    headers = {
        "geometry.hpp": '''
#pragma once
#include <vector>

namespace geometry {
    class Point {
        double x, y;
    public:
        Point(double x, double y);
        double distance_to(const Point& other) const;
    };
    
    class Shape {
    public:
        virtual double area() const = 0;
        virtual ~Shape() = default;
    };
    
    class Circle : public Shape {
        Point center;
        double radius;
    public:
        Circle(const Point& center, double radius);
        double area() const override;
    };
}
''',
        "containers.hpp": '''
#pragma once
#include <vector>
#include <unordered_map>
#include <memory>

template<typename T>
class SmartVector {
    std::vector<std::unique_ptr<T>> data;
public:
    void add(std::unique_ptr<T> item);
    T* get(size_t index) const;
    size_t size() const;
};

class DataManager {
    std::unordered_map<std::string, int> lookup;
    std::vector<std::string> names;
public:
    void add_entry(const std::string& name, int value);
    int find(const std::string& name) const;
};
''',
        "algorithms.hpp": '''
#pragma once
#include <functional>
#include <vector>

namespace algorithms {
    template<typename T, typename Compare = std::less<T>>
    void quick_sort(std::vector<T>& data, Compare comp = Compare{});
    
    template<typename Container, typename Predicate>
    auto filter(const Container& container, Predicate pred);
    
    class SortingStrategy {
    public:
        virtual ~SortingStrategy() = default;
        virtual void sort(std::vector<int>& data) = 0;
    };
    
    class QuickSortStrategy : public SortingStrategy {
    public:
        void sort(std::vector<int>& data) override;
    };
}
'''
    }
    
    # Write header files
    for filename, content in headers.items():
        with open(include_dir / filename, 'w') as f:
            f.write(content)
    
    # Create source directory
    src_dir = project_dir / "src"
    src_dir.mkdir(exist_ok=True)
    
    # Create a simple source file
    with open(src_dir / "main.cpp", 'w') as f:
        f.write('''
#include <iostream>
#include "geometry.hpp"
#include "containers.hpp"

int main() {
    geometry::Point p1(0, 0);
    geometry::Point p2(3, 4);
    
    std::cout << "Distance: " << p1.distance_to(p2) << std::endl;
    return 0;
}
''')

def main():
    """Demonstrate batch processing capabilities"""
    
    print("=" * 80)
    print("Batch Processing & Real-World Integration Examples")
    print("=" * 80)
    
    # Create sample project
    sample_project = Path(__file__).parent / "sample_cpp_project"
    print(f"\n🏗️  Creating sample C++ project at {sample_project}")
    create_sample_project(sample_project)
    
    # Initialize batch processor
    processor = BatchTypeProcessor("batch_demo_db")
    
    # Process the sample project
    print(f"\n📁 Processing sample project...")
    stats = processor.process_directory(sample_project)
    
    # Generate and display report
    report = processor.generate_report(stats, Path(__file__).parent / "batch_report.txt")
    print("\n" + report)
    
    # Demonstrate incremental processing
    print("\n🔄 Demonstrating incremental processing...")
    print("Running again (should skip unchanged files):")
    
    stats2 = processor.process_directory(sample_project)
    print(f"Second run: {stats2.processed_files} files processed, {stats2.total_types_extracted} new types")
    
    # Demonstrate forced reprocessing
    print("\n🔄 Demonstrating forced reprocessing...")
    stats3 = processor.process_directory(sample_project, force_reprocess=True)
    print(f"Forced run: {stats3.processed_files} files processed, {stats3.total_types_extracted} types")
    
    # Search extracted types
    print("\n🔍 Searching extracted types...")
    all_types = processor.storage.get_all_types()
    
    print(f"Total types in storage: {len(all_types)}")
    
    # Group by source file
    files_to_types = {}
    for t in all_types:
        source_file = t.metadata.get('source_file', 'unknown')
        if source_file not in files_to_types:
            files_to_types[source_file] = []
        files_to_types[source_file].append(t)
    
    print("\nTypes by source file:")
    for file_path, types in files_to_types.items():
        if file_path != 'unknown':
            filename = Path(file_path).name
            print(f"  📄 {filename}: {len(types)} types")
            for t in types[:3]:  # Show first 3 types
                print(f"    - {t.name} ({t.type_kind})")
            if len(types) > 3:
                print(f"    ... and {len(types) - 3} more")

if __name__ == "__main__":
    main()