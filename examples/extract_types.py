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
from src.types.polyglot_types import ObjectType

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

    console.print("[bold green]C++ Type Extraction Demo[/bold green]\n")

    # Extract types
    console.print(f"Extracting types from: [blue]{cpp_file}[/blue]")
    extractor = CppTypeExtractor()
    cpp_types = extractor.extract_from_file(str(cpp_file), include_paths)

    console.print(f"Found [bold]{len(cpp_types)}[/bold] types\n")

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
    console.print("\n[bold]Testing search functionality:[/bold]")

    test_queries = [
        "vector container class",
        "3D point structure",
        "data processing interface",
        "template class with size method"
    ]

    for query in test_queries:
        console.print(f"\nSearching for: [yellow]{query}[/yellow]")
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
