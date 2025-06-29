from typing import List, Dict, Any, Optional
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

from ..types.polyglot_types import PolyglotType, ObjectType, FunctionType, TemplateType

class PolyglotRAGStore:
    """Simple RAG storage for polyglot types"""
    
    def __init__(self, storage_path: str = "./polyglot_storage"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Simple JSON-based storage
        self.store = {
            "types": {},
            "search_index": {}
        }
        
        # Type index
        self.type_index = {}
        self._load_index()
        
        # Load existing store
        store_file = self.storage_path / "simple_store.json"
        if store_file.exists():
            try:
                with open(store_file, 'r') as f:
                    self.store = json.load(f)
            except:
                pass
        
        print("Initialized simple RAG store")
    
    def store_type(self, poly_type: PolyglotType):
        """Store a polyglot type"""
        type_id = poly_type.id
        
        # Create searchable text
        doc_text = self._create_document_text(poly_type)
        
        # Store in memory
        self.store["types"][type_id] = {
            "data": poly_type.to_dict(),
            "text": doc_text,
            "canonical_name": poly_type.canonical_name,
            "kind": poly_type.kind.value
        }
        
        # Update search index
        words = doc_text.lower().split()
        for word in set(words):
            if word not in self.store["search_index"]:
                self.store["search_index"][word] = []
            if type_id not in self.store["search_index"][word]:
                self.store["search_index"][word].append(type_id)
        
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
        
        # Save everything
        self._save_all()
    
    def search_types(self, query: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """Search for types using simple text matching"""
        query_words = set(query.lower().split())
        scores = {}
        
        # Score based on word matches
        for word in query_words:
            if word in self.store.get("search_index", {}):
                for type_id in self.store["search_index"][word]:
                    scores[type_id] = scores.get(type_id, 0) + 1
        
        # Check canonical names
        for type_id, type_info in self.store.get("types", {}).items():
            name_lower = type_info["canonical_name"].lower()
            if any(word in name_lower for word in query_words):
                scores[type_id] = scores.get(type_id, 0) + 3
        
        # Sort by score
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n_results]
        
        results = []
        for type_id, score in sorted_results:
            type_data = self.load_type(type_id)
            if type_data:
                results.append({
                    "type": type_data,
                    "score": score / (len(query_words) + 1),
                    "metadata": {
                        "canonical_name": self.store["types"][type_id]["canonical_name"],
                        "kind": self.store["types"][type_id]["kind"]
                    }
                })
        
        return results
    
    def load_type(self, type_id: str) -> Optional[Dict[str, Any]]:
        """Load a type by ID"""
        if type_id in self.type_index:
            type_file = Path(self.type_index[type_id]["file"])
            if type_file.exists():
                with open(type_file, 'r') as f:
                    return json.load(f)
        return None
    
    def _create_document_text(self, poly_type: PolyglotType) -> str:
        """Create searchable document text"""
        parts = [
            f"Type: {poly_type.canonical_name}",
            f"Kind: {poly_type.kind.value}",
            f"Language: {poly_type.source_language}",
        ]
        
        if poly_type.qualifiers:
            parts.append(f"Qualifiers: {', '.join(q.value for q in poly_type.qualifiers)}")
        
        if isinstance(poly_type, ObjectType):
            if poly_type.members:
                parts.append(f"Members: {', '.join(poly_type.members.keys())}")
            if poly_type.methods:
                parts.append(f"Methods: {', '.join(poly_type.methods.keys())}")
        
        return " ".join(parts)
    
    def _load_index(self):
        """Load type index"""
        index_file = self.storage_path / "type_index.json"
        if index_file.exists():
            with open(index_file, 'r') as f:
                self.type_index = json.load(f)
    
    def _save_all(self):
        """Save all data"""
        # Save store
        with open(self.storage_path / "simple_store.json", 'w') as f:
            json.dump(self.store, f)
        
        # Save index
        with open(self.storage_path / "type_index.json", 'w') as f:
            json.dump(self.type_index, f, indent=2)
