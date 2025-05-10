# core/vector_memory.py
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os
import json
from typing import List, Dict, Any, Optional, Tuple, Union
from .enhanced_memory import EnhancedMemory, MemorySegment

class VectorMemorySegment(MemorySegment):
    """Enhanced memory segment with vector embedding"""
    
    def __init__(self, content: str, segment_type: str, source: str = None, 
                 metadata: Dict[str, Any] = None, timestamp: str = None,
                 embedding: Optional[np.ndarray] = None):
        super().__init__(content, segment_type, source, metadata, timestamp)
        self.embedding = embedding
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the memory segment to a dictionary"""
        data = super().to_dict()
        # Convert embedding to list for JSON serialization if it exists
        if self.embedding is not None:
            data["embedding"] = self.embedding.tolist()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VectorMemorySegment':
        """Create a memory segment from a dictionary"""
        embedding = None
        if "embedding" in data:
            embedding = np.array(data["embedding"], dtype=np.float32)
            
        segment = cls(
            content=data["content"],
            segment_type=data["segment_type"],
            source=data.get("source"),
            metadata=data.get("metadata", {}),
            timestamp=data.get("timestamp"),
            embedding=embedding
        )
        segment.id = data["id"]
        segment.importance = data.get("importance", 0.5)
        return segment


class VectorMemory(EnhancedMemory):
    """Enhanced Memory with vector embeddings for semantic search"""
    
    def __init__(self, memory_file: str = 'vector_memory.json', 
                 index_file: str = 'vector_index.bin',
                 model_name: str = 'all-MiniLM-L6-v2',
                 max_segments: int = 1000):
        # Initialize without loading
        self.memory_file = memory_file
        self.index_file = index_file
        self.max_segments = max_segments
        self.segments = []
        self.goals = []
        self.completed_goals = []
        self.agent_contexts = {}
        self.last_output = {}
        self.last_agent_name = None
        
        # Initialize vector search components
        self.model_name = model_name
        print(f"[VectorMemory] Loading embedding model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        self.vector_dim = self.embedding_model.get_sentence_embedding_dimension()
        print(f"[VectorMemory] Vector dimension: {self.vector_dim}")
        
        # Maps segment IDs to their index in the FAISS index
        self.id_to_index = {}
        self.index_to_id = []
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(self.vector_dim)
        
        # Specialized indexes for faster retrieval
        self.segment_by_type = {}
        self.segment_by_source = {}
        
        # Now load existing memory
        self._load_memory()
    
    def _load_memory(self) -> None:
        """Load memory from file and rebuild the vector index"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Convert to VectorMemorySegment objects
                    self.segments = [VectorMemorySegment.from_dict(segment_data) for segment_data in data.get('segments', [])]
                    self.goals = data.get('goals', [])
                    self.completed_goals = data.get('completed_goals', [])
                    self.agent_contexts = data.get('agent_contexts', {})
                    self.last_output = data.get('last_output', {})
                    self.last_agent_name = data.get('last_agent_name')
                    
                    # Rebuild indexes
                    self._rebuild_indexes()
                    
                    # Rebuild vector index
                    self._rebuild_vector_index()
                    
                    print(f"[VectorMemory] Loaded {len(self.segments)} memory segments from {self.memory_file}")
            except Exception as e:
                print(f"[VectorMemory] Error loading memory: {e}")
                # Initialize empty memory structures
                self.segments = []
                self.goals = []
                self.completed_goals = []
                self.agent_contexts = {}
                self.last_output = {}
                self.last_agent_name = None
        else:
            print(f"[VectorMemory] No memory file found at {self.memory_file}, starting with empty memory")
    
    def _save_memory(self) -> None:
        """Save memory to file"""
        try:
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                data = {
                    'segments': [segment.to_dict() for segment in self.segments],
                    'goals': self.goals,
                    'completed_goals': self.completed_goals,
                    'agent_contexts': self.agent_contexts,
                    'last_output': self.last_output,
                    'last_agent_name': self.last_agent_name
                }
                json.dump(data, f, indent=2)
            
            # Save the FAISS index
            if self.index and self.index.ntotal > 0:
                faiss.write_index(self.index, self.index_file)
                
        except Exception as e:
            print(f"[VectorMemory] Error saving memory: {e}")
    
    def _rebuild_vector_index(self) -> None:
        """Rebuild the vector index from memory segments"""
        # Reset index
        self.index = faiss.IndexFlatL2(self.vector_dim)
        self.id_to_index = {}
        self.index_to_id = []
        
        # Add embeddings for all segments
        embeddings = []
        for i, segment in enumerate(self.segments):
            # If segment doesn't have an embedding yet, create one
            if not hasattr(segment, 'embedding') or segment.embedding is None:
                # Convert to VectorMemorySegment if it's a regular MemorySegment
                if not isinstance(segment, VectorMemorySegment):
                    segment = VectorMemorySegment(
                        content=segment.content,
                        segment_type=segment.segment_type,
                        source=segment.source,
                        metadata=segment.metadata,
                        timestamp=segment.timestamp
                    )
                    segment.id = self.segments[i].id
                    segment.importance = self.segments[i].importance
                    self.segments[i] = segment
                
                # Generate embedding
                segment.embedding = self._generate_embedding(segment.content)
            
            if segment.embedding is not None:
                embeddings.append(segment.embedding)
                self.id_to_index[segment.id] = len(self.index_to_id)
                self.index_to_id.append(segment.id)
        
        # Add all embeddings to the index at once (more efficient)
        if embeddings:
            embeddings_array = np.vstack(embeddings).astype(np.float32)
            self.index.add(embeddings_array)
            print(f"[VectorMemory] Rebuilt vector index with {len(embeddings)} embeddings")
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate an embedding for the given text"""
        try:
            embedding = self.embedding_model.encode(text, convert_to_numpy=True)
            return embedding.astype(np.float32)
        except Exception as e:
            print(f"[VectorMemory] Error generating embedding: {e}")
            # Return a zero vector as fallback
            return np.zeros(self.vector_dim, dtype=np.float32)
    
    def add_segment(self, content: str, segment_type: str, source: str = None, 
                   metadata: Dict[str, Any] = None, importance: float = 0.5) -> str:
        """
        Add a new memory segment with vector embedding
        
        Args:
            content: The content of the memory segment
            segment_type: The type of memory (e.g., 'conversation', 'goal', 'task_result')
            source: The source of the memory (e.g., agent name)
            metadata: Additional metadata about the memory
            importance: Importance factor (0.0 to 1.0) for memory retention
            
        Returns:
            The ID of the new memory segment
        """
        metadata = metadata or {}
        metadata['importance'] = importance
        
        # Generate embedding
        embedding = self._generate_embedding(content)
        
        # Create vector memory segment
        segment = VectorMemorySegment(
            content=content,
            segment_type=segment_type,
            source=source,
            metadata=metadata,
            embedding=embedding
        )
        
        # Add to main segments list
        self.segments.append(segment)
        
        # Add to indexes
        if segment.segment_type not in self.segment_by_type:
            self.segment_by_type[segment.segment_type] = []
        self.segment_by_type[segment.segment_type].append(segment)
        
        if segment.source:
            if segment.source not in self.segment_by_source:
                self.segment_by_source[segment.source] = []
            self.segment_by_source[segment.source].append(segment)
        
        # Add to vector index
        if embedding is not None:
            self.index.add(embedding.reshape(1, -1))
            self.id_to_index[segment.id] = len(self.index_to_id)
            self.index_to_id.append(segment.id)
        
        # If we exceed max_segments, remove the least important ones
        if len(self.segments) > self.max_segments:
            self._prune_memory()
        
        # Save memory
        self._save_memory()
        
        return segment.id
    
    def _prune_memory(self) -> None:
        """
        Prune memory by removing least important segments when we exceed max_segments
        
        Algorithm: 
        1. Keep all high importance segments (importance >= 0.8)
        2. Sort remaining segments by importance and timestamp
        3. Remove oldest, least important segments until we're under max_segments
        """
        # Separate high importance segments (always keep these)
        high_importance = [s for s in self.segments if s.importance >= 0.8]
        other_segments = [s for s in self.segments if s.importance < 0.8]
        
        # Calculate how many to keep from other_segments
        to_keep = max(0, self.max_segments - len(high_importance))
        
        if len(other_segments) > to_keep:
            # Sort by importance (primary) and recency (secondary)
            other_segments.sort(key=lambda s: (s.importance, s.timestamp), reverse=True)
            
            # Track removed segments to update the vector index
            segments_to_remove = other_segments[to_keep:]
            
            # Keep only the top 'to_keep' segments
            other_segments = other_segments[:to_keep]
            
            # Remove from vector index
            if segments_to_remove:
                # Need to rebuild the entire index since FAISS doesn't support direct removal
                self.segments = high_importance + other_segments
                self._rebuild_vector_index()
        
        # Combine segments to keep
        self.segments = high_importance + other_segments
        
        # Rebuild indexes
        self._rebuild_indexes()
        
        print(f"[VectorMemory] Pruned memory to {len(self.segments)} segments")
    
    def semantic_search(self, query: str, limit: int = 5, 
                        segment_type: Optional[str] = None, 
                        source: Optional[str] = None) -> List[VectorMemorySegment]:
        """
        Search for memory segments semantically similar to the query
        
        Args:
            query: The query string
            limit: Maximum number of results to return
            segment_type: Optional filter by segment type
            source: Optional filter by source (e.g., agent name)
            
        Returns:
            List of semantically related memory segments
        """
        if self.index.ntotal == 0:
            return []
        
        # Generate query embedding
        query_embedding = self._generate_embedding(query)
        
        # Search the index
        distances, indices = self.index.search(query_embedding.reshape(1, -1), min(limit * 3, self.index.ntotal))
        
        # Get the memory segments
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.index_to_id):
                continue
            
            segment_id = self.index_to_id[idx]
            segment = self.get_memory_by_id(segment_id)
            
            # Apply filters
            if segment and (segment_type is None or segment.segment_type == segment_type) and \
               (source is None or segment.source == source):
                results.append(segment)
                if len(results) >= limit:
                    break
        
        return results
    
    def remember_context(self, query: str, limit: int = 5) -> str:
        """
        Retrieve context relevant to the given query
        
        Args:
            query: The query string
            limit: Maximum number of context items to include
            
        Returns:
            A string with the relevant context
        """
        # Get semantically related segments
        related_segments = self.semantic_search(query, limit=limit)
        
        if not related_segments:
            return ""
        
        # Format the context
        context_parts = ["## Relevant Context from Memory:"]
        for i, segment in enumerate(related_segments):
            source_str = f" from {segment.source}" if segment.source else ""
            context_parts.append(f"### Memory {i+1}{source_str}:")
            context_parts.append(segment.content.strip())
            context_parts.append("")
        
        return "\n".join(context_parts)