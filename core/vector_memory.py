# core/vector_memory.py
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os
import json
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime # Added for default timestamps
import uuid # Added for generating default goal IDs

# Assuming EnhancedMemory and MemorySegment are correctly defined in core.enhanced_memory
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
        if "embedding" in data and data["embedding"] is not None: # Check for None
            embedding = np.array(data["embedding"], dtype=np.float32)

        segment = cls(
            content=data["content"],
            segment_type=data["segment_type"],
            source=data.get("source"),
            metadata=data.get("metadata", {}),
            timestamp=data.get("timestamp"),
            embedding=embedding
        )
        segment.id = data.get("id", segment._generate_id()) # Ensure ID exists
        segment.importance = data.get("importance", 0.5)
        return segment


class VectorMemory(EnhancedMemory):
    """Enhanced Memory with vector embeddings for semantic search"""

    def __init__(self, memory_file: str = 'vector_memory.json',
                 index_file: str = 'vector_index.bin', # Not currently used for saving/loading FAISS index
                 model_name: str = 'all-MiniLM-L6-v2',
                 max_segments: int = 1000):
        # Initialize EnhancedMemory's attributes first
        # super().__init__(memory_file, max_segments) # Call EnhancedMemory's __init__
        # Instead of super(), explicitly initialize attributes inherited or managed by EnhancedMemory
        # to avoid issues if EnhancedMemory's __init__ also calls _load_memory.
        self.memory_file = memory_file
        self.max_segments = max_segments
        self.segments: List[VectorMemorySegment] = [] # Explicitly type as VectorMemorySegment
        self.goals: List[Dict[str, Any]] = []
        self.completed_goals: List[Dict[str, Any]] = []
        self.agent_contexts: Dict[str, Dict[str, Any]] = {}
        self.last_output: Dict[str, str] = {}
        self.last_agent_name: Optional[str] = None
        self.segment_by_type: Dict[str, List[VectorMemorySegment]] = {} # Type hint
        self.segment_by_source: Dict[str, List[VectorMemorySegment]] = {} # Type hint

        # Initialize vector search components
        self.model_name = model_name
        print(f"[VectorMemory] Loading embedding model: {model_name}")
        try:
            self.embedding_model = SentenceTransformer(model_name)
            self.vector_dim = self.embedding_model.get_sentence_embedding_dimension()
            print(f"[VectorMemory] Vector dimension: {self.vector_dim}")
        except Exception as e_model_load:
            print(f"[VectorMemory_ERROR] Failed to load SentenceTransformer model '{model_name}': {e_model_load}")
            print("[VectorMemory_ERROR] Vector search capabilities will be severely limited or disabled.")
            self.embedding_model = None
            self.vector_dim = 384 # Fallback dimension, though operations will likely fail

        # Maps segment IDs to their index in the FAISS index
        self.id_to_index: Dict[str, int] = {}
        self.index_to_id: List[str] = []

        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(self.vector_dim) if self.embedding_model else None

        # Now load existing memory
        self._load_memory()

    def _initialize_empty_memory(self):
        """Initializes memory to an empty state specifically for VectorMemory."""
        super()._initialize_empty_memory() # Call parent's method to clear basic structures
        # Reset FAISS index and mappings
        if self.embedding_model:
            self.index = faiss.IndexFlatL2(self.vector_dim)
        else:
            self.index = None # No index if model failed to load
        self.id_to_index = {}
        self.index_to_id = []
        print("[VectorMemory] Initialized with empty memory structures and vector index.")


    def _load_memory(self) -> None:
        """Load memory from file and rebuild the vector index."""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Convert to VectorMemorySegment objects
                self.segments = [VectorMemorySegment.from_dict(segment_data) for segment_data in data.get('segments', [])]

                # MODIFICATION: Ensure goals have IDs and default fields
                loaded_goals = data.get('goals', [])
                self.goals = []
                for g_data in loaded_goals:
                    if not isinstance(g_data, dict):
                        print(f"[VectorMemory] Warning: Found non-dictionary goal item during load: {g_data}. Skipping.")
                        continue
                    g_data.setdefault('id', str(uuid.uuid4())) # Ensure ID exists
                    g_data.setdefault('status', 'pending')    # Default status
                    g_data.setdefault('priority', g_data.get('priority', 1))
                    g_data.setdefault('created', g_data.get('created', datetime.now().isoformat()))
                    # Ensure other fields expected by PlannerAgent/EnhancedMemory are present or defaulted
                    g_data.setdefault('task', g_data.get('task', 'Untitled Goal'))
                    g_data.setdefault('agent_suggestion', g_data.get('agent_suggestion', None))
                    g_data.setdefault('deadline', g_data.get('deadline', None))
                    g_data.setdefault('retries', g_data.get('retries', 0))
                    g_data.setdefault('last_attempt_time', g_data.get('last_attempt_time', None))
                    g_data.setdefault('processing_plan_id', g_data.get('processing_plan_id', None))
                    self.goals.append(g_data)

                loaded_completed_goals = data.get('completed_goals', [])
                self.completed_goals = []
                for cg_data in loaded_completed_goals:
                    if not isinstance(cg_data, dict):
                        print(f"[VectorMemory] Warning: Found non-dictionary completed goal item: {cg_data}. Skipping.")
                        continue
                    cg_data.setdefault('id', str(uuid.uuid4()))
                    cg_data.setdefault('status', 'completed') # Ensure status
                    # Ensure other fields are present or defaulted
                    cg_data.setdefault('task', cg_data.get('task', 'Untitled Completed Goal'))
                    cg_data.setdefault('completed_time', cg_data.get('completed_time', datetime.now().isoformat()))
                    self.completed_goals.append(cg_data)

                self.agent_contexts = data.get('agent_contexts', {})
                self.last_output = data.get('last_output', {})
                self.last_agent_name = data.get('last_agent_name')

                super()._rebuild_indexes() # Call EnhancedMemory's _rebuild_indexes for type/source
                self._rebuild_vector_index() # Rebuild FAISS index

                print(f"[VectorMemory] Loaded {len(self.segments)} memory segments, {len(self.goals)} active goals from {self.memory_file}")

            except json.JSONDecodeError as e_json:
                print(f"[VectorMemory] Error decoding JSON from {self.memory_file}: {e_json}. Initializing fresh memory.")
                self._initialize_empty_memory()
            except Exception as e:
                import traceback
                print(f"[VectorMemory] Error loading memory: {e}\n{traceback.format_exc()}")
                self._initialize_empty_memory()
        else:
            print(f"[VectorMemory] No memory file found at {self.memory_file}, starting with empty memory.")
            self._initialize_empty_memory()
            # self._save_memory() # Optionally create an empty file on first run

    def _save_memory(self) -> None:
        """Save memory to file"""
        # This method is inherited from EnhancedMemory and should work if VectorMemorySegment.to_dict() is correct.
        # If FAISS index needs separate saving, it would be done here.
        # For simplicity, the FAISS index is rebuilt on load. If persistence is needed:
        # if self.index and self.index.ntotal > 0:
        #     faiss.write_index(self.index, self.index_file_path_for_faiss) # Define self.index_file_path_for_faiss
        super()._save_memory() # Calls EnhancedMemory's save which handles segments, goals etc.

    def _rebuild_vector_index(self) -> None:
        """Rebuild the vector index from memory segments."""
        if not self.embedding_model or not self.index:
            print("[VectorMemory] Embedding model or FAISS index not initialized. Cannot rebuild vector index.")
            return

        # Reset index
        self.index.reset() # Clears the FAISS index
        self.id_to_index = {}
        self.index_to_id = []

        embeddings_to_add = []
        segment_ids_for_index = []

        for segment in self.segments:
            # Ensure segment is VectorMemorySegment and has an embedding
            if not isinstance(segment, VectorMemorySegment):
                # This case should ideally not happen if _load_memory converts correctly
                print(f"[VectorMemory_WARN] Segment {segment.id} is not VectorMemorySegment. Skipping for vector index.")
                continue

            if segment.embedding is None:
                # Generate embedding if missing (e.g., for older segments or if generation failed previously)
                segment.embedding = self._generate_embedding(segment.content)

            if segment.embedding is not None and segment.embedding.ndim == 1 and segment.embedding.size == self.vector_dim:
                embeddings_to_add.append(segment.embedding)
                segment_ids_for_index.append(segment.id)
            elif segment.embedding is not None:
                print(f"[VectorMemory_WARN] Segment {segment.id} has malformed embedding (shape: {segment.embedding.shape}, expected dim: {self.vector_dim}). Skipping.")


        if embeddings_to_add:
            embeddings_array = np.vstack(embeddings_to_add).astype(np.float32)
            self.index.add(embeddings_array)
            for i, seg_id in enumerate(segment_ids_for_index):
                self.id_to_index[seg_id] = i # Store the FAISS index 'i' for this segment ID
            self.index_to_id = segment_ids_for_index # List of segment IDs in the order they were added to FAISS
            print(f"[VectorMemory] Rebuilt vector index with {self.index.ntotal} embeddings.")
        else:
            print("[VectorMemory] No valid embeddings found to rebuild vector index.")

    def _generate_embedding(self, text: str) -> Optional[np.ndarray]:
        """Generate an embedding for the given text"""
        if not self.embedding_model:
            return None
        try:
            embedding = self.embedding_model.encode(text, convert_to_numpy=True)
            return embedding.astype(np.float32)
        except Exception as e:
            print(f"[VectorMemory] Error generating embedding for text '{text[:50]}...': {e}")
            return None # Return None on failure

    def add_segment(self, content: str, segment_type: str, source: str = None,
                   metadata: Dict[str, Any] = None, importance: float = 0.5) -> str:
        """
        Add a new memory segment with vector embedding.
        Overrides EnhancedMemory.add_segment to handle vector specifics.
        """
        metadata = metadata or {}
        metadata['importance'] = importance

        embedding = self._generate_embedding(content) if self.embedding_model else None

        segment = VectorMemorySegment(
            content=content,
            segment_type=segment_type,
            source=source,
            metadata=metadata,
            embedding=embedding # Pass the generated embedding
        )
        # The rest of the logic is similar to EnhancedMemory.add_segment,
        # but we also need to add to the FAISS index.

        self.segments.append(segment)

        # Update type and source indexes (inherited from EnhancedMemory or reimplemented here)
        if segment.segment_type not in self.segment_by_type:
            self.segment_by_type[segment.segment_type] = []
        self.segment_by_type[segment.segment_type].append(segment)

        if segment.source:
            if segment.source not in self.segment_by_source:
                self.segment_by_source[segment.source] = []
            self.segment_by_source[segment.source].append(segment)

        # Add to FAISS index
        if self.index and embedding is not None and embedding.ndim == 1 and embedding.size == self.vector_dim:
            current_faiss_index = self.index.ntotal
            self.index.add(embedding.reshape(1, -1))
            self.id_to_index[segment.id] = current_faiss_index # Map segment ID to its new index in FAISS
            self.index_to_id.append(segment.id) # Add segment ID to the list maintaining order
        elif self.index and embedding is not None:
            print(f"[VectorMemory_WARN] Failed to add segment {segment.id} to FAISS index due to malformed embedding.")


        if len(self.segments) > self.max_segments:
            self._prune_memory() # This needs to be aware of FAISS index

        self._save_memory()
        return segment.id

    def _prune_memory(self) -> None:
        """
        Prune memory by removing least important segments.
        Needs to rebuild FAISS index after pruning.
        """
        if len(self.segments) <= self.max_segments:
            return

        # EnhancedMemory's pruning logic:
        high_importance_segments = [s for s in self.segments if s.importance >= 0.8]
        other_segments = [s for s in self.segments if s.importance < 0.8]

        num_to_remove = len(self.segments) - self.max_segments
        if num_to_remove <= 0: # Should not happen if called correctly
            return

        # Sort segments to remove by importance (ascending) then by timestamp (oldest first)
        other_segments.sort(key=lambda s: (s.importance, s.timestamp))

        # Determine segments to be removed
        segments_actually_removed = other_segments[:num_to_remove]
        segments_to_keep = high_importance_segments + other_segments[num_to_remove:]

        self.segments = segments_to_keep
        print(f"[VectorMemory] Pruned memory. Removed {len(segments_actually_removed)} segments. New count: {len(self.segments)}.")

        # Rebuild all indexes including FAISS
        super()._rebuild_indexes() # For type/source
        self._rebuild_vector_index() # For FAISS

    def semantic_search(self, query: str, limit: int = 5,
                        segment_type: Optional[str] = None,
                        source: Optional[str] = None) -> List[VectorMemorySegment]: # Return type hint
        """
        Search for memory segments semantically similar to the query.
        """
        if not self.index or self.index.ntotal == 0 or not self.embedding_model:
            print("[VectorMemory] Semantic search unavailable (index empty or model not loaded).")
            return []

        query_embedding = self._generate_embedding(query)
        if query_embedding is None:
            return []

        # Search the FAISS index
        # k_search should not exceed the number of items in the index
        k_search = min(limit * 3, self.index.ntotal) # Search for more initially to allow filtering
        if k_search == 0: return []

        distances, faiss_indices = self.index.search(query_embedding.reshape(1, -1), k_search)

        results: List[VectorMemorySegment] = []
        for i, faiss_idx in enumerate(faiss_indices[0]):
            if faiss_idx < 0 or faiss_idx >= len(self.index_to_id): # faiss_idx can be -1 if less than k results found
                continue

            segment_id = self.index_to_id[faiss_idx] # Get segment ID from our mapping
            segment = self.get_memory_by_id(segment_id) # Retrieve the full segment object

            if segment and isinstance(segment, VectorMemorySegment): # Ensure it's the correct type
                # Apply filters
                type_match = (segment_type is None or segment.segment_type == segment_type)
                source_match = (source is None or (segment.source and segment.source.lower() == source.lower()))

                if type_match and source_match:
                    # Optionally, you could store/use the distance if needed
                    # segment.metadata['search_distance'] = float(distances[0][i])
                    results.append(segment)
                    if len(results) >= limit:
                        break
        return results

    def remember_context(self, query: str, limit: int = 5) -> str:
        """
        Retrieve context relevant to the given query using semantic search.
        """
        related_segments = self.semantic_search(query, limit=limit)

        if not related_segments:
            return "No semantically relevant context found in memory."

        context_parts = ["## Relevant Context from Vector Memory:"]
        for i, segment in enumerate(related_segments):
            source_str = f" from {segment.source}" if segment.source else ""
            timestamp_dt = datetime.fromisoformat(segment.timestamp)
            time_str = timestamp_dt.strftime("%Y-%m-%d %H:%M")
            context_parts.append(f"### Memory {i+1} (ID: {segment.id[:8]}..., Type: {segment.segment_type}, Source: {segment.source or 'N/A'}, Time: {time_str}):")
            context_parts.append(segment.content.strip())
            context_parts.append("") # Newline for separation

        return "\n".join(context_parts)

    # Override other methods from EnhancedMemory if they need specific vector handling
    # For example, delete_segment and update_segment would also need to update the FAISS index.

    def delete_segment(self, segment_id: str) -> bool:
        """Delete a memory segment and update FAISS index."""
        # Find the segment to get its original FAISS index
        original_faiss_idx = self.id_to_index.get(segment_id)

        if super().delete_segment(segment_id): # Call parent to remove from lists and type/source indexes
            if original_faiss_idx is not None and self.index:
                # FAISS does not directly support removing by arbitrary ID easily without rebuilding
                # or using IndexIDMap. For simplicity with IndexFlatL2, we rebuild.
                print(f"[VectorMemory] Segment {segment_id} deleted. Rebuilding FAISS index.")
                self._rebuild_vector_index()
            return True
        return False

    def update_segment(self, segment_id: str, content: Optional[str] = None,
                      metadata: Optional[Dict[str, Any]] = None,
                      importance: Optional[float] = None) -> bool:
        """Update a memory segment. If content changes, re-embed and update FAISS."""
        segment_to_update = self.get_memory_by_id(segment_id)
        if not segment_to_update or not isinstance(segment_to_update, VectorMemorySegment):
            return False

        content_changed = False
        if content is not None and segment_to_update.content != content:
            segment_to_update.content = content
            content_changed = True

        if metadata is not None:
            segment_to_update.metadata.update(metadata)
        if importance is not None:
            segment_to_update.importance = importance
            segment_to_update.metadata['importance'] = importance # Keep metadata consistent

        segment_to_update.timestamp = datetime.now().isoformat() # Update timestamp on any change

        if content_changed and self.embedding_model:
            print(f"[VectorMemory] Content changed for segment {segment_id}. Re-embedding and rebuilding FAISS index.")
            segment_to_update.embedding = self._generate_embedding(segment_to_update.content)
            # Rebuilding the index is the simplest way to handle updates with IndexFlatL2
            self._rebuild_vector_index()
        elif content_changed:
             print(f"[VectorMemory_WARN] Content changed for segment {segment_id}, but no embedding model to re-embed.")


        self._save_memory() # Save all changes
        return True
