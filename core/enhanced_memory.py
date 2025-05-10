# core/enhanced_memory.py
"""
Enhanced Memory System for WITS CrewAI.
Provides more sophisticated memory management with context awareness,
memory segmentation, and retrieval mechanisms.
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union
import re
import time

class MemorySegment:
    """Represents a segment of memory with a specific type and content"""
    
    def __init__(self, 
                 content: str, 
                 segment_type: str,
                 source: str = None,
                 metadata: Dict[str, Any] = None,
                 timestamp: str = None):
        self.id = self._generate_id()
        self.content = content
        self.segment_type = segment_type  # e.g., 'conversation', 'goal', 'task_result', 'agent_output'
        self.source = source  # e.g., agent name or system
        self.metadata = metadata or {}
        self.timestamp = timestamp or datetime.now().isoformat()
        self.importance = metadata.get('importance', 0.5)  # Default importance factor (0.0 to 1.0)
        
    def _generate_id(self) -> str:
        """Generate a unique memory segment ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        return f"mem_{timestamp}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the memory segment to a dictionary"""
        return {
            "id": self.id,
            "content": self.content,
            "segment_type": self.segment_type,
            "source": self.source,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "importance": self.importance
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemorySegment':
        """Create a memory segment from a dictionary"""
        segment = cls(
            content=data["content"],
            segment_type=data["segment_type"],
            source=data.get("source"),
            metadata=data.get("metadata", {}),
            timestamp=data.get("timestamp")
        )
        segment.id = data["id"]
        segment.importance = data.get("importance", 0.5)
        return segment
    
    def __str__(self) -> str:
        return f"{self.segment_type} from {self.source}: {self.content[:50]}..."


class EnhancedMemory:
    """Enhanced Memory system for the WITS CrewAI"""
    
    def __init__(self, memory_file: str = 'enhanced_memory.json', max_segments: int = 1000):
        self.memory_file = memory_file
        self.max_segments = max_segments
        self.segments: List[MemorySegment] = []
        self.goals: List[Dict[str, Any]] = []
        self.completed_goals: List[Dict[str, Any]] = []
        self.agent_contexts: Dict[str, Dict[str, Any]] = {}  # Stores per-agent contextual data
        self.last_output: Dict[str, str] = {}
        self.last_agent_name: Optional[str] = None
        
        # Specialized indexes for faster retrieval
        self.segment_by_type: Dict[str, List[MemorySegment]] = {}
        self.segment_by_source: Dict[str, List[MemorySegment]] = {}
        
        # Load existing memory if available
        self._load_memory()
    
    def _load_memory(self) -> None:
        """Load memory from file"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.segments = [MemorySegment.from_dict(segment_data) for segment_data in data.get('segments', [])]
                    self.goals = data.get('goals', [])
                    self.completed_goals = data.get('completed_goals', [])
                    self.agent_contexts = data.get('agent_contexts', {})
                    self.last_output = data.get('last_output', {})
                    self.last_agent_name = data.get('last_agent_name')
                    
                    # Rebuild indexes
                    self._rebuild_indexes()
                    
                    print(f"[EnhancedMemory] Loaded {len(self.segments)} memory segments from {self.memory_file}")
            except Exception as e:
                print(f"[EnhancedMemory] Error loading memory: {e}")
                # Initialize empty memory structures
                self.segments = []
                self.goals = []
                self.completed_goals = []
                self.agent_contexts = {}
                self.last_output = {}
                self.last_agent_name = None
        else:
            print(f"[EnhancedMemory] No memory file found at {self.memory_file}, starting with empty memory")
    
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
        except Exception as e:
            print(f"[EnhancedMemory] Error saving memory: {e}")
    
    def _rebuild_indexes(self) -> None:
        """Rebuild the memory segment indexes"""
        self.segment_by_type = {}
        self.segment_by_source = {}
        
        for segment in self.segments:
            # Index by type
            if segment.segment_type not in self.segment_by_type:
                self.segment_by_type[segment.segment_type] = []
            self.segment_by_type[segment.segment_type].append(segment)
            
            # Index by source
            if segment.source:
                if segment.source not in self.segment_by_source:
                    self.segment_by_source[segment.source] = []
                self.segment_by_source[segment.source].append(segment)
    
    def add_segment(self, content: str, segment_type: str, source: str = None, 
                   metadata: Dict[str, Any] = None, importance: float = 0.5) -> str:
        """
        Add a new memory segment
        
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
        
        segment = MemorySegment(
            content=content,
            segment_type=segment_type,
            source=source,
            metadata=metadata
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
            # Keep only the top 'to_keep' segments
            other_segments = other_segments[:to_keep]
        
        # Combine segments to keep
        self.segments = high_importance + other_segments
        
        # Rebuild indexes
        self._rebuild_indexes()
        
        print(f"[EnhancedMemory] Pruned memory to {len(self.segments)} segments")
    
    def remember_agent_output(self, agent_name: str, content: str, importance: float = 0.6) -> str:
        """
        Remember output produced by an agent
        
        Args:
            agent_name: Name of the agent that produced the output
            content: The output content
            importance: Importance factor (0.0 to 1.0)
            
        Returns:
            The ID of the memory segment
        """
        if agent_name:
            agent_name = agent_name.lower()
            self.last_output[agent_name] = content
            self.last_agent_name = agent_name
            
            # Add as a memory segment
            segment_id = self.add_segment(
                content=content,
                segment_type="agent_output",
                source=agent_name,
                metadata={"output_type": "agent_response"},
                importance=importance
            )
            
            return segment_id
        return None
    
    def recall_agent_output(self, agent_name: str) -> Optional[str]:
        """Recall the most recent output from a specific agent"""
        if agent_name:
            return self.last_output.get(agent_name.lower())
        return None
    
    def get_last_agent(self) -> Optional[str]:
        """Get the name of the agent that last produced output"""
        return self.last_agent_name
    
    def add_goal(self, task_description: str, agent: Optional[str] = None, 
                priority: int = 1, deadline: Optional[str] = None) -> int:
        """
        Add a new goal or task
        
        Args:
            task_description: Description of the goal/task
            agent: The agent assigned to this goal (optional)
            priority: Priority level (1-5, with 5 being highest)
            deadline: Optional deadline in ISO format
            
        Returns:
            The index of the goal
        """
        goal = {
            'task': task_description,
            'created': datetime.now().isoformat()
        }
        
        if agent:
            goal['agent'] = agent
        
        if priority and 1 <= priority <= 5:
            goal['priority'] = priority
        else:
            goal['priority'] = 1
            
        if deadline:
            goal['deadline'] = deadline
        
        self.goals.append(goal)
        
        # Add as a memory segment
        self.add_segment(
            content=task_description,
            segment_type="goal",
            source=agent,
            metadata={
                "priority": priority,
                "deadline": deadline
            },
            importance=0.7 + (priority * 0.06)  # Higher priority = higher importance
        )
        
        self._save_memory()
        return len(self.goals)
    
    def complete_goal(self, index_or_task: Union[int, str], result: Optional[str] = None) -> bool:
        """
        Mark a goal as completed
        
        Args:
            index_or_task: Either the 1-based index of the goal, or a substring of the task
            result: Optional result or outcome information
            
        Returns:
            True if a goal was completed, False otherwise
        """
        goal_entry = None
        index = None
        
        if isinstance(index_or_task, int):
            idx = index_or_task - 1  # Convert to 0-based
            if 0 <= idx < len(self.goals):
                goal_entry = self.goals[idx]
                index = idx
        elif isinstance(index_or_task, str):
            for i, g in enumerate(self.goals):
                if index_or_task.lower() in g.get('task', '').lower():
                    goal_entry = g
                    index = i
                    break
        
        if not goal_entry:
            return False
        
        # Add completion timestamp and result
        goal_entry['completed_time'] = datetime.now().isoformat()
        if result:
            goal_entry['result'] = result
        
        # Move to completed goals
        completed_goal = self.goals.pop(index)
        self.completed_goals.append(completed_goal)
        
        # Add as a memory segment
        self.add_segment(
            content=f"Completed: {completed_goal['task']}" + (f"\nResult: {result}" if result else ""),
            segment_type="goal_completed",
            source=completed_goal.get('agent'),
            metadata=completed_goal,
            importance=0.75  # Completed goals are important to remember
        )
        
        self._save_memory()
        return True
    
    def list_goals(self, agent: Optional[str] = None, priority: Optional[int] = None) -> str:
        """
        List current goals, optionally filtered by agent or priority
        
        Args:
            agent: Filter goals by assigned agent
            priority: Filter goals by priority level
            
        Returns:
            A formatted string of goals
        """
        if not self.goals:
            return "No current goals."
        
        filtered_goals = self.goals
        
        # Apply filters
        if agent:
            filtered_goals = [g for g in filtered_goals if g.get('agent', '').lower() == agent.lower()]
        
        if priority:
            filtered_goals = [g for g in filtered_goals if g.get('priority', 1) == priority]
        
        if not filtered_goals:
            return "No goals matching the specified filters."
        
        # Sort by priority (descending) and then by creation date
        filtered_goals.sort(key=lambda g: (-g.get('priority', 1), g.get('created', '')))
        
        # Format the output
        output_lines = []
        for i, g in enumerate(filtered_goals):
            priority_str = f"[Priority: {g.get('priority', 1)}]"
            agent_str = f"[{g.get('agent', 'General')}]"
            deadline_str = f" (Due: {g.get('deadline', 'No deadline')})" if 'deadline' in g else ""
            
            output_lines.append(f"{i+1}. {priority_str} {agent_str} {g.get('task', '')} {deadline_str}")
        
        return "\n".join(output_lines)
    
    def get_goals_list(self):
        """Returns the raw list of goal dictionaries"""
        return self.goals
    
    def list_completed(self, limit: int = 10) -> str:
        """
        List completed goals, with optional limit
        
        Args:
            limit: Maximum number of completed goals to list (default: 10, 0 for all)
            
        Returns:
            A formatted string of completed goals
        """
        if not self.completed_goals:
            return "No completed goals yet."
        
        # Sort by completion timestamp (most recent first)
        sorted_completed = sorted(
            self.completed_goals,
            key=lambda g: g.get('completed_time', ''),
            reverse=True
        )
        
        # Apply limit
        if limit > 0:
            sorted_completed = sorted_completed[:limit]
        
        output_lines = []
        for i, g in enumerate(sorted_completed):
            agent_str = f"[{g.get('agent', 'General')}]"
            completed_str = f"(completed on {g.get('completed_time', 'unknown')}"
            
            if 'result' in g:
                result_preview = g['result'][:50] + ("..." if len(g['result']) > 50 else "")
                completed_str += f", result: {result_preview})"
            else:
                completed_str += ")"
            
            output_lines.append(f"{i+1}. {agent_str} {g.get('task', '')} {completed_str}")
        
        return "\n".join(output_lines)
    
    def set_agent_context(self, agent_name: str, key: str, value: Any) -> None:
        """
        Set a contextual value for a specific agent
        
        Args:
            agent_name: The name of the agent
            key: The context key
            value: The context value
        """
        agent_name = agent_name.lower()
        if agent_name not in self.agent_contexts:
            self.agent_contexts[agent_name] = {}
        
        self.agent_contexts[agent_name][key] = value
        self._save_memory()
    
    def get_agent_context(self, agent_name: str, key: str, default: Any = None) -> Any:
        """
        Get a contextual value for a specific agent
        
        Args:
            agent_name: The name of the agent
            key: The context key
            default: Default value if the key doesn't exist
            
        Returns:
            The context value or default
        """
        agent_name = agent_name.lower()
        if agent_name not in self.agent_contexts:
            return default
        
        return self.agent_contexts[agent_name].get(key, default)
    
    def search_memory(self, query: str, segment_type: Optional[str] = None, 
                     source: Optional[str] = None, limit: int = 10) -> List[MemorySegment]:
        """
        Search memory segments by content
        
        Args:
            query: The search query string
            segment_type: Optional filter by segment type
            source: Optional filter by source (e.g., agent name)
            limit: Maximum number of results to return
            
        Returns:
            List of matching memory segments
        """
        candidates = self.segments
        
        # Apply segment_type filter if provided
        if segment_type:
            candidates = self.segment_by_type.get(segment_type, [])
        
        # Apply source filter if provided
        if source:
            # If we already filtered by type, further filter by source
            if segment_type:
                candidates = [s for s in candidates if s.source == source]
            else:
                # Otherwise use the source index directly
                candidates = self.segment_by_source.get(source, [])
        
        # Search for the query in content
        results = []
        for segment in candidates:
            if re.search(query, segment.content, re.IGNORECASE):
                results.append(segment)
                if len(results) >= limit:
                    break
        
        return results
    
    def get_recent_memory(self, segment_type: Optional[str] = None, 
                         source: Optional[str] = None, limit: int = 10) -> List[MemorySegment]:
        """
        Get recent memory segments
        
        Args:
            segment_type: Optional filter by segment type
            source: Optional filter by source (e.g., agent name)
            limit: Maximum number of results to return
            
        Returns:
            List of recent memory segments
        """
        candidates = self.segments
        
        # Apply segment_type filter if provided
        if segment_type:
            candidates = self.segment_by_type.get(segment_type, [])
        
        # Apply source filter if provided
        if source:
            # If we already filtered by type, further filter by source
            if segment_type:
                candidates = [s for s in candidates if s.source == source]
            else:
                # Otherwise use the source index directly
                candidates = self.segment_by_source.get(source, [])
        
        # Sort by timestamp (most recent first)
        candidates.sort(key=lambda s: s.timestamp, reverse=True)
        
        return candidates[:limit]
    
    def get_memory_by_id(self, segment_id: str) -> Optional[MemorySegment]:
        """Get a specific memory segment by ID"""
        for segment in self.segments:
            if segment.id == segment_id:
                return segment
        return None
    
    def delete_segment(self, segment_id: str) -> bool:
        """
        Delete a memory segment
        
        Args:
            segment_id: The ID of the segment to delete
            
        Returns:
            True if the segment was deleted, False otherwise
        """
        for i, segment in enumerate(self.segments):
            if segment.id == segment_id:
                # Remove from main list
                deleted_segment = self.segments.pop(i)
                
                # Remove from indexes
                if deleted_segment.segment_type in self.segment_by_type:
                    self.segment_by_type[deleted_segment.segment_type].remove(deleted_segment)
                
                if deleted_segment.source and deleted_segment.source in self.segment_by_source:
                    self.segment_by_source[deleted_segment.source].remove(deleted_segment)
                
                self._save_memory()
                return True
        
        return False
    
    def update_segment(self, segment_id: str, content: Optional[str] = None, 
                      metadata: Optional[Dict[str, Any]] = None, 
                      importance: Optional[float] = None) -> bool:
        """
        Update a memory segment
        
        Args:
            segment_id: The ID of the segment to update
            content: New content (or None to keep existing)
            metadata: New metadata to merge with existing (or None to keep existing)
            importance: New importance factor (or None to keep existing)
            
        Returns:
            True if the segment was updated, False otherwise
        """
        segment = self.get_memory_by_id(segment_id)
        if not segment:
            return False
        
        # Update fields
        if content is not None:
            segment.content = content
        
        if metadata is not None:
            # Merge with existing metadata
            segment.metadata.update(metadata)
        
        if importance is not None:
            segment.importance = importance
        
        self._save_memory()
        return True
    
    def get_memory_summary(self, segment_type: Optional[str] = None, 
                          source: Optional[str] = None, limit: int = 5) -> str:
        """
        Generate a summary of recent memory segments
        
        Args:
            segment_type: Optional filter by segment type
            source: Optional filter by source (e.g., agent name)
            limit: Maximum number of segments to include
            
        Returns:
            A formatted string summary
        """
        recent_segments = self.get_recent_memory(segment_type, source, limit)
        
        if not recent_segments:
            return "No relevant memory segments found."
        
        summary_lines = ["Recent memory segments:"]
        for i, segment in enumerate(recent_segments):
            # Format timestamp for readability
            dt = datetime.fromisoformat(segment.timestamp)
            friendly_time = dt.strftime("%Y-%m-%d %H:%M:%S")
            
            # Create a preview of the content
            content_preview = segment.content[:100] + ("..." if len(segment.content) > 100 else "")
            content_preview = content_preview.replace("\n", " ")
            
            summary_lines.append(
                f"{i+1}. [{segment.segment_type}] From: {segment.source or 'unknown'} at {friendly_time}\n"
                f"   {content_preview}"
            )
        
        return "\n".join(summary_lines)
    
    def get_related_memories(self, query: str, limit: int = 5) -> List[MemorySegment]:
        """
        Find memories related to a query using a simple relevance algorithm
        
        Args:
            query: The query string to find related memories
            limit: Maximum number of results to return
            
        Returns:
            List of related memory segments
        """
        # Basic implementation using keyword matching
        # For more advanced implementation, consider using embeddings or a proper vector database
        
        # Split query into keywords (filtering out common words)
        stopwords = {"the", "a", "an", "of", "to", "in", "for", "on", "with", "by", "at", "from"}
        keywords = [word.lower() for word in re.findall(r'\w+', query) 
                    if word.lower() not in stopwords and len(word) > 2]
        
        if not keywords:
            # If no good keywords, return recent memories
            return self.get_recent_memory(limit=limit)
        
        # Score each memory segment based on keyword matches
        scored_segments = []
        for segment in self.segments:
            score = 0
            content_lower = segment.content.lower()
            
            # Check for keyword matches
            for keyword in keywords:
                # Count occurrences of the keyword
                matches = len(re.findall(r'\b' + re.escape(keyword) + r'\b', content_lower))
                score += matches
            
            # Adjust score by importance factor
            score *= segment.importance
            
            if score > 0:
                scored_segments.append((segment, score))
        
        # Sort by score (descending)
        scored_segments.sort(key=lambda x: x[1], reverse=True)
        
        # Return top segments
        return [segment for segment, score in scored_segments[:limit]]
    
    def flush(self) -> str:
        """Clear all memory, returning a confirmation message"""
        self.segments = []
        self.goals = []
        self.completed_goals = []
        self.agent_contexts = {}
        self.last_output = {}
        self.last_agent_name = None
        self.segment_by_type = {}
        self.segment_by_source = {}
        
        self._save_memory()
        return "[EnhancedMemory] All memory wiped."