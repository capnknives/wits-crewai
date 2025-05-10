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
import uuid # For generating unique goal IDs

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
        # Goals will now be a list of dictionaries, each with an 'id' and 'status'
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

    def _initialize_empty_memory(self):
        """Initializes memory to an empty state."""
        self.segments = []
        self.goals = []
        self.completed_goals = []
        self.agent_contexts = {}
        self.last_output = {}
        self.last_agent_name = None
        self.segment_by_type = {}
        self.segment_by_source = {}
        print("[EnhancedMemory] Initialized with empty memory structures.")

    def _load_memory(self) -> None:
        """Load memory from file, ensuring goals have id and status."""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self.segments = [MemorySegment.from_dict(segment_data) for segment_data in data.get('segments', [])]
                
                loaded_goals = data.get('goals', [])
                self.goals = []
                for g_data in loaded_goals:
                    if not isinstance(g_data, dict): # Skip non-dict goals
                        print(f"[EnhancedMemory] Warning: Found non-dictionary goal item during load: {g_data}. Skipping.")
                        continue
                    g_data.setdefault('id', str(uuid.uuid4())) # Ensure ID exists
                    g_data.setdefault('status', 'pending') # Default status for older goals
                    g_data.setdefault('priority', g_data.get('priority', 1)) # Ensure priority
                    g_data.setdefault('created', g_data.get('created', datetime.now().isoformat())) # Ensure created timestamp
                    self.goals.append(g_data)

                loaded_completed_goals = data.get('completed_goals', [])
                self.completed_goals = []
                for cg_data in loaded_completed_goals:
                    if not isinstance(cg_data, dict):
                        print(f"[EnhancedMemory] Warning: Found non-dictionary completed goal item: {cg_data}. Skipping.")
                        continue
                    cg_data.setdefault('id', str(uuid.uuid4()))
                    cg_data['status'] = 'completed' # Ensure status
                    self.completed_goals.append(cg_data)
                    
                self.agent_contexts = data.get('agent_contexts', {})
                self.last_output = data.get('last_output', {})
                self.last_agent_name = data.get('last_agent_name')
                
                self._rebuild_indexes()
                print(f"[EnhancedMemory] Loaded {len(self.segments)} segments, {len(self.goals)} active goals, {len(self.completed_goals)} completed goals from {self.memory_file}")
            
            except json.JSONDecodeError as e_json:
                print(f"[EnhancedMemory] Error decoding JSON from {self.memory_file}: {e_json}. Initializing fresh memory.")
                self._initialize_empty_memory()
            except Exception as e:
                print(f"[EnhancedMemory] Error loading memory from {self.memory_file}: {e}. Initializing fresh memory.")
                self._initialize_empty_memory()
        else:
            print(f"[EnhancedMemory] No memory file found at {self.memory_file}, starting with empty memory.")
            self._initialize_empty_memory()
            self._save_memory() # Create an empty memory file
    
    def _save_memory(self) -> None:
        """Save memory to file, including new goal structure."""
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
            if segment.segment_type not in self.segment_by_type:
                self.segment_by_type[segment.segment_type] = []
            self.segment_by_type[segment.segment_type].append(segment)
            
            if segment.source:
                if segment.source not in self.segment_by_source:
                    self.segment_by_source[segment.source] = []
                self.segment_by_source[segment.source].append(segment)
    
    def add_segment(self, content: str, segment_type: str, source: str = None, 
                   metadata: Dict[str, Any] = None, importance: float = 0.5) -> str:
        """Add a new memory segment"""
        metadata = metadata or {}
        metadata['importance'] = importance
        
        segment = MemorySegment(
            content=content,
            segment_type=segment_type,
            source=source,
            metadata=metadata
        )
        
        self.segments.append(segment)
        
        if segment.segment_type not in self.segment_by_type:
            self.segment_by_type[segment.segment_type] = []
        self.segment_by_type[segment.segment_type].append(segment)
        
        if segment.source:
            if segment.source not in self.segment_by_source:
                self.segment_by_source[segment.source] = []
            self.segment_by_source[segment.source].append(segment)
        
        if len(self.segments) > self.max_segments:
            self._prune_memory()
        
        self._save_memory()
        return segment.id
    
    def _prune_memory(self) -> None:
        """Prune memory by removing least important segments."""
        high_importance = [s for s in self.segments if s.importance >= 0.8]
        other_segments = [s for s in self.segments if s.importance < 0.8]
        
        to_keep_count = max(0, self.max_segments - len(high_importance))
        
        if len(other_segments) > to_keep_count:
            other_segments.sort(key=lambda s: (s.importance, s.timestamp), reverse=True)
            other_segments = other_segments[:to_keep_count]
        
        self.segments = high_importance + other_segments
        self._rebuild_indexes()
        print(f"[EnhancedMemory] Pruned memory to {len(self.segments)} segments")

    def remember_output(self, agent_name, content): # Alias for compatibility
        return self.remember_agent_output(agent_name, content)

    def remember_agent_output(self, agent_name: str, content: str, importance: float = 0.6) -> Optional[str]:
        """Remember output produced by an agent."""
        if agent_name:
            agent_name_lower = agent_name.lower()
            self.last_output[agent_name_lower] = content
            self.last_agent_name = agent_name_lower
            
            segment_id = self.add_segment(
                content=content,
                segment_type="agent_output",
                source=agent_name_lower,
                metadata={"output_type": "agent_response"},
                importance=importance
            )
            # self._save_memory() # add_segment already saves
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
                priority: int = 1, deadline: Optional[str] = None,
                goal_id: Optional[str] = None) -> str: # Return goal_id
        """
        Add a new goal or task with a unique ID and status.
        Returns the ID of the added goal.
        """
        new_goal_id = goal_id or str(uuid.uuid4())
        goal = {
            'id': new_goal_id,
            'task': task_description,
            'created': datetime.now().isoformat(),
            'status': 'pending',  # Initial status: pending, processing, autonomous_failed, completed
            'agent_suggestion': agent, # Agent suggestion for the planner
            'priority': priority,
            'deadline': deadline,
            'retries': 0, # For autonomous processing attempts
            'last_attempt_time': None, # Timestamp of the last autonomous attempt
            'processing_plan_id': None # ID of the plan currently processing this goal
        }
        
        # Avoid adding duplicate goals if an ID is provided and already exists
        if goal_id and any(g['id'] == goal_id for g in self.goals):
            print(f"[EnhancedMemory] Goal with ID {goal_id} already exists. Not adding again.")
            return goal_id

        self.goals.append(goal)
        
        self.add_segment(
            content=f"PENDING GOAL (ID: {new_goal_id}): {task_description}",
            segment_type="goal_pending",
            source=agent or "System", # Use 'agent' if provided, else 'System'
            metadata={
                "goal_id": new_goal_id,
                "priority": priority,
                "deadline": deadline,
                "assigned_agent_suggestion": agent
            },
            importance=0.7 + (priority * 0.06) 
        )
        
        self._save_memory()
        print(f"[EnhancedMemory] Added Goal ID {new_goal_id}: {task_description[:50]}...")
        return new_goal_id

    def update_goal_status(self, goal_id: str, new_status: str, 
                           processing_plan_id: Optional[str] = None, 
                           result_summary: Optional[str] = None) -> bool:
        """
        Update the status of a goal.
        Valid statuses: pending, processing, autonomous_failed, completed, manual_override.
        """
        for goal in self.goals:
            if goal.get('id') == goal_id:
                goal['status'] = new_status
                goal['updated_time'] = datetime.now().isoformat()
                if new_status == 'processing':
                    goal['processing_plan_id'] = processing_plan_id
                    goal['last_attempt_time'] = datetime.now().isoformat()
                elif new_status == 'autonomous_failed':
                    goal['retries'] = goal.get('retries', 0) + 1
                    goal['last_attempt_time'] = datetime.now().isoformat()
                    goal['processing_plan_id'] = None # Clear plan ID on failure
                elif new_status == 'pending': # Resetting to pending
                    goal['processing_plan_id'] = None
                    # Optionally reset retries if you want it to be picked up fresh
                    # goal['retries'] = 0 
                if result_summary:
                    goal['result_summary'] = result_summary
                
                self._save_memory()
                print(f"[EnhancedMemory] Updated Goal ID {goal_id} to status: {new_status}")
                return True
        
        # If trying to update a completed goal (e.g., add a result summary later)
        for goal in self.completed_goals:
            if goal.get('id') == goal_id:
                if new_status == 'completed': # Reaffirm or update details
                    goal['status'] = 'completed'
                    goal['updated_time'] = datetime.now().isoformat()
                    if result_summary and not goal.get('result_summary'): # Only add if not already set
                        goal['result_summary'] = result_summary
                    self._save_memory()
                    print(f"[EnhancedMemory] Updated details for completed Goal ID {goal_id}.")
                    return True
                else:
                    print(f"[EnhancedMemory] Cannot change status of already completed Goal ID {goal_id} to '{new_status}'.")
                    return False

        print(f"[EnhancedMemory] Update failed: Goal ID {goal_id} not found in active goals.")
        return False

    def get_pending_goals(self) -> List[Dict[str, Any]]:
        """Returns a list of goals with 'pending' status, suitable for autonomous processing."""
        # Filter for goals that are 'pending' AND not currently associated with an active plan
        # (or add more sophisticated logic like retry counts, backoff periods for 'autonomous_failed')
        pending_for_pickup = [
            g for g in self.goals 
            if g.get('status') == 'pending' and not g.get('processing_plan_id')
        ]
        # Sort by priority (desc), then by creation date (asc)
        pending_for_pickup.sort(key=lambda g: (-g.get('priority', 1), g.get('created', '')))
        return pending_for_pickup
    
    def complete_goal(self, goal_id_or_task_desc: Union[str, int], result: Optional[str] = None) -> bool:
        """Mark a goal as completed. Prefers goal_id for accuracy."""
        goal_to_complete = None
        index_to_pop = -1

        # Prioritize finding by ID
        if isinstance(goal_id_or_task_desc, str):
            for i, g in enumerate(self.goals):
                if g.get('id') == goal_id_or_task_desc:
                    goal_to_complete = g
                    index_to_pop = i
                    break
        
        # Fallback to index (if int) or task description (if str and not an ID)
        if not goal_to_complete:
            if isinstance(goal_id_or_task_desc, int):
                user_index = goal_id_or_task_desc - 1 # Assuming 1-based index from user
                # This is tricky because the list order can change. Best to get ID from UI/user.
                # For now, we'll assume the user is referring to the current `self.goals` list order.
                # A safer way would be for the UI to pass the ID.
                if 0 <= user_index < len(self.goals):
                    goal_to_complete = self.goals[user_index]
                    index_to_pop = user_index
            elif isinstance(goal_id_or_task_desc, str): # Match by task description as last resort
                for i, g in enumerate(self.goals):
                    if goal_id_or_task_desc.lower() in g.get('task', '').lower():
                        goal_to_complete = g
                        index_to_pop = i
                        print(f"[EnhancedMemory] Warning: Matched goal for completion by task description '{goal_id_or_task_desc}'. Using Goal ID is preferred.")
                        break
        
        if goal_to_complete and index_to_pop != -1:
            completed_goal_entry = self.goals.pop(index_to_pop)
            completed_goal_entry['status'] = 'completed'
            completed_goal_entry['completed_time'] = datetime.now().isoformat()
            completed_goal_entry['updated_time'] = completed_goal_entry['completed_time']
            if result:
                completed_goal_entry['result'] = result # Store the main result
            
            self.completed_goals.append(completed_goal_entry)
            
            self.add_segment(
                content=f"COMPLETED GOAL (ID: {completed_goal_entry['id']}): {completed_goal_entry['task']}" + (f"\nResult: {result}" if result else ""),
                segment_type="goal_completed",
                source=completed_goal_entry.get('agent_suggestion') or "System",
                metadata={key: val for key, val in completed_goal_entry.items() if key not in ['task']},
                importance=0.85 
            )
            self._save_memory()
            print(f"[EnhancedMemory] Goal ID {completed_goal_entry['id']} marked as completed.")
            return True
            
        print(f"[EnhancedMemory] Could not complete goal: '{goal_id_or_task_desc}' not found or not a valid identifier in active goals.")
        return False

    def get_goal_by_id(self, goal_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific goal by its ID from active or completed goals."""
        for goal in self.goals:
            if goal.get('id') == goal_id:
                return goal
        for goal in self.completed_goals:
             if goal.get('id') == goal_id:
                return goal
        return None
        
    def list_goals(self, agent: Optional[str] = None, priority: Optional[int] = None, status_filter: Optional[str] = None) -> str:
        """List current goals, optionally filtered, and show ID and status."""
        if not self.goals:
            return "No current goals."
        
        display_goals = self.goals
        
        if status_filter:
            display_goals = [g for g in display_goals if g.get('status', 'pending') == status_filter]
        if agent: # Filter by agent_suggestion
            display_goals = [g for g in display_goals if g.get('agent_suggestion', '').lower() == agent.lower()]
        if priority:
            display_goals = [g for g in display_goals if g.get('priority', 1) == priority]
        
        if not display_goals:
            return "No goals matching the specified filters."
            
        # Sort by priority (desc), then by creation date (asc)
        display_goals.sort(key=lambda g: (-g.get('priority', 1), g.get('created', '')))
        
        output_lines = ["Current Goals:"]
        for i, g in enumerate(display_goals): # User-friendly 1-based index for display
            goal_id_str = g.get('id', 'N/A')
            priority_str = f"[Prio: {g.get('priority', 1)}]"
            agent_sugg_str = f"[Suggest: {g.get('agent_suggestion', 'Any')}]"
            deadline_str = f" (Due: {g.get('deadline', 'N/A')})" if g.get('deadline') else ""
            status_str = g.get('status', 'pending')
            plan_id_str = f" (Plan: {g.get('processing_plan_id', 'None')[:8]}...)" if g.get('processing_plan_id') else ""
            
            output_lines.append(
                f"{i+1}. ID: {goal_id_str[:8]}... {priority_str} {agent_sugg_str} "
                f"Task: {g.get('task', '')}{deadline_str} [Status: {status_str}]{plan_id_str}"
            )
        return "\n".join(output_lines)
    
    def get_goals_list(self): # This is used by main.py
        """Returns the raw list of active goal dictionaries."""
        return self.goals
    
    def list_completed(self, limit: int = 10) -> str:
        """List completed goals, with ID and status."""
        if not self.completed_goals:
            return "No completed goals yet."
        
        sorted_completed = sorted(self.completed_goals, key=lambda g: g.get('completed_time', ''), reverse=True)
        if limit > 0:
            sorted_completed = sorted_completed[:limit]
        
        output_lines = ["Recently Completed Goals:"]
        for i, g in enumerate(sorted_completed):
            goal_id_str = g.get('id', 'N/A')
            agent_sugg_str = f"[Suggest: {g.get('agent_suggestion', 'Any')}]" # Show original suggestion
            completed_time_str = g.get('completed_time', 'N/A')
            result_str = g.get('result', '')
            result_preview = result_str[:50] + ("..." if len(result_str) > 50 else "")
            
            output_lines.append(
                f"{i+1}. ID: {goal_id_str[:8]}... {agent_sugg_str} Task: {g.get('task', '')} "
                f"(Completed: {completed_time_str})" + (f" Result: {result_preview}" if result_str else "") +
                f" [Status: {g.get('status', 'completed')}]" # Should always be 'completed' here
            )
        return "\n".join(output_lines)
    
    def set_agent_context(self, agent_name: str, key: str, value: Any) -> None:
        """Set a contextual value for a specific agent."""
        agent_name_lower = agent_name.lower()
        if agent_name_lower not in self.agent_contexts:
            self.agent_contexts[agent_name_lower] = {}
        self.agent_contexts[agent_name_lower][key] = value
        self._save_memory()
    
    def get_agent_context(self, agent_name: str, key: str, default: Any = None) -> Any:
        """Get a contextual value for a specific agent."""
        agent_name_lower = agent_name.lower()
        return self.agent_contexts.get(agent_name_lower, {}).get(key, default)
    
    def search_memory(self, query: str, segment_type: Optional[str] = None, 
                     source: Optional[str] = None, limit: int = 10) -> List[MemorySegment]:
        """Search memory segments by content."""
        candidates = list(self.segments) # Work on a copy for filtering
        
        if segment_type:
            candidates = [s for s in candidates if s.segment_type == segment_type]
        if source:
            candidates = [s for s in candidates if s.source and s.source.lower() == source.lower()]
        
        results = []
        query_lower = query.lower()
        for segment in candidates:
            if query_lower in segment.content.lower(): # Simple substring match
                results.append(segment)
        
        # Sort by timestamp (most recent first) as a simple relevance proxy if no other scoring
        results.sort(key=lambda s: s.timestamp, reverse=True)
        return results[:limit]
    
    def get_recent_memory(self, segment_type: Optional[str] = None, 
                         source: Optional[str] = None, limit: int = 10) -> List[MemorySegment]:
        """Get recent memory segments."""
        candidates = list(self.segments)
        if segment_type:
            candidates = [s for s in candidates if s.segment_type == segment_type]
        if source:
            candidates = [s for s in candidates if s.source and s.source.lower() == source.lower()]
        
        candidates.sort(key=lambda s: s.timestamp, reverse=True)
        return candidates[:limit]
    
    def get_memory_by_id(self, segment_id: str) -> Optional[MemorySegment]:
        """Get a specific memory segment by ID."""
        for segment in self.segments:
            if segment.id == segment_id:
                return segment
        return None
    
    def delete_segment(self, segment_id: str) -> bool:
        """Delete a memory segment."""
        original_len = len(self.segments)
        self.segments = [s for s in self.segments if s.id != segment_id]
        if len(self.segments) < original_len:
            self._rebuild_indexes() # Rebuild if a segment was actually removed
            self._save_memory()
            return True
        return False
    
    def update_segment(self, segment_id: str, content: Optional[str] = None, 
                      metadata: Optional[Dict[str, Any]] = None, 
                      importance: Optional[float] = None) -> bool:
        """Update a memory segment."""
        segment = self.get_memory_by_id(segment_id)
        if not segment:
            return False
        
        if content is not None:
            segment.content = content
        if metadata is not None:
            segment.metadata.update(metadata)
        if importance is not None:
            segment.importance = importance
            if 'importance' in segment.metadata: # Also update metadata if it was there
                 segment.metadata['importance'] = importance
        
        self._save_memory()
        return True
    
    def get_memory_summary(self, segment_type: Optional[str] = None, 
                          source: Optional[str] = None, limit: int = 5) -> str:
        """Generate a summary of recent memory segments."""
        recent_segments = self.get_recent_memory(segment_type, source, limit)
        if not recent_segments:
            return "No relevant memory segments found."
        
        summary_lines = ["Recent memory segments:"]
        for i, segment in enumerate(recent_segments):
            dt = datetime.fromisoformat(segment.timestamp)
            friendly_time = dt.strftime("%Y-%m-%d %H:%M:%S")
            content_preview = segment.content[:100].replace("\n", " ") + ("..." if len(segment.content) > 100 else "")
            summary_lines.append(
                f"{i+1}. [{segment.segment_type}] From: {segment.source or 'unknown'} at {friendly_time}\n"
                f"   Content: {content_preview}"
            )
        return "\n".join(summary_lines)
    
    def get_related_memories(self, query: str, limit: int = 5) -> List[MemorySegment]:
        """Find memories related to a query using keyword matching and importance."""
        stopwords = {"the", "a", "an", "of", "to", "in", "for", "on", "with", "by", "at", "from", "is", "are", "was", "were"}
        query_keywords = {word.lower() for word in re.findall(r'\w+', query) if word.lower() not in stopwords and len(word) > 2}
        
        if not query_keywords:
            return self.get_recent_memory(limit=limit)
        
        scored_segments = []
        for segment in self.segments:
            score = 0
            content_lower = segment.content.lower()
            segment_words = {word.lower() for word in re.findall(r'\w+', content_lower) if word.lower() not in stopwords}
            
            common_keywords = query_keywords.intersection(segment_words)
            score += len(common_keywords) * 2 # Higher weight for direct keyword match
            
            # Bonus for query being a substring
            if query.lower() in content_lower:
                score += 3

            # Consider importance
            score *= (0.5 + segment.importance) # Scale score by importance (0.5 to 1.5 factor)
            
            if score > 0:
                scored_segments.append((segment, score))
        
        scored_segments.sort(key=lambda x: x[1], reverse=True)
        return [segment for segment, score in scored_segments[:limit]]
    
    def flush(self) -> str:
        """Clear all memory, returning a confirmation message"""
        self._initialize_empty_memory() # Use the helper
        self._save_memory()
        return "[EnhancedMemory] All memory wiped."

