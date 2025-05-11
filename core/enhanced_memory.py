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
        segment.id = data.get("id", segment._generate_id()) # Ensure ID exists or generate
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
        self.agent_contexts: Dict[str, Dict[str, Any]] = {}
        self.last_output: Dict[str, str] = {}
        self.last_agent_name: Optional[str] = None
        
        self.segment_by_type: Dict[str, List[MemorySegment]] = {}
        self.segment_by_source: Dict[str, List[MemorySegment]] = {}
        
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
                    if not isinstance(g_data, dict):
                        print(f"[EnhancedMemory] Warning: Found non-dictionary goal item during load: {g_data}. Skipping.")
                        continue
                    g_data.setdefault('id', str(uuid.uuid4())) 
                    g_data.setdefault('status', 'pending') 
                    g_data.setdefault('priority', g_data.get('priority', 1)) 
                    g_data.setdefault('created', g_data.get('created', datetime.now().isoformat()))
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
                        print(f"[EnhancedMemory] Warning: Found non-dictionary completed goal item: {cg_data}. Skipping.")
                        continue
                    cg_data.setdefault('id', str(uuid.uuid4()))
                    cg_data.setdefault('status', 'completed') # Ensure status
                    cg_data.setdefault('task', cg_data.get('task', 'Untitled Completed Goal'))
                    cg_data.setdefault('completed_time', cg_data.get('completed_time', datetime.now().isoformat()))
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
                import traceback
                traceback.print_exc()
                self._initialize_empty_memory()
        else:
            print(f"[EnhancedMemory] No memory file found at {self.memory_file}, starting with empty memory.")
            self._initialize_empty_memory()
            self._save_memory() 
    
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
        """Rebuild the memory segment indexes for type and source."""
        self.segment_by_type = {}
        self.segment_by_source = {}
        
        for segment in self.segments:
            if segment.segment_type not in self.segment_by_type:
                self.segment_by_type[segment.segment_type] = []
            self.segment_by_type[segment.segment_type].append(segment)
            
            if segment.source: # Ensure source exists
                if segment.source not in self.segment_by_source:
                    self.segment_by_source[segment.source] = []
                self.segment_by_source[segment.source].append(segment)
    
    def add_segment(self, content: str, segment_type: str, source: str = None, 
                   metadata: Dict[str, Any] = None, importance: float = 0.5) -> str:
        """Add a new memory segment."""
        metadata = metadata or {}
        metadata['importance'] = importance # Ensure importance is in metadata for MemorySegment
        
        segment = MemorySegment(
            content=content,
            segment_type=segment_type,
            source=source,
            metadata=metadata # Pass full metadata here
        )
        # segment.importance is set by MemorySegment's __init__ from metadata
        
        self.segments.append(segment)
        
        # Update indexes
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
        if len(self.segments) <= self.max_segments:
            return

        # Separate high importance segments (always keep these, e.g., active goals)
        # For simplicity, we'll sort all and take top N, but a more nuanced approach
        # might protect certain segment types or recent items.
        # Current logic from user's file:
        high_importance_segments = [s for s in self.segments if s.importance >= 0.8]
        other_segments = [s for s in self.segments if s.importance < 0.8]
        
        num_to_remove_from_others = len(other_segments) - max(0, self.max_segments - len(high_importance_segments))

        if num_to_remove_from_others > 0:
            # Sort other_segments by importance (ascending) then by timestamp (oldest first) to remove
            other_segments.sort(key=lambda s: (s.importance, s.timestamp))
            segments_to_keep_from_others = other_segments[num_to_remove_from_others:]
            self.segments = high_importance_segments + segments_to_keep_from_others
        else: # Enough space, or only high importance segments
            self.segments = high_importance_segments + other_segments
            # Still, ensure we don't exceed max_segments if high_importance alone is too many
            if len(self.segments) > self.max_segments:
                self.segments.sort(key=lambda s: (s.importance, s.timestamp), reverse=True)
                self.segments = self.segments[:self.max_segments]

        self._rebuild_indexes() # Rebuild type/source indexes
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
                metadata={"output_type": "agent_response"}, # 'importance' will be added by add_segment
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
                goal_id: Optional[str] = None) -> str:
        """Add a new goal or task with a unique ID and status. Returns the ID of the added goal."""
        new_goal_id = goal_id or str(uuid.uuid4())
        goal = {
            'id': new_goal_id,
            'task': task_description,
            'created': datetime.now().isoformat(),
            'status': 'pending',
            'agent_suggestion': agent,
            'priority': priority,
            'deadline': deadline,
            'retries': 0,
            'last_attempt_time': None,
            'processing_plan_id': None,
            'result_summary': None # Initialize result_summary
        }
        
        if goal_id and any(g['id'] == goal_id for g in self.goals):
            print(f"[EnhancedMemory] Goal with ID {goal_id} already exists. Not adding again.")
            return goal_id

        self.goals.append(goal)
        
        self.add_segment(
            content=f"PENDING GOAL (ID: {new_goal_id}): {task_description}",
            segment_type="goal_pending",
            source=agent or "System",
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
        """Update the status of a goal."""
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
                    goal['processing_plan_id'] = None 
                elif new_status == 'pending': 
                    goal['processing_plan_id'] = None
                if result_summary: # Update or set result summary
                    goal['result_summary'] = result_summary
                
                self._save_memory()
                print(f"[EnhancedMemory] Updated Goal ID {goal_id} to status: {new_status}")
                return True
        
        for goal in self.completed_goals: # Check completed goals too for updates like adding a result summary
            if goal.get('id') == goal_id:
                if new_status == 'completed': 
                    goal['status'] = 'completed' # Reaffirm
                    goal['updated_time'] = datetime.now().isoformat()
                    if result_summary and not goal.get('result_summary'): # Add if not already set
                        goal['result_summary'] = result_summary
                    self._save_memory()
                    print(f"[EnhancedMemory] Updated details for completed Goal ID {goal_id}.")
                    return True
                else:
                    print(f"[EnhancedMemory] Cannot change status of already completed Goal ID {goal_id} to '{new_status}'. Only result_summary can be added.")
                    return False

        print(f"[EnhancedMemory] Update failed: Goal ID {goal_id} not found in active or completed goals.")
        return False

    def get_pending_goals(self) -> List[Dict[str, Any]]:
        """Returns a list of goals with 'pending' status, suitable for autonomous processing."""
        pending_for_pickup = [
            g for g in self.goals 
            if g.get('status') == 'pending' and not g.get('processing_plan_id')
        ]
        pending_for_pickup.sort(key=lambda g: (-g.get('priority', 1), g.get('created', '')))
        return pending_for_pickup
    
    def complete_goal(self, goal_id_or_task_desc: Union[str, int], result: Optional[str] = None) -> bool:
        """Mark a goal as completed. Prefers goal_id for accuracy."""
        goal_to_complete = None
        index_to_pop = -1

        # Try to find by ID first
        if isinstance(goal_id_or_task_desc, str): # Could be ID or task description
            for i, g in enumerate(self.goals):
                if g.get('id') == goal_id_or_task_desc:
                    goal_to_complete = g
                    index_to_pop = i
                    break
        
        # If not found by ID and it's an int, try by displayed index (less reliable)
        if not goal_to_complete and isinstance(goal_id_or_task_desc, int):
            # This assumes the list_goals() display order. For UI, passing ID is better.
            # For CLI, if user says "complete goal 1", it refers to the displayed list.
            # We need a way to map this displayed index back to an ID or actual list index.
            # Let's sort self.goals as list_goals does, then use the index.
            temp_sorted_goals = sorted(self.goals, key=lambda g_sort: (-g_sort.get('priority', 1), g_sort.get('created', '')))
            user_index = goal_id_or_task_desc - 1 
            if 0 <= user_index < len(temp_sorted_goals):
                goal_to_find_in_main_list = temp_sorted_goals[user_index]
                for i, g_main in enumerate(self.goals):
                    if g_main.get('id') == goal_to_find_in_main_list.get('id'):
                        goal_to_complete = g_main
                        index_to_pop = i
                        break
                if goal_to_complete:
                    print(f"[EnhancedMemory] Matched goal for completion by displayed index {goal_id_or_task_desc} (ID: {goal_to_complete.get('id')}).")


        # Fallback to task description match if still not found and input was string
        if not goal_to_complete and isinstance(goal_id_or_task_desc, str):
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
            if result: # Store the main result/summary
                completed_goal_entry['result_summary'] = result # Changed from 'result' to 'result_summary'
            
            self.completed_goals.append(completed_goal_entry)
            
            self.add_segment(
                content=f"COMPLETED GOAL (ID: {completed_goal_entry['id']}): {completed_goal_entry['task']}" + (f"\nResult Summary: {result}" if result else ""),
                segment_type="goal_completed",
                source=completed_goal_entry.get('agent_suggestion') or "System",
                metadata={key: val for key, val in completed_goal_entry.items() if key not in ['task', 'content']}, # Avoid duplicating task in content and metadata
                importance=0.85 
            )
            self._save_memory()
            print(f"[EnhancedMemory] Goal ID {completed_goal_entry['id']} marked as completed.")
            return True
            
        print(f"[EnhancedMemory] Could not complete goal: '{goal_id_or_task_desc}' not found or not a valid identifier in active goals.")
        return False

    def delete_goal_permanently(self, goal_id: str) -> bool:
        """
        Permanently deletes a goal from active and completed lists, and its related memory segments.
        """
        goal_found_and_deleted = False
        original_task_desc = None

        # Try to delete from active goals
        for i, goal in enumerate(self.goals):
            if goal.get('id') == goal_id:
                original_task_desc = goal.get('task', 'Unknown task')
                del self.goals[i]
                goal_found_and_deleted = True
                print(f"[EnhancedMemory] Deleted active goal ID {goal_id}: '{original_task_desc}'")
                break
        
        # Try to delete from completed goals if not found in active
        if not goal_found_and_deleted:
            for i, goal in enumerate(self.completed_goals):
                if goal.get('id') == goal_id:
                    original_task_desc = goal.get('task', 'Unknown task')
                    del self.completed_goals[i]
                    goal_found_and_deleted = True
                    print(f"[EnhancedMemory] Deleted completed goal ID {goal_id}: '{original_task_desc}'")
                    break
        
        if goal_found_and_deleted:
            # Remove related memory segments
            segments_to_keep = []
            deleted_segment_count = 0
            for segment in self.segments:
                # Check if segment's metadata links it to the deleted goal
                if segment.metadata.get('goal_id') == goal_id:
                    print(f"[EnhancedMemory] Deleting segment ID {segment.id} related to goal ID {goal_id}")
                    deleted_segment_count += 1
                # Also check if content indicates it's the primary goal_pending/completed segment
                elif segment.segment_type in ["goal_pending", "goal_completed"] and f"(ID: {goal_id})" in segment.content:
                    print(f"[EnhancedMemory] Deleting primary segment ID {segment.id} for goal ID {goal_id}")
                    deleted_segment_count += 1
                else:
                    segments_to_keep.append(segment)
            
            if deleted_segment_count > 0:
                self.segments = segments_to_keep
                self._rebuild_indexes() # Rebuild if segments were removed

            self._save_memory()
            return True
        
        print(f"[EnhancedMemory] Permanent delete failed: Goal ID {goal_id} not found.")
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
        
        display_goals = list(self.goals) # Create a copy for filtering
        
        if status_filter:
            display_goals = [g for g in display_goals if g.get('status', 'pending').lower() == status_filter.lower()]
        if agent: 
            display_goals = [g for g in display_goals if g.get('agent_suggestion', '').lower() == agent.lower()]
        if priority is not None: # Allow priority 0
            display_goals = [g for g in display_goals if g.get('priority', 1) == priority]
        
        if not display_goals:
            return "No goals matching the specified filters."
            
        display_goals.sort(key=lambda g: (-g.get('priority', 1), g.get('created', '')))
        
        output_lines = ["Current Goals:"]
        for i, g in enumerate(display_goals):
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
    
    def get_goals_list(self): # This is used by main.py and app.py
        """Returns the raw list of active goal dictionaries."""
        return self.goals
    
    def list_completed(self, limit: int = 10) -> str:
        """List completed goals, with ID and status."""
        if not self.completed_goals:
            return "No completed goals yet."
        
        # Sort by completion time, most recent first
        sorted_completed = sorted(self.completed_goals, key=lambda g: g.get('completed_time', ''), reverse=True)
        if limit > 0:
            sorted_completed = sorted_completed[:limit]
        
        output_lines = ["Recently Completed Goals:"]
        for i, g in enumerate(sorted_completed):
            goal_id_str = g.get('id', 'N/A')
            agent_sugg_str = f"[Original Suggest: {g.get('agent_suggestion', 'Any')}]"
            completed_time_str = g.get('completed_time', 'N/A')
            result_summary = g.get('result_summary', '') # Changed from 'result'
            result_preview = result_summary[:50] + ("..." if len(result_summary) > 50 else "")
            
            output_lines.append(
                f"{i+1}. ID: {goal_id_str[:8]}... {agent_sugg_str} Task: {g.get('task', '')} "
                f"(Completed: {completed_time_str})" + (f" Result: {result_preview}" if result_summary else "") +
                f" [Status: {g.get('status', 'completed')}]"
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
        """Search memory segments by content (simple keyword match)."""
        candidates = list(self.segments)
        
        if segment_type:
            candidates = [s for s in candidates if s.segment_type == segment_type]
        if source:
            candidates = [s for s in candidates if s.source and s.source.lower() == source.lower()]
        
        results = []
        query_lower = query.lower()
        for segment in candidates:
            if query_lower in segment.content.lower():
                results.append(segment)
        
        results.sort(key=lambda s: (s.importance, s.timestamp), reverse=True) # Sort by importance then recency
        return results[:limit]
    
    def get_recent_memory(self, segment_type: Optional[str] = None, 
                         source: Optional[str] = None, limit: int = 10) -> List[MemorySegment]:
        """Get recent memory segments, optionally filtered."""
        # This method from user's file is good.
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
        """Delete a memory segment by its ID."""
        original_len = len(self.segments)
        self.segments = [s for s in self.segments if s.id != segment_id]
        if len(self.segments) < original_len:
            self._rebuild_indexes() 
            self._save_memory()
            print(f"[EnhancedMemory] Deleted segment ID {segment_id}.")
            return True
        print(f"[EnhancedMemory] Segment ID {segment_id} not found for deletion.")
        return False
    
    def update_segment(self, segment_id: str, content: Optional[str] = None, 
                      metadata: Optional[Dict[str, Any]] = None, 
                      importance: Optional[float] = None) -> bool:
        """Update a memory segment."""
        segment = self.get_memory_by_id(segment_id)
        if not segment:
            return False
        
        changed = False
        if content is not None and segment.content != content:
            segment.content = content
            changed = True
        if metadata is not None:
            segment.metadata.update(metadata)
            changed = True # Assume metadata change is significant
        if importance is not None and segment.importance != importance:
            segment.importance = importance
            segment.metadata['importance'] = importance 
            changed = True
        
        if changed:
            segment.timestamp = datetime.now().isoformat() # Update timestamp on any change
            self._save_memory()
        return True
    
    def get_memory_summary(self, segment_type: Optional[str] = None, 
                          source: Optional[str] = None, limit: int = 5) -> str:
        """Generate a summary of recent memory segments."""
        # This method from user's file is good.
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
        # This method from user's file is good.
        stopwords = {"the", "a", "an", "of", "to", "in", "for", "on", "with", "by", "at", "from", "is", "are", "was", "were"}
        query_keywords = {word.lower() for word in re.findall(r'\w+', query) if word.lower() not in stopwords and len(word) > 2}
        
        if not query_keywords: # If query is all stopwords or too short, return recent
            return self.get_recent_memory(limit=limit)
        
        scored_segments = []
        for segment in self.segments:
            score = 0
            content_lower = segment.content.lower()
            segment_words = {word.lower() for word in re.findall(r'\w+', content_lower) if word.lower() not in stopwords}
            
            common_keywords = query_keywords.intersection(segment_words)
            score += len(common_keywords) * 2 
            
            if query.lower() in content_lower: # Bonus for query being a substring
                score += 3

            score *= (0.5 + segment.importance) # Scale score by importance
            
            if score > 0:
                scored_segments.append((segment, score))
        
        scored_segments.sort(key=lambda x: x[1], reverse=True)
        return [segment for segment, score in scored_segments[:limit]]
    
    def flush(self) -> str:
        """Clear all memory, returning a confirmation message"""
        self._initialize_empty_memory()
        self._save_memory()
        return "[EnhancedMemory] All memory wiped."