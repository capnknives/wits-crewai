"""
Memory system for storing user goals, progress, and caching recent agent outputs.
"""
import json
import os
from datetime import datetime

class Memory:
    def __init__(self, memory_file='memory.json'):
        self.memory_file = memory_file
        self.goals = []
        self.completed = []
        self.last_output = {}
        self.last_agent_name = None # Added to track the last agent providing output

        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.goals = data.get('goals', [])
                    self.completed = data.get('completed', [])
                    self.last_output = data.get('last_output', {})
                    # Optionally load last_agent_name if you decide to persist it
                    # self.last_agent_name = data.get('last_agent_name', None)
            except json.JSONDecodeError:
                print("[Memory] Warning: Memory file corrupted. Reinitializing.")
        else:
            self._save()

        if not isinstance(self.goals, list):
            self.goals = []
        if not isinstance(self.completed, list):
            self.completed = []
        if not isinstance(self.last_output, dict):
            self.last_output = {}

    def _save(self):
        data = {
            'goals': self.goals,
            'completed': self.completed,
            'last_output': self.last_output
            # Optionally save last_agent_name if you decide to persist it
            # 'last_agent_name': self.last_agent_name
        }
        try:
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"[Memory] Save failed: {e}")

    def list_goals(self):
        if not self.goals:
            return "No current goals."
        return "\n".join(
            f"{i+1}. [{g.get('agent','General')}] {g.get('task','')} (pending)" # Added 'General' default
            for i, g in enumerate(self.goals)
        )

    def get_goals_list(self):
        """Returns the raw list of goal dictionaries."""
        return self.goals

    def list_completed(self):
        if not self.completed:
            return "No completed goals yet."
        return "\n".join(
            f"{i+1}. [{g.get('agent','General')}] {g.get('task','')} (completed on {g.get('completed_time','')})" # Added 'General' default
            for i, g in enumerate(self.completed)
        )

    def add_goal(self, task_description, agent=None):
        new_goal = {'task': task_description}
        if agent:
            new_goal['agent'] = agent
        self.goals.append(new_goal)
        self._save()
        return True

    def complete_goal(self, index_or_task):
        goal_entry = None
        if isinstance(index_or_task, int):
            idx = index_or_task - 1
            if 0 <= idx < len(self.goals):
                goal_entry = self.goals.pop(idx)
        elif isinstance(index_or_task, str):
            for i, g in enumerate(self.goals):
                if index_or_task.lower() in g.get('task', '').lower():
                    goal_entry = self.goals.pop(i)
                    break
        if not goal_entry:
            return False

        goal_entry['completed_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.completed.append(goal_entry)
        self._save()
        return True

    def remember_output(self, agent_name, content):
        if agent_name:
            agent_name_lower = agent_name.lower()
            self.last_output[agent_name_lower] = content
            self.last_agent_name = agent_name_lower # Track the last agent
            self._save()

    def recall_output(self, agent_name):
        if agent_name:
            return self.last_output.get(agent_name.lower())
        return None

    def get_last_agent(self):
        """Returns the name of the agent that last provided an output."""
        return self.last_agent_name

    def flush(self):
        self.goals = []
        self.completed = []
        self.last_output = {}
        self.last_agent_name = None
        self._save()
        return "[Memory] All memory wiped."