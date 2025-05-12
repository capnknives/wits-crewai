# agents/planner_agent.py
import ollama
import re
import json
from datetime import datetime
import os
from typing import List, Dict, Any, Optional, Tuple
import time

# Assuming these are in the same directory or accessible via Python path
from .tools.base_tool import Tool, ToolException # If Planner uses tools directly
from .sentinel_agent import SentinelException # For type hinting or direct use

class PlannerAgent:
    def __init__(self, config, memory, quartermaster, sentinel, tools=None, agents=None, message_bus_client=None):
        self.config = config
        self.memory = memory # Instance of EnhancedMemory or VectorMemory
        self.qm = quartermaster # Instance of QuartermasterAgent
        self.sentinel = sentinel # Instance of SentinelAgent
        self.tools = tools if tools else [] # Tools Planner might use directly (if any)
        self.agents = agents if agents else {} # Dictionary of other agent instances for delegation
        model_conf = config.get("models", {})
        self.model_name = model_conf.get("planner", model_conf.get("default", "llama2"))

        # Ensure output_dir is correctly accessed from quartermaster
        self.plans_dir = os.path.join(self.qm.output_dir, "plans")
        os.makedirs(self.plans_dir, exist_ok=True)
        
        self.plans: Dict[str, Dict[str, Any]] = {}
        self.current_plan_id: Optional[str] = None
        self.current_goal_id_for_plan: Optional[str] = None # Track associated goal ID for a plan being created/executed
        self._load_plans()

        self.message_bus_client = message_bus_client # For inter-agent communication if needed

    def _generate_plan_id(self, goal_summary: str) -> str:
        """Generates a unique ID for a plan."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Sanitize goal summary for use in ID
        sanitized_goal = re.sub(r'\W+', '_', goal_summary.lower())[:30].strip('_')
        return f"plan_{timestamp}_{sanitized_goal}"

    def _load_plans(self):
        """Loads all plan .json files from the self.plans_dir directory, prioritizing plans_index.json."""
        print(f"[PlannerAgent] Loading plans from: {self.plans_dir}")
        self.plans = {} 
        try:
            if not os.path.exists(self.plans_dir):
                print(f"[PlannerAgent] Plans directory '{self.plans_dir}' does not exist. Creating it.")
                os.makedirs(self.plans_dir)
                return

            index_file_path = os.path.join(self.plans_dir, "plans_index.json")
            if os.path.exists(index_file_path):
                try:
                    with open(index_file_path, 'r', encoding='utf-8') as f:
                        loaded_index_plans_data = json.load(f)
                    
                    self.plans = loaded_index_plans_data
                    print(f"[PlannerAgent] Loaded {len(self.plans)} plans directly from index: {index_file_path}")
                    # Optional: Add verification here to ensure individual files for these plans exist
                    # and reconcile if necessary. For now, trusting the index if it exists.
                    return 
                except json.JSONDecodeError as e_json_idx:
                    print(f"[PlannerAgent] Warning: plans_index.json is corrupted ({e_json_idx}). Attempting to load individual plan files.")
                except Exception as e_idx: 
                    print(f"[PlannerAgent] Error loading from plans_index.json: {e_idx}. Attempting individual plan files.")
            
            print(f"[PlannerAgent] No valid index file found or error during index load. Scanning directory for individual plan files...")
            loaded_count = 0
            for filename in os.listdir(self.plans_dir):
                if filename.startswith("plan_") and filename.endswith(".json") and filename != "plans_index.json":
                    file_path = os.path.join(self.plans_dir, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            plan_data = json.load(f)
                        
                        plan_id_from_content = plan_data.get("id")
                        plan_id_from_filename = filename[:-5] 

                        current_plan_id_to_use = plan_id_from_content if plan_id_from_content else plan_id_from_filename
                        if not plan_id_from_content: 
                            plan_data["id"] = plan_id_from_filename
                        
                        if plan_id_from_content and plan_id_from_content != plan_id_from_filename:
                             print(f"[PlannerAgent] Warning: Plan ID '{plan_id_from_content}' in content of {filename} differs from filename-derived ID '{plan_id_from_filename}'. Using ID from content.")
                        
                        self.plans[current_plan_id_to_use] = plan_data
                        loaded_count += 1
                    except json.JSONDecodeError:
                        print(f"[PlannerAgent] Warning: Plan file {filename} is corrupted. Skipping.")
                    except Exception as e_load_single:
                        print(f"[PlannerAgent] Error loading plan file {filename}: {e_load_single}. Skipping.")
            
            if loaded_count > 0:
                 print(f"[PlannerAgent] Loaded {loaded_count} plans by scanning individual files.")
                 self._save_plans_index() 
            elif not os.path.exists(index_file_path): 
                 print(f"[PlannerAgent] No plan files found in {self.plans_dir} and no index file to load from.")
        except Exception as e: 
            print(f"[PlannerAgent] Critical error loading plans: {e}")
            import traceback; traceback.print_exc()

    def _save_plan_to_file(self, plan_id: str):
        """Saves a single plan to its JSON file AND updates the main plan index file."""
        if plan_id not in self.plans:
            print(f"[PlannerAgent] Error: Plan {plan_id} not found in memory to save.")
            return
        plan_data = self.plans[plan_id]
        individual_plan_file_path = os.path.join(self.plans_dir, f"{plan_id}.json")
        try:
            with open(individual_plan_file_path, 'w', encoding='utf-8') as f:
                json.dump(plan_data, f, indent=2)
            print(f"[PlannerAgent] Saved individual plan '{plan_id}' to {individual_plan_file_path}")
            # After saving the individual file, always update and save the main index
            self._save_plans_index()
        except Exception as e:
            print(f"[PlannerAgent] Error saving individual plan {plan_id} to file: {e}")

    def _save_plans_index(self):
        """Saves the self.plans dictionary (which is the index of all plans) to plans_index.json."""
        index_file_path = os.path.join(self.plans_dir, "plans_index.json")
        try:
            with open(index_file_path, 'w', encoding='utf-8') as f:
                json.dump(self.plans, f, indent=2) # Save the entire self.plans dictionary
            print(f"[PlannerAgent] Plans index (containing data for all plans) saved to {index_file_path}")
        except Exception as e:
            print(f"[PlannerAgent] Error saving plans index: {e}")

    def _get_agent_capabilities_description(self) -> str:
        """Generates a string describing available agents and their roles for the LLM."""
        descriptions = ["\nAVAILABLE AGENTS (for delegation - use the 'Agent Key' for 'assigned_agent' field):"]
        agent_roles = {
            "analyst": "Analyzes data, performs research, and provides insights. Can use tools like web search, file reading, data visualization.",
            "engineer": "Generates, debugs, and explains code (primarily Python). Can write scripts or functions.",
            "scribe": "Creates written content like articles, stories, reports, or summaries. Can draft and refine text.",
            "researcher": "Conducts in-depth, methodical research on complex topics, synthesizes information, and creates comprehensive research reports with citations.",
            "quartermaster": "Manages files (read, write, list, delete), project goals. Does not use an LLM directly for its core tasks.",
        }
        for agent_key, agent_instance in self.agents.items():
            agent_class_name = agent_instance.__class__.__name__ if hasattr(agent_instance, '__class__') else agent_key.capitalize() + "Agent"
            role = agent_roles.get(agent_key.lower(), "Performs specialized tasks.")
            descriptions.append(f"- Agent Key: \"{agent_key}\" (Class: {agent_class_name}): {role}")
        return "\n".join(descriptions)

    def _generate_planning_prompt(self, goal: str, context: str = "") -> str:
        """Generates the prompt for the LLM to create a plan, emphasizing data piping."""
        agent_capabilities = self._get_agent_capabilities_description()
        prompt = (
            f"You are a Planner Agent. Your role is to break down a complex USER GOAL into a sequence of manageable, concrete tasks. "
            f"Each task must be assigned to the most appropriate agent from AVAILABLE AGENTS using their 'Agent Key'.\n"
            f"{agent_capabilities}\n"
            f"\nUSER GOAL: \"{goal}\"\n"
            f"{context}\n"
            f"Based on this goal, provide a detailed plan as a single JSON object. Structure:\n"
            "{\n"
            "  \"id\": \"plan_YYYYMMDD_HHMMSS_summary\",\n"
            "  \"title\": \"Plan Title\",\n"
            "  \"description\": \"Plan summary.\",\n"
            "  \"tasks\": [\n"
            "    {{\n"
            "      \"index\": 1, \"description\": \"Task 1 description.\", \"assigned_agent\": \"agent_key\", ...\n"
            "      \"dependencies\": []\n"
            "    }},\n"
            "    {{\n"
            "      \"index\": 2, \"description\": \"Task 2. Use output from Task 1: [OUTPUT_OF_TASK_1]\", \"assigned_agent\": \"agent_key\", ...\n"
            "      \"dependencies\": [1]\n"
            "    }},\n"
            "    {{\n"
            "      \"index\": 3, \"description\": \"Quartermaster, save content: [OUTPUT_OF_TASK_2] to file 'specific_filename.txt'\", \"assigned_agent\": \"quartermaster\", ...\n"
            "      \"dependencies\": [2]\n"
            "    }}\n"
            "  ]\n"
            "}}\n"
            "CRITICAL INSTRUCTIONS FOR TASK DESCRIPTIONS & DATA FLOW:\n"
            "1.  **Data Piping**: If a task (e.g., Task 2) needs the output of a previous task (e.g., Task 1), its 'description' MUST include the placeholder `[OUTPUT_OF_TASK_X]` where X is the index of the prerequisite task. For example: 'Scribe, summarize the following research: [OUTPUT_OF_TASK_1]'.\n"
            "2.  **File Saving (Quartermaster)**: If a task is to save content to a file using the 'quartermaster', the 'description' MUST be very specific: 'Quartermaster, save content: [CONTENT_PLACEHOLDER_OR_ACTUAL_TEXT] to file \\'your_exact_filename.ext\\''. Replace [CONTENT_PLACEHOLDER_OR_ACTUAL_TEXT] with the actual text or an output placeholder like `[OUTPUT_OF_TASK_X]`.\n"
            "3.  **Dependencies**: 'dependencies' list must contain 'index' numbers of tasks that *must* be completed before the current task can start. Ensure logical flow and no circular dependencies.\n"
            "4.  **Agent Assignment**: 'assigned_agent' MUST be one of the 'Agent Key' values (e.g., \"scribe\", \"analyst\").\n"
            "5.  **Clarity**: Task descriptions must be clear and actionable for the assigned agent.\n"
            "Output ONLY the JSON plan. No other text, explanations, or markdown formatting."
        )
        return prompt

    def create_plan(self, goal_text: str, associated_goal_id: Optional[str] = None) -> Tuple[Optional[str], str]:
        """Creates a plan for a given goal using the LLM. Returns (plan_id, formatted_plan_string_or_error)."""
        print(f"[PlannerAgent] Creating plan for goal: '{goal_text[:100]}...' (Associated Goal ID: {associated_goal_id})")
        prompt = self._generate_planning_prompt(goal_text)
        plan_id_generated: Optional[str] = None # Explicitly define for clarity
        raw_plan_json_str = "" # Initialize for error logging

        try:
            print(f"[PlannerAgent] Sending planning prompt to LLM (model: {self.model_name}). Prompt length: {len(prompt)}")
            ollama_response = ollama.generate(model=self.model_name, prompt=prompt, format="json")
            raw_plan_json_str = ollama_response['response'].strip()
            print(f"[PlannerAgent] LLM Raw Output for plan (first 500 chars):\n{raw_plan_json_str[:500]}...")

            plan_data = json.loads(raw_plan_json_str) # This can raise JSONDecodeError
            
            plan_id_from_llm = plan_data.get("id")
            plan_title_from_llm = plan_data.get("title", goal_text[:50] + "...") # Default title
            plan_id_generated = plan_id_from_llm if plan_id_from_llm and plan_id_from_llm.startswith("plan_") else self._generate_plan_id(plan_title_from_llm)
            
            plan_data["id"] = plan_id_generated # Ensure ID is consistently set
            self.current_plan_id = plan_id_generated # Set as current plan being worked on

            plan_data.setdefault("title", plan_title_from_llm)
            plan_data.setdefault("description", plan_data.get("description", goal_text)) # Use goal_text if LLM omits description
            plan_data["created"] = datetime.now().isoformat()
            plan_data["updated"] = plan_data["created"] # Initially same as created
            plan_data["status"] = "pending" # Initial status
            plan_data["progress"] = 0.0
            plan_data["original_goal_text"] = goal_text # Store the original goal text
            plan_data["original_goal_id"] = associated_goal_id # Store associated memory goal ID
            plan_data.setdefault("plan_file", os.path.join("plans", f"{plan_id_generated}.json")) # Relative path for UI reference

            tasks = plan_data.get("tasks", [])
            if not isinstance(tasks, list): # Ensure tasks is a list, even if empty
                print(f"[PlannerAgent] Warning: 'tasks' field in LLM output for plan {plan_id_generated} is not a list. Defaulting to empty. Tasks: {tasks}")
                tasks = []
            
            valid_tasks = []
            for i, task_item_data in enumerate(tasks):
                if not isinstance(task_item_data, dict): # Ensure each task is a dictionary
                     print(f"[PlannerAgent] Warning: Task at index {i} in plan {plan_id_generated} is not a dictionary. Skipping. Task: {task_item_data}")
                     continue # Skip malformed task
                task_item_data["id"] = f"{plan_id_generated}_task_{i+1}" # Standardize task ID
                task_item_data["index"] = i + 1 # Ensure sequential index
                task_item_data.setdefault("status", "pending")
                task_item_data.setdefault("result", None)
                task_item_data.setdefault("started", None)
                task_item_data.setdefault("completed", None)
                # Validate and clean dependencies
                raw_deps = task_item_data.get("dependencies", [])
                if isinstance(raw_deps, list):
                    clean_deps = []
                    for dep in raw_deps:
                        if isinstance(dep, int):
                            clean_deps.append(dep)
                        elif isinstance(dep, str): # Try to convert string numbers to int
                            try: clean_deps.append(int(re.sub(r'\D', '', dep))) # Extract numbers from "Task 1" or "1"
                            except ValueError: print(f"[PlannerAgent] Warning: Could not parse dependency '{dep}' to int for task {i+1}")
                    task_item_data["dependencies"] = clean_deps
                else: # If dependencies is not a list, default to empty
                    print(f"[PlannerAgent] Warning: Task {i+1} dependencies not a list. Defaulting to empty. Deps: {raw_deps}")
                    task_item_data["dependencies"] = []
                valid_tasks.append(task_item_data)
            plan_data["tasks"] = valid_tasks # Use the validated list of tasks
            
            self.plans[plan_id_generated] = plan_data
            self._save_plan_to_file(plan_id_generated) # Save the individual plan file and update index

            # If associated with a memory goal, update its status
            if associated_goal_id and hasattr(self.memory, 'update_goal_status'): # Check if memory has this method
                self.memory.update_goal_status(associated_goal_id, "processing_planned", processing_plan_id=plan_id_generated)
            
            return plan_id_generated, self._format_plan_for_display(plan_id_generated)

        except json.JSONDecodeError as e:
            error_msg = f"Error: LLM plan output was not valid JSON. Error: {e}. Raw output was:\n{raw_plan_json_str}"
            print(f"[PlannerAgent] {error_msg}")
            if associated_goal_id and hasattr(self.memory, 'update_goal_status'):
                self.memory.update_goal_status(associated_goal_id, "autonomous_failed", result_summary=f"Plan creation failed (JSON error): {e}")
            if self.memory: self.memory.remember_agent_output("planner_status", f"Failed to create plan (JSON parse error for goal '{goal_text[:30]}...').")
            return None, error_msg
        except Exception as e:
            import traceback
            error_msg = f"Error creating plan: {type(e).__name__} - {e}\n{traceback.format_exc()}"
            print(f"[PlannerAgent] {error_msg}")
            if associated_goal_id and hasattr(self.memory, 'update_goal_status'):
                self.memory.update_goal_status(associated_goal_id, "autonomous_failed", result_summary=f"Plan creation failed: {e}")
            if self.memory: self.memory.remember_agent_output("planner_status", f"Failed to create plan for goal '{goal_text[:30]}...': {e}")
            return None, error_msg

    def _format_plan_for_display(self, plan_id: str) -> str:
        """Formats a plan for user display."""
        # This method is identical to the one in the user's provided file.
        if plan_id not in self.plans:
            return f"Plan with ID '{plan_id}' not found."
        plan = self.plans[plan_id]
        display = [
            f"Plan ID: {plan.get('id', 'N/A')}",
            f"Title: {plan.get('title', 'N/A')}",
            f"Description: {plan.get('description', 'N/A')}",
            f"Status: {plan.get('status', 'pending')}",
            f"Progress: {plan.get('progress', 0.0):.1f}%",
            f"Created: {plan.get('created', 'N/A')}",
            f"Last Updated: {plan.get('updated', 'N/A')}",
            f"Associated Memory Goal ID: {plan.get('original_goal_id', 'None')}" # Changed from 'original_goal_id'
        ]
        tasks = plan.get("tasks", [])
        if tasks:
            display.append("Tasks:")
            for task in tasks:
                dependencies_str = ", ".join(map(str, task.get("dependencies", []))) or "None"
                task_result_preview = ""
                if task.get("result"): # Check if result exists and is not None
                    result_str = str(task.get("result")) # Convert to string for safety
                    task_result_preview = f" Result: {result_str[:70]}..." if len(result_str) > 70 else f" Result: {result_str}"
                
                display.append(
                    f"  - Task {task.get('index', 'N/A')}: {task.get('description', 'N/A')}\n"
                    f"    Assigned to: {task.get('assigned_agent', 'N/A')}\n"
                    f"    Status: {task.get('status', 'pending')}{task_result_preview}\n" # Append preview here
                    f"    Dependencies: {dependencies_str}"
                )
        else:
            display.append("Tasks: No tasks defined in this plan yet.")
        return "\n".join(display)

    def update_task_status(self, plan_id: str, task_index: int, new_status: str, result: Optional[str] = None) -> bool:
        """Updates the status of a specific task in a plan and the overall plan."""
        # This method is identical to the one in the user's provided file.
        if plan_id not in self.plans:
            print(f"[PlannerAgent] Update task status failed: Plan {plan_id} not found.")
            return False
        
        plan = self.plans[plan_id]
        task_found_for_update = False
        
        for task in plan.get("tasks", []):
            if task.get("index") == task_index:
                task["status"] = new_status
                task["updated_time"] = datetime.now().isoformat() # Add an updated time for the task
                if new_status == "in_progress" and not task.get("started"):
                    task["started"] = datetime.now().isoformat()
                elif new_status == "completed":
                    task["completed"] = datetime.now().isoformat()
                    task["result"] = result if result is not None else task.get("result", "Completed without specific result.")
                elif new_status == "error":
                    task["result"] = result if result is not None else "Task failed with an unspecified error."
                elif new_status == "blocked": # Explicitly handle "blocked" status
                    task["result"] = result if result is not None else "Task blocked by Sentinel or other restriction."
                task_found_for_update = True
                break # Found and updated the task

        if task_found_for_update:
            plan["updated"] = datetime.now().isoformat() # Update plan's last updated time
            
            # Calculate overall plan progress
            completed_tasks_count = sum(1 for t in plan.get("tasks", []) if t.get("status") == "completed")
            total_tasks_count = len(plan.get("tasks", []))
            plan["progress"] = (completed_tasks_count / total_tasks_count) * 100 if total_tasks_count > 0 else 0.0
            
            # Update plan status if all tasks are completed or if a task error/block halts the plan
            current_plan_status = plan.get("status")
            all_tasks_are_terminal_completed = total_tasks_count > 0 and all(t.get("status") == "completed" for t in plan.get("tasks", []))

            if all_tasks_are_terminal_completed and current_plan_status != "completed":
                plan["status"] = "completed"
                print(f"[PlannerAgent] Plan {plan_id} ('{plan.get('title')}') fully marked as COMPLETED.")
                if self.message_bus_client:
                     self.message_bus_client.broadcast(f"Plan '{plan.get('title', plan_id)}' has completed!", context=f"plan_completion:{plan_id}")
                
                # Update associated memory goal if it exists
                associated_goal_id = plan.get("original_goal_id")
                if associated_goal_id and hasattr(self.memory, 'complete_goal'): # Check if memory has complete_goal
                    plan_summary_for_goal = f"Plan '{plan.get('title', plan_id)}' (ID: {plan_id}) completed successfully. All {total_tasks_count} tasks finished."
                    self.memory.complete_goal(associated_goal_id, result=plan_summary_for_goal)
                    print(f"[PlannerAgent] Marked associated Goal ID {associated_goal_id} as completed in memory.")
            # If a task errors or is blocked, and the plan isn't already completed/halted, mark it as halted.
            elif new_status in ["error", "blocked"] and current_plan_status not in ["error", "blocked", "completed", "halted_due_to_task_issue"]:
                plan["status"] = "halted_due_to_task_issue" # More specific status
                print(f"[PlannerAgent] Plan {plan_id} status changed to '{plan['status']}' due to task {task_index} status '{new_status}'.")
                associated_goal_id = plan.get("original_goal_id")
                if associated_goal_id and hasattr(self.memory, 'update_goal_status'):
                    self.memory.update_goal_status(associated_goal_id, "autonomous_failed", result_summary=f"Plan '{plan.get('title')}' halted. Task {task_index} {new_status}: {result}")


            self._save_plan_to_file(plan_id) # Save changes to the individual plan file and the index
            print(f"[PlannerAgent] Task {task_index} in plan {plan_id} status updated to {new_status}. Progress: {plan['progress']:.1f}%")
            if self.memory: self.memory.remember_agent_output("planner_status", f"Plan {plan_id}: Task {task_index} is now {new_status}. Overall progress {plan['progress']:.1f}%.")
            return True
        else: # Task not found
            print(f"[PlannerAgent] Update task status failed: Task {task_index} not found in plan {plan_id}.")
            return False

    def get_next_task(self, plan_id: str) -> Tuple[Optional[Dict[str, Any]], str]:
        """Gets the next pending task whose dependencies are met."""
        # This method is identical to the one in the user's provided file.
        if plan_id not in self.plans:
            return None, f"Plan {plan_id} not found."
        plan = self.plans[plan_id]

        if plan.get("status") == "completed":
            return None, f"Plan {plan_id} is already completed."
        if plan.get("status") == "halted_due_to_task_issue": # Check for halted status
            return None, f"Plan {plan_id} is halted due to a previous task issue. Manual review or new plan may be needed."


        tasks_by_index_lookup = {task["index"]: task for task in plan.get("tasks", [])}

        for task_to_check in plan.get("tasks", []): # Iterate in defined order
            if task_to_check.get("status") == "pending":
                dependencies_are_met = True
                for dep_idx_val in task_to_check.get("dependencies", []): # dep_idx_val is the index from dependencies list
                    dep_idx_to_find = -1 # Default to invalid index
                    if isinstance(dep_idx_val, int):
                        dep_idx_to_find = dep_idx_val
                    elif isinstance(dep_idx_val, str): # Handles "Task 1" or just "1"
                        try:
                            # Extract numbers from string like "Task 1", "1", "task1"
                            dep_idx_to_find = int(re.sub(r'\D', '', dep_idx_val)) 
                        except ValueError:
                            print(f"[PlannerAgent] Warning: Could not parse dependency '{dep_idx_val}' to an integer index for task {task_to_check.get('index')}.")
                            pass # Could not parse as an integer
                    
                    dependent_task = tasks_by_index_lookup.get(dep_idx_to_find)
                    
                    if not dependent_task or dependent_task.get("status") != "completed":
                        dependencies_are_met = False
                        break # A dependency is not met
                if dependencies_are_met:
                    return task_to_check, f"Next task {task_to_check['index']} is ready."
        
        # If loop finishes, no pending task with met dependencies was found.
        # Check if the plan should now be considered complete.
        if all(t.get("status") == "completed" for t in plan.get("tasks", [])): # Only "completed" signifies overall plan completion
            if plan.get("status") != "completed": # If status wasn't updated yet by update_task_status
                plan["status"] = "completed"
                plan["progress"] = 100.0
                plan["updated"] = datetime.now().isoformat()
                self._save_plan_to_file(plan_id)
                print(f"[PlannerAgent] Plan {plan_id} auto-marked as completed (all tasks done).")
                if self.message_bus_client:
                    self.message_bus_client.broadcast(f"Plan '{plan.get('title', plan_id)}' auto-completed!", context=f"plan_completion:{plan_id}")
                # Update associated memory goal
                associated_goal_id = plan.get("original_goal_id")
                if associated_goal_id and hasattr(self.memory, 'complete_goal'):
                    self.memory.complete_goal(associated_goal_id, result=f"Plan '{plan.get('title')}' auto-completed successfully.")
            return None, f"Plan {plan_id} has no more tasks and is complete."
        else: # Some tasks are not completed (e.g. error, blocked, in_progress) but no pending task is ready
             return None, f"Plan {plan_id}: No pending tasks with met dependencies are currently available. Plan status: {plan.get('status')}."

    def delegate_task(self, plan_id: str, task_index: int) -> str:
        """Delegates a task to its assigned agent. Updates task status before and after.
           Injects outputs from dependent tasks into the current task's description if placeholders are found.
        """
        if plan_id not in self.plans:
            return f"Error: Plan {plan_id} not found."
        plan = self.plans[plan_id]
        
        task_to_delegate = next((t for t in plan.get("tasks", []) if t.get("index") == task_index), None)
        
        if not task_to_delegate:
            return f"Error: Task {task_index} not found in plan {plan_id}."
        if task_to_delegate.get("status") not in ["pending", "retry"]: # Allow retry status for re-delegation
            return f"Error: Task {task_index} (status: {task_to_delegate.get('status')}) is not pending or ready for retry."

        assigned_agent_key = task_to_delegate.get("assigned_agent", "").lower() # LLM should use lowercase key
        task_description_for_agent = task_to_delegate.get("description", "") # Get the raw description

        # --- **NEW: Substitute placeholders with outputs from dependent tasks** ---
        substituted_task_description = task_description_for_agent
        # Find all placeholders like [OUTPUT_OF_TASK_X]
        placeholders = re.findall(r'\[OUTPUT_OF_TASK_(\d+)\]', substituted_task_description, re.IGNORECASE)
        
        if placeholders:
            print(f"[PlannerAgent] Found placeholders in task {task_index} description: {placeholders}")
            all_deps_resolved = True
            for dep_task_idx_str in placeholders:
                try:
                    dep_task_idx = int(dep_task_idx_str)
                    # Find the dependent task in the plan's task list
                    dependent_task_data = next((t for t in plan.get("tasks", []) if t.get("index") == dep_task_idx), None)
                    
                    if dependent_task_data and dependent_task_data.get("status") == "completed":
                        dep_result = str(dependent_task_data.get("result", "")) # Ensure it's a string
                        # Replace the placeholder (case-insensitive)
                        placeholder_regex = re.compile(rf'\[OUTPUT_OF_TASK_{dep_task_idx}\]', re.IGNORECASE)
                        substituted_task_description = placeholder_regex.sub(dep_result, substituted_task_description)
                        print(f"[PlannerAgent] Substituted [OUTPUT_OF_TASK_{dep_task_idx}] with result: '{dep_result[:50]}...'")
                    else:
                        err_msg_placeholder = f"Could not substitute placeholder [OUTPUT_OF_TASK_{dep_task_idx}] for task {task_index}: Dependent task {dep_task_idx} not found or not completed."
                        print(f"[PlannerAgent] {err_msg_placeholder}")
                        # This is a critical issue if a placeholder cannot be filled.
                        # The task should probably be marked as error or blocked.
                        self.update_task_status(plan_id, task_index, "error", result=err_msg_placeholder)
                        return f"Error: {err_msg_placeholder}"
                except ValueError:
                    err_msg_val = f"Invalid task index in placeholder: [OUTPUT_OF_TASK_{dep_task_idx_str}] for task {task_index}."
                    print(f"[PlannerAgent] {err_msg_val}")
                    self.update_task_status(plan_id, task_index, "error", result=err_msg_val)
                    return f"Error: {err_msg_val}"
            task_description_for_agent = substituted_task_description # Use the description with substitutions
        # --- **END OF NEW PLACEHOLDER SUBSTITUTION LOGIC** ---

        target_agent_instance = self.agents.get(assigned_agent_key) 

        if not target_agent_instance: # Fallback to check by class name if key not found
            original_assigned_str = task_to_delegate.get("assigned_agent", "") # Get original string from plan
            for key_iter, agent_inst_iter in self.agents.items():
                if hasattr(agent_inst_iter, '__class__') and agent_inst_iter.__class__.__name__.lower() == original_assigned_str.lower():
                    target_agent_instance = agent_inst_iter
                    assigned_agent_key = key_iter # Correct to the actual key for logging
                    print(f"[PlannerAgent] Warning: Agent for task {task_index} matched by class name '{original_assigned_str}' (should be key '{assigned_agent_key}'). LLM should use agent keys from prompt.")
                    break
        
        if not target_agent_instance:
            error_message = f"Assigned agent key/name '{task_to_delegate.get('assigned_agent')}' for task {task_index} not found in configured agents."
            self.update_task_status(plan_id, task_index, "error", result=error_message)
            return f"Error: {error_message}"

        agent_class_name_for_log = target_agent_instance.__class__.__name__ # For logging
        print(f"[PlannerAgent] Delegating Task {task_index} ('{task_description_for_agent[:50]}...') to agent key '{assigned_agent_key}' (Instance: {agent_class_name_for_log})")
        self.update_task_status(plan_id, task_index, "in_progress") # Mark as in_progress BEFORE execution
        if self.memory: self.memory.remember_agent_output("planner_status", f"Plan {plan_id}: Delegating task {task_index} ('{task_description_for_agent[:30]}...') to {assigned_agent_key}.")

        task_execution_result: Optional[str] = None # Initialize
        try:
            # Sentinel approval for delegation action itself
            self.sentinel.approve_action(
                agent_name="PlannerAgent",
                action_type=f"delegate_to:{assigned_agent_key}", # Use the resolved agent key
                detail=f"Task for plan {plan_id}: {task_description_for_agent}" # Send the (potentially substituted) description
            )
            
            # Execute the task via the agent's run method
            # If the target agent is also a PlannerAgent (for sub-plans), pass the associated goal ID
            if isinstance(target_agent_instance, PlannerAgent):
                # The task_description_for_agent here IS the goal for the sub-planner
                # We might need to extract the actual goal if it's wrapped, or ensure the sub-planner's run can handle it.
                # For now, assume task_description_for_agent is the goal for the sub-plan.
                # The sub-planner will create its own plan ID.
                task_execution_result = target_agent_instance.run(task_description_for_agent, associated_goal_id_for_new_plan=plan.get("original_goal_id"))
            else:
                task_execution_result = target_agent_instance.run(task_description_for_agent)

            # Store the result and mark as completed
            self.update_task_status(plan_id, task_index, "completed", result=str(task_execution_result))
            if self.memory: self.memory.remember_agent_output("planner_status", f"Plan {plan_id}: Task {task_index} completed by {assigned_agent_key}.")
            return f"Task {task_index} delegated to {assigned_agent_key} and completed. Result snippet: {str(task_execution_result)[:100]}..."

        except SentinelException as se:
            error_details = f"Delegation for task {task_index} blocked by Sentinel: {str(se)}"
            print(f"[PlannerAgent] {error_details}")
            self.update_task_status(plan_id, task_index, "blocked", result=error_details) # Use "blocked" status
            if self.memory: self.memory.remember_agent_output("planner_status", f"Plan {plan_id}: Task {task_index} delegation blocked by Sentinel.")
            return error_details
        except Exception as e_delegate: # Catch any other exception during agent.run()
            import traceback
            error_details = f"Error during task {task_index} delegation to {assigned_agent_key}: {str(e_delegate)}\n{traceback.format_exc()}"
            print(f"[PlannerAgent] {error_details}")
            self.update_task_status(plan_id, task_index, "error", result=f"Agent {assigned_agent_key} execution error: {str(e_delegate)}")
            if self.memory: self.memory.remember_agent_output("planner_status", f"Plan {plan_id}: Task {task_index} failed during execution by {assigned_agent_key}.")
            return error_details

    def assess_plan(self, plan_id: str) -> str:
        """Assesses the current status and next steps of a plan."""
        # This method is identical to the one in the user's provided file.
        if plan_id not in self.plans:
            return f"Plan {plan_id} not found."
        
        plan_display = self._format_plan_for_display(plan_id)
        next_task_info, message_from_get_next = self.get_next_task(plan_id)

        assessment = [
            f"Assessment for Plan ID: {plan_id} (Title: {self.plans[plan_id].get('title', 'N/A')})",
            f"Current Status: {self.plans[plan_id].get('status', 'N/A')}",
            f"Overall Progress: {self.plans[plan_id].get('progress', 0.0):.1f}%",
            f"Associated Memory Goal ID: {self.plans[plan_id].get('original_goal_id', 'None')}",
            "\n--- Plan Details ---",
            plan_display,
            "\n--- Next Action ---"
        ]
        if next_task_info:
            assessment.append(f"Next task to execute is Task {next_task_info['index']}: '{next_task_info['description']}' assigned to {next_task_info['assigned_agent']}.")
            assessment.append("If plan is in auto-execution mode, this task will be attempted next. Otherwise, use 'Planner, execute task X in plan Y' for manual step.")
        else:
            assessment.append(message_from_get_next)
        
        return "\n".join(assessment)

    def list_plans(self) -> str:
        """Lists all available plans with their status and associated goal ID."""
        # This method is identical to the one in the user's provided file.
        if not self.plans:
            return "No plans found."
        
        plan_summaries = ["Available Plans:"]
        # Sort plans by creation time (descending, newest first) if 'created' field exists
        sorted_plan_ids = sorted(self.plans.keys(), key=lambda pid: self.plans[pid].get("created", ""), reverse=True)

        for plan_id in sorted_plan_ids:
            plan_data = self.plans[plan_id]
            status_emoji = "✅" if plan_data.get('status') == 'completed' else ("⏳" if plan_data.get('status') in ['active', 'processing', 'in_progress'] else "📋") # Added more active statuses
            goal_id_str = f"(Goal: {plan_data.get('original_goal_id', 'N/A')[:8]}...)" if plan_data.get('original_goal_id') else "" # Show more of goal ID if available
            plan_summaries.append(
                f"  {status_emoji} ID: {plan_id} - Title: {plan_data.get('title', 'Untitled Plan')} "
                f"(Status: {plan_data.get('status', 'pending')}, Progress: {plan_data.get('progress', 0.0):.1f}%) {goal_id_str}"
            )
        return "\n".join(plan_summaries)

    def set_current_plan(self, plan_id: str) -> str:
        """Sets the currently active plan for context."""
        # This method is identical to the one in the user's provided file.
        if plan_id in self.plans:
            self.current_plan_id = plan_id
            return f"Current plan set to {plan_id} ('{self.plans[plan_id].get('title', '')}')."
        return f"Error: Plan ID {plan_id} not found."

    def _execute_plan_loop(self, plan_id: str) -> str:
        """Internal loop to automatically execute tasks in a plan. Updates associated goal status."""
        # This method is largely identical to the one in the user's provided file, with minor logging enhancements.
        if plan_id not in self.plans:
            return f"Error: Plan {plan_id} not found for execution loop."

        plan = self.plans[plan_id]
        plan_title = plan.get('title', 'Untitled Plan')
        associated_goal_id = plan.get("original_goal_id") # Get the associated goal ID
        
        execution_log = [f"Starting automated execution of plan: {plan_title} ({plan_id}) for Goal ID: {associated_goal_id or 'N/A'}"]
        print(f"[PlannerAgent] Starting automated execution of plan: {plan_id} - '{plan_title}' (Goal ID: {associated_goal_id})")
        if self.memory: self.memory.remember_agent_output("planner_status", f"Plan {plan_id} ('{plan_title}'): Execution started for Goal ID {associated_goal_id}.")
        
        # Ensure plan status is active if it was pending
        if plan.get("status") == "pending":
            plan["status"] = "active" # Mark as active when loop starts
            self._save_plan_to_file(plan_id)
            if associated_goal_id and hasattr(self.memory, 'update_goal_status'): # Update memory goal status
                self.memory.update_goal_status(associated_goal_id, "processing_automated", processing_plan_id=plan_id)


        max_steps_in_loop = len(plan.get("tasks", [])) * 2 + 10 # Allow for retries or additional small steps, increased buffer
        current_step_count = 0
        plan_final_outcome_message = "" # To store the final message of the loop

        while current_step_count < max_steps_in_loop:
            current_step_count += 1
            plan = self.plans[plan_id] # Refresh plan data from memory in each iteration

            if plan.get("status") == "completed":
                plan_final_outcome_message = f"Plan {plan_id} '{plan_title}' is marked as completed."
                break # Exit loop if plan is completed
            if plan.get("status") == "halted_due_to_task_issue": # Check for halted status
                plan_final_outcome_message = f"Plan {plan_id} '{plan_title}' execution halted due to a task issue."
                break # Exit loop if plan is halted

            task_to_execute, message_from_get_next = self.get_next_task(plan_id)

            if not task_to_execute: # No task is ready or plan is finished.
                plan_final_outcome_message = f"Plan {plan_id} '{plan_title}': No further executable tasks. {message_from_get_next}"
                # If get_next_task determined completion, its message will reflect that.
                if self.plans[plan_id].get("status") == "completed": # Double check if get_next_task completed it
                     plan_final_outcome_message = f"Plan {plan_id} '{plan_title}' successfully completed all tasks."
                break # Exit loop

            task_log_desc = f"Task {task_to_execute['index']} ('{task_to_execute['description'][:40]}...')"
            log_msg = f"[PlannerAgent] Plan {plan_id}: Attempting to auto-execute {task_log_desc} with agent {task_to_execute['assigned_agent']}"
            print(log_msg); execution_log.append(log_msg)
            if self.memory: self.memory.remember_agent_output("planner_status", f"Plan {plan_id}: Executing {task_log_desc} via {task_to_execute['assigned_agent']}.")

            # delegate_task handles updating task status to in_progress, completed, error, or blocked
            delegation_outcome_message = self.delegate_task(plan_id, task_to_execute["index"])
            execution_log.append(f"Task {task_to_execute['index']} delegation attempt: {delegation_outcome_message[:150]}...") # Log more of the outcome
            print(f"[PlannerAgent] Plan {plan_id}: Task {task_to_execute['index']} delegation outcome: {delegation_outcome_message[:100]}...")
            
            # Re-fetch task to check its status after delegation, as delegate_task updates it
            updated_task_instance = next((t for t in self.plans[plan_id].get("tasks", []) if t.get("index") == task_to_execute["index"]), None)
            
            if updated_task_instance and updated_task_instance.get("status") in ["error", "blocked"]:
                plan_final_outcome_message = (
                    f"Plan {plan_id} '{plan_title}' execution halted at task {task_to_execute['index']} "
                    f"(Status: {updated_task_instance['status']}). Result: {updated_task_instance.get('result', '')}"
                )
                break # Exit loop on task error/block

            time.sleep(0.1) # Small delay for system to breathe / logs to catch up

            # Explicitly check plan status again after a task attempt, as update_task_status (called by delegate_task) might have marked it completed
            if self.plans[plan_id].get("status") == "completed":
                plan_final_outcome_message = f"Plan {plan_id} '{plan_title}' marked as fully completed after processing task {task_to_execute['index']}."
                break # Exit loop
        
        # After loop finishes (either by break or max_steps)
        if not plan_final_outcome_message: # If loop exited due to max_steps
            plan_final_outcome_message = f"Plan {plan_id} '{plan_title}' execution loop reached max steps ({max_steps_in_loop}). Halting. Current plan status: {plan.get('status', 'unknown')}."
            # If timed out and not completed, and associated with a goal, mark goal as failed
            if associated_goal_id and plan.get("status") != "completed" and hasattr(self.memory, 'update_goal_status'):
                self.memory.update_goal_status(associated_goal_id, "autonomous_failed", result_summary=plan_final_outcome_message)
        
        print(f"[PlannerAgent] {plan_final_outcome_message}")
        execution_log.append(plan_final_outcome_message)
        if self.memory: self.memory.remember_agent_output("planner_status", f"Plan {plan_id}: Execution loop finished. {plan_final_outcome_message}")

        # Final check on associated goal status if plan didn't complete successfully
        if plan.get("status") != "completed" and associated_goal_id and hasattr(self.memory, 'update_goal_status'):
            # Check memory goal status to avoid overwriting if it was already set to failed by update_task_status
            goal_in_memory = self.memory.get_goal_by_id(associated_goal_id) if hasattr(self.memory, 'get_goal_by_id') else None
            if goal_in_memory and goal_in_memory.get('status') not in ['completed', 'autonomous_failed', 'failed']: # Avoid double-marking
                 self.memory.update_goal_status(associated_goal_id, "autonomous_failed", result_summary=f"Plan '{plan.get('title')}' did not complete. Final plan status: {plan.get('status')}")
        
        return "\n".join(execution_log)

    def run(self, command: str, associated_goal_id_for_new_plan: Optional[str] = None) -> str:
        """
        Handles commands for the PlannerAgent.
        The 'command' parameter is the task description for the planner.
        'associated_goal_id_for_new_plan' is used if this run is creating a plan for a specific goal from memory.
        """
        # --- Robust Whitespace Normalization ---
        # First, replace non-breaking spaces, then normalize all other whitespace sequences to a single space.
        command_cleaned_nbsp = command.replace('\xa0', ' ') 
        command_normalized = ' '.join(command_cleaned_nbsp.split()) 
        cmd_lower = command_normalized.lower() # Use this for all matching

        print(f"[PlannerAgent] === PlannerAgent.run START ===")
        print(f"[PlannerAgent] Original command (passed to run): '{command}'") # Log original for comparison
        print(f"[PlannerAgent] Normalized cmd_lower for parsing: '{cmd_lower}'") # Log normalized
        print(f"[PlannerAgent] Associated Goal ID for new plan (if any): {associated_goal_id_for_new_plan}")

        goal_text_for_plan = "" # Initialize to store the extracted goal for plan creation

        try:
            # --- Command Parsing (using more flexible regex and startswith for clarity) ---
            
            # Pattern for "create and execute [a] plan [for] ..." or "create [a] plan and run [for] ..."
            # Using startswith for these primary commands is more direct if "Planner," prefix is stripped by router.
            if cmd_lower.startswith("create and execute plan") or cmd_lower.startswith("create plan and run"):
                # Determine the base phrase to strip
                base_phrase = ""
                if cmd_lower.startswith("create and execute plan"):
                    base_phrase = "create and execute plan"
                elif cmd_lower.startswith("create plan and run"):
                    base_phrase = "create plan and run"
                
                # Extract goal text: everything after "base_phrase", then strip "for/to/about" if present
                temp_goal_text = command_normalized[len(base_phrase):].strip()
                # Check if it starts with "for", "to", or "about" and strip that too
                match_conj = re.match(r"^(?:for|to|about)\s+(.+)", temp_goal_text, re.IGNORECASE)
                if match_conj:
                    goal_text_for_plan = match_conj.group(1).strip()
                else:
                    goal_text_for_plan = temp_goal_text # No "for/to/about", so use the rest
                
                print(f"[PlannerAgent_DEBUG] Matched 'create and execute/run'. Goal text: '{goal_text_for_plan}'")
                if not goal_text_for_plan: # Check if anything remains after stripping
                    if self.memory: self.memory.remember_agent_output("planner_status", "Failed to start plan: Goal description missing for 'create and execute'.")
                    return "Error: Please provide a clear goal description after '... plan [for] ...'."
                
                if self.memory: self.memory.remember_agent_output("planner_status", f"Request: Create & Execute Plan for: {goal_text_for_plan[:50]}...")
                plan_id, plan_creation_output = self.create_plan(goal_text_for_plan, associated_goal_id=associated_goal_id_for_new_plan)

                if not plan_id or "Error" in plan_creation_output: # Check if plan_id is None or error in output
                    if self.memory: self.memory.remember_agent_output("planner_status", f"Failed to create plan: {plan_creation_output}")
                    return f"Failed to create plan for execution: {plan_creation_output}"
                
                if self.memory: self.memory.remember_agent_output("planner_status", f"Plan {plan_id} created for '{goal_text_for_plan[:50]}...'. Starting auto-execution.")
                plan_execution_log = self._execute_plan_loop(plan_id)
                final_plan_status = self.plans.get(plan_id, {}).get("status", "unknown") # Safe get
                if self.memory: self.memory.remember_agent_output("planner_status", f"Plan {plan_id} auto-execution finished. Status: {final_plan_status}.")
                return f"Plan Creation Summary:\n{plan_creation_output}\n\n--- Automated Plan Execution Log ---\n{plan_execution_log}"

            # Pattern for "start main plan for ..." or "execute main plan for ..."
            elif cmd_lower.startswith("start main plan for") or cmd_lower.startswith("execute main plan for"):
                base_phrase = "start main plan for" if cmd_lower.startswith("start main plan for") else "execute main plan for"
                goal_text_for_plan = command_normalized[len(base_phrase):].strip()
                print(f"[PlannerAgent_DEBUG] Matched 'start/execute main plan for'. Goal text: '{goal_text_for_plan}'")
                if not goal_text_for_plan:
                    return "Error: Please provide a goal after '... main plan for ...'."
                if self.memory: self.memory.remember_agent_output("planner_status", f"Request: Start Main Plan for: {goal_text_for_plan[:50]}...")
                plan_id, plan_creation_output = self.create_plan(goal_text_for_plan, associated_goal_id=associated_goal_id_for_new_plan)
                if not plan_id or "Error" in plan_creation_output:
                    return f"Failed to create plan for execution: {plan_creation_output}"
                plan_execution_log = self._execute_plan_loop(plan_id)
                return f"Plan Creation Summary:\n{plan_creation_output}\n\n--- Automated Plan Execution Log ---\n{plan_execution_log}"

            # Pattern for "execute plan ..." or "run plan ..." or "start executing plan ..."
            elif cmd_lower.startswith("execute plan") or cmd_lower.startswith("run plan") or cmd_lower.startswith("start executing plan"):
                execute_existing_match = re.match(r"^(execute plan|run plan|start executing plan)\s+([\w\d_\-\.:]+)", cmd_lower, re.IGNORECASE)
                if execute_existing_match:
                    plan_id_to_exec = execute_existing_match.group(2).strip()
                    print(f"[PlannerAgent_DEBUG] Matched 'execute existing plan'. Plan ID: '{plan_id_to_exec}'")
                    if plan_id_to_exec not in self.plans: return f"Error: Plan '{plan_id_to_exec}' not found."
                    self.current_plan_id = plan_id_to_exec # Set current plan when executing existing
                    if self.memory: self.memory.remember_agent_output("planner_status", f"Starting execution of existing plan: {plan_id_to_exec}.")
                    exec_log = self._execute_plan_loop(plan_id_to_exec)
                    final_status_exec = self.plans.get(plan_id_to_exec, {}).get("status", "unknown")
                    if self.memory: self.memory.remember_agent_output("planner_status", f"Exec loop for plan {plan_id_to_exec} finished. Status: {final_status_exec}.")
                    return f"--- Execution Log for Plan {plan_id_to_exec} ---\n{exec_log}"
                else:
                    return "Error: Invalid format for executing an existing plan. Use 'execute plan [plan_id]'."
            
            # Pattern for "create [a] plan [for] ..." or "make [a] plan [for] ..." or "new [a] plan [for] ..."
            elif cmd_lower.startswith("create plan") or cmd_lower.startswith("make plan") or cmd_lower.startswith("new plan"):
                base_phrase = ""
                if cmd_lower.startswith("create plan"): base_phrase = "create plan"
                elif cmd_lower.startswith("make plan"): base_phrase = "make plan"
                elif cmd_lower.startswith("new plan"): base_phrase = "new plan"
                
                temp_goal_text = command_normalized[len(base_phrase):].strip()
                match_conj_create = re.match(r"^(?:for|to|about)\s+(.+)", temp_goal_text, re.IGNORECASE)
                if match_conj_create:
                    goal_text_for_plan = match_conj_create.group(1).strip()
                else:
                    goal_text_for_plan = temp_goal_text # No "for/to/about", so use the rest
                
                print(f"[PlannerAgent_DEBUG] Matched 'create only plan'. Goal text: '{goal_text_for_plan}'")
                if not goal_text_for_plan:
                    if self.memory: self.memory.remember_agent_output("planner_status", "Failed to create plan: Goal description missing.")
                    return "Error: Please provide a goal for the plan (e.g., 'Planner, create plan for organizing my research')."
                
                if self.memory: self.memory.remember_agent_output("planner_status", f"Creating plan (manual) for: {goal_text_for_plan[:50]}...")
                plan_id_manual, plan_disp_manual = self.create_plan(goal_text_for_plan, associated_goal_id=associated_goal_id_for_new_plan)
                if plan_id_manual: # create_plan returns (plan_id, display_string)
                    if self.memory: self.memory.remember_agent_output("planner_status", f"Plan created (manual). ID: {plan_id_manual}")
                    return plan_disp_manual # Return the display string
                else: # plan_id_manual is None, meaning error
                    if self.memory: self.memory.remember_agent_output("planner_status", f"Failed to create plan (manual): {plan_disp_manual}")
                    return plan_disp_manual # Return the error message from create_plan
            
            # --- Other commands (list, show, update, etc.) ---
            elif "list plans" in cmd_lower or "show plans" in cmd_lower:
                print(f"[PlannerAgent_DEBUG] Matched 'list plans'.")
                return self.list_plans()
            
            elif cmd_lower.startswith("show plan") or cmd_lower.startswith("plan details") or cmd_lower.startswith("view plan"):
                show_plan_match = re.match(r"^(show plan|plan details|view plan)\s+([\w\d_\-\.:]+)", cmd_lower, re.IGNORECASE)
                pid_show = None
                if show_plan_match:
                    pid_show = show_plan_match.group(2).strip()
                    print(f"[PlannerAgent_DEBUG] Matched 'show plan [id]'. Plan ID: '{pid_show}'")
                elif self.current_plan_id: # If no ID provided, show current plan if set
                    pid_show = self.current_plan_id
                    print(f"[PlannerAgent_DEBUG] Matched 'show plan' (no ID), using current_plan_id: '{pid_show}'")
                
                if pid_show and pid_show in self.plans: return self._format_plan_for_display(pid_show)
                elif pid_show: return f"Error: Plan ID '{pid_show}' not found."
                else: return "Error: Specify plan ID or set a current plan to show details."

            elif cmd_lower.startswith("update task"):
                update_task_match = re.match(r'update task\s+(\d+)\s+in plan\s+([\w\d_\-\.:]+)\s+to\s+(\w+)(?:\s+with result:\s*(.+))?', cmd_lower, re.IGNORECASE)
                if update_task_match:
                    task_idx, plan_id_upd, status_upd, res_text_upd = update_task_match.groups()
                    print(f"[PlannerAgent_DEBUG] Matched 'update task'. Task: {task_idx}, Plan: {plan_id_upd}, Status: {status_upd}")
                    if self.update_task_status(plan_id_upd.strip(), int(task_idx), status_upd.lower().strip(), result=res_text_upd.strip() if res_text_upd else None):
                        return f"Task {task_idx} in plan {plan_id_upd} updated to {status_upd.lower()}."
                    return f"Error: Could not update task {task_idx} in plan {plan_id_upd}."
                else:
                    return "Error: Invalid 'update task' format. Use: 'Planner, update task [index] in plan [id] to [status] [with result: text]'"

            elif cmd_lower.startswith("next task") or cmd_lower.startswith("get next task"): # Simplified "next task"
                next_task_match = re.match(r"^(next task|get next task)(?:\s+in plan\s+([\w\d_\-\.:]+))?", cmd_lower, re.IGNORECASE)
                pid_next = self.current_plan_id # Default to current plan
                if next_task_match and next_task_match.group(2): # If plan ID is specified in command
                    pid_next = next_task_match.group(2).strip()
                
                print(f"[PlannerAgent_DEBUG] Matched 'next task'. Plan ID to check: '{pid_next}'")
                if not pid_next: return "Error: Specify plan ID (e.g., 'next task in plan X') or set a current plan."
                if pid_next not in self.plans: return f"Error: Plan ID '{pid_next}' not found for 'next task'."
                
                task_next, message_next = self.get_next_task(pid_next)
                if task_next: return f"Next task for {pid_next} (Task {task_next['index']}): '{task_next['description']}' (Agent: {task_next['assigned_agent']}, Status: {task_next['status']})"
                return f"For plan {pid_next}: {message_next}"

            elif cmd_lower.startswith("delegate task") or cmd_lower.startswith("execute task"): # Manual execution of a single task from plan
                delegate_task_match = re.match(r"^(delegate task|execute task)\s+(\d+)\s*(?:in plan\s+([\w\d_\-\.:]+))?", cmd_lower, re.IGNORECASE)
                if delegate_task_match:
                    task_idx_del = int(delegate_task_match.group(2)) # Corrected group index for task number
                    pid_del_match = delegate_task_match.group(3) # Group for optional plan ID
                    pid_del = pid_del_match.strip() if pid_del_match else self.current_plan_id
                    
                    print(f"[PlannerAgent_DEBUG] Matched 'delegate/execute task'. Task: {task_idx_del}, Plan ID: '{pid_del}'")
                    if not pid_del: return "Error: No plan specified for task delegation and no current plan set."
                    if pid_del not in self.plans: return f"Error: Plan ID '{pid_del}' not found for task delegation."
                    return self.delegate_task(pid_del, task_idx_del)
                else:
                    return "Error: Invalid format for task delegation. Use 'Planner, delegate task [index] [in plan id_xyz]'"

            elif cmd_lower.startswith("assess plan"):
                assess_plan_match = re.match(r"^assess plan\s+([\w\d_\-\.:]+)", cmd_lower, re.IGNORECASE)
                pid_assess = None
                if assess_plan_match:
                    pid_assess = assess_plan_match.group(1).strip()
                elif self.current_plan_id:
                    pid_assess = self.current_plan_id
                
                print(f"[PlannerAgent_DEBUG] Matched 'assess plan'. Plan ID: '{pid_assess}'")
                if not pid_assess: return "Error: Specify plan ID or set a current plan to assess."
                if pid_assess not in self.plans: return f"Error: Plan ID '{pid_assess}' not found for assessment."
                return self.assess_plan(pid_assess)
            
            elif cmd_lower.startswith("set current plan"):
                set_current_match = re.match(r"^set current plan\s+([\w\d_\-\.:]+)", cmd_lower, re.IGNORECASE)
                if set_current_match:
                    plan_id_to_set = set_current_match.group(1).strip()
                    print(f"[PlannerAgent_DEBUG] Matched 'set current plan'. Plan ID: '{plan_id_to_set}'")
                    return self.set_current_plan(plan_id_to_set)
                else:
                    return "Error: Specify plan ID to set as current. Format: 'set current plan [plan_id]'"
            
            else: # Fallback if no other specific command structure matched
                print(f"[PlannerAgent_DEBUG] Command '{cmd_lower}' did not match any specific known Planner command structures. Returning generic help.")
                unrec_msg = ("PlannerAgent: Command not recognized. Try 'create and execute plan for [goal]', "
                             "'execute plan [plan_id]', 'create plan for [goal]', 'list plans', 'show plan [plan_id]', etc.")
                if self.memory: self.memory.remember_agent_output("planner_status", "Unrecognized command received by Planner.")
                return unrec_msg

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            critical_err_msg = f"PlannerAgent CRITICAL ERROR in run method: {type(e).__name__} - {str(e)}"
            print(f"[PlannerAgent] {critical_err_msg}\n{error_details}")
            if self.memory: self.memory.remember_agent_output("planner_status", f"Critical error in Planner: {str(e)}")
            # If this run was associated with a memory goal, mark that goal as failed due to planner error
            if associated_goal_id_for_new_plan and hasattr(self.memory, 'update_goal_status'):
                self.memory.update_goal_status(associated_goal_id_for_new_plan, "autonomous_failed", result_summary=f"Planner critical error: {e}")
            return critical_err_msg
