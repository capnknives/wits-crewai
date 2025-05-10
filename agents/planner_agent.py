# agents/planner_agent.py
import ollama
import re
import json
from datetime import datetime
import os
from typing import List, Dict, Any, Optional, Tuple
import time

from .tools.base_tool import Tool, ToolException
from .sentinel_agent import SentinelException

class PlannerAgent:
    def __init__(self, config, memory, quartermaster, sentinel, tools=None, agents=None, message_bus_client=None):
        self.config = config
        self.memory = memory # Instance of EnhancedMemory
        self.qm = quartermaster
        self.sentinel = sentinel
        self.tools = tools if tools else []
        self.agents = agents if agents else {} # Dictionary of other agent instances
        model_conf = config.get("models", {})
        self.model_name = model_conf.get("planner", model_conf.get("default", "llama2"))

        self.plans_dir = os.path.join(self.qm.output_dir, "plans")
        os.makedirs(self.plans_dir, exist_ok=True)
        
        self.plans: Dict[str, Dict[str, Any]] = {}
        self.current_plan_id: Optional[str] = None
        self._load_plans()

        self.message_bus_client = message_bus_client

    def _generate_plan_id(self, goal_summary: str) -> str:
        """Generates a unique ID for a plan."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sanitized_goal = re.sub(r'\W+', '_', goal_summary.lower())[:30].strip('_')
        return f"plan_{timestamp}_{sanitized_goal}"

    def _load_plans(self):
        """Loads all plan .json files from the self.plans_dir directory."""
        print(f"[PlannerAgent] Loading plans from: {self.plans_dir}")
        self.plans = {} 
        try:
            if not os.path.exists(self.plans_dir):
                print(f"[PlannerAgent] Plans directory '{self.plans_dir}' does not exist. Creating it.")
                os.makedirs(self.plans_dir)
                return # No plans to load if directory was just created

            index_file_path = os.path.join(self.plans_dir, "plans_index.json")
            if os.path.exists(index_file_path):
                try:
                    with open(index_file_path, 'r', encoding='utf-8') as f:
                        # The index itself contains the full plan data as per previous logic
                        loaded_index_plans_data = json.load(f) 
                    
                    # Validate that plans in index still have corresponding files (optional sanity check)
                    # For now, we trust the index if it exists, as _save_plans_index saves the full self.plans
                    self.plans = loaded_index_plans_data
                    print(f"[PlannerAgent] Loaded {len(self.plans)} plans directly from index: {index_file_path}")
                    return 
                except json.JSONDecodeError as e_json_idx:
                    print(f"[PlannerAgent] Warning: plans_index.json is corrupted ({e_json_idx}). Attempting to load individual files.")
                except Exception as e_idx:
                    print(f"[PlannerAgent] Error loading from plans_index.json: {e_idx}. Attempting individual files.")
            
            # Fallback: Load individual plan files if index is missing or corrupt
            print(f"[PlannerAgent] Scanning directory for individual plan files...")
            loaded_count = 0
            for filename in os.listdir(self.plans_dir):
                if filename.startswith("plan_") and filename.endswith(".json") and filename != "plans_index.json":
                    file_path = os.path.join(self.plans_dir, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            plan_data = json.load(f)
                        plan_id_from_content = plan_data.get("id")
                        plan_id_from_filename = filename[:-5] # Remove .json

                        # Prioritize ID from content if available and matches filename convention
                        current_plan_id_to_use = plan_id_from_content if plan_id_from_content else plan_id_from_filename
                        if not plan_id_from_content: # If no ID in content, inject it from filename
                            plan_data["id"] = plan_id_from_filename
                        
                        if plan_id_from_content and plan_id_from_content != plan_id_from_filename:
                             print(f"[PlannerAgent] Warning: Plan ID '{plan_id_from_content}' in {filename} differs from filename. Using ID from content.")
                        
                        self.plans[current_plan_id_to_use] = plan_data
                        loaded_count += 1
                    except json.JSONDecodeError:
                        print(f"[PlannerAgent] Warning: Plan file {filename} is corrupted. Skipping.")
                    except Exception as e_load_single:
                        print(f"[PlannerAgent] Error loading plan file {filename}: {e_load_single}. Skipping.")
            
            if loaded_count > 0:
                 print(f"[PlannerAgent] Loaded {loaded_count} plans by scanning directory.")
                 self._save_plans_index() # Create/update the index file
            elif not os.path.exists(index_file_path): # Only print if index also doesn't exist
                 print(f"[PlannerAgent] No plan files found in {self.plans_dir} and no index file to load from.")
        except Exception as e:
            print(f"[PlannerAgent] Critical error loading plans: {e}")

    def _save_plan_to_file(self, plan_id: str):
        """Saves a single plan to its JSON file and updates the main plan index."""
        if plan_id not in self.plans:
            print(f"[PlannerAgent] Error: Plan {plan_id} not found in memory to save.")
            return
        plan_data = self.plans[plan_id]
        plan_file_path = os.path.join(self.plans_dir, f"{plan_id}.json")
        try:
            with open(plan_file_path, 'w', encoding='utf-8') as f:
                json.dump(plan_data, f, indent=2)
            print(f"[PlannerAgent] Saved plan '{plan_id}' to {plan_file_path}")
            # The index (self.plans) is already updated in memory, save the index file.
            self._save_plans_index() 
        except Exception as e:
            print(f"[PlannerAgent] Error saving plan {plan_id} to file: {e}")

    def _save_plans_index(self):
        """Saves the self.plans dictionary (which is the index of all plans) to plans_index.json."""
        index_file_path = os.path.join(self.plans_dir, "plans_index.json")
        try:
            with open(index_file_path, 'w', encoding='utf-8') as f:
                # Save the entire self.plans dictionary, which now holds full plan data
                json.dump(self.plans, f, indent=2) 
            print(f"[PlannerAgent] Plans index (containing full plan data) saved to {index_file_path}")
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
            "quartermaster": "Manages files (read, write, list, delete), project goals, and provides system information. Does not use an LLM directly for its core tasks.",
        }
        for agent_key, agent_instance in self.agents.items():
            agent_class_name = agent_instance.__class__.__name__ if hasattr(agent_instance, '__class__') else agent_key.capitalize() + "Agent"
            role = agent_roles.get(agent_key.lower(), "Performs specialized tasks.") # Ensure key is lowercase for lookup
            descriptions.append(f"- Agent Key: \"{agent_key}\" (Class: {agent_class_name}): {role}")
        return "\n".join(descriptions)

    def _generate_planning_prompt(self, goal: str, context: str = "") -> str:
        agent_capabilities = self._get_agent_capabilities_description()
        prompt = (
            f"You are a Planner Agent. Your role is to break down a complex USER GOAL into a sequence of manageable, concrete tasks. "
            f"Each task must be assigned to the most appropriate agent from AVAILABLE AGENTS using their 'Agent Key' (e.g., \"analyst\", \"scribe\"). "
            f"Consider dependencies between tasks.\n"
            f"{agent_capabilities}\n"
            f"\nUSER GOAL: \"{goal}\"\n"
            f"{context}\n" 
            f"Based on this goal, provide a detailed plan. The plan must be a single JSON object with the following structure:\n"
            "{\n"
            "  \"id\": \"plan_YYYYMMDD_HHMMSS_short_goal_summary\",\n" 
            "  \"title\": \"Descriptive Title of the Plan\",\n"
            "  \"description\": \"A brief summary of what this plan aims to achieve.\",\n"
            "  \"tasks\": [\n"
            "    {\n"
            "      \"id\": \"plan_id_task_1\",\n" 
            "      \"index\": 1, \n"
            "      \"description\": \"Clear, specific description of the first task.\",\n"
            "      \"assigned_agent\": \"agent_key_from_list\",\n" 
            "      \"complexity\": \"Low/Medium/High\",\n"
            "      \"dependencies\": []\n"
            "    }\n"
            "    // ... more tasks (ensure 'index' is sequential and 'dependencies' use these indices)\n"
            "  ]\n"
            "}\n"
            "IMPORTANT:\n"
            "- 'assigned_agent' MUST be one of the 'Agent Key' values provided in AVAILABLE AGENTS.\n"
            "- Task descriptions should be actionable and clear for the assigned agent.\n"
            "- 'dependencies' must be a list of 'index' numbers of tasks that need to be completed before this task can start.\n"
            "- If the goal is simple and can be handled by one agent in one step, create a plan with a single task.\n"
            "Output ONLY the JSON plan. No other text, explanations, or markdown formatting."
        )
        return prompt

    def create_plan(self, goal_text: str, associated_goal_id: Optional[str] = None) -> Tuple[Optional[str], str]:
        """
        Creates a plan for a given goal using the LLM.
        Args:
            goal_text (str): The goal description.
            associated_goal_id (Optional[str]): The ID of the goal in EnhancedMemory this plan is for.
        Returns:
            Tuple[Optional[str], str]: (plan_id, message_or_plan_display_string)
        """
        print(f"[PlannerAgent] Creating plan for goal: '{goal_text[:100]}...' (Associated Goal ID: {associated_goal_id})")
        prompt = self._generate_planning_prompt(goal_text)
        plan_id_generated: Optional[str] = None

        try:
            print(f"[PlannerAgent] Sending planning prompt to LLM (model: {self.model_name}).")
            ollama_response = ollama.generate(model=self.model_name, prompt=prompt, format="json")
            raw_plan_json_str = ollama_response['response'].strip()
            print(f"[PlannerAgent] LLM Raw Output for plan (first 500 chars):\n{raw_plan_json_str[:500]}...")

            plan_data = json.loads(raw_plan_json_str)
            
            # Generate or use LLM-provided plan ID
            plan_id_generated = plan_data.get("id") or self._generate_plan_id(plan_data.get("title", goal_text[:30]))
            plan_data["id"] = plan_id_generated
            self.current_plan_id = plan_id_generated # Set as current active plan being worked on

            # Standardize plan structure
            plan_data.setdefault("title", goal_text[:50] + "...")
            plan_data.setdefault("description", goal_text)
            plan_data["created"] = datetime.now().isoformat()
            plan_data["updated"] = plan_data["created"]
            plan_data["status"] = "pending" # Initial status of the plan itself
            plan_data["progress"] = 0.0
            plan_data["original_goal_text"] = goal_text 
            plan_data["original_goal_id"] = associated_goal_id # Link to the EnhancedMemory goal ID
            plan_data.setdefault("plan_file", os.path.join("plans", f"{plan_id_generated}.json")) # Relative path

            tasks = plan_data.get("tasks", [])
            if not isinstance(tasks, list):
                raise ValueError(f"'tasks' field in LLM output for plan {plan_id_generated} is not a list.")
            
            valid_tasks = []
            for i, task_item_data in enumerate(tasks):
                if not isinstance(task_item_data, dict):
                     print(f"[PlannerAgent] Warning: Task at index {i} in plan {plan_id_generated} is not a dictionary. Skipping. Task: {task_item_data}")
                     continue
                task_item_data["id"] = f"{plan_id_generated}_task_{i+1}" # Standardize task ID
                task_item_data["index"] = i + 1 # Ensure correct index
                task_item_data.setdefault("status", "pending")
                task_item_data.setdefault("result", None)
                task_item_data.setdefault("started", None)
                task_item_data.setdefault("completed", None)
                task_item_data.setdefault("dependencies", [])
                if not isinstance(task_item_data["dependencies"], list): 
                    print(f"[PlannerAgent] Warning: Task {task_item_data.get('index')} dependencies not a list. Defaulting to empty. Deps: {task_item_data['dependencies']}")
                    task_item_data["dependencies"] = []
                valid_tasks.append(task_item_data)
            plan_data["tasks"] = valid_tasks
            
            self.plans[plan_id_generated] = plan_data # Store in memory
            self._save_plan_to_file(plan_id_generated) # Save to file and update index

            # If this plan is for a specific goal, update that goal's status in EnhancedMemory
            if associated_goal_id:
                self.memory.update_goal_status(associated_goal_id, "processing", processing_plan_id=plan_id_generated)
            
            return plan_id_generated, self._format_plan_for_display(plan_id_generated)

        except json.JSONDecodeError as e:
            error_msg = f"Error: LLM plan output was not valid JSON. Error: {e}. Raw: {raw_plan_json_str}"
            print(f"[PlannerAgent] {error_msg}")
            if associated_goal_id:
                self.memory.update_goal_status(associated_goal_id, "autonomous_failed", result_summary=f"Plan creation failed (JSON error): {e}")
            self.memory.remember_agent_output("planner_status", f"Failed to create plan (JSON parse error for goal '{goal_text[:30]}...').")
            return None, error_msg
        except Exception as e:
            import traceback
            error_msg = f"Error creating plan: {type(e).__name__} - {e}\n{traceback.format_exc()}"
            print(f"[PlannerAgent] {error_msg}")
            if associated_goal_id:
                self.memory.update_goal_status(associated_goal_id, "autonomous_failed", result_summary=f"Plan creation failed: {e}")
            self.memory.remember_agent_output("planner_status", f"Failed to create plan for goal '{goal_text[:30]}...': {e}")
            return None, error_msg

    def _format_plan_for_display(self, plan_id: str) -> str:
        """Formats a plan for user display."""
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
            f"Associated Memory Goal ID: {plan.get('original_goal_id', 'None')}"
        ]
        tasks = plan.get("tasks", [])
        if tasks:
            display.append("Tasks:")
            for task in tasks:
                dependencies_str = ", ".join(map(str, task.get("dependencies", []))) or "None"
                task_result_preview = ""
                if task.get("result"):
                    result_str = str(task.get("result"))
                    task_result_preview = f" Result: {result_str[:70]}..." if len(result_str) > 70 else f" Result: {result_str}"
                
                display.append(
                    f"  - Task {task.get('index', 'N/A')}: {task.get('description', 'N/A')}\n"
                    f"    Assigned to: {task.get('assigned_agent', 'N/A')}\n"
                    f"    Status: {task.get('status', 'pending')}{task_result_preview}\n"
                    f"    Dependencies: {dependencies_str}"
                )
        else:
            display.append("Tasks: No tasks defined in this plan yet.")
        return "\n".join(display)

    def update_task_status(self, plan_id: str, task_index: int, new_status: str, result: Optional[str] = None) -> bool:
        """Updates the status of a specific task in a plan and the overall plan."""
        if plan_id not in self.plans:
            print(f"[PlannerAgent] Update task status failed: Plan {plan_id} not found.")
            return False
        
        plan = self.plans[plan_id]
        task_found_for_update = False
        
        for task in plan.get("tasks", []):
            if task.get("index") == task_index:
                task["status"] = new_status
                task["updated_time"] = datetime.now().isoformat() # Add updated time for task
                if new_status == "in_progress" and not task.get("started"):
                    task["started"] = datetime.now().isoformat()
                elif new_status == "completed":
                    task["completed"] = datetime.now().isoformat()
                    task["result"] = result if result is not None else task.get("result", "Completed without specific result.")
                elif new_status == "error":
                    task["result"] = result if result is not None else "Task failed with an unspecified error."
                elif new_status == "blocked":
                    task["result"] = result if result is not None else "Task blocked by Sentinel or other restriction."
                task_found_for_update = True
                break # Found and updated the task

        if task_found_for_update:
            plan["updated"] = datetime.now().isoformat()
            
            completed_tasks_count = sum(1 for t in plan.get("tasks", []) if t.get("status") == "completed")
            total_tasks_count = len(plan.get("tasks", []))
            plan["progress"] = (completed_tasks_count / total_tasks_count) * 100 if total_tasks_count > 0 else 0.0
            
            current_plan_status = plan.get("status")
            # Check if all tasks are completed
            all_tasks_completed = total_tasks_count > 0 and all(t.get("status") == "completed" for t in plan.get("tasks", []))

            if all_tasks_completed and current_plan_status != "completed":
                plan["status"] = "completed"
                print(f"[PlannerAgent] Plan {plan_id} ('{plan.get('title')}') fully marked as COMPLETED.")
                if self.message_bus_client:
                     self.message_bus_client.broadcast(f"Plan '{plan.get('title', plan_id)}' has completed!", context=f"plan_completion:{plan_id}")
                
                associated_goal_id = plan.get("original_goal_id")
                if associated_goal_id:
                    plan_summary_for_goal = f"Plan '{plan.get('title', plan_id)}' (ID: {plan_id}) completed successfully. All {total_tasks_count} tasks finished."
                    self.memory.complete_goal(associated_goal_id, result=plan_summary_for_goal)
                    print(f"[PlannerAgent] Marked associated Goal ID {associated_goal_id} as completed in memory.")
            # If any task fails or is blocked, the plan itself might be considered 'failed' or 'halted'
            elif new_status in ["error", "blocked"] and current_plan_status not in ["error", "blocked", "completed"]:
                plan["status"] = "halted_due_to_task_issue" # Or a more specific status
                print(f"[PlannerAgent] Plan {plan_id} status changed to '{plan['status']}' due to task {task_index} status '{new_status}'.")
                associated_goal_id = plan.get("original_goal_id")
                if associated_goal_id: # Update memory goal to reflect failure
                    self.memory.update_goal_status(associated_goal_id, "autonomous_failed", result_summary=f"Plan '{plan.get('title')}' halted. Task {task_index} {new_status}: {result}")


            self._save_plan_to_file(plan_id)
            print(f"[PlannerAgent] Task {task_index} in plan {plan_id} status updated to {new_status}. Progress: {plan['progress']:.1f}%")
            self.memory.remember_agent_output("planner_status", f"Plan {plan_id}: Task {task_index} is now {new_status}. Overall progress {plan['progress']:.1f}%.")
            return True
        else:
            print(f"[PlannerAgent] Update task status failed: Task {task_index} not found in plan {plan_id}.")
            return False

    def get_next_task(self, plan_id: str) -> Tuple[Optional[Dict[str, Any]], str]:
        """Gets the next pending task whose dependencies are met."""
        if plan_id not in self.plans:
            return None, f"Plan {plan_id} not found."
        plan = self.plans[plan_id]

        if plan.get("status") == "completed":
            return None, f"Plan {plan_id} is already completed."
        if plan.get("status") == "halted_due_to_task_issue": # If plan is halted, no next task
            return None, f"Plan {plan_id} is halted due to a previous task issue. Manual review or new plan may be needed."


        tasks_by_index_lookup = {task["index"]: task for task in plan.get("tasks", [])}

        for task_to_check in plan.get("tasks", []):
            if task_to_check.get("status") == "pending":
                dependencies_are_met = True
                for dep_idx_val in task_to_check.get("dependencies", []):
                    dep_idx_to_find = -1
                    if isinstance(dep_idx_val, int):
                        dep_idx_to_find = dep_idx_val
                    elif isinstance(dep_idx_val, str):
                        try: dep_idx_to_find = int(re.sub(r'\D', '', dep_idx_val))
                        except ValueError: pass
                    
                    dependent_task = tasks_by_index_lookup.get(dep_idx_to_find)
                    if not dependent_task or dependent_task.get("status") != "completed":
                        dependencies_are_met = False
                        break 
                if dependencies_are_met:
                    return task_to_check, f"Next task {task_to_check['index']} is ready."
        
        # If loop finishes, no pending task with met dependencies.
        # Check again if all tasks are actually completed (could have been updated by another process)
        if all(t.get("status") == "completed" for t in plan.get("tasks", [])):
            if plan.get("status") != "completed":
                plan["status"] = "completed"
                plan["progress"] = 100.0
                plan["updated"] = datetime.now().isoformat()
                self._save_plan_to_file(plan_id)
                print(f"[PlannerAgent] Plan {plan_id} auto-marked as completed (all tasks done).")
                if self.message_bus_client:
                    self.message_bus_client.broadcast(f"Plan '{plan.get('title', plan_id)}' auto-completed!", context=f"plan_completion:{plan_id}")
                # Also update the original goal in EnhancedMemory if linked
                associated_goal_id = plan.get("original_goal_id")
                if associated_goal_id:
                    self.memory.complete_goal(associated_goal_id, result=f"Plan '{plan.get('title')}' auto-completed.")
            return None, f"Plan {plan_id} has no more tasks and is complete."
        else:
             return None, f"Plan {plan_id}: No pending tasks with met dependencies. Plan status: {plan.get('status')}."


    def delegate_task(self, plan_id: str, task_index: int) -> str:
        """Delegates a task to its assigned agent. Updates task status before and after."""
        if plan_id not in self.plans:
            return f"Error: Plan {plan_id} not found."
        plan = self.plans[plan_id]
        
        task_to_delegate = next((t for t in plan.get("tasks", []) if t.get("index") == task_index), None)
        
        if not task_to_delegate:
            return f"Error: Task {task_index} not found in plan {plan_id}."
        if task_to_delegate.get("status") not in ["pending", "retry"]: # "retry" could be a status for re-attempting
            return f"Error: Task {task_index} (status: {task_to_delegate.get('status')}) is not pending or ready for retry."

        assigned_agent_key = task_to_delegate.get("assigned_agent", "").lower()
        task_description_for_agent = task_to_delegate.get("description")
        target_agent_instance = self.agents.get(assigned_agent_key)

        if not target_agent_instance:
            # Attempt fallback match by class name if key fails (for robustness against LLM output)
            original_assigned_str = task_to_delegate.get("assigned_agent", "")
            for key, inst in self.agents.items():
                if hasattr(inst, '__class__') and inst.__class__.__name__ == original_assigned_str:
                    target_agent_instance = inst
                    assigned_agent_key = key # Use the correct key now
                    print(f"[PlannerAgent] Warning: Agent for task {task_index} matched by class name '{original_assigned_str}'. LLM should use key '{assigned_agent_key}'.")
                    break
        
        if not target_agent_instance:
            err_msg = f"Assigned agent '{task_to_delegate.get('assigned_agent')}' for task {task_index} not found."
            self.update_task_status(plan_id, task_index, "error", result=err_msg)
            return f"Error: {err_msg}"

        agent_class_name_for_log = target_agent_instance.__class__.__name__
        print(f"[PlannerAgent] Delegating Task {task_index} ('{task_description_for_agent[:50]}...') to agent '{assigned_agent_key}' ({agent_class_name_for_log})")
        self.update_task_status(plan_id, task_index, "in_progress")
        self.memory.remember_agent_output("planner_status", f"Plan {plan_id}: Delegating task {task_index} ('{task_description_for_agent[:30]}...') to {assigned_agent_key}.")

        task_execution_result: Optional[str] = None
        try:
            self.sentinel.approve_action(
                agent_name="PlannerAgent",
                action_type=f"delegate_to:{assigned_agent_key}",
                detail=f"Task for plan {plan_id}: {task_description_for_agent}"
            )
            task_execution_result = target_agent_instance.run(task_description_for_agent)
            self.update_task_status(plan_id, task_index, "completed", result=str(task_execution_result))
            self.memory.remember_agent_output("planner_status", f"Plan {plan_id}: Task {task_index} completed by {assigned_agent_key}.")
            return f"Task {task_index} delegated to {assigned_agent_key} and completed. Result snippet: {str(task_execution_result)[:100]}..."

        except SentinelException as se:
            error_details = f"Delegation for task {task_index} blocked by Sentinel: {str(se)}"
            print(f"[PlannerAgent] {error_details}")
            self.update_task_status(plan_id, task_index, "blocked", result=error_details)
            self.memory.remember_agent_output("planner_status", f"Plan {plan_id}: Task {task_index} delegation blocked by Sentinel.")
            return error_details # Return the error message to the execution loop
        except Exception as e_delegate:
            import traceback
            error_details = f"Error during task {task_index} delegation to {assigned_agent_key}: {str(e_delegate)}\n{traceback.format_exc()}"
            print(f"[PlannerAgent] {error_details}")
            self.update_task_status(plan_id, task_index, "error", result=f"Agent {assigned_agent_key} execution error: {str(e_delegate)}")
            self.memory.remember_agent_output("planner_status", f"Plan {plan_id}: Task {task_index} failed during execution by {assigned_agent_key}.")
            return error_details # Return the error message

    def assess_plan(self, plan_id: str) -> str:
        """Assesses the current status and next steps of a plan."""
        # ... (remains the same as your provided version, but ensure it uses self.plans[plan_id].get("original_goal_id"))
        if plan_id not in self.plans:
            return f"Plan {plan_id} not found."
        
        plan_display = self._format_plan_for_display(plan_id) # This now includes original_goal_id
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
        if not self.plans:
            return "No plans found."
        
        plan_summaries = ["Available Plans:"]
        sorted_plan_ids = sorted(self.plans.keys(), key=lambda pid: self.plans[pid].get("created", ""), reverse=True)

        for plan_id in sorted_plan_ids:
            plan_data = self.plans[plan_id]
            status_emoji = "✅" if plan_data.get('status') == 'completed' else ("⏳" if plan_data.get('status') == 'active' or plan_data.get('status') == 'processing' else "📋")
            goal_id_str = f"(Goal: {plan_data.get('original_goal_id', 'N/A')[:8]}...)" if plan_data.get('original_goal_id') else ""
            plan_summaries.append(
                f"  {status_emoji} ID: {plan_id} - Title: {plan_data.get('title', 'Untitled Plan')} "
                f"(Status: {plan_data.get('status', 'pending')}, Progress: {plan_data.get('progress', 0.0):.1f}%) {goal_id_str}"
            )
        return "\n".join(plan_summaries)

    def set_current_plan(self, plan_id: str) -> str:
        # ... (remains the same)
        if plan_id in self.plans:
            self.current_plan_id = plan_id
            return f"Current plan set to {plan_id} ('{self.plans[plan_id].get('title', '')}')."
        return f"Error: Plan ID {plan_id} not found."

    def _execute_plan_loop(self, plan_id: str) -> str:
        """Internal loop to automatically execute tasks in a plan. Updates associated goal status."""
        if plan_id not in self.plans:
            return f"Error: Plan {plan_id} not found for execution loop."

        plan = self.plans[plan_id]
        plan_title = plan.get('title', 'Untitled Plan')
        associated_goal_id = plan.get("original_goal_id")
        
        execution_log = [f"Starting automated execution of plan: {plan_title} ({plan_id}) for Goal ID: {associated_goal_id or 'N/A'}"]
        print(f"[PlannerAgent] Starting automated execution of plan: {plan_id} - '{plan_title}' (Goal ID: {associated_goal_id})")
        self.memory.remember_agent_output("planner_status", f"Plan {plan_id} ('{plan_title}'): Execution started for Goal ID {associated_goal_id}.")
        
        if plan.get("status") == "pending":
            plan["status"] = "active"
            self._save_plan_to_file(plan_id)
            if associated_goal_id: # Also update goal status in memory
                self.memory.update_goal_status(associated_goal_id, "processing", processing_plan_id=plan_id)

        max_steps = len(plan.get("tasks", [])) * 2 + 10 
        current_step_count = 0
        plan_final_outcome_message = "" # To store the final outcome message

        while current_step_count < max_steps:
            current_step_count += 1
            plan = self.plans[plan_id] # Refresh plan data

            if plan.get("status") == "completed":
                plan_final_outcome_message = f"Plan {plan_id} '{plan_title}' is marked as completed."
                break 
            if plan.get("status") == "halted_due_to_task_issue":
                plan_final_outcome_message = f"Plan {plan_id} '{plan_title}' execution halted due to a task issue."
                break

            task_to_execute, message_from_get_next = self.get_next_task(plan_id)

            if not task_to_execute:
                plan_final_outcome_message = f"Plan {plan_id} '{plan_title}': No further executable tasks. {message_from_get_next}"
                break 

            task_log_desc = f"Task {task_to_execute['index']} ('{task_to_execute['description'][:40]}...')"
            log_msg = f"[PlannerAgent] Plan {plan_id}: Attempting to auto-execute {task_log_desc} with agent {task_to_execute['assigned_agent']}"
            print(log_msg); execution_log.append(log_msg)
            self.memory.remember_agent_output("planner_status", f"Plan {plan_id}: Executing {task_log_desc} via {task_to_execute['assigned_agent']}.")

            # delegate_task now returns the error/block message directly if it occurs, or a success summary.
            delegation_outcome_message = self.delegate_task(plan_id, task_to_execute["index"])
            execution_log.append(f"Task {task_to_execute['index']} delegation attempt: {delegation_outcome_message[:150]}...")
            print(f"[PlannerAgent] Plan {plan_id}: Task {task_to_execute['index']} delegation outcome: {delegation_outcome_message[:100]}...")
            
            # Check task status after delegation attempt
            # The plan object (self.plans[plan_id]) is updated by delegate_task via update_task_status
            updated_task_instance = next((t for t in self.plans[plan_id].get("tasks", []) if t.get("index") == task_to_execute["index"]), None)
            
            if updated_task_instance and updated_task_instance.get("status") in ["error", "blocked"]:
                plan_final_outcome_message = (
                    f"Plan {plan_id} '{plan_title}' execution halted at task {task_to_execute['index']} "
                    f"(Status: {updated_task_instance['status']}). Result: {updated_task_instance.get('result', '')}"
                )
                # The update_task_status method (called by delegate_task) should have already set plan status to "halted_due_to_task_issue"
                # and updated the associated memory goal to "autonomous_failed".
                break # Halt on task error/block

            time.sleep(0.1) # Small delay

            # Check plan status again, as the last task might have completed the plan
            if self.plans[plan_id].get("status") == "completed":
                plan_final_outcome_message = f"Plan {plan_id} '{plan_title}' marked as fully completed after processing task {task_to_execute['index']}."
                break
        
        # After loop finishes (either by break or max_steps)
        if not plan_final_outcome_message: # If loop finished due to max_steps
            plan_final_outcome_message = f"Plan {plan_id} '{plan_title}' execution loop reached max steps ({max_steps}). Halting. Current plan status: {plan.get('status', 'unknown')}."
            if associated_goal_id and plan.get("status") != "completed":
                self.memory.update_goal_status(associated_goal_id, "autonomous_failed", result_summary=plan_final_outcome_message)
        
        print(f"[PlannerAgent] {plan_final_outcome_message}")
        execution_log.append(plan_final_outcome_message)
        self.memory.remember_agent_output("planner_status", f"Plan {plan_id}: Execution loop finished. {plan_final_outcome_message}")

        # Final check on associated goal if plan didn't complete successfully
        if plan.get("status") != "completed" and associated_goal_id:
            goal_in_memory = self.memory.get_goal_by_id(associated_goal_id)
            if goal_in_memory and goal_in_memory.get('status') not in ['completed', 'autonomous_failed']:
                 self.memory.update_goal_status(associated_goal_id, "autonomous_failed", result_summary=f"Plan '{plan.get('title')}' did not complete. Final plan status: {plan.get('status')}")
        
        return "\n".join(execution_log)

    def run(self, command: str, associated_goal_id_for_new_plan: Optional[str] = None) -> str:
        """
        Handles commands for the PlannerAgent.
        If associated_goal_id_for_new_plan is provided, it's used when 'create plan' or 
        'create and execute plan' commands are processed for that specific goal.
        """
        print(f"[PlannerAgent] === PlannerAgent.run START === RAW COMMAND: '{command}', Assoc. Goal ID: {associated_goal_id_for_new_plan}")
        cmd_lower = command.lower().strip()
        
        try:
            if cmd_lower.startswith("create and execute plan") or \
               cmd_lower.startswith("create plan and run") or \
               cmd_lower.startswith("start main plan for") or \
               cmd_lower.startswith("execute main plan for"):
                
                goal_text_for_plan = ""
                if cmd_lower.startswith("create and execute plan"):
                    goal_text_for_plan = command[len("create and execute plan"):].strip()
                    if goal_text_for_plan.lower().startswith("for "): goal_text_for_plan = goal_text_for_plan[len("for "):].strip()
                elif cmd_lower.startswith("create plan and run"):
                    goal_text_for_plan = command[len("create plan and run"):].strip()
                    if goal_text_for_plan.lower().startswith("for "): goal_text_for_plan = goal_text_for_plan[len("for "):].strip()
                # Add other extraction logic if needed for "start main plan for", etc.
                else: # Fallback for simpler extraction
                     goal_text_for_plan = re.sub(r"^(create and execute plan|create plan and run|start main plan for|execute main plan for)\s*(?:for|to|about)?\s*", "", command, flags=re.IGNORECASE).strip()


                if not goal_text_for_plan:
                    self.memory.remember_agent_output("planner_status", "Failed to start plan: Goal description missing.")
                    return "Error: Please provide a clear goal for the plan (e.g., 'Planner, create and execute plan for X')."
                
                self.memory.remember_agent_output("planner_status", f"Request: Create & Execute Plan for: {goal_text_for_plan[:50]}...")
                
                # Use associated_goal_id_for_new_plan if provided, otherwise it's None
                plan_id, plan_creation_output = self.create_plan(goal_text_for_plan, associated_goal_id=associated_goal_id_for_new_plan)

                if not plan_id or "Error" in plan_creation_output: # Check if plan_id is None
                    self.memory.remember_agent_output("planner_status", f"Failed to create plan: {plan_creation_output}")
                    # The create_plan method already updates goal status on failure if associated_goal_id was passed.
                    return f"Failed to create plan for execution: {plan_creation_output}"
                
                self.memory.remember_agent_output("planner_status", f"Plan {plan_id} created for '{goal_text_for_plan[:50]}...'. Starting auto-execution.")
                plan_execution_log = self._execute_plan_loop(plan_id)
                final_plan_status = self.plans.get(plan_id, {}).get("status", "unknown")
                self.memory.remember_agent_output("planner_status", f"Plan {plan_id} auto-execution finished. Status: {final_plan_status}.")
                return f"Plan Creation Summary:\n{plan_creation_output}\n\n--- Automated Plan Execution Log ---\n{plan_execution_log}"

            elif cmd_lower.startswith("execute plan") or cmd_lower.startswith("run plan") or cmd_lower.startswith("start executing plan"):
                plan_id_match = re.search(r'(?:execute|run|start executing) plan\s+([\w\d_\-\.:]+)', command, re.IGNORECASE)
                if plan_id_match:
                    plan_id_to_exec = plan_id_match.group(1).strip()
                    if plan_id_to_exec not in self.plans: return f"Error: Plan '{plan_id_to_exec}' not found."
                    self.current_plan_id = plan_id_to_exec
                    self.memory.remember_agent_output("planner_status", f"Starting execution of existing plan: {plan_id_to_exec}.")
                    exec_log = self._execute_plan_loop(plan_id_to_exec)
                    final_status_exec = self.plans.get(plan_id_to_exec, {}).get("status", "unknown")
                    self.memory.remember_agent_output("planner_status", f"Exec loop for plan {plan_id_to_exec} finished. Status: {final_status_exec}.")
                    return f"--- Execution Log for Plan {plan_id_to_exec} ---\n{exec_log}"
                else: return "Error: Please specify Plan ID to execute (e.g., 'execute plan plan_YYYYMMDD_HHMMSS_id')."

            elif cmd_lower.startswith("create plan") or cmd_lower.startswith("make plan") or cmd_lower.startswith("new plan"):
                goal_for_manual_plan = re.sub(r'^(create|make|new)\s+plan\s*(?:for|to|about)?\s*', '', command, flags=re.IGNORECASE).strip()
                if goal_for_manual_plan:
                    self.memory.remember_agent_output("planner_status", f"Creating plan (manual) for: {goal_for_manual_plan[:50]}...")
                    # Use associated_goal_id_for_new_plan if this command is for an autonomous goal
                    plan_id_manual, plan_disp_manual = self.create_plan(goal_for_manual_plan, associated_goal_id=associated_goal_id_for_new_plan)
                    if plan_id_manual:
                        self.memory.remember_agent_output("planner_status", f"Plan created (manual). ID: {plan_id_manual}")
                        return plan_disp_manual
                    else: # Error in plan_disp_manual
                        self.memory.remember_agent_output("planner_status", f"Failed to create plan (manual): {plan_disp_manual}")
                        return plan_disp_manual 
                else: return "Error: Please provide a goal for the plan (e.g., 'Planner, create plan for organizing my research')."
            
            # ... (rest of the command parsing like list_plans, show_plan, update_task, etc. remain the same)
            elif "list plans" in cmd_lower or "show plans" in cmd_lower:
                return self.list_plans()
            elif cmd_lower.startswith("show plan") or cmd_lower.startswith("plan details") or cmd_lower.startswith("view plan"):
                plan_id_match = re.search(r'(?:show|view|get) plan (?:details for )?([\w\d_\-\.:]+)', command, re.IGNORECASE)
                pid_show = plan_id_match.group(1).strip() if plan_id_match else self.current_plan_id
                if pid_show and pid_show in self.plans: return self._format_plan_for_display(pid_show)
                elif pid_show: return f"Error: Plan ID '{pid_show}' not found."
                else: return "Error: Specify plan ID or set current plan."
            elif cmd_lower.startswith("update task"):
                match = re.search(r'update task\s+(\d+)\s+in plan\s+([\w\d_\-\.:]+)\s+to\s+(\w+)(?:\s+with result:\s*(.+))?', command, re.IGNORECASE)
                if match:
                    task_idx, plan_id_upd, status_upd, res_text_upd = match.groups()
                    if self.update_task_status(plan_id_upd, int(task_idx), status_upd.lower(), result=res_text_upd): return f"Task {task_idx} in plan {plan_id_upd} updated to {status_upd.lower()}."
                    else: return f"Error: Could not update task {task_idx} in plan {plan_id_upd}."
                else: return "Error: Invalid 'update task' format. Use: update task [index] in plan [plan_id] to [status] [with result: text]"
            elif cmd_lower.startswith("next task in plan") or cmd_lower.startswith("get next task for plan"):
                plan_id_match_next = re.search(r'(?:next task in plan|get next task for plan)\s+([\w\d_\-\.:]+)', command, re.IGNORECASE)
                pid_next = plan_id_match_next.group(1).strip() if plan_id_match_next else self.current_plan_id
                if not pid_next: return "Error: Specify plan ID or set current plan."
                if pid_next not in self.plans: return f"Error: Plan ID '{pid_next}' not found."
                task_next, message_next = self.get_next_task(pid_next)
                if task_next: return f"Next task for {pid_next} (Task {task_next['index']}): '{task_next['description']}' (Agent: {task_next['assigned_agent']}, Status: {task_next['status']})"
                else: return f"For plan {pid_next}: {message_next}"
            elif cmd_lower.startswith("delegate task") or cmd_lower.startswith("execute task"): # Manual execution of a single task
                match_delegate = re.search(r'(?:delegate task|execute task)\s+(\d+)\s*(?:in plan\s+([\w\d_\-\.:]+))?', command, re.IGNORECASE)
                if match_delegate:
                    task_idx_del = int(match_delegate.group(1))
                    pid_del = match_delegate.group(2).strip() if match_delegate.group(2) else self.current_plan_id
                    if not pid_del: return "Error: No plan specified for task delegation and no current plan set."
                    if pid_del not in self.plans: return f"Error: Plan ID '{pid_del}' not found."
                    return self.delegate_task(pid_del, task_idx_del)
                else: return "Error: Invalid format. Use: delegate task [index] in plan [plan_id]"
            elif cmd_lower.startswith("assess plan"):
                plan_id_match_assess = re.search(r'assess plan\s+([\w\d_\-\.:]+)', command, re.IGNORECASE)
                pid_assess = plan_id_match_assess.group(1).strip() if plan_id_match_assess else self.current_plan_id
                if not pid_assess: return "Error: Specify plan ID or set current plan."
                if pid_assess not in self.plans: return f"Error: Plan ID '{pid_assess}' not found."
                return self.assess_plan(pid_assess)
            elif cmd_lower.startswith("set current plan"):
                plan_id_match_set = re.search(r'set current plan\s+([\w\d_\-\.:]+)', command, re.IGNORECASE)
                if plan_id_match_set: return self.set_current_plan(plan_id_match_set.group(1).strip())
                else: return "Error: Specify plan ID to set as current."
            else:
                unrec_msg = ("PlannerAgent: Command not recognized. Try 'create and execute plan for [goal]', "
                             "'execute plan [plan_id]', 'create plan for [goal]', 'list plans', 'show plan [plan_id]', etc.")
                self.memory.remember_agent_output("planner_status", "Unrecognized command received.")
                return unrec_msg
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            critical_err_msg = f"PlannerAgent CRITICAL ERROR in run method: {type(e).__name__} - {str(e)}"
            print(f"[PlannerAgent] {critical_err_msg}\n{error_details}")
            self.memory.remember_agent_output("planner_status", f"Critical error in Planner: {str(e)}")
            # If an associated_goal_id was involved and an error occurred before plan creation or during it.
            if associated_goal_id_for_new_plan:
                self.memory.update_goal_status(associated_goal_id_for_new_plan, "autonomous_failed", result_summary=f"Planner critical error: {e}")
            return critical_err_msg

