# agents/planner_agent.py
import ollama
import re
import json
from datetime import datetime
import os
from typing import List, Dict, Any, Optional, Tuple

from .tools.base_tool import Tool, ToolException
from .sentinel_agent import SentinelException

class PlannerAgent:
    """
    A specialized agent that focuses on task decomposition, workflow management,
    and coordinating multi-stage goals. The Planner creates structured plans, 
    tracks progress, and can delegate subtasks to other agents.
    """
    
    def __init__(self, config, memory, quartermaster, sentinel, tools=None, agents=None):
        self.config = config
        self.memory = memory
        self.qm = quartermaster
        self.sentinel = sentinel
        self.tools = tools if tools else []
        self.agents = agents if agents else {}  # Dictionary of agent_name: agent_instance
        model_conf = config.get("models", {})
        # Default to using the analyst model if no specific planner model
        self.model_name = model_conf.get("planner", model_conf.get("analyst", model_conf.get("default", "llama2")))
        
        # Plan tracking directory and files
        self.plans_dir = "plans"
        try:
            # Create the plans directory if it doesn't exist
            self.qm.list_files(self.plans_dir)
        except:
            # If this fails, the directory might not exist - we'll create it when needed
            pass
        
        # Plan tracking
        self.plans = {}
        self.current_plan_id = None
        
        # Load existing plans if available
        self._load_plans()
    
    def _load_plans(self):
        """Load existing plans from the plans directory"""
        try:
            plans_index_file = f"{self.plans_dir}/plans_index.json"
            content = self.qm.read_file(plans_index_file)
            self.plans = json.loads(content)
            print(f"[PlannerAgent] Loaded {len(self.plans)} existing plans")
        except Exception as e:
            print(f"[PlannerAgent] No existing plans index found or error loading: {e}")
            self.plans = {}
    
    def _save_plans(self):
        """Save plans index to file"""
        try:
            # Ensure plans directory exists
            try:
                self.qm.list_files(self.plans_dir)
            except:
                # Create directory if it doesn't exist
                os.makedirs(os.path.join(self.qm.output_dir, self.plans_dir), exist_ok=True)
            
            plans_index_file = f"{self.plans_dir}/plans_index.json"
            self.qm.write_file(plans_index_file, json.dumps(self.plans, indent=2))
        except Exception as e:
            print(f"[PlannerAgent] Error saving plans index: {e}")
    
    def _generate_plan_id(self, title):
        """Generate a unique ID for a plan"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        title_slug = re.sub(r'[^a-zA-Z0-9]', '_', title.lower())[:20]
        return f"plan_{timestamp}_{title_slug}"
    
    def _generate_task_id(self, plan_id, task_index):
        """Generate a unique ID for a task within a plan"""
        return f"{plan_id}_task_{task_index}"
    
    def _call_llm(self, prompt: str):
        """Call the LLM with the given prompt"""
        print(f"\n[PlannerAgent] Sending prompt to LLM (model: {self.model_name}):\n{prompt[:500]}...\n-----------------------------")
        ollama_response = ollama.generate(model=self.model_name, prompt=prompt)
        llm_output = ollama_response['response'].strip()
        print(f"[PlannerAgent] LLM Output (truncated):\n{llm_output[:500]}...\n-----------------------------")
        return llm_output
    
    def _get_agent_capabilities_description(self):
        """Generate a description of all available agents for the LLM prompt"""
        if not self.agents:
            return "No specialized agents are currently available for task delegation."
        
        descriptions = ["\nAVAILABLE AGENTS FOR DELEGATION:"]
        for agent_name, agent in self.agents.items():
            agent_class = agent.__class__.__name__
            # Extract the docstring or provide a default description
            doc = agent.__doc__ or f"{agent_class} - No description available"
            # Format the docstring - take just the first line
            description = doc.strip().split('\n')[0]
            descriptions.append(f"- {agent_name}: {description}")
        return "\n".join(descriptions)
    
    def _get_tool_descriptions_for_llm(self):
        """Generate descriptions of available tools for the LLM prompt"""
        if not self.tools:
            return "No tools are currently available to you for direct use."
        
        descriptions = ["\nAVAILABLE TOOLS:"]
        for tool in self.tools:
            descriptions.append(f"- {tool.get_full_description_for_llm()}")
        return "\n".join(descriptions)
    
    def _generate_planning_prompt(self, goal, context="", mode="create"):
        """Generate a prompt for the LLM based on the planning mode"""
        # Gather available contexts
        agent_capabilities = self._get_agent_capabilities_description()
        tool_descriptions = self._get_tool_descriptions_for_llm()
        
        # System prompt
        system_prompt = (
            "You are a Planner Agent, specialized in breaking down complex goals into clear, achievable tasks. "
            "Your expertise is in task decomposition, dependency management, and orchestrating workflows "
            "across multiple specialized agents.\n\n"
        )
        
        # Context includes any existing plan details if provided
        plan_context = f"\nCURRENT CONTEXT:\n{context}\n" if context else ""
        
        # Mode-specific instructions
        if mode == "create":
            mode_instructions = (
                "\nPLAN CREATION MODE:\n"
                "1. Analyze the goal to understand its full scope and requirements\n"
                "2. Break down the goal into 3-7 sequential and concrete tasks\n"
                "3. For each task, specify:\n"
                "   - A clear, actionable description\n"
                "   - Which agent would be best suited to execute it\n"
                "   - Estimated complexity (Low, Medium, High)\n"
                "   - Any dependencies on other tasks (by task number)\n"
                "4. Format your plan as a JSON object with the following structure:\n"
                "```json\n"
                "{\n"
                "  \"plan_title\": \"Descriptive title for the plan\",\n"
                "  \"plan_description\": \"Brief overview of what this plan accomplishes\",\n"
                "  \"tasks\": [\n"
                "    {\n"
                "      \"task_description\": \"Clear description of the task\",\n"
                "      \"assigned_agent\": \"agent_name\",\n"
                "      \"complexity\": \"Low/Medium/High\",\n"
                "      \"dependencies\": [1, 2]  // Task numbers (1-indexed) this task depends on\n"
                "    },\n"
                "    // Additional tasks...\n"
                "  ]\n"
                "}\n"
                "```\n"
            )
        elif mode == "update":
            mode_instructions = (
                "\nPLAN UPDATE MODE:\n"
                "1. Review the current plan and progress\n"
                "2. Consider any new information or changes in requirements\n"
                "3. Recommend one of the following actions:\n"
                "   - Continue with the current plan (if it remains valid)\n"
                "   - Modify specific tasks (specify which ones and how)\n"
                "   - Add new tasks (provide full details as in plan creation)\n"
                "   - Remove or replace tasks (specify which ones and why)\n"
                "4. Format your update recommendation as a JSON object with the following structure:\n"
                "```json\n"
                "{\n"
                "  \"update_type\": \"continue/modify/add/remove\",\n"
                "  \"update_reason\": \"Brief explanation of why this update is needed\",\n"
                "  \"updates\": [  // Only for modify/add/remove\n"
                "    {\n"
                "      \"task_index\": 2,  // For modify/remove; omit for add\n"
                "      \"action\": \"modify/add/remove\",\n"
                "      \"new_task\": {  // Only for modify/add\n"
                "        \"task_description\": \"Updated task description\",\n"
                "        \"assigned_agent\": \"agent_name\",\n"
                "        \"complexity\": \"Low/Medium/High\",\n"
                "        \"dependencies\": [1]  // Task numbers this task depends on\n"
                "      }\n"
                "    },\n"
                "    // Additional updates...\n"
                "  ]\n"
                "}\n"
                "```\n"
            )
        elif mode == "assess":
            mode_instructions = (
                "\nPLAN ASSESSMENT MODE:\n"
                "1. Review the current plan, progress, and any execution results\n"
                "2. Assess whether each completed task has met its objectives\n"
                "3. Identify any issues, obstacles, or unexpected outcomes\n"
                "4. Determine if the plan remains viable or needs adjustment\n"
                "5. Provide an assessment report with the following sections:\n"
                "   - Overall Progress (percentage complete, on track/behind)\n"
                "   - Task-by-Task Assessment (brief evaluation of each task)\n"
                "   - Issues and Risks (any concerns or potential problems)\n"
                "   - Recommendations (continue as planned, adjust specific tasks, or revise plan)\n"
            )
        else:  # Default to execution guidance
            mode_instructions = (
                "\nPLAN EXECUTION GUIDANCE MODE:\n"
                "1. Review the current plan and its next pending task\n"
                "2. Provide detailed guidance on how to execute this specific task\n"
                "3. Consider any relevant context from previously completed tasks\n"
                "4. Suggest specific tools, approaches, or techniques the assigned agent should use\n"
                "5. Format your guidance as clear, actionable instructions\n"
            )
        
        # Combine all components
        full_prompt = (
            f"{system_prompt}{plan_context}{agent_capabilities}\n{tool_descriptions}\n{mode_instructions}\n\n"
            f"GOAL: {goal}\n\n"
            "Based on the above, please provide your planning output in the requested format."
        )
        
        return full_prompt
    
    def create_plan(self, goal_description):
        """Create a new plan for a goal"""
        # Generate a prompt for plan creation
        creation_prompt = self._generate_planning_prompt(goal_description, mode="create")
        
        # Get the plan from the LLM
        llm_output = self._call_llm(creation_prompt)
        
        # Extract the JSON plan
        try:
            # Look for JSON in the output
            json_match = re.search(r'```json\s*(.*?)```', llm_output, re.DOTALL)
            if json_match:
                plan_json = json_match.group(1).strip()
            else:
                # If no JSON code block, try to find JSON directly
                json_start = llm_output.find('{')
                json_end = llm_output.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    plan_json = llm_output[json_start:json_end]
                else:
                    raise ValueError("No valid JSON found in LLM output")
            
            plan_data = json.loads(plan_json)
            
            # Validate the plan structure
            if not isinstance(plan_data, dict):
                raise ValueError("Plan must be a dictionary")
            if "plan_title" not in plan_data:
                raise ValueError("Plan must have a title")
            if "tasks" not in plan_data or not isinstance(plan_data["tasks"], list):
                raise ValueError("Plan must have a list of tasks")
            
            # Generate a plan ID
            plan_id = self._generate_plan_id(plan_data["plan_title"])
            
            # Build the complete plan object
            plan = {
                "id": plan_id,
                "title": plan_data["plan_title"],
                "description": plan_data.get("plan_description", ""),
                "created": datetime.now().isoformat(),
                "updated": datetime.now().isoformat(),
                "status": "active",
                "progress": 0.0,
                "tasks": [],
                "original_goal": goal_description,
                "plan_file": f"{self.plans_dir}/{plan_id}.json"
            }
            
            # Process tasks
            for i, task_data in enumerate(plan_data["tasks"]):
                task_id = self._generate_task_id(plan_id, i+1)
                task = {
                    "id": task_id,
                    "index": i+1,
                    "description": task_data["task_description"],
                    "assigned_agent": task_data["assigned_agent"],
                    "complexity": task_data.get("complexity", "Medium"),
                    "dependencies": task_data.get("dependencies", []),
                    "status": "pending",
                    "result": None,
                    "started": None,
                    "completed": None
                }
                plan["tasks"].append(task)
            
            # Save the plan
            self.plans[plan_id] = plan
            self._save_plans()
            
            # Save the detailed plan to its own file
            self.qm.write_file(plan["plan_file"], json.dumps(plan, indent=2))
            
            # Set as the current plan
            self.current_plan_id = plan_id
            
            # Add goals to the memory system for each task
            for task in plan["tasks"]:
                self.memory.add_goal(
                    f"[Plan: {plan['title']}] {task['description']}",
                    agent=task['assigned_agent'].capitalize()
                )
            
            # Format a human-readable version of the plan
            return self._format_plan_for_display(plan_id)
        
        except json.JSONDecodeError as e:
            print(f"[PlannerAgent] JSON decode error: {e}")
            return f"Planning agent error: {str(e)}""Error creating plan: Invalid JSON format in LLM output.\n\nRaw output:\n{llm_output}"
        except ValueError as e:
            print(f"[PlannerAgent] Value error: {e}")
            return f"Error creating plan: {str(e)}.\n\nRaw output:\n{llm_output}"
        except Exception as e:
            print(f"[PlannerAgent] Error creating plan: {e}")
            return f"Error creating plan: {str(e)}"
    
    def _format_plan_for_display(self, plan_id):
        """Format a plan for human-readable display"""
        if plan_id not in self.plans:
            return f"Plan {plan_id} not found."
        
        plan = self.plans[plan_id]
        
        # Header
        output = [
            f"# {plan['title']}",
            f"Plan ID: {plan_id}",
            f"Status: {plan['status'].upper()}",
            f"Progress: {int(plan['progress'] * 100)}%",
            f"Goal: {plan['original_goal']}",
            f"Description: {plan['description']}",
            "\n## Tasks:"
        ]
        
        # Tasks
        for task in plan["tasks"]:
            status_marker = "✅" if task["status"] == "completed" else "⏳" if task["status"] == "in_progress" else "🔲"
            dependencies = f"Dependencies: Tasks {', '.join(map(str, task['dependencies']))}" if task["dependencies"] else "No dependencies"
            
            task_entry = [
                f"{status_marker} Task {task['index']}: {task['description']}",
                f"   Agent: {task['assigned_agent']}, Complexity: {task['complexity']}, Status: {task['status'].upper()}",
                f"   {dependencies}"
            ]
            
            if task["result"]:
                task_entry.append(f"   Result: {task['result'][:100]}..." if len(task["result"]) > 100 else f"   Result: {task['result']}")
            
            output.extend(task_entry)
            output.append("")
        
        return "\n".join(output)
    
    def update_task_status(self, plan_id, task_index, new_status, result=None):
        """Update the status of a task in a plan"""
        if plan_id not in self.plans:
            return f"Plan {plan_id} not found."
        
        plan = self.plans[plan_id]
        
        # Find the task
        task = None
        for t in plan["tasks"]:
            if t["index"] == task_index:
                task = t
                break
        
        if not task:
            return f"Task {task_index} not found in plan {plan_id}."
        
        # Update the task
        old_status = task["status"]
        task["status"] = new_status
        
        if new_status == "in_progress" and not task["started"]:
            task["started"] = datetime.now().isoformat()
        
        if new_status == "completed" and not task["completed"]:
            task["completed"] = datetime.now().isoformat()
        
        if result:
            task["result"] = result
        
        # Update the plan's progress
        total_tasks = len(plan["tasks"])
        completed_tasks = sum(1 for t in plan["tasks"] if t["status"] == "completed")
        in_progress_tasks = sum(1 for t in plan["tasks"] if t["status"] == "in_progress")
        
        # Each completed task counts as 1.0, each in-progress as 0.5
        plan["progress"] = (completed_tasks + (in_progress_tasks * 0.5)) / total_tasks
        
        # If all tasks are completed, mark the plan as completed
        if completed_tasks == total_tasks:
            plan["status"] = "completed"
        
        # Update timestamps
        plan["updated"] = datetime.now().isoformat()
        
        # Save the updated plan
        self.plans[plan_id] = plan
        self._save_plans()
        self.qm.write_file(plan["plan_file"], json.dumps(plan, indent=2))
        
        # Mark the corresponding goal as completed if task is completed
        if new_status == "completed":
            for i, goal in enumerate(self.memory.get_goals_list()):
                if f"[Plan: {plan['title']}]" in goal.get('task', '') and f"Task {task_index}:" in goal.get('task', ''):
                    self.memory.complete_goal(i + 1)
                    break
        
        return f"Updated task {task_index} status from '{old_status}' to '{new_status}' in plan {plan_id}."
    
    def get_next_task(self, plan_id):
        """Get the next executable task in a plan"""
        if plan_id not in self.plans:
            return None, f"Plan {plan_id} not found."
        
        plan = self.plans[plan_id]
        
        if plan["status"] == "completed":
            return None, f"Plan {plan_id} is already completed."
        
        # Find tasks that are pending and have all dependencies met
        executable_tasks = []
        for task in plan["tasks"]:
            if task["status"] != "pending":
                continue
            
            # Check if all dependencies are completed
            dependencies_met = True
            for dep_idx in task["dependencies"]:
                dep_task = next((t for t in plan["tasks"] if t["index"] == dep_idx), None)
                if not dep_task or dep_task["status"] != "completed":
                    dependencies_met = False
                    break
            
            if dependencies_met:
                executable_tasks.append(task)
        
        if not executable_tasks:
            # Check if there are any in-progress tasks
            in_progress_tasks = [t for t in plan["tasks"] if t["status"] == "in_progress"]
            if in_progress_tasks:
                return in_progress_tasks[0], f"Task {in_progress_tasks[0]['index']} is currently in progress."
            else:
                # Check if there are any pending tasks with unmet dependencies
                pending_tasks = [t for t in plan["tasks"] if t["status"] == "pending"]
                if pending_tasks:
                    blocked_task = pending_tasks[0]
                    missing_deps = []
                    for dep_idx in blocked_task["dependencies"]:
                        dep_task = next((t for t in plan["tasks"] if t["index"] == dep_idx), None)
                        if not dep_task or dep_task["status"] != "completed":
                            missing_deps.append(dep_idx)
                    
                    return None, f"No executable tasks available. Task {blocked_task['index']} is blocked by incomplete dependencies: Tasks {', '.join(map(str, missing_deps))}."
                else:
                    # This shouldn't happen if the plan isn't completed
                    return None, "No pending tasks found, but plan is not marked as completed."
        
        # Return the first executable task (usually the one with the lowest index)
        executable_tasks.sort(key=lambda t: t["index"])
        return executable_tasks[0], f"Task {executable_tasks[0]['index']} is ready to execute."
    
    def delegate_task(self, plan_id, task_index):
        """Delegate a task to the assigned agent"""
        if plan_id not in self.plans:
            return f"Plan {plan_id} not found."
        
        plan = self.plans[plan_id]
        
        # Find the task
        task = None
        for t in plan["tasks"]:
            if t["index"] == task_index:
                task = t
                break
        
        if not task:
            return f"Task {task_index} not found in plan {plan_id}."
        
        # Check if task is executable
        if task["status"] != "pending":
            return f"Task {task_index} is not pending (current status: {task['status']})."
        
        # Check dependencies
        for dep_idx in task["dependencies"]:
            dep_task = next((t for t in plan["tasks"] if t["index"] == dep_idx), None)
            if not dep_task or dep_task["status"] != "completed":
                return f"Task {task_index} cannot be executed because it depends on incomplete task {dep_idx}."
        
        # Get the assigned agent
        agent_name = task["assigned_agent"].lower()
        if agent_name not in self.agents:
            return f"Agent '{agent_name}' not found. Available agents: {', '.join(self.agents.keys())}."
        
        agent = self.agents[agent_name]
        
        # Update task status to in_progress
        self.update_task_status(plan_id, task_index, "in_progress")
        
        # Generate context for the agent from completed dependencies
        context = []
        for dep_idx in task["dependencies"]:
            dep_task = next((t for t in plan["tasks"] if t["index"] == dep_idx), None)
            if dep_task and dep_task["status"] == "completed" and dep_task["result"]:
                context.append(f"Result from Task {dep_idx}: {dep_task['result']}")
        
        context_str = "\n\n".join(context)
        
        # Prepare the task command with context
        task_command = f"[Plan: {plan['title']}] Task {task_index}: {task['description']}"
        if context_str:
            task_command += f"\n\nContext from dependencies:\n{context_str}"
        
        # Execute the task with the agent
        try:
            print(f"[PlannerAgent] Delegating task {task_index} to {agent_name} agent")
            result = agent.run(task_command)
            
            # Update task status to completed
            self.update_task_status(plan_id, task_index, "completed", result=result)
            
            return f"Task {task_index} completed by {agent_name} agent.\n\nResult:\n{result}"
        except Exception as e:
            print(f"[PlannerAgent] Error delegating task: {e}")
            return f"Error executing task {task_index} with {agent_name} agent: {str(e)}"
    
    def assess_plan(self, plan_id):
        """Assess the current state of a plan"""
        if plan_id not in self.plans:
            return f"Plan {plan_id} not found."
        
        plan = self.plans[plan_id]
        
        # Generate a prompt for plan assessment
        assessment_prompt = self._generate_planning_prompt(
            plan["original_goal"],
            context=json.dumps(plan, indent=2),
            mode="assess"
        )
        
        # Get the assessment from the LLM
        assessment = self._call_llm(assessment_prompt)
        
        # Save the assessment to the plan
        plan["last_assessment"] = {
            "timestamp": datetime.now().isoformat(),
            "content": assessment
        }
        self.plans[plan_id] = plan
        self._save_plans()
        self.qm.write_file(plan["plan_file"], json.dumps(plan, indent=2))
        
        return f"Plan Assessment for {plan_id}:\n\n{assessment}"
    
    def list_plans(self):
        """List all plans"""
        if not self.plans:
            return "No plans found."
        
        plans_list = []
        for plan_id, plan in self.plans.items():
            status_emoji = "✅" if plan["status"] == "completed" else "🔄"
            progress = int(plan["progress"] * 100)
            plans_list.append(f"{status_emoji} [{plan_id}] {plan['title']} - {progress}% complete")
        
        return "Available Plans:\n" + "\n".join(plans_list)
    
    def run(self, command):
        """Main entry point for the Planner Agent"""
        print(f"[PlannerAgent] Processing command: {command[:100]}...")
        
        try:
            # Parse the command to determine the action
            cmd_lower = command.lower()
            
            # Create a new plan
            if "create plan" in cmd_lower or "make plan" in cmd_lower or "new plan" in cmd_lower:
                # Extract the goal description
                goal_match = re.search(r'(?:create|make|new) plan (?:for|about|to) (.+)', command, re.IGNORECASE)
                if goal_match:
                    goal = goal_match.group(1).strip()
                    return self.create_plan(goal)
                else:
                    # If no clear goal is found in the pattern, use the whole command minus the "create plan" part
                    goal = re.sub(r'(?:create|make|new) plan(?:\s+for|\s+about|\s+to)?', '', command, flags=re.IGNORECASE).strip()
                    if goal:
                        return self.create_plan(goal)
                    else:
                        return "Please provide a goal for the plan. Example: 'Create plan for building a personal website'"
            
            # List plans
            if "list plans" in cmd_lower or "show plans" in cmd_lower:
                return self.list_plans()
            
            # Show plan details
            if "show plan" in cmd_lower or "plan details" in cmd_lower:
                # Extract the plan ID
                plan_id_match = re.search(r'(?:show|view|get) plan (?:details for )?([\w\d_]+)', command, re.IGNORECASE)
                if plan_id_match:
                    plan_id = plan_id_match.group(1).strip()
                    return self._format_plan_for_display(plan_id)
                else:
                    # If no plan ID is specified and there's a current plan, show that
                    if self.current_plan_id:
                        return self._format_plan_for_display(self.current_plan_id)
                    else:
                        return "Please specify which plan to show. Example: 'Show plan details for plan_20220101_123456'"
            
            # Update task status
            if "update task" in cmd_lower or "mark task" in cmd_lower:
                # Extract plan ID, task index, and new status
                update_match = re.search(
                    r'(?:update|mark) task (\d+) (?:in plan )?([\w\d_]+)? (?:as|to) (completed|in progress|pending)',
                    command,
                    re.IGNORECASE
                )
                if update_match:
                    task_index = int(update_match.group(1))
                    plan_id = update_match.group(2)
                    new_status = update_match.group(3).lower().replace(' ', '_')
                    
                    # If plan ID is not specified, use the current plan
                    if not plan_id and self.current_plan_id:
                        plan_id = self.current_plan_id
                    
                    if plan_id:
                        return self.update_task_status(plan_id, task_index, new_status)
                    else:
                        return "Please specify which plan to update. Example: 'Update task 2 in plan plan_20220101_123456 as completed'"
            
            # Get next task
            if "next task" in cmd_lower:
                # Extract the plan ID if specified
                plan_id_match = re.search(r'next task (?:in plan |for )?([\w\d_]+)?', command, re.IGNORECASE)
                plan_id = None
                if plan_id_match:
                    plan_id = plan_id_match.group(1)
                
                # If plan ID is not specified, use the current plan
                if not plan_id and self.current_plan_id:
                    plan_id = self.current_plan_id
                
                if plan_id:
                    task, message = self.get_next_task(plan_id)
                    if task:
                        return f"Next task in plan {plan_id}:\n\nTask {task['index']}: {task['description']}\nAssigned to: {task['assigned_agent']}\nStatus: {task['status']}\n\n{message}"
                    else:
                        return message
                else:
                    return "Please specify which plan to check. Example: 'Next task in plan plan_20220101_123456'"
            
            # Execute next task
            if "execute task" in cmd_lower or "run task" in cmd_lower or "do task" in cmd_lower:
                # Extract plan ID and task index if specified
                task_match = re.search(
                    r'(?:execute|run|do) task (\d+)?(?: in plan| for plan)? ?([\w\d_]+)?',
                    command,
                    re.IGNORECASE
                )
                
                plan_id = None
                task_index = None
                
                if task_match:
                    # Both task index and plan ID might be optional
                    if task_match.group(1):
                        task_index = int(task_match.group(1))
                    if task_match.group(2):
                        plan_id = task_match.group(2)
                
                # If plan ID is not specified, use the current plan
                if not plan_id and self.current_plan_id:
                    plan_id = self.current_plan_id
                
                if not plan_id:
                    return "Please specify which plan to execute a task from. Example: 'Execute task 2 in plan plan_20220101_123456'"
                
                # If task index is specified, execute that task
                if task_index:
                    return self.delegate_task(plan_id, task_index)
                else:
                    # Get the next executable task
                    task, message = self.get_next_task(plan_id)
                    if task:
                        return self.delegate_task(plan_id, task["index"])
                    else:
                        return message
            
            # Assess plan
            if "assess plan" in cmd_lower or "evaluate plan" in cmd_lower:
                # Extract the plan ID
                plan_id_match = re.search(r'(?:assess|evaluate) plan ?([\w\d_]+)?', command, re.IGNORECASE)
                plan_id = None
                if plan_id_match:
                    plan_id = plan_id_match.group(1)
                
                # If plan ID is not specified, use the current plan
                if not plan_id and self.current_plan_id:
                    plan_id = self.current_plan_id
                
                if plan_id:
                    return self.assess_plan(plan_id)
                else:
                    return "Please specify which plan to assess. Example: 'Assess plan plan_20220101_123456'"
            
            # Set current plan
            if "use plan" in cmd_lower or "switch to plan" in cmd_lower or "set current plan" in cmd_lower:
                # Extract the plan ID
                plan_id_match = re.search(r'(?:use|switch to|set current) plan ([\w\d_]+)', command, re.IGNORECASE)
                if plan_id_match:
                    plan_id = plan_id_match.group(1).strip()
                    if plan_id in self.plans:
                        self.current_plan_id = plan_id
                        return f"Current plan set to: {plan_id} - {self.plans[plan_id]['title']}"
                    else:
                        return f"Plan {plan_id} not found."
                else:
                    return "Please specify which plan to use. Example: 'Use plan plan_20220101_123456'"
            
            # Execute next task in current plan (simplified command)
            if "execute next task" in cmd_lower or "run next task" in cmd_lower or "do next task" in cmd_lower:
                if self.current_plan_id:
                    task, message = self.get_next_task(self.current_plan_id)
                    if task:
                        return self.delegate_task(self.current_plan_id, task["index"])
                    else:
                        return message
                else:
                    return "No current plan is set. Please specify a plan or create one first."
            
            # Default to creating a new plan if no specific command is recognized
            return self.create_plan(command)
        
        except SentinelException as se:
            print(f"[PlannerAgent] Action blocked by Sentinel: {se}")
            return f"Planning action blocked: {se}"
        except ToolException as te:
            print(f"[PlannerAgent] Tool error: {te}")
            return f"Planning tool error: {te}"
        except Exception as e:
            print(f"[PlannerAgent] Error processing command: {e}")
            import traceback
            print(traceback.format_exc())
            return f