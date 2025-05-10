# agents/planner_agent.py
import ollama
import re
import json
from datetime import datetime
import os
from typing import List, Dict, Any, Optional, Tuple
import time # Import time for delays

from .tools.base_tool import Tool, ToolException
from .sentinel_agent import SentinelException

class PlannerAgent:
    # ... (existing __init__ and other methods: _load_plans, _save_plans, _generate_plan_id, etc.) ...
    # Add message_bus_client if you want to use MessageBus for alerts
    def __init__(self, config, memory, quartermaster, sentinel, tools=None, agents=None, message_bus_client=None):
        self.config = config
        self.memory = memory
        self.qm = quartermaster
        self.sentinel = sentinel
        self.tools = tools if tools else []
        self.agents = agents if agents else {}
        model_conf = config.get("models", {})
        self.model_name = model_conf.get("planner", model_conf.get("analyst", model_conf.get("default", "llama2")))
        self.plans_dir = "plans"
        try:
            self.qm.list_files(self.plans_dir)
        except:
            pass # Directory will be created if it doesn't exist by QM or plan saving
        self.plans = {}
        self.current_plan_id = None
        self._load_plans()
        self.message_bus_client = message_bus_client # Store the message bus client

    # ... (existing _get_agent_capabilities_description, _get_tool_descriptions_for_llm, _generate_planning_prompt) ...
    # ... (existing create_plan, _format_plan_for_display, update_task_status, get_next_task, delegate_task, assess_plan, list_plans) ...

    def _execute_plan_loop(self, plan_id: str) -> str:
        """
        Internal loop to automatically execute tasks in a plan.
        Returns a summary log of the execution.
        """
        if plan_id not in self.plans:
            return f"Error: Plan {plan_id} not found for execution loop."

        plan = self.plans[plan_id]
        plan_title = plan.get('title', 'Untitled Plan')
        execution_log = [f"Starting automated execution of plan: {plan_title} ({plan_id})"]
        print(f"[PlannerAgent] Starting automated execution of plan: {plan_id} - '{plan_title}'")
        self.memory.remember_agent_output("planner_status", f"Plan {plan_id}: Execution started for '{plan_title}'.")

        # Safety break for very long or stuck loops
        max_steps = len(plan.get("tasks", [])) * 2 + 5 # Allow for retries or additional small steps
        current_step_count = 0

        while current_step_count < max_steps:
            current_step_count += 1
            # Refresh plan data in case it was modified externally (though unlikely in this single-threaded agent model)
            plan = self.plans[plan_id]

            if plan["status"] == "completed":
                completion_message = f"Plan {plan_id} '{plan_title}' is marked as completed."
                print(f"[PlannerAgent] {completion_message}")
                execution_log.append(completion_message)
                if self.message_bus_client:
                    self.message_bus_client.broadcast(f"Plan '{plan_title}' ({plan_id}) has completed!", context=f"plan_completion:{plan_id}")
                else:
                    print(f"ALERT: Plan '{plan_title}' ({plan_id}) COMPLETED.")
                self.memory.remember_agent_output("planner_status", f"Plan {plan_id}: '{plan_title}' completed.")
                return "\n".join(execution_log)

            task, message = self.get_next_task(plan_id)

            if not task:
                # No task is ready or plan is finished.
                # get_next_task returns None if plan is completed or stuck.
                # update_task_status is responsible for setting plan to "completed".
                if plan["status"] == "completed": # Double check status
                    final_message = f"Plan {plan_id} '{plan_title}' successfully completed."
                    print(f"[PlannerAgent] {final_message}")
                    execution_log.append(final_message)
                    if self.message_bus_client:
                        self.message_bus_client.broadcast(f"Plan '{plan_title}' ({plan_id}) has completed!", context=f"plan_completion:{plan_id}")
                    else:
                        print(f"ALERT: Plan '{plan_title}' ({plan_id}) COMPLETED.")
                    self.memory.remember_agent_output("planner_status", f"Plan {plan_id}: '{plan_title}' completed.")
                else:
                    stuck_message = f"Plan {plan_id} '{plan_title}': No further executable tasks. Status: {plan['status']}. Reason: {message}"
                    print(f"[PlannerAgent] {stuck_message}")
                    execution_log.append(stuck_message)
                    self.memory.remember_agent_output("planner_status", f"Plan {plan_id}: '{plan_title}' execution stalled. {message}")
                return "\n".join(execution_log) # Exit loop

            task_log_desc = f"Task {task['index']} ('{task['description'][:40]}...')"
            log_msg = f"[PlannerAgent] Plan {plan_id}: Executing {task_log_desc} assigned to {task['assigned_agent']}"
            print(log_msg)
            execution_log.append(f"Attempting to execute {task_log_desc} with agent {task['assigned_agent']}.")
            self.memory.remember_agent_output("planner_status", f"Plan {plan_id}: Executing {task_log_desc} via {task['assigned_agent']}.")

            try:
                # delegate_task should handle:
                # 1. Updating task status to "in_progress".
                # 2. Calling the sub-agent.
                # 3. Updating task status to "completed" (with result) or "error" on sub-agent failure.
                # 4. Updating overall plan progress and status (if all tasks done).
                task_result_output = self.delegate_task(plan_id, task["index"])

                execution_log.append(f"Task {task['index']} delegated. Agent '{task['assigned_agent']}' output snippet: {str(task_result_output)[:100]}...")
                print(f"[PlannerAgent] Plan {plan_id}: Task {task['index']} processed by {task['assigned_agent']}.")
                self.memory.remember_agent_output("planner_status", f"Plan {plan_id}: Task {task['index']} processed by {task['assigned_agent']}.")
                
                time.sleep(1) # Small delay for readability or to allow other system events

            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                error_message = f"CRITICAL ERROR in PlannerAgent while trying to delegate task {task['index']} of plan {plan_id}: {str(e)}"
                print(f"[PlannerAgent] {error_message}\n{error_details}")
                # Ensure task is marked as error if delegate_task didn't catch it
                self.update_task_status(plan_id, task["index"], "error", result=f"Planner delegation error: {str(e)}")
                execution_log.append(error_message)
                execution_log.append(f"Plan {plan_id} '{plan_title}' execution halted due to critical error in task {task['index']}.")
                self.memory.remember_agent_output("planner_status", f"Plan {plan_id}: Critical error in task {task['index']}. Execution halted.")
                return "\n".join(execution_log) # Halt on critical error

            # After a task is processed, check overall plan status again explicitly.
            # This is because update_task_status (called within delegate_task) should set the plan to 'completed'
            # if that was the last task.
            if self.plans[plan_id]["status"] == "completed":
                final_completion_message = f"Plan {plan_id} '{plan_title}' fully completed after task {task['index']}."
                print(f"[PlannerAgent] {final_completion_message}")
                execution_log.append(final_completion_message)
                self.memory.remember_agent_output("planner_status", f"Plan {plan_id}: '{plan_title}' all tasks completed.")
                if self.message_bus_client:
                     self.message_bus_client.broadcast(f"Plan '{plan_title}' ({plan_id}) has completed!", context=f"plan_completion:{plan_id}")
                else:
                    print(f"ALERT: Plan '{plan_title}' ({plan_id}) COMPLETED.")
                return "\n".join(execution_log)
        
        if current_step_count >= max_steps:
            timeout_msg = f"Plan {plan_id} '{plan_title}' execution loop reached max steps ({max_steps}). Halting to prevent infinite loop. Current status: {plan['status']}."
            print(f"[PlannerAgent] {timeout_msg}")
            execution_log.append(timeout_msg)
            self.memory.remember_agent_output("planner_status", f"Plan {plan_id}: Execution timed out (max steps).")
            
        return "\n".join(execution_log)


    def run(self, command: str) -> str:
        print(f"[PlannerAgent] Processing command: {command[:100]}...")
        cmd_lower = command.lower().strip()

        # Command: Create and Execute Plan
        if "create and execute plan" in cmd_lower or \
           "create plan and run" in cmd_lower or \
           "start main plan for" in cmd_lower: # Renamed for clarity
            goal_match = re.search(r'(?:create and execute plan|create plan and run|start main plan) (?:for|about|to) (.+)', command, re.IGNORECASE)
            goal = ""
            if goal_match:
                goal = goal_match.group(1).strip()
            else:
                goal = re.sub(r'(?:create and execute plan|create plan and run|start main plan)(?:\s+for|\s+about|\s+to)?', '', command, flags=re.IGNORECASE).strip()

            if not goal:
                return "Please provide a goal for the plan. Example: 'Planner, create and execute plan for launching a new product'"
            
            print(f"[PlannerAgent] Request to create and execute plan for: {goal}")
            try:
                plan_summary_or_error = self.create_plan(goal) # create_plan sets self.current_plan_id
                if "Error creating plan" in plan_summary_or_error or not self.current_plan_id:
                    return f"Failed to create plan for execution: {plan_summary_or_error}"
                
                print(f"[PlannerAgent] Plan created: {self.current_plan_id}. Now starting execution loop.")
                plan_execution_log = self._execute_plan_loop(self.current_plan_id)
                return f"Initial Plan Summary:\n{plan_summary_or_error}\n\nPlan Execution Log:\n{plan_execution_log}"
            except Exception as e:
                import traceback
                return f"Error during 'create and execute plan': {str(e)}\n{traceback.format_exc()}"

        # Command: Execute an Existing Plan by ID
        elif "execute plan" in cmd_lower or "run plan" in cmd_lower or "start executing plan" in cmd_lower:
            plan_id_match = re.search(r'(?:execute|run|start executing) plan ([\w\d_]+)', command, re.IGNORECASE)
            if plan_id_match:
                plan_id = plan_id_match.group(1).strip()
                if plan_id not in self.plans:
                    return f"Plan '{plan_id}' not found for execution."
                self.current_plan_id = plan_id # Ensure current_plan_id is set
                print(f"[PlannerAgent] Request to execute existing plan: {plan_id}")
                return self._execute_plan_loop(plan_id)
            else:
                return "Please specify the Plan ID to execute. Example: 'Planner, execute plan plan_xxxxxxxx_xxxx'"

        # ... (Keep your existing commands for list_plans, show plan, update task, get_next_task (manual), delegate_task (manual), assess_plan, create_plan (manual), set current plan) ...
        # Example: Create Plan (Manual - does not auto-execute)
        if "create plan" in cmd_lower or "make plan" in cmd_lower or "new plan" in cmd_lower:
            goal_match = re.search(r'(?:create|make|new) plan (?:for|about|to) (.+)', command, re.IGNORECASE)
            if goal_match:
                goal = goal_match.group(1).strip()
                return self.create_plan(goal)
            else:
                goal = re.sub(r'(?:create|make|new) plan(?:\s+for|\s+about|\s+to)?', '', command, flags=re.IGNORECASE).strip()
                if goal:
                    return self.create_plan(goal)
                else:
                    return "Please provide a goal for the plan. Example: 'Create plan for building a personal website'"
        
        if "list plans" in cmd_lower or "show plans" in cmd_lower:
            return self.list_plans()

        if "show plan" in cmd_lower or "plan details" in cmd_lower:
            plan_id_match = re.search(r'(?:show|view|get) plan (?:details for )?([\w\d_]+)', command, re.IGNORECASE)
            if plan_id_match:
                plan_id = plan_id_match.group(1).strip()
                return self._format_plan_for_display(plan_id)
            elif self.current_plan_id:
                return self._format_plan_for_display(self.current_plan_id)
            else:
                return "Please specify which plan to show or set a current plan."

        # ... (other specific commands like 'next task', 'execute task X', 'update task Y')

        return "PlannerAgent: Command not fully recognized. Try 'create and execute plan for [goal]', 'execute plan [plan_id]', 'create plan for [goal]', 'list plans', etc."