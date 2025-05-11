# app.py
"""
Web interface for WITS CrewAI.
Provides a simple Flask-based web interface for interacting with the WITS CrewAI system.
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory
import os
import json
import time
import threading
import queue
import yaml
import sys
from datetime import datetime
import re
from werkzeug.utils import secure_filename # For file uploads
from typing import List, Dict, Any, Optional, Tuple, Union # Make sure Union is here

# Add the parent directory to the path so we can import from the WITS modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import WITS components
from core.enhanced_memory import EnhancedMemory

# Attempt to import VectorMemory and set a flag
VECTOR_MEMORY_AVAILABLE = False
_ImportedVectorMemory = None # Will hold the class if imported successfully
try:
    from core.vector_memory import VectorMemory as ActualVectorMemory # Use an alias
    _ImportedVectorMemory = ActualVectorMemory # Store the actual imported class
    VECTOR_MEMORY_AVAILABLE = True
    print("[WEB_APP_DEBUG] Successfully imported core.vector_memory.VectorMemory.")
except ImportError:
    print("[WEB_APP_IMPORTERROR] core.vector_memory.VectorMemory not found or import error. Using EnhancedMemory as fallback.")
except Exception as e_vm_import:
    print(f"[WEB_APP_EXCEPTION_IMPORT] Error importing core.vector_memory.VectorMemory: {e_vm_import}. Using EnhancedMemory as fallback.")

from ethics.ethics_rules import EthicsFilter, EthicsViolation
from core.router import route_task
from core.message_bus import MessageBus, MessageBusClient, Message

# Agent classes
from agents.scribe_agent import ScribeAgent
from agents.analyst_agent import AnalystAgent
from agents.engineer_agent import EngineerAgent
from agents.researcher_agent import ResearcherAgent
from agents.planner_agent import PlannerAgent
from agents.quartermaster_agent import QuartermasterAgent, QuartermasterException
from agents.sentinel_agent import SentinelAgent, SentinelException

# Tool classes
from agents.tools.base_tool import ToolException
from agents.tools.calculator_tool import CalculatorTool
from agents.tools.datetime_tool import DateTimeTool
from agents.tools.web_search_tool import WebSearchTool
from agents.tools.file_read_tool import ReadFileTool
from agents.tools.file_list_tool import ListFilesTool
from agents.tools.file_write_tool import WriteFileTool
from agents.tools.weather_tool import WeatherTool
from agents.tools.data_visualization_tool import DataVisualizationTool
from agents.tools.pdf_generator_tool import PdfGeneratorTool

app = Flask(__name__)

# --- Global Variables for Task Processing ---
task_queue = queue.Queue()
task_results: Dict[str, Any] = {}
task_status: Dict[str, str] = {}
command_history: List[Dict[str, Any]] = []

# --- Configuration Loading ---
config: Dict[str, Any] = {}
config_path = "config.yaml"
if os.path.exists(config_path):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
            if config_data and isinstance(config_data, dict):
                config.update(config_data)
                print(f"[WEB_APP_CONFIG] Loaded configuration from '{config_path}'.")
            else:
                print(f"[WEB_APP_CONFIG] Warning: '{config_path}' is empty or not valid YAML. Using defaults.")
    except Exception as e_cfg:
        print(f"[WEB_APP_CONFIG] Error loading '{config_path}': {e_cfg}. Using defaults.")
else:
    print(f"[WEB_APP_CONFIG] Warning: '{config_path}' not found. Using defaults.")

config.setdefault("internet_access", True)
config.setdefault("voice_input", False)
config.setdefault("allow_code_execution", False)
config.setdefault("output_directory", "output")
os.makedirs(config.get("output_directory", "output"), exist_ok=True)
config.setdefault("ethics_enabled", True)
config.setdefault("router", {"fallback_agent": "analyst"})
config.setdefault("models", {
    "default": "llama2", "scribe": "llama2", "analyst": "llama2",
    "engineer": "codellama:7b", "researcher": "llama2", "planner": "llama2"
})
config.setdefault("web_interface", {
    "port": 5000, "host": "0.0.0.0", "debug": True,
    "enable_file_uploads": True, "max_file_size": 5
})

# --- Initialize Memory System ---
# Use a string literal for 'core.vector_memory.VectorMemory' in the type hint
# to prevent NameError if the import failed during a specific reload phase.
# The actual type of 'memory' will be determined by the runtime logic below.
memory: "Optional[Union[EnhancedMemory, Any]]" # Type hint for the 'memory' variable

memory_instance: Union[EnhancedMemory, Any] # Variable to hold the instantiated memory object

if VECTOR_MEMORY_AVAILABLE and _ImportedVectorMemory is not None:
    try:
        memory_instance = _ImportedVectorMemory(memory_file='vector_memory.json', index_file='vector_index.bin')
        print("[WEB_APP] Using Vector Memory system for app.py.")
    except Exception as e_vm_init:
        print(f"[WEB_APP] Vector Memory instantiation failed ({e_vm_init}), falling back to Enhanced Memory for app.py.")
        memory_instance = EnhancedMemory(memory_file='enhanced_memory.json') # Fallback
        print("[WEB_APP] Using Enhanced Memory system (VectorMemory fallback) for app.py.")
else:
    memory_instance = EnhancedMemory(memory_file='enhanced_memory.json')
    print("[WEB_APP] Using Enhanced Memory system for app.py (VectorMemory not available or failed to import).")

memory = memory_instance # Assign the instantiated object to the module-level 'memory' variable

# --- Initialize Core Components & Agents for app.py's own task processing ---
message_bus_for_app: Optional[MessageBus] = MessageBus(save_path='message_history_webapp.json')

ethics_filter_for_app: Optional[EthicsFilter] = EthicsFilter(overlay_path="ethics/ethics_overlay.md", config=config)
sentinel_agent_for_app: Optional[SentinelAgent] = SentinelAgent(config=config, ethics=ethics_filter_for_app, memory=memory) if memory and ethics_filter_for_app else None
quartermaster_for_app: Optional[QuartermasterAgent] = QuartermasterAgent(config=config, memory=memory, sentinel=sentinel_agent_for_app) if memory and sentinel_agent_for_app else None

# Tools for app-specific agent instances
if quartermaster_for_app:
    calculator_tool_app = CalculatorTool()
    datetime_tool_app = DateTimeTool()
    web_search_tool_app = WebSearchTool(quartermaster=quartermaster_for_app)
    read_file_tool_app = ReadFileTool(quartermaster=quartermaster_for_app)
    list_files_tool_app = ListFilesTool(quartermaster=quartermaster_for_app)
    write_file_tool_app = WriteFileTool(quartermaster=quartermaster_for_app)
    weather_tool_app = WeatherTool()
    data_visualization_tool_app = DataVisualizationTool(quartermaster=quartermaster_for_app)
    pdf_generator_tool_app = PdfGeneratorTool(quartermaster=quartermaster_for_app)

    common_tools_app = [calculator_tool_app, datetime_tool_app, read_file_tool_app, list_files_tool_app]
    analyst_tools_app = common_tools_app + [web_search_tool_app, write_file_tool_app, weather_tool_app, data_visualization_tool_app]
    scribe_tools_app = common_tools_app + [write_file_tool_app, pdf_generator_tool_app]
    engineer_tools_app = common_tools_app + [write_file_tool_app]
    researcher_tools_app = common_tools_app + [web_search_tool_app, write_file_tool_app, pdf_generator_tool_app]
    planner_tools_app = common_tools_app
else:
    common_tools_app, analyst_tools_app, scribe_tools_app, engineer_tools_app, researcher_tools_app, planner_tools_app = [],[],[],[],[],[]
    print("[WEB_APP_ERROR] Quartermaster not initialized, tools for app agents will be empty.")

scribe_agent_app: Optional[ScribeAgent] = ScribeAgent(config, memory, quartermaster_for_app, sentinel_agent_for_app, tools=scribe_tools_app) if all([memory, quartermaster_for_app, sentinel_agent_for_app]) else None
engineer_agent_app: Optional[EngineerAgent] = EngineerAgent(config, memory, quartermaster_for_app, sentinel_agent_for_app, tools=engineer_tools_app) if all([memory, quartermaster_for_app, sentinel_agent_for_app]) else None
analyst_agent_app: Optional[AnalystAgent] = AnalystAgent(config, memory, quartermaster_for_app, sentinel_agent_for_app, engineer=engineer_agent_app, tools=analyst_tools_app) if all([memory, quartermaster_for_app, sentinel_agent_for_app]) else None
researcher_agent_app: Optional[ResearcherAgent] = ResearcherAgent(config, memory, quartermaster_for_app, sentinel_agent_for_app, tools=researcher_tools_app) if all([memory, quartermaster_for_app, sentinel_agent_for_app]) else None

agents_for_planner_app: Dict[str, Any] = {
    "scribe": scribe_agent_app, "engineer": engineer_agent_app,
    "analyst": analyst_agent_app, "researcher": researcher_agent_app,
    "quartermaster": quartermaster_for_app
}
planner_mb_client_app: Optional[MessageBusClient] = MessageBusClient("planner_webapp", message_bus_for_app) if message_bus_for_app else None
planner_agent_app: Optional[PlannerAgent] = PlannerAgent(config, memory, quartermaster_for_app, sentinel_agent_for_app,
                                 tools=planner_tools_app, agents=agents_for_planner_app,
                                 message_bus_client=planner_mb_client_app) if all([memory, quartermaster_for_app, sentinel_agent_for_app]) else None

agents_app_instances: Dict[str, Any] = {
    "scribe": scribe_agent_app, "engineer": engineer_agent_app,
    "analyst": analyst_agent_app, "researcher": researcher_agent_app,
    "planner": planner_agent_app, "quartermaster": quartermaster_for_app,
    "sentinel": sentinel_agent_for_app
}
agents_app_instances = {k: v for k, v in agents_app_instances.items() if v is not None}

# --- Task Processing Thread ---
def process_task_queue_thread_target():
    """Processes tasks from the web app's queue."""
    while True:
        task_id_local, agent_key_routed_local, task_content_local = "", "", ""
        try:
            task_id_local, agent_key_routed_local, task_content_local = task_queue.get(block=True)
            print(f"[WEB_APP_TASK_PROCESSOR] Processing task {task_id_local}: Agent='{agent_key_routed_local}', Content='{task_content_local[:60]}...'")
            task_status[task_id_local] = "processing"

            command_history.append({
                "id": task_id_local, "agent": agent_key_routed_local,
                "command": task_content_local, "timestamp": datetime.now().isoformat()
            })
            if len(command_history) > 100: command_history.pop(0)

            agent_instance_to_run = agents_app_instances.get(agent_key_routed_local)
            if agent_instance_to_run:
                assoc_goal_id = None
                # Logic for assoc_goal_id if UI supports linking web tasks to memory goals
                # For now, web UI tasks are generally ad-hoc or manage goals via Quartermaster commands

                if isinstance(agent_instance_to_run, PlannerAgent):
                    result_from_agent = agent_instance_to_run.run(
                        task_content_local, # For Planner, this is the 'command' argument
                        associated_goal_id_for_new_plan=assoc_goal_id
                    )
                else:
                    result_from_agent = agent_instance_to_run.run(task_content_local)

                task_results[task_id_local] = {
                    "agent": agent_key_routed_local, "command": task_content_local,
                    "result": str(result_from_agent), "status": "completed",
                    "timestamp": datetime.now().isoformat()
                }
                task_status[task_id_local] = "completed"
                print(f"[WEB_APP_TASK_PROCESSOR] Task {task_id_local} completed. Result snippet: {str(result_from_agent)[:100]}...")
            else:
                error_msg = f"Unknown agent key for web task: {agent_key_routed_local}"
                task_results[task_id_local] = {"agent": agent_key_routed_local, "command": task_content_local, "result": error_msg, "status": "error", "timestamp": datetime.now().isoformat()}
                task_status[task_id_local] = "error"
                print(f"[WEB_APP_TASK_PROCESSOR] Task {task_id_local} error: {error_msg}")

        except (EthicsViolation, SentinelException, ToolException, QuartermasterException) as controlled_exc:
            error_msg = f"[{type(controlled_exc).__name__}] {controlled_exc}"
            task_results[task_id_local] = {"agent": agent_key_routed_local, "command": task_content_local,
                                           "result": error_msg, "status": "blocked",
                                           "timestamp": datetime.now().isoformat()}
            task_status[task_id_local] = "blocked"
            print(f"[WEB_APP_TASK_PROCESSOR] Task {task_id_local} blocked/tool_error: {error_msg}")
        except Exception as e_proc:
            import traceback
            error_details = traceback.format_exc()
            # Ensure agent_key_routed_local is defined, default if not
            agent_name_for_error = agent_key_routed_local if agent_key_routed_local else "UnknownAgent"
            error_msg = f"[Agent Execution Error - {agent_name_for_error}] {str(e_proc)}"
            task_results[task_id_local] = {"agent": agent_name_for_error, "command": task_content_local,
                                           "result": f"{error_msg}\nTraceback (see server logs for full trace):\n{error_details.splitlines()[-1] if error_details else 'No traceback'}",
                                           "status": "error", "timestamp": datetime.now().isoformat()}
            task_status[task_id_local] = "error"
            print(f"[WEB_APP_TASK_PROCESSOR] Task {task_id_local} critical error: {error_msg}\n{error_details}")
        finally:
            if task_id_local: # Ensure task_id_local was set before calling task_done
                 task_queue.task_done()


app_task_processor_thread = threading.Thread(target=process_task_queue_thread_target, daemon=True)
if not app.debug or os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
    app_task_processor_thread.start()

# --- Helper Function for Parsing Goals for Templates ---
def parse_goals_for_template(goals_text_list_str: str) -> List[Dict[str, str]]:
    parsed_goals = []
    if not memory:
        print("[WEB_APP_WARN] parse_goals_for_template: Memory not initialized.")
        return parsed_goals

    if goals_text_list_str and "No current goals" not in goals_text_list_str:
        lines = goals_text_list_str.split('\n')
        if lines and lines[0].strip().startswith("Current Goals:"):
            lines = lines[1:]

        for line_content in lines:
            line_content = line_content.strip()
            if not line_content: continue

            match = re.match(
                r"^\d+\.\s+ID:\s+([a-f0-9N\/A\-]{1,36}).*?\s+Task:\s+(.+?)\s+(?:\[Status:\s*([^\]]+)\])",
                line_content,
                re.IGNORECASE
            )
            if match:
                goal_id_short_or_full = match.group(1).strip()
                task_desc_raw = match.group(2).strip()
                task_desc = re.sub(r'\s*\(Plan: [a-f0-9]{8}.*?\)\s*$', '', task_desc_raw).strip()
                task_desc = re.sub(r'^\[Prio: \d+\]\s*', '', task_desc).strip()
                task_desc = re.sub(r'^\[Suggest: \w+\]\s*', '', task_desc).strip()
                status = match.group(3).strip() if match.group(3) else "unknown"

                full_goal_id = goal_id_short_or_full
                # If it's a short ID (e.g., 8 chars) or N/A, try to find the full one
                if memory and (goal_id_short_or_full == "N/A" or (len(goal_id_short_or_full) == 8 and '-' not in goal_id_short_or_full)):
                    raw_goals = memory.get_goals_list()
                    if isinstance(raw_goals, list):
                        for g_dict in raw_goals:
                            if isinstance(g_dict, dict) and g_dict.get('task') == task_desc:
                                current_g_id = g_dict.get('id', '')
                                if goal_id_short_or_full == "N/A" or current_g_id.startswith(goal_id_short_or_full):
                                    full_goal_id = current_g_id
                                    break
                parsed_goals.append({
                    "id": full_goal_id,
                    "display_text": line_content,
                    "task_description": task_desc,
                    "status": status
                })
            else:
                print(f"[WEB_APP_WARN] Line from list_goals did not match expected pattern: '{line_content}'")
                parsed_goals.append({"id": None, "display_text": line_content, "task_description": "Could not parse task.", "status": "unknown"})
    return parsed_goals

# --- Flask Routes (Remain the same as previously provided) ---
@app.route('/')
def index_route():
    if not memory or not quartermaster_for_app :
        return "Error: Core WITS components (Memory or Quartermaster) not initialized for web app.", 500

    agent_ui_descriptions = {
        "analyst": "Research, analyze information, and provide insights",
        "engineer": "Generate and modify code", "scribe": "Create written content",
        "researcher": "Conduct in-depth research", "planner": "Break down goals and coordinate",
        "quartermaster": "Manage files, resources, goals", "sentinel": "Monitor ethics and approve actions"
    }
    agents_list_for_ui = [{"name": name, "description": agent_ui_descriptions.get(name, "A WITS agent.")}
                          for name in agents_app_instances.keys()]

    files_for_ui = []
    try:
        files_output_str = quartermaster_for_app.list_files()
        if files_output_str and "No files found" not in files_output_str:
            files_list_lines = files_output_str.split('\n')
            if len(files_list_lines) > 1:
                files_for_ui = [line.lstrip("- ").split('/ (directory)')[0].strip()
                                for line in files_list_lines[1:] if line.strip()]
    except Exception as e_qm_list_idx:
        files_for_ui = [f"Error listing files: {str(e_qm_list_idx)}"]

    goals_text_from_memory = memory.list_goals()
    current_goals_for_ui = parse_goals_for_template(goals_text_from_memory)

    return render_template('index.html', agents=agents_list_for_ui, files=files_for_ui, goals=current_goals_for_ui)

@app.route('/goals')
def goals_view_route():
    if not memory: return "Error: Memory system not initialized.", 500

    goals_text_current = memory.list_goals()
    current_goals_parsed = parse_goals_for_template(goals_text_current)

    completed_text_list = memory.list_completed()
    completed_goals_parsed = []
    if completed_text_list and "No completed goals yet" not in completed_text_list:
        lines = completed_text_list.split('\n')
        if lines and lines[0].strip().startswith("Recently Completed Goals:"): lines = lines[1:]
        for line_content in lines:
            line_content = line_content.strip()
            if not line_content: continue
            completed_goals_parsed.append({"display_text": line_content})

    return render_template('goals.html', goals=current_goals_parsed, completed=completed_goals_parsed)

@app.route('/api/submit_task', methods=['POST'])
def submit_task_api():
    data = request.json
    command_from_user = data.get('command', '').strip()
    if not command_from_user:
        return jsonify({"status": "error", "message": "Command cannot be empty"}), 400

    if command_from_user.lower().startswith("er,"):
        command_from_user = command_from_user[3:].strip()

    router_fallback = config.get("router", {}).get("fallback_agent", "analyst")
    agent_key_routed, task_content_for_agent = route_task(command_from_user, fallback_agent=router_fallback)

    if "quartermaster" in command_from_user.lower() and "complete goal by id" in command_from_user.lower():
        agent_key_routed = "quartermaster"
        match_complete_cmd = re.search(r"complete goal by id\s+([a-f0-9\-]+)(?:\s+with result:(.*))?", command_from_user, re.IGNORECASE)
        if match_complete_cmd:
            goal_id_part = match_complete_cmd.group(1)
            result_part = match_complete_cmd.group(2)
            task_content_for_agent = f"complete goal by id {goal_id_part}"
            if result_part:
                task_content_for_agent += f" with result:{result_part.strip()}"
            print(f"[WEB_APP_API] Refined task for Quartermaster (complete by ID): {task_content_for_agent}")
        else:
            idx = command_from_user.lower().find("complete goal by id")
            if idx != -1:
                task_content_for_agent = command_from_user[idx:]

    task_id = f"webtask_{int(time.time())}_{os.urandom(4).hex()}_{agent_key_routed}"
    print(f"[WEB_APP_API] Final Routed: Agent='{agent_key_routed}', Task='{task_content_for_agent}' (ID: {task_id})")

    task_queue.put((task_id, agent_key_routed, task_content_for_agent))
    task_status[task_id] = "queued"

    return jsonify({"status": "success", "task_id": task_id,
                    "agent_routed_to": agent_key_routed,
                    "task_content_for_agent": task_content_for_agent,
                    "message": f"Task submitted to {agent_key_routed} (ID: {task_id})"})

@app.route('/api/task_status/<task_id>', methods=['GET'])
def get_task_status_api(task_id: str):
    current_status = task_status.get(task_id, "unknown")
    result_data = task_results.get(task_id, {})
    return jsonify({"task_id": task_id, "status": current_status, "result": result_data})

@app.route('/api/planner_status', methods=['GET'])
def get_planner_status_api():
    if not memory: return jsonify({"status": "error", "message": "Memory not initialized"}), 500
    try:
        planner_status_message = memory.recall_agent_output("planner_status")
        if planner_status_message:
            return jsonify({"status": "success", "planner_status": planner_status_message})
        else:
            return jsonify({"status": "success", "planner_status": "Planner status not available or no recent updates."})
    except Exception as e:
        print(f"[WEB_APP_API_ERROR] Error fetching planner status: {e}")
        return jsonify({"status": "error", "message": f"Error fetching planner status: {str(e)}"}), 500

@app.route('/api/agents', methods=['GET'])
def get_agents_api():
    agent_ui_descriptions = {
        "analyst": "Research, analyze information, and provide insights",
        "engineer": "Generate and modify code", "scribe": "Create written content",
        "researcher": "Conduct in-depth research", "planner": "Break down goals and coordinate",
        "quartermaster": "Manage files, resources, goals", "sentinel": "Monitor ethics and approve actions"
    }
    agents_list_for_ui = [{"name": name, "description": agent_ui_descriptions.get(name, "A WITS agent.")}
                          for name in agents_app_instances.keys()]
    return jsonify(agents_list_for_ui)

@app.route('/api/files', methods=['GET'])
def get_files_api():
    if not quartermaster_for_app: return jsonify({"status": "error", "message": "Quartermaster not initialized"}), 500
    try:
        files_output_str = quartermaster_for_app.list_files()
        files_for_ui = []
        if files_output_str and "No files found" not in files_output_str:
            files_list_lines = files_output_str.split('\n')
            if len(files_list_lines) > 1:
                files_for_ui = [line.lstrip("- ").split('/ (directory)')[0].strip()
                                for line in files_list_lines[1:] if line.strip()]
        return jsonify({"status": "success", "files": files_for_ui})
    except Exception as e_qm_list_api:
        return jsonify({"status": "error", "message": str(e_qm_list_api)}), 500

@app.route('/api/file/<path:filename>', methods=['GET'])
def get_file_content_api(filename: str):
    if not quartermaster_for_app or not sentinel_agent_for_app:
        return jsonify({"status": "error", "message": "Core components not initialized"}), 500
    try:
        output_dir_abs = os.path.abspath(config.get("output_directory", "output"))
        safe_filename = secure_filename(filename)
        if safe_filename != filename:
            print(f"[WEB_APP_SECURITY] Potentially unsafe filename in get_file_content_api: '{filename}', sanitized to '{safe_filename}'")
            return jsonify({"status": "error", "message": "Invalid filename."}), 400

        path_for_sentinel = safe_filename
        sentinel_agent_for_app.approve_action("WebAppFileContent", "file_read", detail=path_for_sentinel)
        content = quartermaster_for_app.read_file(safe_filename)

        binary_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.ico', '.pdf', '.zip', '.gz', '.tar', '.exe', '.dll', '.o', '.so', '.class', '.pyc')
        is_binary = safe_filename.lower().endswith(binary_extensions)
        if is_binary:
             return jsonify({"status": "success", "filename": safe_filename, "content": f"Binary file. Use download link.", "is_binary": True, "download_url": url_for('serve_file_from_output_route', filename=safe_filename)})

        return jsonify({"status": "success", "filename": safe_filename, "content": content, "is_binary": False})
    except FileNotFoundError:
         return jsonify({"status": "error", "message": "File not found."}), 404
    except SentinelException as se:
        return jsonify({"status": "error", "message": f"Access denied by Sentinel: {se}"}), 403
    except Exception as e_get_file_api:
        return jsonify({"status": "error", "message": str(e_get_file_api)}), 500

@app.route('/files/<path:filename>')
def serve_file_from_output_route(filename: str):
    if not sentinel_agent_for_app: return "Error: Sentinel not initialized", 500

    output_dir_to_serve_from = os.path.abspath(config.get("output_directory", "output"))
    safe_filename = secure_filename(filename)
    if safe_filename != filename:
        return "Access Denied: Invalid filename characters.", 400

    requested_file_abs = os.path.abspath(os.path.join(output_dir_to_serve_from, safe_filename))
    if not requested_file_abs.startswith(output_dir_to_serve_from):
        return "Access Denied: Path traversal attempt detected.", 403

    try:
        sentinel_agent_for_app.approve_action("WebAppFileServe", "file_read", detail=safe_filename)
    except SentinelException as se:
        return f"Access to file blocked by Sentinel: {se}", 403

    return send_from_directory(output_dir_to_serve_from, safe_filename)

@app.route('/api/goals', methods=['GET', 'POST'])
def goals_api_route():
    if not memory: return jsonify({"status": "error", "message": "Memory not initialized"}), 500

    if request.method == 'POST':
        data = request.json
        task_description = data.get('task', '').strip()
        agent_assigned = data.get('agent')
        priority = data.get('priority', 1)

        if not task_description:
            return jsonify({"status": "error", "message": "Task description cannot be empty"}), 400

        try:
            goal_id = memory.add_goal(task_description, agent=agent_assigned if agent_assigned else None, priority=int(priority))
            return jsonify({"status": "success", "message": "Goal added successfully", "goal_id": goal_id})
        except Exception as e_add_goal:
            print(f"[WEB_APP_API_ERROR] Adding goal: {e_add_goal}")
            return jsonify({"status": "error", "message": f"Failed to add goal: {e_add_goal}"}), 500

    goals_text_current = memory.list_goals()
    current_goals_parsed = parse_goals_for_template(goals_text_current)

    completed_text_list = memory.list_completed()
    completed_goals_parsed = []
    if completed_text_list and "No completed goals yet" not in completed_text_list:
        lines = completed_text_list.split('\n')
        if lines and lines[0].strip().startswith("Recently Completed Goals:"): lines = lines[1:]
        for line_content in lines:
            line_content = line_content.strip()
            if not line_content: continue
            completed_goals_parsed.append({"display_text": line_content})

    return jsonify({"status": "success", "current_goals": current_goals_parsed, "completed_goals": completed_goals_parsed})

@app.route('/api/history', methods=['GET'])
def get_history_api():
    return jsonify({"status": "success", "history": list(command_history)})

@app.route('/api/upload', methods=['POST'])
def upload_file_api():
    if not config.get("web_interface", {}).get("enable_file_uploads", False):
        return jsonify({"status": "error", "message": "File uploads are disabled."}), 403
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part in the request."}), 400

    file_to_upload = request.files['file']
    if file_to_upload.filename == '':
        return jsonify({"status": "error", "message": "No file selected for upload."}), 400

    if file_to_upload:
        filename_secured = secure_filename(file_to_upload.filename)
        if not filename_secured:
             return jsonify({"status": "error", "message": "Invalid filename after sanitization."}), 400

        max_size_bytes = config.get("web_interface", {}).get("max_file_size", 5) * 1024 * 1024

        file_to_upload.seek(0, os.SEEK_END)
        file_length = file_to_upload.tell()
        file_to_upload.seek(0)
        if file_length > max_size_bytes:
            return jsonify({"status": "error", "message": f"File too large (max {config.get('web_interface', {}).get('max_file_size', 5)}MB)"}), 413

        save_path = os.path.join(config.get("output_directory", "output"), filename_secured)

        try:
            if sentinel_agent_for_app:
                 sentinel_agent_for_app.approve_action("WebAppUpload", "file_write", detail=save_path)
            file_to_upload.save(save_path)
            return jsonify({"status": "success", "message": f"File '{filename_secured}' uploaded successfully to output directory.", "filename": filename_secured})
        except SentinelException as se:
             return jsonify({"status": "error", "message": f"File upload blocked by Sentinel: {se}"}), 403
        except Exception as e_upload:
            print(f"[WEB_APP_ERROR] File upload failed: {e_upload}")
            return jsonify({"status": "error", "message": f"File upload failed: {str(e_upload)}"}), 500
    return jsonify({"status": "error", "message": "File upload failed for unknown reasons."}), 500

@app.route('/agent/<agent_name_route_param>')
def agent_view_route(agent_name_route_param: str):
    if agent_name_route_param not in agents_app_instances:
        return redirect(url_for('index_route'))

    agent_ui_descriptions = {
        "analyst": "Research, analyze information, and provide insights",
        "engineer": "Generate and modify code", "scribe": "Create written content",
        "researcher": "Conduct in-depth research", "planner": "Break down goals and coordinate",
        "quartermaster": "Manage files, resources, goals", "sentinel": "Monitor ethics and approve actions"
    }
    agent_desc = agent_ui_descriptions.get(agent_name_route_param, "A WITS agent.")

    plans_data = None
    research_projects_data = None
    current_agent_instance = agents_app_instances.get(agent_name_route_param)

    if agent_name_route_param == "planner" and current_agent_instance:
        try:
            plans_output = current_agent_instance.list_plans()
            if isinstance(plans_output, str) and "No plans found" not in plans_output:
                 plans_data = [p.strip() for p in plans_output.split('\n') if p.strip() and not p.startswith("Available Plans:")]
            elif isinstance(plans_output, list):
                 plans_data = plans_output
            else: plans_data = []
        except Exception as e_planner_ui: plans_data = [f"Error loading plans: {e_planner_ui}"]

    elif agent_name_route_param == "researcher" and current_agent_instance:
        try:
            research_output = current_agent_instance.list_research_projects()
            if isinstance(research_output, str) and "No research projects found" not in research_output:
                research_projects_data = [p.strip() for p in research_output.split('\n') if p.strip() and not p.startswith("Research Projects:")]
            elif isinstance(research_output, list):
                research_projects_data = research_output
            else: research_projects_data = []
        except Exception as e_research_ui: research_projects_data = [f"Error loading research: {e_research_ui}"]

    recent_output_from_memory = memory.recall_agent_output(agent_name_route_param) if memory else "Memory not available."

    return render_template('agent.html',
                           agent_name=agent_name_route_param,
                           agent_description=agent_desc,
                           plans=plans_data,
                           research_projects=research_projects_data,
                           recent_output=recent_output_from_memory)

@app.route('/files')
def files_view_route():
    if not quartermaster_for_app: return "Error: Quartermaster not initialized.", 500
    files_for_ui_page = []
    try:
        files_output_str = quartermaster_for_app.list_files()
        if files_output_str and "No files found" not in files_output_str:
            files_list_lines = files_output_str.split('\n')
            if len(files_list_lines) > 1:
                files_for_ui_page = [line.lstrip("- ").split('/ (directory)')[0].strip()
                                     for line in files_list_lines[1:] if line.strip()]
    except Exception as e_qm_files_page:
        files_for_ui_page = [f"Error listing files: {str(e_qm_files_page)}"]
    return render_template('files.html', files=files_for_ui_page)

@app.route('/history')
def history_view_route():
    return render_template('history.html', history=list(command_history))

@app.route('/settings')
def settings_view_route():
    return render_template('settings.html', config=config)

@app.route('/api/settings', methods=['GET', 'POST'])
def settings_api_route():
    global config # Ensure we are modifying the global config object

    if request.method == 'POST':
        data_from_request = request.json
        changes_made = False

        # System keys
        for key in ["internet_access", "ethics_enabled", "allow_code_execution", "output_directory", "voice_input"]:
            if key in data_from_request:
                if isinstance(config.get(key), bool): # Check current type in config for boolean conversion
                    config[key] = bool(data_from_request[key])
                elif isinstance(config.get(key), int) and key not in ["port", "max_file_size", "voice_input_duration"]:
                    try: config[key] = int(data_from_request[key])
                    except ValueError: print(f"[WEB_APP_SETTINGS] Invalid int value for {key}: {data_from_request[key]}")
                else: # string or other
                    config[key] = data_from_request[key]
                changes_made = True
                print(f"[WEB_APP_SETTINGS] Updated '{key}' to: {config[key]}")

        # Router settings
        if "router" in data_from_request and isinstance(data_from_request["router"], dict):
            config.setdefault("router", {}) # Ensure 'router' key exists
            if "fallback_agent" in data_from_request["router"]:
                config["router"]["fallback_agent"] = str(data_from_request["router"]["fallback_agent"])
                changes_made = True
                print(f"[WEB_APP_SETTINGS] Updated router.fallback_agent to: {config['router']['fallback_agent']}")

        # Models settings
        if "models" in data_from_request and isinstance(data_from_request["models"], dict):
            config.setdefault("models", {}) # Ensure 'models' key exists
            for agent_model_key, model_name_val in data_from_request["models"].items():
                if agent_model_key in ["default", "scribe", "analyst", "engineer", "researcher", "planner"]:
                    config["models"][agent_model_key] = str(model_name_val)
                    changes_made = True
                    print(f"[WEB_APP_SETTINGS] Updated models.{agent_model_key} to: {model_name_val}")

        # Voice specific settings
        for key in ["whisper_model", "voice_input_duration", "whisper_fp16"]:
            if key in data_from_request:
                if key == "voice_input_duration":
                    try: config[key] = int(data_from_request[key])
                    except ValueError: print(f"[WEB_APP_SETTINGS] Invalid int for {key}")
                elif key == "whisper_fp16":
                    config[key] = bool(data_from_request[key])
                else: # whisper_model
                    config[key] = str(data_from_request[key])
                changes_made = True
                print(f"[WEB_APP_SETTINGS] Updated '{key}' to: {config[key]}")

        # Web interface settings
        if "web_interface" in data_from_request and isinstance(data_from_request["web_interface"], dict):
            config.setdefault("web_interface", {}) # Ensure 'web_interface' key exists
            for web_key, web_val in data_from_request["web_interface"].items():
                if web_key in ["port", "host", "debug", "enable_file_uploads", "max_file_size"]:
                    if web_key in ["port", "max_file_size"]:
                        try: config["web_interface"][web_key] = int(web_val)
                        except ValueError: print(f"[WEB_APP_SETTINGS] Invalid int for web_interface.{web_key}")
                    elif web_key in ["debug", "enable_file_uploads"]:
                        config["web_interface"][web_key] = bool(web_val)
                    else: # host
                         config["web_interface"][web_key] = str(web_val)
                    changes_made = True
                    print(f"[WEB_APP_SETTINGS] Updated web_interface.{web_key} to: {config['web_interface'][web_key]}")

        if changes_made:
            try:
                with open(config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                return jsonify({"status": "success", "message": "Settings updated. Some may require restart."})
            except Exception as e_save_cfg:
                return jsonify({"status": "error", "message": f"Error saving config: {e_save_cfg}"}), 500
        else:
            return jsonify({"status": "success", "message": "No recognized settings were changed."})

    # GET request
    settings_to_expose = {key: config.get(key) for key in [
        "internet_access", "ethics_enabled", "allow_code_execution",
        "voice_input", "output_directory", "router", "models",
        "whisper_model", "voice_input_duration", "whisper_fp16", "web_interface"
    ]}
    return jsonify({"status": "success", "settings": settings_to_expose})


if __name__ == '__main__':
    print("[WEB_APP] Starting Flask development server directly from app.py...")
    web_run_config = config.get("web_interface", {})
    app.run(
        host=web_run_config.get("host", "0.0.0.0"),
        port=web_run_config.get("port", 5000),
        debug=web_run_config.get("debug", True)
    )