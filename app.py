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

# Add the parent directory to the path so we can import from the WITS modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import WITS components
from core.memory import Memory # Basic memory as a fallback
from core.enhanced_memory import EnhancedMemory
# Attempt to import VectorMemory, fall back if not available or issues
try:
    from core.vector_memory import VectorMemory
    VECTOR_MEMORY_AVAILABLE = True
except ImportError:
    print("[WEB] VectorMemory module not found or import error. Vector search capabilities will be disabled.")
    VECTOR_MEMORY_AVAILABLE = False
except Exception as e:
    print(f"[WEB] Error importing VectorMemory: {e}. Vector search capabilities will be disabled.")
    VECTOR_MEMORY_AVAILABLE = False


from ethics.ethics_rules import EthicsFilter, EthicsViolation
from core.router import route_task
from core.message_bus import MessageBus, MessageBusClient, Message # Assuming Message is also needed for type hinting or direct use

from agents.scribe_agent import ScribeAgent
from agents.analyst_agent import AnalystAgent
from agents.engineer_agent import EngineerAgent
from agents.researcher_agent import ResearcherAgent
from agents.planner_agent import PlannerAgent
from agents.quartermaster_agent import QuartermasterAgent
from agents.sentinel_agent import SentinelAgent, SentinelException

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

# Create a task queue and result cache
task_queue = queue.Queue()
task_results = {} # Stores {task_id: {agent, command, result, status, timestamp}}
task_status = {}  # Stores {task_id: "queued" | "processing" | "completed" | "error" | "blocked"}
command_history = [] # List of {"id", "agent", "command", "timestamp"}

# Load configuration
config = {}
# Assuming app.py is at the root, config.yaml is also at the root
config_path = "config.yaml" # Adjusted path relative to app.py if it's at the root

if os.path.exists(config_path):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
            if config_data is not None and isinstance(config_data, dict):
                config.update(config_data)
                print(f"[WEB_CONFIG] Loaded configuration from '{config_path}'.")
            else:
                print(f"[WEB_CONFIG] Warning: '{config_path}' is empty or not valid YAML dictionary. Using defaults.")
    except Exception as e:
        print(f"[WEB_CONFIG] Error loading '{config_path}': {e}. Using defaults.")
else:
    print(f"[WEB_CONFIG] Warning: '{config_path}' not found. Using defaults.")

# Set defaults if not specified in config
config.setdefault("internet_access", True)
config.setdefault("voice_input", False) # Voice input primarily for CLI, but config might be shared
config.setdefault("voice_input_duration", 5)
config.setdefault("whisper_model", "base")
config.setdefault("whisper_fp16", False)
config.setdefault("allow_code_execution", False)
# Output directory path should be relative to the project root where app.py is
config.setdefault("output_directory", "output") # Assuming 'output' is in the same dir as app.py
os.makedirs(config.get("output_directory", "output"), exist_ok=True)
config.setdefault("ethics_enabled", True)
config.setdefault("router", {"fallback_agent": "analyst"})
config.setdefault("models", {
    "default": "llama2",
    "scribe": "llama2",
    "analyst": "llama2",
    "engineer": "codellama:7b",
    "researcher": "llama2",
    "planner": "llama2"
})
config.setdefault("web_interface", {
    "port": 5000,
    "host": "0.0.0.0",
    "debug": True,
    "enable_file_uploads": True,
    "max_file_size": 5  # in MB
})

# Initialize the memory system
if VECTOR_MEMORY_AVAILABLE:
    try:
        memory = VectorMemory(memory_file='vector_memory.json', index_file='vector_index.bin')
        print("[WEB] Using Vector Memory system")
    except Exception as e_vm:
        print(f"[WEB] Vector Memory instantiation failed ({e_vm}), falling back to Enhanced Memory.")
        memory = EnhancedMemory(memory_file='enhanced_memory.json')
        print("[WEB] Using Enhanced Memory system due to VectorMemory fallback.")
else:
    memory = EnhancedMemory(memory_file='enhanced_memory.json')
    print("[WEB] Using Enhanced Memory system.")


# Initialize message bus
message_bus = MessageBus(save_path='message_history.json') # Path relative to app.py

# Initialize components
ethics_filter = EthicsFilter(overlay_path="ethics/ethics_overlay.md", config=config) # Path relative to app.py
sentinel_agent = SentinelAgent(config=config, ethics=ethics_filter, memory=memory)
quartermaster = QuartermasterAgent(config=config, memory=memory, sentinel=sentinel_agent)

# Initialize tools
calculator = CalculatorTool()
datetime_tool = DateTimeTool()
web_search_tool = WebSearchTool(quartermaster=quartermaster)
read_file_tool = ReadFileTool(quartermaster=quartermaster)
list_files_tool = ListFilesTool(quartermaster=quartermaster)
write_file_tool = WriteFileTool(quartermaster=quartermaster)
weather_tool = WeatherTool() # Consider adding API key via config
data_visualization_tool = DataVisualizationTool(quartermaster=quartermaster)
pdf_generator_tool = PdfGeneratorTool(quartermaster=quartermaster)

# Tool sets for different agents
common_tools = [calculator, datetime_tool, read_file_tool, list_files_tool]
analyst_tools = common_tools + [web_search_tool, write_file_tool, weather_tool, data_visualization_tool]
researcher_tools = common_tools + [web_search_tool, write_file_tool, pdf_generator_tool]
engineer_tools = common_tools + [write_file_tool]
scribe_tools = common_tools + [write_file_tool, pdf_generator_tool]
planner_tools = common_tools # Planner might mostly delegate, but can have basic tools

# Initialize agents
scribe_agent = ScribeAgent(
    config=config, memory=memory, quartermaster=quartermaster, sentinel=sentinel_agent,
    tools=scribe_tools
)
engineer_agent = EngineerAgent(
    config=config, memory=memory, quartermaster=quartermaster, sentinel=sentinel_agent,
    tools=engineer_tools
)
analyst_agent = AnalystAgent(
    config=config, memory=memory, quartermaster=quartermaster, sentinel=sentinel_agent,
    engineer=engineer_agent, # For delegation
    tools=analyst_tools
)
researcher_agent = ResearcherAgent(
    config=config, memory=memory, quartermaster=quartermaster, sentinel=sentinel_agent,
    tools=researcher_tools
)

# Agent dictionary (needed for Planner)
agents_dict_for_planner = {
    "scribe": scribe_agent,
    "engineer": engineer_agent,
    "analyst": analyst_agent,
    "researcher": researcher_agent,
    "quartermaster": quartermaster, # Quartermaster can also be "called" by planner for tasks
    "sentinel": sentinel_agent    # Sentinel isn't typically delegated to, but good to have a reference
}

planner_message_client = MessageBusClient("planner", message_bus)
planner_agent = PlannerAgent(
    config=config, memory=memory, quartermaster=quartermaster, sentinel=sentinel_agent,
    tools=planner_tools, agents=agents_dict_for_planner, message_bus_client=planner_message_client
)

# Main dictionary of all available agents for routing
agents = {
    "scribe": scribe_agent,
    "engineer": engineer_agent,
    "analyst": analyst_agent,
    "researcher": researcher_agent,
    "planner": planner_agent,
    "quartermaster": quartermaster,
    "sentinel": sentinel_agent
}

# Initialize message bus clients for other agents if they need to proactively send/receive
# Example:
# analyst_message_client = MessageBusClient("analyst", message_bus)
# if hasattr(analyst_agent, 'set_message_bus_client'): # Or add to __init__
#     analyst_agent.set_message_bus_client(analyst_message_client)


def process_task_queue():
    """Background thread to process tasks from the queue"""
    while True:
        try:
            task_id, agent_key_from_router, task_content_for_agent = task_queue.get(block=True)
            task_status[task_id] = "processing"
            actual_agent_name_used = agent_key_from_router # For logging/history

            try:
                print(f"[WEB_TASK_PROCESSOR] Processing task {task_id}: Agent='{actual_agent_name_used}', Content='{task_content_for_agent[:60]}...'")

                # Record in command history before execution
                command_history.append({
                    "id": task_id,
                    "agent": actual_agent_name_used,
                    "command": task_content_for_agent, # Use the content given to the agent
                    "timestamp": datetime.now().isoformat()
                })
                # Keep history trimmed if it grows too large (e.g., last 100 commands)
                if len(command_history) > 100:
                    command_history.pop(0)


                if actual_agent_name_used in agents:
                    agent_instance = agents[actual_agent_name_used]
                    # The agent's .run() method should handle its own logic, tool use, LLM calls, etc.
                    result_from_agent = agent_instance.run(task_content_for_agent)

                    task_results[task_id] = {
                        "agent": actual_agent_name_used,
                        "command": task_content_for_agent,
                        "result": result_from_agent,
                        "status": "completed",
                        "timestamp": datetime.now().isoformat()
                    }
                    task_status[task_id] = "completed"
                    print(f"[WEB_TASK_PROCESSOR] Task {task_id} completed. Result snippet: {str(result_from_agent)[:100]}...")
                else:
                    error_msg = f"Unknown agent key determined by router: {actual_agent_name_used}"
                    task_results[task_id] = {
                        "agent": actual_agent_name_used, # Log what router decided
                        "command": task_content_for_agent,
                        "result": error_msg,
                        "status": "error",
                        "timestamp": datetime.now().isoformat()
                    }
                    task_status[task_id] = "error"
                    print(f"[WEB_TASK_PROCESSOR] Task {task_id} error: {error_msg}")

            except (EthicsViolation, SentinelException, ToolException) as controlled_exception:
                error_msg = f"[{type(controlled_exception).__name__}] {controlled_exception}"
                task_results[task_id] = {
                    "agent": actual_agent_name_used,
                    "command": task_content_for_agent,
                    "result": error_msg,
                    "status": "blocked", # Or "error" depending on how you want to classify
                    "timestamp": datetime.now().isoformat()
                }
                task_status[task_id] = "blocked"
                print(f"[WEB_TASK_PROCESSOR] Task {task_id} blocked/tool_error: {error_msg}")

            except Exception as e_agent_processing:
                import traceback
                error_details = traceback.format_exc()
                error_msg = f"[Agent Execution Error - {actual_agent_name_used}] {str(e_agent_processing)}"
                task_results[task_id] = {
                    "agent": actual_agent_name_used,
                    "command": task_content_for_agent,
                    "result": f"{error_msg}\nFull Traceback:\n{error_details}",
                    "status": "error",
                    "timestamp": datetime.now().isoformat()
                }
                task_status[task_id] = "error"
                print(f"[WEB_TASK_PROCESSOR] Task {task_id} critical error: {error_msg}\n{error_details}")

            finally:
                task_queue.task_done()

        except Exception as e_queue_loop:
            print(f"[WEB_TASK_PROCESSOR] Critical error in task queue loop: {e_queue_loop}")
            import traceback
            print(traceback.format_exc())
            time.sleep(1) # Avoid tight loop on recurring error

# Start the background task processing thread
task_processor_thread = threading.Thread(target=process_task_queue, daemon=True)
task_processor_thread.start()


@app.route('/')
def index_route(): # Renamed to avoid conflict with 'index' variable name
    """Render the main index page"""
    # Agent descriptions for the UI
    agents_list_for_ui = [
        {"name": name, "description": (getattr(agent, 'description', "A helpful WITS agent.") if hasattr(agent, '__doc__') and agent.__doc__ else "A WITS agent.")}
        for name, agent in agents.items()
    ]
    # A more direct way if agents don't have a 'description' attribute:
    agent_ui_descriptions = {
        "analyst": "Research, analyze information, and provide insights",
        "engineer": "Generate and modify code",
        "scribe": "Create written content like articles, stories, and documentation",
        "researcher": "Conduct thorough, methodical research on complex topics",
        "planner": "Break down complex goals into tasks and coordinate workflows",
        "quartermaster": "Manage files, resources, and goals",
        "sentinel": "Monitor ethical guidelines and approve actions"
    }
    agents_list_for_ui = [{"name": name, "description": agent_ui_descriptions.get(name, "A WITS agent.")} for name in agents.keys()]


    # Get recent files from Quartermaster
    try:
        # Assuming Quartermaster's list_files returns a string that needs parsing
        files_output_str = quartermaster.list_files() # Default lists output_dir
        if "No files found" in files_output_str or not files_output_str:
            files_for_ui = []
        else:
            # Example parsing: "Files in <path>:\n- file1.txt\n- folderA"
            files_list_lines = files_output_str.split('\n')
            if len(files_list_lines) > 1:
                files_for_ui = [line.lstrip("- ").strip() for line in files_list_lines[1:] if line.strip()]
            else:
                files_for_ui = []
    except Exception as e_qm_list:
        files_for_ui = [f"Error listing files: {str(e_qm_list)}"]

    # Get current goals from memory
    goals_text_from_memory = memory.list_goals()
    current_goals_for_ui = [g.strip() for g in goals_text_from_memory.split('\n') if g.strip() and "No current goals" not in g]

    return render_template('index.html', agents=agents_list_for_ui, files=files_for_ui, goals=current_goals_for_ui)


@app.route('/api/submit_task', methods=['POST'])
def submit_task_api(): # Renamed
    data = request.json
    command_from_user = data.get('command', '').strip()

    if not command_from_user:
        return jsonify({"status": "error", "message": "Command cannot be empty"}), 400

    # Clean the command if needed (e.g., "er," prefix from voice)
    if command_from_user.lower().startswith("er,"):
        command_from_user = command_from_user[3:].strip()
        print(f"[WEB_API] Removed 'er,' prefix from command: '{command_from_user}'")

    # Use the router to determine which agent and the actual task content
    router_fallback = config.get("router", {}).get("fallback_agent", "analyst")
    agent_key_routed, task_content_for_agent = route_task(command_from_user, fallback_agent=router_fallback)

    task_id = f"task_{int(time.time())}_{os.urandom(4).hex()}_{agent_key_routed}"

    print(f"[WEB_API] Routed command '{command_from_user}' to agent '{agent_key_routed}'. Task for agent: '{task_content_for_agent}' (ID: {task_id})")

    task_queue.put((task_id, agent_key_routed, task_content_for_agent))
    task_status[task_id] = "queued"

    return jsonify({
        "status": "success",
        "task_id": task_id,
        "agent_routed_to": agent_key_routed,
        "task_content_for_agent": task_content_for_agent,
        "message": f"Task submitted to {agent_key_routed} (ID: {task_id})"
    })

@app.route('/api/task_status/<task_id>', methods=['GET'])
def get_task_status_api(task_id): # Renamed
    current_status = task_status.get(task_id, "unknown")
    result_data = task_results.get(task_id, {}) # Contains full details if completed/error

    return jsonify({
        "task_id": task_id,
        "status": current_status,
        "result": result_data # This will include agent, command, result string, final status
    })


@app.route('/api/agents', methods=['GET'])
def get_agents_api(): # Renamed
    agent_ui_descriptions = {
        "analyst": "Research, analyze information, and provide insights",
        "engineer": "Generate and modify code",
        "scribe": "Create written content like articles, stories, and documentation",
        "researcher": "Conduct thorough, methodical research on complex topics",
        "planner": "Break down complex goals into tasks and coordinate workflows",
        "quartermaster": "Manage files, resources, and goals",
        "sentinel": "Monitor ethical guidelines and approve actions"
    }
    agents_list_for_ui = [{"name": name, "description": agent_ui_descriptions.get(name, "A WITS agent.")} for name in agents.keys()]
    return jsonify(agents_list_for_ui)


@app.route('/api/files', methods=['GET'])
def get_files_api(): # Renamed
    try:
        files_output_str = quartermaster.list_files() # Default lists output_dir
        if "No files found" in files_output_str or not files_output_str:
            files_for_ui = []
        else:
            files_list_lines = files_output_str.split('\n')
            if len(files_list_lines) > 1: # Header + files
                files_for_ui = [line.lstrip("- ").strip() for line in files_list_lines[1:] if line.strip()]
            else: # Only header or empty
                files_for_ui = []
        return jsonify({"status": "success", "files": files_for_ui})
    except Exception as e_qm_list_api:
        return jsonify({"status": "error", "message": str(e_qm_list_api)}), 500


@app.route('/api/file/<path:filename>', methods=['GET'])
def get_file_content_api(filename): # Renamed
    """API endpoint to get the content of a file for display (not download)"""
    try:
        # Security: Ensure filename is within the output directory
        # Quartermaster's read_file should handle this via Sentinel's approval.
        # However, an additional check here for API safety is good.
        output_dir_abs = os.path.abspath(config.get("output_directory", "output"))
        requested_file_abs = os.path.abspath(os.path.join(output_dir_abs, filename))

        if not requested_file_abs.startswith(output_dir_abs):
            return jsonify({"status": "error", "message": "Access denied to this file path."}), 403

        # Check if it's a binary file type where content display is not ideal
        binary_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.ico', '.pdf', '.zip', '.gz', '.tar', '.exe', '.dll', '.o', '.so', '.class', '.pyc')
        if filename.lower().endswith(binary_extensions):
            # For binary files, it's better to use the direct /files/<filename> download/serve link
            return jsonify({
                "status": "success",
                "filename": filename,
                "content": f"This appears to be a binary file. Please use the 'View' or 'Download' link directly: /files/{filename}",
                "is_binary": True,
                "download_url": url_for('serve_file_from_output', filename=filename)
            })

        content = quartermaster.read_file(filename) # QM uses Sentinel
        return jsonify({
            "status": "success",
            "filename": filename,
            "content": content,
            "is_binary": False
        })
    except FileNotFoundError:
         return jsonify({"status": "error", "message": "File not found."}), 404
    except Exception as e_get_file_api:
        return jsonify({"status": "error", "message": str(e_get_file_api)}), 500


@app.route('/files/<path:filename>')
def serve_file_from_output(filename): # Renamed
    """Serve a file from the output directory (for viewing/downloading)"""
    # Security: Ensure filename doesn't try to escape the output directory
    # send_from_directory handles this reasonably well but an explicit check is safer.
    output_dir_abs = os.path.abspath(config.get("output_directory", "output"))
    file_abs_path = os.path.abspath(os.path.join(output_dir_abs, filename))

    if not file_abs_path.startswith(output_dir_abs):
        return "Access Denied", 403 # Or a proper error page

    # Ensure Quartermaster (and thus Sentinel) would approve reading this for consistency,
    # though send_from_directory doesn't go through QM.
    try:
        # This is a conceptual check; we don't actually use the content here.
        # We rely on send_from_directory for the actual serving.
        sentinel_agent.approve_action("WebAppFileServe", "file_read", detail=filename)
    except SentinelException as se:
        return f"Access to file blocked by Sentinel: {se}", 403

    return send_from_directory(config.get("output_directory", "output"), filename)


@app.route('/api/goals', methods=['GET'])
def get_goals_api(): # Renamed
    goals_text = memory.list_goals()
    goals_list = [g.strip() for g in goals_text.split('\n') if g.strip() and "No current goals" not in g]

    completed_text = memory.list_completed()
    completed_list = [g.strip() for g in completed_text.split('\n') if g.strip() and "No completed goals yet" not in g]

    return jsonify({
        "status": "success",
        "current_goals": goals_list,
        "completed_goals": completed_list
        })


@app.route('/api/goals', methods=['POST'])
def add_goal_api(): # Renamed
    data = request.json
    task_description = data.get('task', '').strip()
    agent_assigned = data.get('agent') # Can be None or empty

    if not task_description:
        return jsonify({"status": "error", "message": "Task description cannot be empty"}), 400

    memory.add_goal(task_description, agent=agent_assigned if agent_assigned else None)
    return jsonify({"status": "success", "message": "Goal added successfully"})


@app.route('/api/history', methods=['GET'])
def get_history_api(): # Renamed
    # Return a copy to prevent modification if needed, though direct access is fine for read-only here
    return jsonify({"status": "success", "history": list(command_history)})


@app.route('/api/upload', methods=['POST'])
def upload_file_api(): # Renamed
    if not config.get("web_interface", {}).get("enable_file_uploads", False):
        return jsonify({"status": "error", "message": "File uploads are disabled."}), 403

    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part in the request."}), 400

    file_to_upload = request.files['file']

    if file_to_upload.filename == '':
        return jsonify({"status": "error", "message": "No file selected for upload."}), 400

    if file_to_upload:
        # Security: Sanitize filename
        from werkzeug.utils import secure_filename
        filename_secured = secure_filename(file_to_upload.filename)
        if not filename_secured: # If filename becomes empty after securing
             return jsonify({"status": "error", "message": "Invalid filename."}), 400


        # Check file size (example: 5MB limit)
        max_size_bytes = config.get("web_interface", {}).get("max_file_size", 5) * 1024 * 1024
        # Werkzeug's SpooledTemporaryFile (used by Flask for file uploads)
        # might require seeking to end and telling to get size if not already read into memory.
        file_to_upload.seek(0, os.SEEK_END)
        file_length = file_to_upload.tell()
        file_to_upload.seek(0) # Reset stream position

        if file_length > max_size_bytes:
            return jsonify({"status": "error", "message": f"File too large (max {config.get('web_interface', {}).get('max_file_size', 5)}MB)"}), 413

        # Path to save the file (within the configured output directory)
        save_path = os.path.join(config.get("output_directory", "output"), filename_secured)

        try:
            # Before saving, check with Sentinel if writing this file is okay.
            # This requires Quartermaster to have a method that can check without writing, or we simulate.
            # For simplicity, we can let Quartermaster handle the write, which includes Sentinel approval.
            # However, for uploads, the content comes from user, not agent.
            # A direct sentinel check for "user_upload" might be better.
            # sentinel_agent.approve_action("WebAppUpload", "file_write", detail=filename_secured) # Conceptual

            file_to_upload.save(save_path) # Saves the file
            # Optionally, have Quartermaster "register" this uploaded file if QM manages a manifest
            # quartermaster.register_uploaded_file(filename_secured, "user_upload")

            return jsonify({
                "status": "success",
                "message": f"File '{filename_secured}' uploaded successfully.",
                "filename": filename_secured
            })
        except SentinelException as se:
             return jsonify({"status": "error", "message": f"File upload blocked by Sentinel: {se}"}), 403
        except Exception as e_upload:
            return jsonify({"status": "error", "message": f"File upload failed: {str(e_upload)}"}), 500


@app.route('/agent/<agent_name_route_param>') # Renamed param
def agent_view_route(agent_name_route_param): # Renamed param
    """Render the agent-specific view"""
    if agent_name_route_param not in agents:
        return redirect(url_for('index_route')) # Use correct route function name

    agent_ui_descriptions = {
        "analyst": "Research, analyze information, and provide insights",
        "engineer": "Generate and modify code",
        "scribe": "Create written content like articles, stories, and documentation",
        "researcher": "Conduct thorough, methodical research on complex topics",
        "planner": "Break down complex goals into tasks and coordinate workflows",
        "quartermaster": "Manage files, resources, and goals",
        "sentinel": "Monitor ethical guidelines and approve actions"
    }

    plans_data = None
    research_projects_data = None

    if agent_name_route_param == "planner":
        try:
            plans_output = planner_agent.list_plans() # Assumes this returns a string or list of strings
            if isinstance(plans_output, str) and "No plans found" in plans_output:
                 plans_data = []
            elif isinstance(plans_output, str):
                 plans_data = [p.strip() for p in plans_output.split('\n') if p.strip()]
            else: # Assuming it might return a list directly
                 plans_data = plans_output if isinstance(plans_output, list) else []
        except Exception as e_planner_list:
            print(f"Error getting plans for UI: {e_planner_list}")
            plans_data = ["Error loading plans."]
            
    elif agent_name_route_param == "researcher":
        try:
            research_output = researcher_agent.list_research_projects()
            if isinstance(research_output, str) and "No research projects found" in research_output:
                research_projects_data = []
            elif isinstance(research_output, str):
                research_projects_data = [p.strip() for p in research_output.split('\n') if p.strip()]
            else:
                research_projects_data = research_output if isinstance(research_output, list) else []
        except Exception as e_researcher_list:
            print(f"Error getting research projects for UI: {e_researcher_list}")
            research_projects_data = ["Error loading research projects."]

    recent_output_from_memory = memory.recall_agent_output(agent_name_route_param)

    return render_template(
        'agent.html',
        agent_name=agent_name_route_param,
        agent_description=agent_ui_descriptions.get(agent_name_route_param, "A WITS agent."),
        plans=plans_data, # Pass the actual data
        research_projects=research_projects_data, # Pass the actual data
        recent_output=recent_output_from_memory
    )


@app.route('/files')
def files_view_route(): # Renamed
    try:
        files_output_str = quartermaster.list_files()
        if "No files found" in files_output_str or not files_output_str:
            files_for_ui = []
        else:
            files_list_lines = files_output_str.split('\n')
            if len(files_list_lines) > 1:
                files_for_ui = [line.lstrip("- ").strip() for line in files_list_lines[1:] if line.strip()]
            else:
                files_for_ui = []
    except Exception as e_qm_files_route:
        files_for_ui = [f"Error listing files: {str(e_qm_files_route)}"]
    return render_template('files.html', files=files_for_ui)


@app.route('/goals')
def goals_view_route(): # Renamed
    goals_text = memory.list_goals()
    current_goals = [g.strip() for g in goals_text.split('\n') if g.strip() and "No current goals" not in g]
    completed_text = memory.list_completed()
    completed_goals = [g.strip() for g in completed_text.split('\n') if g.strip() and "No completed goals yet" not in g]
    return render_template('goals.html', goals=current_goals, completed=completed_goals)


@app.route('/history')
def history_view_route(): # Renamed
    return render_template('history.html', history=list(command_history))


@app.route('/settings')
def settings_view_route(): # Renamed
    # Ensure config is up-to-date if it can be changed elsewhere (though unlikely for web settings)
    return render_template('settings.html', config=config)


@app.route('/api/settings', methods=['GET'])
def get_settings_api():
    """API endpoint to get current settings"""
    # Return a copy of the relevant parts of the config
    # Be careful not to expose sensitive information if any were in config
    settings_to_expose = {
        "internet_access": config.get("internet_access"),
        "ethics_enabled": config.get("ethics_enabled"),
        "allow_code_execution": config.get("allow_code_execution"),
        "voice_input": config.get("voice_input"), # For UI toggle consistency
        "output_directory": config.get("output_directory"),
        "router": config.get("router"),
        "models": config.get("models"),
        "whisper_model": config.get("whisper_model"),
        "voice_input_duration": config.get("voice_input_duration"),
        "whisper_fp16": config.get("whisper_fp16"),
        "web_interface": config.get("web_interface") # Expose web settings for the UI to manage them
    }
    return jsonify({"status": "success", "settings": settings_to_expose})


@app.route('/api/settings', methods=['POST'])
def update_settings_api(): # Renamed
    """API endpoint to update settings"""
    data_from_request = request.json
    changes_made = False

    # Define which keys are safe/expected to be updated from the UI
    # System settings
    system_keys = ["internet_access", "ethics_enabled", "allow_code_execution", "voice_input", "output_directory"]
    for key in system_keys:
        if key in data_from_request:
            config[key] = data_from_request[key]
            changes_made = True
            print(f"[WEB_SETTINGS] Updated '{key}' to: {config[key]}")

    # Router settings
    if "router" in data_from_request and isinstance(data_from_request["router"], dict):
        if "fallback_agent" in data_from_request["router"]:
            config.setdefault("router", {})["fallback_agent"] = data_from_request["router"]["fallback_agent"]
            changes_made = True
            print(f"[WEB_SETTINGS] Updated router.fallback_agent to: {config['router']['fallback_agent']}")


    # Model settings
    if "models" in data_from_request and isinstance(data_from_request["models"], dict):
        config.setdefault("models", {})
        for agent_model_key, model_name_val in data_from_request["models"].items():
            config["models"][agent_model_key] = model_name_val
            changes_made = True
            print(f"[WEB_SETTINGS] Updated models.{agent_model_key} to: {model_name_val}")


    # Voice specific settings (that are not just the voice_input toggle)
    voice_specific_keys = ["whisper_model", "voice_input_duration", "whisper_fp16"]
    for key in voice_specific_keys:
        if key in data_from_request:
            config[key] = data_from_request[key]
            changes_made = True
            print(f"[WEB_SETTINGS] Updated '{key}' to: {config[key]}")


    # Web interface settings (some might require restart, UI should note this)
    if "web_interface" in data_from_request and isinstance(data_from_request["web_interface"], dict):
        config.setdefault("web_interface", {})
        for web_key, web_val in data_from_request["web_interface"].items():
            # Only update keys that are typically managed by config.yaml for web settings
            if web_key in ["port", "host", "debug", "enable_file_uploads", "max_file_size"]:
                config["web_interface"][web_key] = web_val
                changes_made = True
                print(f"[WEB_SETTINGS] Updated web_interface.{web_key} to: {web_val}")


    if changes_made:
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            return jsonify({"status": "success", "message": "Settings updated successfully. Some changes may require a restart."})
        except Exception as e_save_config:
            return jsonify({"status": "error", "message": f"Error saving settings to config file: {str(e_save_config)}"}), 500
    else:
        return jsonify({"status": "success", "message": "No recognized settings were changed."})


if __name__ == '__main__':
    # This block is usually for direct execution of app.py (python app.py)
    # If main.py is the primary entry point and launches app.py, this might not be hit
    # when running through main.py --web
    web_run_config = config.get("web_interface", {})
    app.run(
        host=web_run_config.get("host", "0.0.0.0"),
        port=web_run_config.get("port", 5000),
        debug=web_run_config.get("debug", True)
    )