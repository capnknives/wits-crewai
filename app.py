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
from core.memory import Memory
from core.enhanced_memory import EnhancedMemory
from ethics.ethics_rules import EthicsFilter, EthicsViolation
from core.router import route_task
from core.message_bus import MessageBus, MessageBusClient, Message

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
task_results = {}
task_status = {}
command_history = []

# Load configuration
config = {}
config_path = "../config.yaml"

if os.path.exists(config_path):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
            if config_data is not None and isinstance(config_data, dict):
                config.update(config_data)
                print(f"[CONFIG] Loaded configuration from '{config_path}'.")
            else:
                print(f"[CONFIG] Warning: '{config_path}' is empty or not valid YAML dictionary. Using defaults.")
    except Exception as e:
        print(f"[CONFIG] Error loading '{config_path}': {e}. Using defaults.")
else:
    print(f"[CONFIG] Warning: '{config_path}' not found. Using defaults.")

# Set defaults if not specified in config
config.setdefault("internet_access", True)
config.setdefault("voice_input", False)
config.setdefault("voice_input_duration", 5)
config.setdefault("whisper_model", "base")
config.setdefault("whisper_fp16", False)
config.setdefault("allow_code_execution", False)
config.setdefault("output_directory", "../output")
os.makedirs(config.get("output_directory", "../output"), exist_ok=True)
config.setdefault("ethics_enabled", True)
config.setdefault("router", {"fallback_agent": "analyst"})
config.setdefault("models", {
    "default": "llama2",
    "scribe": "llama2",
    "analyst": "llama2",
    "engineer": "codellama:7b"
})
config.setdefault("web_interface", {
    "port": 5000,
    "host": "0.0.0.0",
    "debug": True,
    "enable_file_uploads": True,
    "max_file_size": 5  # in MB
})

# Initialize the memory system (using enhanced memory if available)
try:
    memory = EnhancedMemory(memory_file='../enhanced_memory.json')
    print("[WEB] Using Enhanced Memory system")
except Exception as e:
    print(f"[WEB] Enhanced Memory not available, falling back to basic Memory: {e}")
    memory = Memory(memory_file='../memory.json')

# Initialize message bus
message_bus = MessageBus(save_path='../message_history.json')

# Initialize components
ethics_filter = EthicsFilter(overlay_path="../ethics/ethics_overlay.md", config=config)
sentinel_agent = SentinelAgent(config=config, ethics=ethics_filter, memory=memory)
quartermaster = QuartermasterAgent(config=config, memory=memory, sentinel=sentinel_agent)

# Initialize tools
calculator = CalculatorTool()
datetime_tool = DateTimeTool()
web_search_tool = WebSearchTool(quartermaster=quartermaster)
read_file_tool = ReadFileTool(quartermaster=quartermaster)
list_files_tool = ListFilesTool(quartermaster=quartermaster)
write_file_tool = WriteFileTool(quartermaster=quartermaster)
weather_tool = WeatherTool()
data_visualization_tool = DataVisualizationTool(quartermaster=quartermaster)
pdf_generator_tool = PdfGeneratorTool(quartermaster=quartermaster)

# Tool sets for different agents
common_tools = [
    calculator,
    datetime_tool,
    read_file_tool,
    list_files_tool
]

analyst_tools = common_tools + [
    web_search_tool,
    write_file_tool,
    weather_tool,
    data_visualization_tool
]

researcher_tools = common_tools + [
    web_search_tool,
    write_file_tool,
    pdf_generator_tool
]

engineer_tools = common_tools + [
    write_file_tool
]

scribe_tools = common_tools + [
    write_file_tool,
    pdf_generator_tool
]

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
    engineer=engineer_agent,
    tools=analyst_tools
)

researcher_agent = ResearcherAgent(
    config=config, memory=memory, quartermaster=quartermaster, sentinel=sentinel_agent,
    tools=researcher_tools
)

# Create a dictionary of agents for easier access
agents = {
    "scribe": scribe_agent,
    "engineer": engineer_agent,
    "analyst": analyst_agent,
    "researcher": researcher_agent,
    "quartermaster": quartermaster,
    "sentinel": sentinel_agent
}

# Initialize Planner Agent with access to other agents
planner_agent = PlannerAgent(
    config=config, memory=memory, quartermaster=quartermaster, sentinel=sentinel_agent,
    tools=common_tools, agents=agents
)

# Add planner to the agents dictionary
agents["planner"] = planner_agent

# Initialize message bus clients for each agent
message_clients = {}
for agent_name, agent in agents.items():
    message_clients[agent_name] = MessageBusClient(agent_name, message_bus)


def process_task_queue():
    """Background thread to process tasks from the queue"""
    while True:
        try:
            task_id, agent_name, task_content = task_queue.get(block=True)
            task_status[task_id] = "processing"
            
            try:
                print(f"[WEB] Processing task {task_id}: {agent_name} - {task_content[:50]}...")
                
                # Record in command history
                command_history.append({
                    "id": task_id,
                    "agent": agent_name,
                    "command": task_content,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Route to appropriate agent
                if agent_name in agents:
                    agent = agents[agent_name]
                    result = agent.run(task_content)
                    
                    # Store the result
                    task_results[task_id] = {
                        "agent": agent_name,
                        "command": task_content,
                        "result": result,
                        "status": "completed",
                        "timestamp": datetime.now().isoformat()
                    }
                    task_status[task_id] = "completed"
                else:
                    error_msg = f"Unknown agent: {agent_name}"
                    task_results[task_id] = {
                        "agent": agent_name,
                        "command": task_content,
                        "result": error_msg,
                        "status": "error",
                        "timestamp": datetime.now().isoformat()
                    }
                    task_status[task_id] = "error"
            
            except (EthicsViolation, SentinelException, ToolException) as se_blocked:
                error_msg = f"[Blocked or Tool Error by WITS System] {se_blocked}"
                task_results[task_id] = {
                    "agent": agent_name,
                    "command": task_content,
                    "result": error_msg,
                    "status": "blocked",
                    "timestamp": datetime.now().isoformat()
                }
                task_status[task_id] = "blocked"
            
            except Exception as e_agent_run:
                import traceback
                error_msg = f"[Agent Error - {agent_name}] An unexpected issue occurred: {str(e_agent_run)}\n{traceback.format_exc()}"
                task_results[task_id] = {
                    "agent": agent_name,
                    "command": task_content,
                    "result": error_msg,
                    "status": "error",
                    "timestamp": datetime.now().isoformat()
                }
                task_status[task_id] = "error"
            
            finally:
                task_queue.task_done()
        
        except Exception as e:
            print(f"[WEB] Error in task queue processing: {e}")
            time.sleep(1)  # Avoid tight loop if recurring error


# Start the background task processing thread
task_processor = threading.Thread(target=process_task_queue, daemon=True)
task_processor.start()


@app.route('/')
def index():
    """Render the main index page"""
    agents_list = [
        {"name": "analyst", "description": "Research, analyze information, and provide insights"},
        {"name": "engineer", "description": "Generate and modify code"},
        {"name": "scribe", "description": "Create written content like articles, stories, and documentation"},
        {"name": "researcher", "description": "Conduct thorough, methodical research on complex topics"},
        {"name": "planner", "description": "Break down complex goals into tasks and coordinate workflows"},
        {"name": "quartermaster", "description": "Manage files, resources, and goals"},
        {"name": "sentinel", "description": "Monitor ethical guidelines and approve actions"}
    ]
    
    # Get recent files
    try:
        files_output = quartermaster.list_files()
        files_list = files_output.split('\n')[1:]  # Skip the header line
        files = [f.strip('- ') for f in files_list]
    except Exception as e:
        files = [f"Error listing files: {str(e)}"]
    
    # Get current goals
    goals_text = memory.list_goals()
    goals = [g for g in goals_text.split('\n') if g.strip()]
    
    return render_template('index.html', agents=agents_list, files=files, goals=goals)


@app.route('/api/submit_task', methods=['POST'])
def submit_task():
    """API endpoint to submit a task to an agent"""
    data = request.json
    command = data.get('command', '').strip()
    
    if not command:
        return jsonify({"status": "error", "message": "Command cannot be empty"}), 400
    
    # Use the router to determine which agent should handle this
    agent_name, task_content = route_task(command, fallback_agent=config["router"].get("fallback_agent", "analyst"))
    
    # Generate a task ID
    task_id = f"task_{int(time.time())}_{agent_name}"
    
    # Add to the task queue
    task_queue.put((task_id, agent_name, task_content))
    task_status[task_id] = "queued"
    
    return jsonify({
        "status": "success",
        "task_id": task_id,
        "agent": agent_name,
        "message": f"Task submitted to {agent_name}"
    })


@app.route('/api/task_status/<task_id>', methods=['GET'])
def get_task_status(task_id):
    """API endpoint to check the status of a task"""
    status = task_status.get(task_id, "unknown")
    result = task_results.get(task_id, {})
    
    return jsonify({
        "task_id": task_id,
        "status": status,
        "result": result
    })


@app.route('/api/agents', methods=['GET'])
def get_agents():
    """API endpoint to get a list of all available agents"""
    agents_list = [
        {"name": "analyst", "description": "Research, analyze information, and provide insights"},
        {"name": "engineer", "description": "Generate and modify code"},
        {"name": "scribe", "description": "Create written content like articles, stories, and documentation"},
        {"name": "researcher", "description": "Conduct thorough, methodical research on complex topics"},
        {"name": "planner", "description": "Break down complex goals into tasks and coordinate workflows"},
        {"name": "quartermaster", "description": "Manage files, resources, and goals"},
        {"name": "sentinel", "description": "Monitor ethical guidelines and approve actions"}
    ]
    
    return jsonify(agents_list)


@app.route('/api/files', methods=['GET'])
def get_files():
    """API endpoint to get a list of files in the output directory"""
    try:
        files_output = quartermaster.list_files()
        files_list = files_output.split('\n')[1:]  # Skip the header line
        files = [f.strip('- ') for f in files_list]
        return jsonify({"status": "success", "files": files})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/file/<path:filename>', methods=['GET'])
def get_file(filename):
    """API endpoint to get the content of a file"""
    try:
        content = quartermaster.read_file(filename)
        
        # Determine file type and set appropriate headers
        if filename.endswith(('.png', '.jpg', '.jpeg', '.gif', '.svg')):
            # For binary files, we need to redirect to a proper file endpoint
            return redirect(url_for('serve_file', filename=filename))
        
        return jsonify({
            "status": "success",
            "filename": filename,
            "content": content
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/files/<path:filename>')
def serve_file(filename):
    """Serve a file from the output directory"""
    return send_from_directory(config.get("output_directory", "../output"), filename)


@app.route('/api/goals', methods=['GET'])
def get_goals():
    """API endpoint to get a list of current goals"""
    goals_text = memory.list_goals()
    goals = [g for g in goals_text.split('\n') if g.strip()]
    return jsonify({"status": "success", "goals": goals})


@app.route('/api/goals', methods=['POST'])
def add_goal():
    """API endpoint to add a new goal"""
    data = request.json
    task = data.get('task', '').strip()
    agent = data.get('agent')
    
    if not task:
        return jsonify({"status": "error", "message": "Task description cannot be empty"}), 400
    
    memory.add_goal(task, agent=agent)
    
    return jsonify({
        "status": "success",
        "message": f"Goal added successfully"
    })


@app.route('/api/history', methods=['GET'])
def get_history():
    """API endpoint to get command history"""
    return jsonify({
        "status": "success",
        "history": command_history
    })


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """API endpoint to upload a file"""
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"}), 400
    
    if file:
        # Check file size
        max_size = config.get("web_interface", {}).get("max_file_size", 5) * 1024 * 1024  # Convert to bytes
        if request.content_length > max_size:
            return jsonify({"status": "error", "message": f"File too large (max {max_size/1024/1024}MB)"}), 413
        
        filename = os.path.join(config.get("output_directory", "../output"), file.filename)
        file.save(filename)
        return jsonify({
            "status": "success",
            "message": f"File uploaded successfully",
            "filename": file.filename
        })


@app.route('/agent/<agent_name>')
def agent_view(agent_name):
    """Render the agent-specific view"""
    if agent_name not in agents:
        return redirect(url_for('index'))
    
    agent_descriptions = {
        "analyst": "Research, analyze information, and provide insights",
        "engineer": "Generate and modify code",
        "scribe": "Create written content like articles, stories, and documentation",
        "researcher": "Conduct thorough, methodical research on complex topics",
        "planner": "Break down complex goals into tasks and coordinate workflows",
        "quartermaster": "Manage files, resources, and goals",
        "sentinel": "Monitor ethical guidelines and approve actions"
    }
    
    # Get agent-specific information
    if agent_name == "planner":
        # Get all plans for the planner
        try:
            plans = planner_agent.list_plans()
        except:
            plans = "No plans available"
    elif agent_name == "researcher":
        # Get all research projects for the researcher
        try:
            research_projects = researcher_agent.list_research_projects()
        except:
            research_projects = "No research projects available"
    else:
        plans = None
        research_projects = None
    
    # Get recent outputs from this agent
    recent_output = memory.recall_agent_output(agent_name)
    
    return render_template(
        'agent.html',
        agent_name=agent_name,
        agent_description=agent_descriptions.get(agent_name, ""),
        plans=plans,
        research_projects=research_projects,
        recent_output=recent_output
    )


@app.route('/files')
def files_view():
    """Render the files view"""
    try:
        files_output = quartermaster.list_files()
        files_list = files_output.split('\n')[1:]  # Skip the header line
        files = [f.strip('- ') for f in files_list]
    except Exception as e:
        files = [f"Error listing files: {str(e)}"]
    
    return render_template('files.html', files=files)


@app.route('/goals')
def goals_view():
    """Render the goals view"""
    goals_text = memory.list_goals()
    goals = [g for g in goals_text.split('\n') if g.strip()]
    
    completed_text = memory.list_completed()
    completed = [g for g in completed_text.split('\n') if g.strip()]
    
    return render_template('goals.html', goals=goals, completed=completed)


@app.route('/history')
def history_view():
    """Render the command history view"""
    return render_template('history.html', history=command_history)


@app.route('/settings')
def settings_view():
    """Render the settings view"""
    return render_template('settings.html', config=config)


@app.route('/api/settings', methods=['POST'])
def update_settings():
    """API endpoint to update settings"""
    data = request.json
    
    # Update configuration
    for key, value in data.items():
        if key in config:
            config[key] = value
    
    # Save configuration
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False)
        return jsonify({"status": "success", "message": "Settings updated successfully"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    web_config = config.get("web_interface", {})
    app.run(
        host=web_config.get("host", "0.0.0.0"),
        port=web_config.get("port", 5000),
        debug=web_config.get("debug", True)
    )