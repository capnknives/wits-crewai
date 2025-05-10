"""
Main control loop for WITS CrewAI.
Listens for voice or text commands and dispatches them to the appropriate agent.
"""

import os
import re
import sounddevice as sd
import yaml
import json
import argparse
import sys

from core.memory import Memory
from core.enhanced_memory import EnhancedMemory
from ethics.ethics_rules import EthicsFilter, EthicsViolation
from core.voice import get_voice_input
from core.router import route_task
from core.message_bus import MessageBus, MessageBusClient

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


def parse_arguments():
    parser = argparse.ArgumentParser(description="WITS CrewAI - Multi-agent AI system")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--web", action="store_true", help="Start the web interface instead of CLI")
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Load configuration
    config = {}
    config_path = args.config

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

    config.setdefault("internet_access", True)
    config.setdefault("voice_input", False)
    config.setdefault("voice_input_duration", 5)
    config.setdefault("whisper_model", "base")
    config.setdefault("whisper_fp16", False)
    config.setdefault("allow_code_execution", False)
    config.setdefault("output_directory", "output")
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
        "enabled": True,
        "port": 5000,
        "host": "0.0.0.0",
        "debug": True,
        "enable_file_uploads": True,
        "max_file_size": 5
    })

    # Start web interface if requested
    if args.web or config.get("web_interface", {}).get("enabled", False):
        # Import here to avoid unnecessary dependencies when running CLI mode
        from app import app
        web_config = config.get("web_interface", {})
        print(f"[WEB] Starting web interface on {web_config.get('host', '0.0.0.0')}:{web_config.get('port', 5000)}")
        app.run(
            host=web_config.get("host", "0.0.0.0"),
            port=web_config.get("port", 5000),
            debug=web_config.get("debug", True)
        )
        return

    # Initialize memory system (prefer vector memory first, then enhanced memory, then basic memory)
    try:
        from core.vector_memory import VectorMemory
        memory = VectorMemory(memory_file='vector_memory.json', index_file='vector_index.bin')
        print("[SYSTEM] Using Vector Memory system")
    except Exception as e:
        print(f"[SYSTEM] Vector Memory not available, falling back to Enhanced Memory: {e}")
        try:
            from core.enhanced_memory import EnhancedMemory
            memory = EnhancedMemory(memory_file='enhanced_memory.json')
            print("[SYSTEM] Using Enhanced Memory system")
        except Exception as e2:
            print(f"[SYSTEM] Enhanced Memory not available, falling back to basic Memory: {e2}")
            from core.memory import Memory
            memory = Memory()
    # Initialize message bus
    message_bus = MessageBus(save_path='message_history.json')

    # Initialize components
    ethics_filter = EthicsFilter(overlay_path="ethics/ethics_overlay.md", config=config)
    sentinel_agent = SentinelAgent(config=config, ethics=ethics_filter, memory=memory)
    quartermaster = QuartermasterAgent(config=config, memory=memory, sentinel=sentinel_agent)

    # Initialize common tools
    calculator = CalculatorTool()
    datetime_tool = DateTimeTool()
    web_search_tool = WebSearchTool(quartermaster=quartermaster)
    read_file_tool = ReadFileTool(quartermaster=quartermaster)
    list_files_tool = ListFilesTool(quartermaster=quartermaster)
    write_file_tool = WriteFileTool(quartermaster=quartermaster)
    weather_tool = WeatherTool()
    data_visualization_tool = DataVisualizationTool(quartermaster=quartermaster)
    pdf_generator_tool = PdfGeneratorTool(quartermaster=quartermaster)

    # Define tool sets for different agents
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

    planner_tools = common_tools

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
    # Also pass the message_bus_client if Planner needs to broadcast alerts
    planner_message_client = MessageBusClient("planner", message_bus)
    planner_agent = PlannerAgent(
        config=config, memory=memory, quartermaster=quartermaster, sentinel=sentinel_agent,
        tools=planner_tools, agents=agents, message_bus_client=planner_message_client
    )

    # Add planner to the agents dictionary
    agents["planner"] = planner_agent

    # Initialize message bus clients for each agent (if they need to actively use the bus)
    # For now, only Planner explicitly takes a client for potential alerts.
    # Other agents can be given clients if they need to send/receive messages proactively.
    # Example:
    # analyst_message_client = MessageBusClient("analyst", message_bus)
    # analyst_agent.message_bus_client = analyst_message_client # If AnalystAgent is designed to use it

    # Set up voice input if enabled
    use_voice = config.get("voice_input", False)
    if use_voice:
        try:
            sd.default.samplerate = 16000
            sd.default.channels = 1
            sd.default.dtype = 'float32'
            print(f"Voice input enabled. Whisper model '{config.get('whisper_model')}' will be used.")
        except Exception as e_voice_setup:
            print(f"Sounddevice/Voice input initialization failed: {e_voice_setup}")
            print("Reverting to text input mode.")
            use_voice = False

    # Display current goals if any
    goals_list = memory.list_goals()
    if goals_list and "No current goals" not in goals_list:
        print("\nCurrent Goals:")
        print(goals_list)

    # Welcome message
    print("\nWelcome to WITS CrewAI - Enhanced Edition!")
    print("Available agents: Analyst, Engineer, Scribe, Researcher, Planner, Quartermaster, Sentinel")
    print("Example: 'Analyst, search for today's weather in London and create a visualization'")
    print("Example: 'Researcher, research on artificial general intelligence'")
    print("Example: 'Planner, create and execute plan for building a personal website'") # Updated example
    print("Say 'exit' or 'quit' to stop.\n")

    # Main interaction loop
    while True:
        try:
            user_input_text = ""
            if use_voice:
                print("Listening for voice command...")
                whisper_use_fp16 = config.get("whisper_fp16", False)
                current_whisper_model = config.get("whisper_model", "base")
                voice_duration = config.get("voice_input_duration", 5)
                user_input_text = get_voice_input(
                    duration=voice_duration, model_name=current_whisper_model, whisper_fp16=whisper_use_fp16
                )
                if not user_input_text:
                    print("...No voice detected or transcribed. Try again or use text.")
                    continue
                else:
                    print(f'You said: "{user_input_text}"')
            else:
                user_input_text = input(">> ").strip()
                if not user_input_text:
                    continue

            if user_input_text.lower() in ["exit", "quit", "bye", "shutdown"]:
                print("Shutting down WITS CrewAI. Goodbye!")
                message_bus.shutdown() # Shutdown message bus gracefully
                break

            router_settings = config.get("router", {})
            fallback_agent_name = router_settings.get("fallback_agent", "analyst")
            agent_name, task_content = route_task(user_input_text, fallback_agent=fallback_agent_name)

            if agent_name in agents:
                goal_ref = re.search(r'goal\s+(\d+)', task_content, flags=re.IGNORECASE)
                if goal_ref:
                    goal_index = int(goal_ref.group(1))
                    current_goals_data = memory.get_goals_list() # Assumes Memory has get_goals_list()
                    if 1 <= goal_index <= len(current_goals_data):
                        goal_task_text = current_goals_data[goal_index - 1].get('task')
                        if goal_task_text:
                            task_content = re.sub(r'goal\s+\d+', goal_task_text, task_content, flags=re.IGNORECASE).strip()
                            print(f"(Substituted goal {goal_index} ('{goal_task_text[:30]}...') into task for {agent_name})")

            result_text = ""
            try:
                if agent_name in agents:
                    agent = agents[agent_name]
                    print(f"[MAIN] Dispatching to Agent: {agent_name.capitalize()}, Task: {task_content[:70]}...")
                    result_text = agent.run(task_content)
                    if result_text and isinstance(result_text, str):
                        memory.remember_agent_output(agent_name, result_text) # Using enhanced memory method
                else:
                    result_text = f"Agent '{agent_name}' was routed but not recognized. Check router/main.py."

            except (EthicsViolation, SentinelException, ToolException) as se_blocked:
                print(f"[CONTROLLED_EXCEPTION] Type: {type(se_blocked).__name__}, Message: {se_blocked}")
                result_text = f"[Blocked or Tool Error by WITS System] {se_blocked}"
            except Exception as e_agent_run:
                print(f"[AGENT_EXECUTION_ERROR] Agent: '{agent_name}', Type: {type(e_agent_run).__name__}, Error: {e_agent_run}")
                import traceback
                print(traceback.format_exc())
                result_text = f"[Agent Error - {agent_name}] An unexpected issue occurred. Please check logs or try rephrasing."

            print(f"\n{result_text}\n")

        except KeyboardInterrupt:
            print("\nKeyboard interrupt detected. Shutting down WITS CrewAI...")
            message_bus.shutdown() # Shutdown message bus gracefully
            break
        except Exception as e_main_loop:
            print(f"\n[FATAL ERROR IN MAIN LOOP] Type: {type(e_main_loop).__name__}, Error: {e_main_loop}")
            import traceback
            print(traceback.format_exc())
            print("The AI crew encountered a critical problem and needs to shut down. Please check logs.")
            message_bus.shutdown() # Attempt to shutdown message bus
            break


if __name__ == "__main__":
    main()