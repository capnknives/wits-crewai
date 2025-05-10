"""
Main control loop for WITS CrewAI.
Listens for voice or text commands and dispatches them to the appropriate agent.
Includes autonomous goal processing by the PlannerAgent.
"""

import os
import re
import sounddevice as sd
import yaml
import json
import argparse
import sys
import time # Added for autonomous check interval
from datetime import datetime # Potentially used by memory/goal updates

# Core WITS components
from core.enhanced_memory import EnhancedMemory # Using your updated EnhancedMemory
from ethics.ethics_rules import EthicsFilter, EthicsViolation
from core.voice import get_voice_input
from core.router import route_task
from core.message_bus import MessageBus, MessageBusClient 

# Agent classes
from agents.scribe_agent import ScribeAgent
from agents.analyst_agent import AnalystAgent
from agents.engineer_agent import EngineerAgent
from agents.researcher_agent import ResearcherAgent
from agents.planner_agent import PlannerAgent # Your updated PlannerAgent
from agents.quartermaster_agent import QuartermasterAgent
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


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="WITS CrewAI - Multi-agent AI system")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--web", action="store_true", help="Start the web interface instead of CLI")
    return parser.parse_args()


def main():
    """Main function to run the WITS CrewAI system."""
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
                    print(f"[CONFIG] Warning: '{config_path}' is empty or not valid YAML. Using defaults.")
        except Exception as e:
            print(f"[CONFIG] Error loading '{config_path}': {e}. Using defaults.")
    else:
        print(f"[CONFIG] Warning: '{config_path}' not found. Using defaults.")

    # Set defaults for various configuration sections
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
        "default": "llama2", "scribe": "llama2", "analyst": "llama2",
        "engineer": "codellama:7b", "researcher": "llama2", "planner": "llama2"
    })
    config.setdefault("web_interface", {
        "enabled": config.get("web_interface", {}).get("enabled", False), # Preserve user's web setting if it exists
        "port": 5000, "host": "0.0.0.0", "debug": True,
        "enable_file_uploads": True, "max_file_size": 5
    })
    config.setdefault("autonomous_planner", { # Autonomous planner settings
        "enabled": True, 
        "check_interval_seconds": 30, 
        "max_goal_retries": 2 
    })

    # Start web interface if --web flag is used or web_interface.enabled is true in config
    if args.web or config.get("web_interface", {}).get("enabled", False):
        try:
            from app import app 
            web_cfg = config.get("web_interface", {})
            print(f"[WEB] Starting web interface on {web_cfg.get('host')}:{web_cfg.get('port')}")
            # Note: app.run() is blocking. If CLI needs to run too, this structure needs change.
            app.run(host=web_cfg.get('host'), port=web_cfg.get('port'), debug=web_cfg.get('debug'))
        except ImportError:
            print("[ERROR] Web interface components (e.g., Flask) not found. Cannot start web mode.")
        except Exception as e_webapp:
            print(f"[ERROR] Failed to start web application: {e_webapp}")
        return # Exit main if web mode is started or fails to start

    # Initialize Memory System (Using EnhancedMemory)
    try:
        # If you have VectorMemory and want to use it, uncomment and adapt:
        # from core.vector_memory import VectorMemory
        # memory = VectorMemory(memory_file='vector_memory.json', index_file='vector_index.bin')
        # print("[SYSTEM] Using Vector Memory system")
        memory = EnhancedMemory(memory_file='enhanced_memory.json') # Ensure this uses your updated version
        print("[SYSTEM] Using Enhanced Memory system.")
    except ImportError:
        print("[SYSTEM_ERROR] EnhancedMemory module not found. Ensure 'core/enhanced_memory.py' is correct. Exiting.")
        return
    except Exception as e_mem_init:
        print(f"[SYSTEM_ERROR] Error initializing EnhancedMemory: {e_mem_init}. Exiting.")
        return

    # Initialize Message Bus
    message_bus = MessageBus(save_path='message_history.json')

    # Initialize Core Components (Ethics, Sentinel, Quartermaster)
    ethics_filter = EthicsFilter(overlay_path="ethics/ethics_overlay.md", config=config)
    sentinel_agent = SentinelAgent(config=config, ethics=ethics_filter, memory=memory) 
    quartermaster = QuartermasterAgent(config=config, memory=memory, sentinel=sentinel_agent)

    # Initialize Tools
    calculator = CalculatorTool()
    datetime_tool = DateTimeTool()
    web_search_tool = WebSearchTool(quartermaster=quartermaster)
    read_file_tool = ReadFileTool(quartermaster=quartermaster)
    list_files_tool = ListFilesTool(quartermaster=quartermaster)
    write_file_tool = WriteFileTool(quartermaster=quartermaster)
    weather_tool = WeatherTool()
    data_visualization_tool = DataVisualizationTool(quartermaster=quartermaster)
    pdf_generator_tool = PdfGeneratorTool(quartermaster=quartermaster)

    common_tools = [calculator, datetime_tool, read_file_tool, list_files_tool]
    analyst_tools = common_tools + [web_search_tool, write_file_tool, weather_tool, data_visualization_tool]
    researcher_tools = common_tools + [web_search_tool, write_file_tool, pdf_generator_tool]
    engineer_tools = common_tools + [write_file_tool]
    scribe_tools = common_tools + [write_file_tool, pdf_generator_tool]
    planner_tools = common_tools

    # Initialize Agents
    scribe_agent = ScribeAgent(config=config, memory=memory, quartermaster=quartermaster, sentinel=sentinel_agent, tools=scribe_tools)
    engineer_agent = EngineerAgent(config=config, memory=memory, quartermaster=quartermaster, sentinel=sentinel_agent, tools=engineer_tools)
    analyst_agent = AnalystAgent(config=config, memory=memory, quartermaster=quartermaster, sentinel=sentinel_agent, engineer=engineer_agent, tools=analyst_tools)
    researcher_agent = ResearcherAgent(config=config, memory=memory, quartermaster=quartermaster, sentinel=sentinel_agent, tools=researcher_tools)
    
    agents_for_planner = {
        "scribe": scribe_agent, "engineer": engineer_agent, 
        "analyst": analyst_agent, "researcher": researcher_agent,
        "quartermaster": quartermaster
    }
    planner_mb_client = MessageBusClient("planner", message_bus) if message_bus else None
    planner_agent = PlannerAgent(
        config=config, memory=memory, quartermaster=quartermaster, sentinel=sentinel_agent,
        tools=planner_tools, agents=agents_for_planner, message_bus_client=planner_mb_client
    )

    agents = {
        "scribe": scribe_agent, "engineer": engineer_agent, "analyst": analyst_agent,
        "researcher": researcher_agent, "planner": planner_agent,
        "quartermaster": quartermaster, "sentinel": sentinel_agent
    }

    # Voice Input Setup
    use_voice = config.get("voice_input", False)
    if use_voice:
        try:
            sd.default.samplerate = 16000; sd.default.channels = 1; sd.default.dtype = 'float32'
            print(f"[VOICE] Voice input enabled. Whisper model '{config.get('whisper_model')}' will be used.")
        except Exception as e_voice_setup:
            print(f"[VOICE_ERROR] Sounddevice/Voice input initialization failed: {e_voice_setup}. Reverting to text input.")
            use_voice = False

    # Display initial goals
    goals_list_str = memory.list_goals() 
    if goals_list_str and "No current goals" not in goals_list_str:
        print("\nCurrent Goals:")
        print(goals_list_str)

    # Welcome Message
    print("\nWelcome to WITS CrewAI - Autonomous Edition!")
    print("Available agents: Analyst, Engineer, Scribe, Researcher, Planner, Quartermaster, Sentinel")
    print("Example: 'Analyst, research recent AI advancements.'")
    print("Example: 'Planner, create and execute plan for writing a short story about a space cat.'")
    print("Say 'exit' or 'quit' to stop.\n")

    # Autonomous Planner Variables
    last_autonomous_check_time = time.time() 
    autonomous_planner_cfg = config.get("autonomous_planner", {})
    autonomous_enabled = autonomous_planner_cfg.get("enabled", False)
    autonomous_interval = autonomous_planner_cfg.get("check_interval_seconds", 30)
    autonomous_max_retries = autonomous_planner_cfg.get("max_goal_retries", 2)

    if autonomous_enabled:
        print(f"[AUTONOMOUS_PLANNER] Mode: ENABLED. Check interval: {autonomous_interval}s, Max retries/goal: {autonomous_max_retries}.")
    else:
        print("[AUTONOMOUS_PLANNER] Mode: DISABLED.")

    # Main interaction loop
    while True:
        user_input_text = ""
        processed_user_command_this_iteration = False
        
        try:
            # --- User Input Handling ---
            if use_voice:
                print("Listening for voice command (or say 'skip autonomous' to yield)...")
                user_input_text = get_voice_input(
                    duration=config.get("voice_input_duration", 5), 
                    model_name=config.get("whisper_model", "base"), 
                    whisper_fp16=config.get("whisper_fp16", False)
                )
                if user_input_text and user_input_text.lower().strip() not in ["skip autonomous", "skip"]:
                    print(f'You said: "{user_input_text}"')
                    processed_user_command_this_iteration = True
                elif user_input_text.lower().strip() in ["skip autonomous", "skip"]:
                    print("Skipping user input for this cycle to allow autonomous check.")
                    user_input_text = "" 
                else:
                    print("...No voice detected or transcribed this cycle.")
                    user_input_text = ""
            else: 
                user_input_text = input(">> ").strip()
                if user_input_text:
                    processed_user_command_this_iteration = True

            if user_input_text.lower() in ["exit", "quit", "bye", "shutdown"]:
                print("Shutting down WITS CrewAI. Goodbye!")
                if message_bus: message_bus.shutdown()
                break
            
            if processed_user_command_this_iteration and user_input_text:
                print(f"\n--- Processing User Command: {user_input_text[:80]} ---")
                router_settings = config.get("router", {})
                fallback_agent_name = router_settings.get("fallback_agent", "analyst")
                agent_name, task_content = route_task(user_input_text, fallback_agent=fallback_agent_name)
                
                # Goal substitution logic
                if agent_name in agents: # Check if agent_name is valid before proceeding
                    goal_ref_match = re.search(r'goal\s+(\d+)', task_content, flags=re.IGNORECASE)
                    if goal_ref_match:
                        try:
                            goal_idx_from_user = int(goal_ref_match.group(1))
                            temp_goals_list = memory.get_goals_list() 
                            temp_goals_list.sort(key=lambda g: (-g.get('priority', 1), g.get('created', '')))

                            if 1 <= goal_idx_from_user <= len(temp_goals_list):
                                actual_goal_obj = temp_goals_list[goal_idx_from_user - 1]
                                goal_task_text = actual_goal_obj.get('task')
                                goal_id_ref = actual_goal_obj.get('id')
                                if goal_task_text:
                                    task_content = re.sub(r'goal\s+\d+', goal_task_text, task_content, flags=re.IGNORECASE).strip()
                                    print(f"(Info: Substituted goal {goal_idx_from_user} (ID: {goal_id_ref[:8]}...) into task for {agent_name})")
                            else:
                                print(f"(Warning: Goal index {goal_idx_from_user} out of range for current goals)")
                        except ValueError:
                            print(f"(Warning: Could not parse goal index from '{goal_ref_match.group(1)}')")
                
                result_text = ""
                try:
                    if agent_name in agents:
                        agent_instance = agents[agent_name]
                        print(f"[MAIN_USER_CMD] Dispatching to Agent: {agent_name.capitalize()}, Task: {task_content[:70]}...")
                        result_text = agent_instance.run(task_content, associated_goal_id_for_new_plan=None) 
                        if result_text and isinstance(result_text, str):
                            memory.remember_agent_output(agent_name, result_text) # Use the correct memory method
                    else:
                        result_text = f"Agent '{agent_name}' was routed but not recognized. Check router/main.py."
                except (EthicsViolation, SentinelException, ToolException) as controlled_exc:
                    print(f"[CONTROLLED_EXCEPTION] User Command: {type(controlled_exc).__name__}, Message: {controlled_exc}")
                    result_text = f"[Blocked or Tool Error by WITS System] {controlled_exc}"
                except Exception as e_agent_run:
                    print(f"[AGENT_EXECUTION_ERROR] User Command, Agent: '{agent_name}', Error: {e_agent_run}")
                    import traceback; print(traceback.format_exc())
                    result_text = f"[Agent Error - {agent_name}] An unexpected issue occurred. Please check logs."
                
                print(f"\n{result_text}\n")
                last_autonomous_check_time = time.time() 

            # --- Autonomous Planner Goal Processing ---
            if autonomous_enabled and \
               (not processed_user_command_this_iteration or \
                (time.time() - last_autonomous_check_time >= autonomous_interval)):
                
                print(f"\n--- [AUTONOMOUS_CHECK] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
                last_autonomous_check_time = time.time() 
                
                pending_goals = memory.get_pending_goals() 
                
                if pending_goals:
                    goal_to_process_autonomously = None
                    for goal_candidate in pending_goals: # Iterate to find eligible goal
                        if goal_candidate.get('retries', 0) < autonomous_max_retries:
                            goal_to_process_autonomously = goal_candidate
                            break 
                    
                    if goal_to_process_autonomously:
                        goal_id_auto = goal_to_process_autonomously.get('id')
                        goal_task_auto = goal_to_process_autonomously.get('task')
                        print(f"[AUTONOMOUS_PLANNER] Picked up Goal ID: {goal_id_auto[:8]}... Task: '{goal_task_auto[:60]}...'")
                        
                        memory.update_goal_status(goal_id_auto, "processing", 
                                                  result_summary="Autonomous planner picked up goal for planning.")
                        
                        planner_autonomous_command = f"Planner, create and execute plan for {goal_task_auto}"
                        
                        try:
                            print(f"[AUTONOMOUS_PLANNER] Dispatching to PlannerAgent for Goal ID {goal_id_auto[:8]}...")
                            planner_execution_result = planner_agent.run(
                                command=planner_autonomous_command, 
                                associated_goal_id_for_new_plan=goal_id_auto
                            )
                            print(f"[AUTONOMOUS_PLANNER] Autonomous execution result for Goal ID {goal_id_auto[:8]}...:\n{planner_execution_result}\n")
                            
                            final_goal_obj = memory.get_goal_by_id(goal_id_auto)
                            if final_goal_obj:
                                print(f"[AUTONOMOUS_PLANNER] Final status for Goal ID {goal_id_auto[:8]}... in memory: {final_goal_obj.get('status')}")
                            else:
                                print(f"[AUTONOMOUS_PLANNER] Warning: Goal ID {goal_id_auto[:8]}... no longer found in memory after planning attempt.")

                        except Exception as e_auto_dispatch:
                            print(f"[AUTONOMOUS_PLANNER_ERROR] Critical error during autonomous dispatch for Goal ID {goal_id_auto[:8]}...: {e_auto_dispatch}")
                            import traceback; print(traceback.format_exc())
                            memory.update_goal_status(goal_id_auto, "autonomous_failed", 
                                                      result_summary=f"Main loop autonomous dispatch error: {e_auto_dispatch}")
                    else:
                        print("[AUTONOMOUS_CHECK] No pending goals eligible for autonomous processing (all might have exceeded retries).")
                else:
                    print("[AUTONOMOUS_CHECK] No pending goals found.")
            elif not processed_user_command_this_iteration and not autonomous_enabled:
                time.sleep(0.1) # Brief pause if idle and autonomous is off

        except KeyboardInterrupt:
            print("\nKeyboard interrupt detected. Shutting down WITS CrewAI...")
            if message_bus: message_bus.shutdown()
            break
        except Exception as e_main_loop:
            print(f"\n[FATAL ERROR IN MAIN LOOP] Type: {type(e_main_loop).__name__}, Error: {e_main_loop}")
            import traceback; print(traceback.format_exc())
            print("The AI crew encountered a critical problem. Please check logs and consider restarting.")
            if message_bus: message_bus.shutdown()
            break

if __name__ == "__main__":
    main()
