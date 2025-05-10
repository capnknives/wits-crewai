
"""
Main control loop for WITS CrewAI.
Listens for voice or text commands and dispatches them to the appropriate agent.
"""
import os
import re
import sounddevice as sd
import numpy as np
import whisper
import yaml

from core.memory import Memory
from core.router import route_task
from core.voice import get_voice_input
from ethics.ethics_rules import EthicsFilter

from agents.scribe_agent import ScribeAgent
from agents.analyst_agent import AnalystAgent
from agents.engineer_agent import EngineerAgent
from agents.quartermaster_agent import QuartermasterAgent
from agents.sentinel_agent import SentinelAgent

# Load config
config = {}
config_path = "config.yaml"
if os.path.exists(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
else:
    config = {
        "internet_access": False,
        "voice_input": False,
        "voice_input_duration": 5,
        "whisper_model": "base",
        "allow_code_execution": False,
        "output_directory": "output"
    }

# Init subsystems
memory = Memory()
ethics = EthicsFilter("ethics/ethics_overlay.md", config)
sentinel = SentinelAgent(config, memory, ethics)
quartermaster = QuartermasterAgent(config, memory, ethics)
scribe_agent = ScribeAgent(config, memory, ethics)
analyst_agent = AnalystAgent(config, memory, ethics)
engineer_agent = EngineerAgent(config, memory, ethics)

# Voice setup
use_voice = config.get("voice_input", False)
whisper_model = None
if use_voice:
    try:
        whisper_model = whisper.load_model(config.get("whisper_model", "base"))
        sd.default.samplerate = 16000
        sd.default.channels = 1
        sd.default.dtype = 'float32'
    except Exception as e:
        print(f"[VOICE ERROR] {e}")
        use_voice = False

print("\nWITS CrewAI Initialized. Use commands like:")
print(" - Scribe, write a chapter on AI")
print(" - Quartermaster, list goals")
print(" - Sentinel, list rules")
print("Say 'exit' or press Ctrl+C to quit.\n")

# Main loop
while True:
    try:
        if use_voice and whisper_model:
            print("Listening...")
            audio = sd.rec(int(config["voice_input_duration"] * sd.default.samplerate))
            sd.wait()
            audio_data = audio.flatten().astype('float32')
            result = whisper_model.transcribe(audio_data)
            user_input = result.get("text", "").strip()
            print(f"You said: {user_input}")
        else:
            user_input = input(">> ").strip()

        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Shutting down.")
            break

        agent_key, task = route_task(user_input)

        if agent_key == "scribe":
            output = scribe_agent.run(task)
        elif agent_key == "analyst":
            output = analyst_agent.run(task)
        elif agent_key == "engineer":
            output = engineer_agent.run(task)
        elif agent_key == "quartermaster":
            output = quartermaster.run(task)
        elif agent_key == "sentinel":
            output = sentinel.run(task)
        else:
            output = "Agent not recognized."

        sentinel.ethics.check_text(output)
        print(output)

    except KeyboardInterrupt:
        print("\n[CTRL+C] Exiting...")
        break
    except Exception as e:
        print(f"[ERROR] {e}")
