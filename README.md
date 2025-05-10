# WITS-CREWAI

## Overview

WITS-CREWAI is an advanced multi-agent AI system designed to perform a variety of tasks through collaborative intelligence. It features specialized agents, a suite of tools, an enhanced memory system, and both command-line and web-based interfaces for interaction. The system leverages Ollama for its language model capabilities and incorporates an ethics filter for safe and responsible operation.

## Features

* **Multi-Agent Architecture:** Employs a crew of specialized AI agents, each designed for specific types of tasks.
* **Extensible Toolset:** Agents can utilize a variety of tools to perform actions like calculations, web searches, file operations, data visualization, and more.
* **Enhanced Memory System:** Features an `EnhancedMemory` component with an optional `VectorMemory` backend for more sophisticated information retrieval and context management. [cite: 29, 30]
* **Dual Interfaces:**
    * **Command-Line Interface (CLI):** For direct interaction and scripting, managed by `main.py`. [cite: 4]
    * **Web Interface:** A Flask-based application (`app.py`) providing a user-friendly way to interact with agents, manage files, track goals, and view history. [cite: 1]
* **Configurable:** System behavior, model selection, and features like internet access and code execution can be configured via `config.yaml`. [cite: 2]
* **Ethics and Safety:** Includes an `EthicsFilter` and a `SentinelAgent` to monitor and enforce ethical guidelines. [cite: 4, 27, 28]
* **Inter-Agent Communication:** Utilizes a `MessageBus` for structured communication between agents. [cite: 3, 4, 31]
* **Autonomous Planner:** Capable of autonomous goal processing and task coordination by the `PlannerAgent`. [cite: 4, 35]
* **Task Routing:** A `Router` component intelligently dispatches user commands to the most appropriate agent. [cite: 4, 32]
* **Voice Input:** Optional voice command input via Whisper. [cite: 2, 4, 33]

## Architecture

The WITS-CREWAI system is built around the following core components:

* **Main Control Loop (`main.py`):** Initializes the system, loads configuration, and manages the primary interaction loop (CLI or Web). [cite: 4]
* **Web Application (`app.py`):** Provides a Flask-based web interface for user interaction, task submission, file management, goal tracking, and system settings. [cite: 1]
* **Configuration (`config.yaml`):** A YAML file to configure system settings, agent models, internet access, code execution permissions, and web interface parameters. [cite: 2]
* **Agents:** Specialized AI entities responsible for different functions:
    * **AnalystAgent:** Researches, analyzes information, and provides insights. Can use tools like web search and file reading. [cite: 34]
    * **EngineerAgent:** Generates, modifies, and debugs code. [cite: 35]
    * **ScribeAgent:** Creates written content such as articles, stories, and reports. [cite: 37]
    * **ResearcherAgent:** Conducts in-depth, methodical research, manages research projects, and produces comprehensive reports. [cite: 3, 36]
    * **PlannerAgent:** Breaks down complex goals into tasks, manages dependencies, and coordinates workflows across agents. [cite: 3, 36]
    * **QuartermasterAgent:** Manages files, resources, and system goals. [cite: 38]
    * **SentinelAgent:** Monitors actions for ethical compliance and approves or blocks operations based on predefined rules. [cite: 39]
* **Tools:** A collection of utilities that agents can use to perform specific actions:
    * `CalculatorTool` [cite: 41]
    * `DateTimeTool` [cite: 43]
    * `WebSearchTool` [cite: 48]
    * `ReadFileTool` [cite: 45]
    * `ListFilesTool` [cite: 44]
    * `WriteFileTool` [cite: 46]
    * `WeatherTool` [cite: 3, 47]
    * `DataVisualizationTool` [cite: 3, 42]
    * `PdfGeneratorTool` [cite: 3, 40]
* **Core Systems:**
    * **EnhancedMemory / VectorMemory:** Manages short-term and long-term memory, goals, and agent outputs. [cite: 29, 30, 31]
    * **EthicsFilter & Ethics Overlay:** Defines and enforces ethical guidelines for agent behavior. [cite: 27, 28]
    * **Router:** Directs user input to the most appropriate agent. [cite: 32]
    * **MessageBus:** Facilitates communication between agents. [cite: 3, 31]
    * **Voice Input:** Handles voice-to-text transcription. [cite: 33]
* **Output Directories:** The system uses an `output` directory by default, which can contain subdirectories for `visualizations`, `documents`, `research_notes`, and `plans`. [cite: 2, 3]

## Setup and Installation

1.  **Prerequisites:**
    * Python (version not explicitly stated, but common Python 3 features are used).
    * Ollama installed and running with the desired language models (e.g., `llama2`, `codellama:7b` as per default `config.yaml` [cite: 2]).

2.  **Clone the repository (if applicable) or ensure all project files are in place.**

3.  **Install Dependencies:**
    Open your terminal in the project's root directory and run:
    ```bash
    pip install -r requirements.txt
    ```
    This will install all necessary Python packages listed in `requirements.txt`[cite: 7], including Flask, Ollama client, Whisper, ReportLab, Matplotlib, etc.

4.  **Create Required Directories (if not present):**
    The system will attempt to create these, but you can also pre-create them: [cite: 3]
    ```bash
    mkdir -p output/visualizations
    mkdir -p output/documents
    mkdir -p output/research_notes
    mkdir -p output/plans
    mkdir -p research_notes # For ResearcherAgent if output dir is customized
    ```

## Configuration

The primary configuration for WITS-CREWAI is managed through the `config.yaml` file [cite: 2] located in the project root. This file allows you to customize:

* **General Settings:**
    * `internet_access`: Enable/disable internet access for agents.
    * `allow_code_execution`: Enable/disable the execution of code generated by agents (use with caution).
    * `ethics_enabled`: Enable/disable the ethics filter.
    * `output_directory`: Specify the default directory for agent-generated files.
* **Voice Input Settings:**
    * `voice_input`: Enable/disable voice command input.
    * `voice_input_duration`: Maximum listening duration for voice commands.
    * `whisper_model`: Whisper model to use for transcription.
    * `whisper_fp16`: Use half-precision for Whisper model.
* **Model Selection:**
    * Specify default and agent-specific Ollama models (e.g., `scribe: llama2`, `engineer: codellama:7b`).
* **Router Settings:**
    * `fallback_agent`: Default agent if a command cannot be routed.
* **Web Interface Settings:** [cite: 1, 2]
    * `enabled`: Enable/disable the web interface.
    * `port`: Port for the web server.
    * `host`: Host address for the web server.
    * `debug`: Run Flask in debug mode.
    * `enable_file_uploads`: Allow file uploads via the web interface.
    * `max_file_size`: Maximum file upload size in MB.
* **Autonomous Planner Settings:**
    * `enabled`: Enable autonomous goal processing.
    * `check_interval_seconds`: How often the planner checks for pending goals.
    * `max_goal_retries`: Maximum retries for an autonomous goal.

Make sure to review and adjust `config.yaml` according to your needs and local Ollama setup before running the application.

## Usage

WITS-CREWAI can be run in two modes:

### 1. Command-Line Interface (CLI) Mode

To run the system in CLI mode:
```bash
python main.py
```
Or, if you have a custom config file:
```bash
python main.py --config your_config.yaml
```
You can then interact with the agents by typing commands like:
* `Analyst, research recent AI advancements.`
* `Scribe, write a short story about a space cat.`
* `Engineer, generate a Python script to list files in a directory.`
* `Planner, create and execute plan for building a personal website.` [cite: 3]
* `Quartermaster, list goals.`
* `Sentinel, list rules.`

### 2. Web Interface Mode

To run the system with the web interface:
```bash
python main.py --web
```
Alternatively, you can enable the web interface by setting `web_interface.enabled: true` in your `config.yaml` file [cite: 2] and then running `python main.py`.

Once started, the web interface is typically accessible at `http://<host>:<port>` (e.g., `http://0.0.0.0:5000` or `http://localhost:5000` by default [cite: 1, 2]).

The web interface provides: [cite: 1, 3]
* **Dashboard:** Quick access to agents, command input, and overview of files/goals.
* **Agent Pages:** Dedicated interfaces for each agent.
* **Files Management:** Upload, view, and delete files.
* **Goals Tracking:** Create, manage, and track goals.
* **Command History:** View past commands and results.
* **Settings:** Configure system parameters via a UI.

## Implemented Agents

The system includes the following specialized agents:

* **ScribeAgent:** Responsible for content creation, such as writing articles, stories, or reports. [cite: 37]
* **AnalystAgent:** Focuses on research, data analysis, and providing insights. [cite: 34]
* **EngineerAgent:** Specialized in generating, modifying, and debugging code. [cite: 35]
* **ResearcherAgent:** Conducts in-depth, methodical research, manages research projects, tracks citations, and provides comprehensive research reports. [cite: 3, 36]
* **PlannerAgent:** Breaks down complex goals into manageable tasks, manages dependencies between tasks, and coordinates the workflow across different agents. [cite: 3, 36]
* **QuartermasterAgent:** Manages files (reading, writing, listing), resources, and system goals. Also handles external actions like internet searches. [cite: 38]
* **SentinelAgent:** Monitors system activities, enforces ethical guidelines, and approves or blocks potentially sensitive actions. [cite: 39]

## Implemented Tools

Agents can leverage the following tools to perform their tasks:

* **CalculatorTool:** Evaluates simple mathematical arithmetic expressions. [cite: 41]
* **DateTimeTool:** Returns the current date and time, with optional timezone and formatting. [cite: 43]
* **WebSearchTool:** Performs internet searches via DuckDuckGo (through Quartermaster) and returns a summary. [cite: 48]
* **ReadFileTool:** Reads the content of a specified file. [cite: 45]
* **ListFilesTool:** Lists files and subdirectories within a specified directory. [cite: 44]
* **WriteFileTool:** Writes or overwrites content to a specified file. [cite: 46]
* **WeatherTool:** Fetches current weather information for a given location. [cite: 3, 47]
* **DataVisualizationTool:** Creates visualizations (bar charts, line graphs, pie charts, etc.) from provided data. [cite: 3, 42]
* **PdfGeneratorTool:** Creates PDF documents from text content, supporting formatting, tables, and images. [cite: 3, 40]

## Future Enhancements

Based on the project's enhancement summary[cite: 3], potential future enhancements include:

* User authentication and multi-user support for the web interface.
* A plugin system for third-party extensions.
* Further integration with vector databases for improved memory retrieval (VectorMemory is a step in this direction).
* Fine-tuning specific language models for different agents.
* WebSocket integration for real-time updates in the web interface.
* Containerization (e.g., Docker) for easier deployment.

## Contributing

(Placeholder for contribution guidelines. If you'd like to contribute, please outline the process here, e.g., fork the repository, create a feature branch, submit a pull request, coding standards, etc.)

## License

(Placeholder for license information. Please specify the license under which this project is distributed, e.g., MIT, Apache 2.0, GPL, etc.)

---

This README provides a comprehensive overview of the WITS-CREWAI project based on the provided files. You can further customize it with specific examples, screenshots (for the web UI), and more detailed architectural diagrams if needed.
