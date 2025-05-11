# agents/researcher_agent.py
import ollama
import re
import json
import os # Added for os.path.join and os.path.exists
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# Assuming these are in the same directory or accessible via Python path
from .tools.base_tool import Tool, ToolException
from .sentinel_agent import SentinelException # Assuming SentinelException is in sentinel_agent
# If QuartermasterException is needed and defined elsewhere, import it.
# from ..quartermaster_agent import QuartermasterException # Example if it's in a parent agent dir

class ResearcherAgent:
    """
    A specialized agent that focuses on in-depth research tasks, synthesizing information
    from multiple sources, and creating comprehensive research reports.
    The Researcher is capable of methodical exploration of topics, citation tracking,
    and maintaining a research notebook for multi-stage inquiries.
    """

    def __init__(self, config, memory, quartermaster, sentinel, tools=None):
        self.config = config
        self.memory = memory # Should be an instance of EnhancedMemory or VectorMemory
        self.qm = quartermaster # Should be an instance of QuartermasterAgent
        self.sentinel = sentinel # Should be an instance of SentinelAgent
        self.tools = tools if tools else []
        model_conf = config.get("models", {})
        self.model_name = model_conf.get("researcher", model_conf.get("analyst", model_conf.get("default", "llama2")))

        # Ensure output_dir is correctly accessed from quartermaster if it's the source of truth for paths
        # If self.qm.output_dir is the base for all agent outputs:
        self.notebook_dir = os.path.join(self.qm.output_dir, "research_notes")
        try:
            if not os.path.exists(self.notebook_dir):
                os.makedirs(self.notebook_dir)
                print(f"[ResearcherAgent] Created research_notes directory at: {self.notebook_dir}")
        except Exception as e_mkdir:
            print(f"[ResearcherAgent_WARN] Could not create or verify notebook directory {self.notebook_dir}: {e_mkdir}")

        self.citation_index: Dict[str, Dict[str, Any]] = {} # Stores metadata about research projects
        self.current_research_id: Optional[str] = None

        self._load_research_history()

    def _load_research_history(self):
        """Load previous research history if available from research_notes/research_index.json."""
        # The path for QM should be relative to its output_dir if that's how QM operates
        research_index_path_relative_to_output = os.path.join("research_notes", "research_index.json")
        try:
            content = self.qm.read_file(research_index_path_relative_to_output)
            self.citation_index = json.loads(content)
            print(f"[ResearcherAgent] Loaded {len(self.citation_index)} previous research entries from '{research_index_path_relative_to_output}'.")
        except Exception as e: # Catching a broader exception as QM might raise its own
             print(f"[ResearcherAgent] No research_index.json found or error loading via Quartermaster ('{research_index_path_relative_to_output}'): {type(e).__name__} - {e}. Starting fresh.")
             self.citation_index = {}

    def _save_research_history(self):
        """Save research history to research_notes/research_index.json."""
        research_index_path_relative_to_output = os.path.join("research_notes", "research_index.json")
        # Check if the directory for the index file exists, create if not
        # This check is more for the self.notebook_dir which is where research_index.json resides.
        # The Quartermaster should handle creation of subdirectories if the base (output_dir) exists.
        try:
            # Ensure the parent directory for the index file exists within QM's scope
            # QM's write_file should ideally handle subdirectory creation within its output_dir.
            # For example, if research_index_path_relative_to_output is "research_notes/index.json",
            # QM.write_file should create "research_notes" if it doesn't exist under output_dir.
            # We assume self.notebook_dir (e.g., output/research_notes) has been ensured at __init__.
            
            # Only save if there's something to save or if the file previously existed (to overwrite with empty if cleared)
            # This logic might be too cautious; usually, we want to save the current state.
            # Forcing a save if the directory exists:
            if not self.citation_index and not os.path.exists(os.path.join(self.notebook_dir, "research_index.json")):
                 print("[ResearcherAgent] No research history to save and index file doesn't exist. Skipping save.")
                 return

            self.qm.write_file(
                research_index_path_relative_to_output,
                json.dumps(self.citation_index, indent=2)
            )
            print(f"[ResearcherAgent] Research index saved to '{research_index_path_relative_to_output}' via Quartermaster.")
        except Exception as e:
            print(f"[ResearcherAgent_ERROR] Error saving research index: {e}")

    def _get_tool_descriptions_for_llm(self) -> str:
        """Generate descriptions of available tools for the LLM prompt."""
        if not self.tools:
            return "No tools are currently available to you for direct use."
        descriptions = ["\nAVAILABLE TOOLS (use these to gather information if essential for the current step):"]
        for tool in self.tools:
            descriptions.append(f"- {tool.get_full_description_for_llm()}")
        return "\n".join(descriptions)

    def _generate_research_id(self, topic: str) -> str:
        """Generate a unique ID for this research topic."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Sanitize topic for use in filename/ID: replace non-alphanumeric with underscore, limit length
        topic_slug = re.sub(r'\W+', '_', topic.lower())[:30].strip('_')
        return f"research_{timestamp}_{topic_slug}"

    def _call_llm(self, prompt: str, research_id: Optional[str] = None):
        """Call the LLM with the given prompt."""
        print(f"\n[ResearcherAgent] Sending prompt to LLM (model: {self.model_name}, Research ID: {research_id or 'N/A'}):\n{prompt[:700]}...\n-----------------------------")
        try:
            ollama_response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={"temperature": 0.2, "num_ctx": 4096} # Lower temperature for more factual research
            )
            llm_output = ollama_response['response'].strip()
            print(f"[ResearcherAgent] LLM Output (Research ID: {research_id or 'N/A'}, first 600 chars):\n{llm_output[:600]}...\n-----------------------------")
            return llm_output
        except Exception as e_llm:
            print(f"[ResearcherAgent_ERROR] LLM call failed for Research ID {research_id or 'N/A'}: {e_llm}")
            return f"LLM Error: Could not get a response. Details: {str(e_llm)}"

    def _execute_tool(self, tool_name: str, tool_args: Dict[str, Any], research_id: str) -> str:
        """Execute a tool by name with the given arguments, and update notebook/sources."""
        print(f"[ResearcherAgent] Executing tool: {tool_name} with args: {tool_args} for Research ID: {research_id}")

        tool_instance = next((t for t in self.tools if t.name == tool_name), None)
        if not tool_instance:
            raise ToolException(f"Tool '{tool_name}' not found or not available to ResearcherAgent.")

        self.sentinel.approve_action(
            agent_name="ResearcherAgent",
            action_type=f"tool_use:{tool_instance.name}",
            detail=f"ResearchID: {research_id}, Args: {str(tool_args)}"
        )
        tool_output = tool_instance.execute(**tool_args)

        notebook_update_content = (
            f"\n### Tool Used: {tool_name}\n"
            f"**Arguments:** `{json.dumps(tool_args)}`\n"
            f"**Timestamp:** {datetime.now().isoformat()}\n"
            f"**Output:**\n```\n{tool_output}\n```\n"
        )
        self.append_to_notebook_section(research_id, "Information Gathering", notebook_update_content)

        if tool_name == "internet_search":
            source_title = f"Web search: {tool_args.get('query', 'Unknown query')}"
            self.add_source(research_id, {"type": "web_search", "title": source_title, "query": tool_args.get('query'), "summary": tool_output[:300]+"..."})
        elif tool_name == "read_file_content":
            source_title = f"File read: {tool_args.get('file_path', 'Unknown file')}"
            self.add_source(research_id, {"type": "file_read", "title": source_title, "path": tool_args.get('file_path'), "summary": tool_output[:300]+"..."})
        return tool_output

    def _generate_research_prompt(self, research_query: str, current_research_stage: str, accumulated_context: str, research_id: str) -> str:
        """Generates a prompt for the LLM based on the research stage and context."""
        tool_descriptions = self._get_tool_descriptions_for_llm()
        research_project_details_str = self.get_research_details(research_id) # Get current state as string

        system_prompt = (
            "You are a methodical Researcher Agent. Your goal is to conduct thorough research on the given query. "
            "You operate in stages: Planning, Information Gathering, Analysis, and Synthesis (Report Generation)."
            "Break down complex queries, use tools to gather information, analyze findings, and produce a comprehensive report."
        )
        tool_usage_instructions = (
            "\nTOOL USAGE:\n"
            "If you need to use a tool to gather more information or perform an action, respond ONLY with a JSON object like this:\n"
            "```json\n{\"tool_to_use\": \"tool_name\", \"tool_arguments\": {\"arg1\": \"value1\"}}\n```\n"
            "Do NOT explain why you are using the tool before the JSON. Just output the JSON. I will execute it and give you the observation."
            "Use tools one at a time. After receiving the tool's output (Observation), decide your next step."
        )
        stage_specific_guidance = ""
        if current_research_stage == "planning":
            stage_specific_guidance = (
                "\nCURRENT STAGE: PLANNING\n"
                "Your immediate task is to create a research plan. "
                "First, analyze the query. What are the main questions to answer? "
                "Second, outline the steps you will take for this research. "
                "For each step in your plan, note what information you'd need and what type of tool might be useful "
                "(e.g., 'Use internet_search for recent developments on X', 'Use read_file_content for internal_report.md on Y'). "
                "Do NOT attempt to list all tool JSONs now. Only create the textual research plan. "
                "If, and only if, you need to use a tool RIGHT NOW to gather some preliminary information *before you can even create your plan* "
                "(e.g., to understand a core concept in the query if the accumulated context is empty or insufficient), then output the JSON for that single tool. "
                "Otherwise, provide your textual research plan as your response for this step. Focus on the plan structure."
            )
        elif current_research_stage == "information_gathering":
            stage_specific_guidance = (
                "\nCURRENT STAGE: INFORMATION GATHERING\n"
                "1. Review your research plan (from the notebook or previous context) and the original query.\n"
                "2. Identify the next piece of information needed according to your plan or the most pressing current gap.\n"
                "3. If a tool is needed to get this information, output the JSON for that tool.\n"
                "4. If you believe you have gathered enough information for now, or for a specific sub-topic, you can state your intention to move to analysis or summarize current findings. Otherwise, continue gathering information with tools."
            )
        elif current_research_stage == "analysis":
            stage_specific_guidance = (
                "\nCURRENT STAGE: ANALYSIS\n"
                "1. Review all gathered information (from Accumulated Context and your notebook).\n"
                "2. Identify patterns, connections, contradictions, and key insights related to the research query.\n"
                "3. If more information is critically needed to complete your analysis, you can use a tool (output JSON).\n"
                "4. Otherwise, synthesize your analysis. Explain your findings based on the evidence. This analysis will form a core part of your final report."
            )
        elif current_research_stage == "synthesis_report":
            stage_specific_guidance = (
                "\nCURRENT STAGE: SYNTHESIS & REPORT GENERATION\n"
                "1. Consolidate all your findings and analysis from previous stages.\n"
                "2. Structure and write a comprehensive research report that directly addresses the original query.\n"
                "3. The report should include sections like: Introduction/Background, Methodology (briefly describe how you gathered and analyzed info), Key Findings, Detailed Analysis, Conclusion, and References (these will be listed from your research notebook automatically).\n"
                "4. Ensure clarity, coherence, and cite information based on the sources gathered (refer to them by their source ID if available in context).\n"
                "This is your final output for this research project. Make it complete and well-structured."
            )

        full_prompt = (
            f"{system_prompt}\n\n"
            f"RESEARCH PROJECT DETAILS (ID: {research_id}):\n{research_project_details_str}\n\n" # Provide current state
            f"ORIGINAL RESEARCH QUERY: \"{research_query}\"\n\n"
            f"{stage_specific_guidance}\n\n"
            f"ACCUMULATED CONTEXT & OBSERVATIONS (most recent first):\n{accumulated_context if accumulated_context else 'No context accumulated in this session yet. Refer to notebook for full history.'}\n\n"
            f"{tool_descriptions}\n{tool_usage_instructions}\n\n"
            "Based on the current stage and available information, what is your next single, actionable step? "
            "If using a tool, provide ONLY the JSON. If providing analysis, a plan, or a report section, write it directly."
        )
        return full_prompt

    def start_new_research(self, topic: str) -> str:
        """Begin a new research project on the given topic."""
        research_id = self._generate_research_id(topic)
        self.current_research_id = research_id # Set as active research project

        self.citation_index[research_id] = {
            "topic": topic,
            "created": datetime.now().isoformat(),
            "updated": datetime.now().isoformat(),
            "status": "planning", # Initial stage
            "current_stage_index": 0, # Index for research_stages list
            "stages_completed": [], # List of completed stage names
            "sources": [], # List of source dicts
            "notebook_file": os.path.join("research_notes", f"{research_id}_notebook.md") # Relative to output_dir
        }

        notebook_content = (
            f"# Research Notebook: {topic}\n\n"
            f"## Research ID: {research_id}\n"
            f"## Created: {self.citation_index[research_id]['created']}\n\n"
            f"## Original Query:\n{topic}\n\n"
            "------------------------------------\n"
            "## Research Plan\n\n" # LLM will fill this
            "------------------------------------\n"
            "## Information Gathering\n\n" # Tools and observations go here
            "------------------------------------\n"
            "## Analysis\n\n" # LLM's analysis of gathered info
            "------------------------------------\n"
            "## Synthesis & Report\n\n" # Final report content
            "------------------------------------\n"
            "## References\n\n" # Auto-populated by add_source
        )
        # Use Quartermaster to write the notebook file
        self.qm.write_file(self.citation_index[research_id]["notebook_file"], notebook_content)
        self._save_research_history() # Save the updated citation_index
        print(f"[ResearcherAgent] Started new research project ID: {research_id} for topic: '{topic}'")
        return research_id

    def append_to_notebook_section(self, research_id: str, section_title: str, content_to_append: str) -> bool:
        """Appends content to a specific section in the research notebook."""
        if research_id not in self.citation_index:
            print(f"[ResearcherAgent_ERROR] Cannot update notebook: Research ID {research_id} not found.")
            return False
        try:
            notebook_file_path_relative = self.citation_index[research_id]["notebook_file"]
            current_content = self.qm.read_file(notebook_file_path_relative)

            # Regex to find the section and append. Handles existing content.
            section_pattern = re.compile(r"(##\s*" + re.escape(section_title) + r"\s*\n)([\s\S]*?)(?=\n##\s*|\Z)", re.IGNORECASE)
            match = section_pattern.search(current_content)

            timestamped_append = f"\n\n---\n*Entry added: {datetime.now().isoformat()}*\n{content_to_append.strip()}\n---"

            if match:
                # Append new content below existing content in the section
                updated_section_block = f"{match.group(1)}{match.group(2).strip()}{timestamped_append}"
                new_notebook_content = current_content.replace(match.group(0), updated_section_block)
            else: # Section not found, append it at the end
                new_notebook_content = current_content.strip() + f"\n\n## {section_title}{timestamped_append}"

            self.qm.write_file(notebook_file_path_relative, new_notebook_content)
            self.citation_index[research_id]["updated"] = datetime.now().isoformat()
            self._save_research_history() # Save history as notebook was updated
            return True
        except Exception as e:
            print(f"[ResearcherAgent_ERROR] Error appending to notebook for {research_id}, section {section_title}: {e}")
            return False

    def add_source(self, research_id: str, source_info: Dict[str, Any]):
        """Add a source to the research project and update notebook."""
        if research_id not in self.citation_index: return False # Or raise error
        sources_list = self.citation_index[research_id].get("sources", [])
        source_id = len(sources_list) + 1
        source_entry = {
            "id": source_id,
            "type": source_info.get("type", "generic"),
            "title": source_info.get("title", f"Source {source_id}"),
            "detail": source_info.get("query") or source_info.get("path") or source_info.get("url", "N/A"), # More specific detail
            "timestamp": datetime.now().isoformat(),
            "summary": source_info.get("summary", "No summary available.") # Optional summary
        }
        sources_list.append(source_entry)
        self.citation_index[research_id]["sources"] = sources_list
        self.citation_index[research_id]["updated"] = datetime.now().isoformat()
        self._update_references_in_notebook(research_id) # This will also save history
        return source_id

    def _update_references_in_notebook(self, research_id: str):
        """Updates the 'References' section of the notebook with current sources."""
        if research_id not in self.citation_index: return
        sources = self.citation_index[research_id].get("sources", [])
        references_text = "\n".join([
            f"- **Source ID {s['id']}** ({s['type']}): {s['title']}\n  - Detail: {s['detail']}\n  - Added: {s['timestamp']}\n  - Summary: {s.get('summary', 'N/A')}"
            for s in sources
        ]) if sources else "No sources recorded yet."

        try:
            notebook_file_path_relative = self.citation_index[research_id]["notebook_file"]
            current_content = self.qm.read_file(notebook_file_path_relative)
            # This regex replaces the entire content under "## References"
            section_pattern = re.compile(r"(##\s*References\s*\n)([\s\S]*?)(?=\n##\s*|\Z)", re.IGNORECASE)
            match = section_pattern.search(current_content)
            if match:
                new_notebook_content = current_content.replace(match.group(0), f"{match.group(1)}{references_text}\n")
            else: # Section not found, append it
                new_notebook_content = current_content.strip() + f"\n\n## References\n{references_text}\n"
            self.qm.write_file(notebook_file_path_relative, new_notebook_content)
            self._save_research_history() # Save history as notebook (references section) was updated
        except Exception as e:
            print(f"[ResearcherAgent_ERROR] Failed to update references in notebook {research_id}: {e}")


    def complete_research(self, research_id: str, final_report_content: str) -> Optional[str]:
        """Marks research as complete and saves the final report."""
        if research_id not in self.citation_index:
            print(f"[ResearcherAgent_ERROR] Cannot complete: Research ID {research_id} not found.")
            return None
        self.citation_index[research_id]["status"] = "completed"
        self.citation_index[research_id]["updated"] = datetime.now().isoformat()
        self.citation_index[research_id]["completed_time"] = datetime.now().isoformat()

        report_filename = f"{research_id}_final_report.md"
        report_path_in_output_relative = os.path.join("research_notes", report_filename)
        self.qm.write_file(report_path_in_output_relative, final_report_content)
        self.citation_index[research_id]["report_file"] = report_path_in_output_relative

        self.append_to_notebook_section(research_id, "Synthesis & Report", f"Final report generated and saved to {report_path_in_output_relative}\n\n---\n{final_report_content[:500]}...")
        # _save_research_history is called by append_to_notebook_section
        if self.current_research_id == research_id:
            self.current_research_id = None # Clear active research if it was this one
        print(f"[ResearcherAgent] Research project {research_id} completed. Report: {report_path_in_output_relative}")
        return report_path_in_output_relative

    def list_research_projects(self) -> str:
        """Lists all research projects with their status."""
        if not self.citation_index:
            return "No research projects found."
        output = ["Current Research Projects:"]
        # Sort by creation date, most recent first
        sorted_rids = sorted(self.citation_index.keys(), key=lambda r_id: self.citation_index[r_id].get("created", ""), reverse=True)
        for rid in sorted_rids:
            data = self.citation_index[rid]
            status_emoji = "✅" if data.get('status') == 'completed' else ("⏳" if 'in_progress' in data.get('status', '') else "📋")
            output.append(f"  {status_emoji} ID: {rid} - Topic: {data.get('topic', 'N/A')} (Status: {data.get('status', 'N/A')})")
        return "\n".join(output)

    def get_research_details(self, research_id: str) -> str:
        """Returns a string with details of a specific research project."""
        if research_id not in self.citation_index:
            return f"Research project with ID '{research_id}' not found."
        data = self.citation_index[research_id]
        details = [
            f"Research Project Details (ID: {research_id})",
            f"  Topic: {data.get('topic', 'N/A')}",
            f"  Status: {data.get('status', 'N/A')}",
            f"  Current Stage Index: {data.get('current_stage_index', 'N/A')}", # Useful for debugging
            f"  Created: {data.get('created', 'N/A')}",
            f"  Last Updated: {data.get('updated', 'N/A')}",
            f"  Notebook File: {data.get('notebook_file', 'N/A')}",
        ]
        if data.get('report_file'):
            details.append(f"  Final Report File: {data.get('report_file')}")
        sources = data.get('sources', [])
        if sources:
            details.append("\n  Sources:")
            for src in sources:
                details.append(f"    - [{src.get('id')}] ({src.get('type')}) {src.get('title')} - Detail: {src.get('detail')}")
        else:
            details.append("\n  Sources: No sources recorded yet.")
        return "\n".join(details)

    def run(self, command: str, max_iterations=15):
        """Main execution loop for the ResearcherAgent."""
        print(f"[ResearcherAgent] Received command: '{command[:100]}...'")
        original_user_command = command.strip()
        accumulated_context = "" # Stores observations from tool uses for the current run/iteration

        cmd_lower = original_user_command.lower()
        if cmd_lower.startswith("researcher,"):
            cmd_lower = cmd_lower[len("researcher,"):].strip()
            original_user_command = original_user_command[len("researcher,"):].strip()

        active_research_id = self.current_research_id
        research_query = ""

        # --- Command Parsing and Research Project Management ---
        if cmd_lower.startswith("list research") or cmd_lower.startswith("show research"):
            return self.list_research_projects()
        elif cmd_lower.startswith("details for research") or cmd_lower.startswith("get research details"):
            match = re.search(r"(?:details for research|get research details)\s+([\w\d_\-]+)", cmd_lower)
            if match: return self.get_research_details(match.group(1).strip())
            return "Please specify Research ID for details (e.g., 'details for research research_xxxx')."
        elif cmd_lower.startswith("continue research"):
            match = re.search(r"continue research\s+([\w\d_\-]+)", cmd_lower)
            if match:
                active_research_id = match.group(1).strip()
                if active_research_id not in self.citation_index:
                    return f"Error: Research ID '{active_research_id}' not found to continue."
                self.current_research_id = active_research_id
                research_query = self.citation_index[active_research_id]["topic"]
                accumulated_context = f"Continuing research on ID {active_research_id}. Original topic: '{research_query}'. User provided additional direction: '{original_user_command}'\n"
                print(f"[ResearcherAgent] Continuing research ID: {active_research_id} on topic: '{research_query}'")
            else: return "Please specify Research ID to continue (e.g., 'continue research research_xxxx')."
        elif cmd_lower.startswith("complete research"):
            match = re.search(r"complete research\s+([\w\d_\-]+)", cmd_lower)
            if match:
                research_id_to_complete = match.group(1).strip()
                if research_id_to_complete not in self.citation_index:
                     return f"Error: Research ID '{research_id_to_complete}' not found to complete."
                notebook_file_path = self.citation_index[research_id_to_complete]["notebook_file"]
                notebook_content = ""
                try:
                    notebook_content = self.qm.read_file(notebook_file_path)
                except Exception as e_read_nb_complete:
                    print(f"[ResearcherAgent_WARN] Could not read notebook {notebook_file_path} for final synthesis: {e_read_nb_complete}")
                    notebook_content = "Notebook content unavailable for final synthesis."

                synthesis_prompt = self._generate_research_prompt(
                    self.citation_index[research_id_to_complete]["topic"],
                    "synthesis_report",
                    f"Current Notebook Content for {research_id_to_complete}:\n{notebook_content}\n\nTask: Summarize all findings and generate the final comprehensive research report.",
                    research_id_to_complete
                )
                final_report_text = self._call_llm(synthesis_prompt, research_id=research_id_to_complete)
                if "LLM Error:" in final_report_text: return final_report_text
                saved_report_path = self.complete_research(research_id_to_complete, final_report_text)
                return f"Research {research_id_to_complete} marked complete. Final report generated and saved to {saved_report_path}.\nReport Preview:\n{final_report_text[:500]}..."
            return "Please specify Research ID to complete (e.g., 'complete research ID_xxx')."
        elif any(keyword in cmd_lower for keyword in ["research", "study", "investigate", "find out about", "report on", "analyze topic"]):
            if self.current_research_id and self.citation_index.get(self.current_research_id, {}).get("status") != "completed":
                if not any(kw in cmd_lower for kw in ["new research", "start research"]):
                    active_research_id = self.current_research_id
                    research_query = self.citation_index[active_research_id]["topic"]
                    accumulated_context = f"Continuing research (ID: {active_research_id}). User input for refinement/direction: '{original_user_command}'\n"
                    print(f"[ResearcherAgent] Interpreting as refinement for active research ID: {active_research_id}")
                else:
                    active_research_id = self.start_new_research(original_user_command)
                    research_query = original_user_command
                    accumulated_context = f"Starting new research (ID: {active_research_id}) on: {research_query}\n"
            else:
                active_research_id = self.start_new_research(original_user_command)
                research_query = original_user_command
                accumulated_context = f"Starting new research (ID: {active_research_id}) on: {research_query}\n"
        else:
            if self.current_research_id and self.citation_index.get(self.current_research_id, {}).get("status") != "completed":
                active_research_id = self.current_research_id
                research_query = self.citation_index[active_research_id]["topic"]
                accumulated_context = f"Continuing research (ID: {active_research_id}). User input: '{original_user_command}'\n"
                print(f"[ResearcherAgent] Interpreting as continuation for active research ID: {active_research_id}")
            else:
                return "ResearcherAgent: Please specify a research topic (e.g., 'research quantum computing') or manage projects ('list research', 'continue research ID_xxx')."

        if not active_research_id or active_research_id not in self.citation_index:
            return "ResearcherAgent: Critical error - could not determine or initialize a valid research project."

        # --- Main ReAct Loop for the active research project ---
        research_stages = ["planning", "information_gathering", "analysis", "synthesis_report"]
        current_stage_idx = self.citation_index[active_research_id].get("current_stage_index", 0)
        if not (0 <= current_stage_idx < len(research_stages)):
            current_stage_idx = 0
            self.citation_index[active_research_id]["current_stage_index"] = 0

        for iteration in range(max_iterations):
            current_research_data = self.citation_index[active_research_id]
            current_stage = research_stages[current_stage_idx]
            print(f"\n[ResearcherAgent] --- Research ID: {active_research_id}, Stage: {current_stage}, Iteration {iteration + 1}/{max_iterations} ---")

            current_research_data["status"] = f"in_progress_{current_stage}"
            current_research_data["updated"] = datetime.now().isoformat()
            self._save_research_history()

            prompt = self._generate_research_prompt(research_query, current_stage, accumulated_context, active_research_id)
            llm_response = self._call_llm(prompt, research_id=active_research_id)

            if "LLM Error:" in llm_response:
                self.append_to_notebook_section(active_research_id, current_stage.capitalize(), f"LLM Error encountered: {llm_response}")
                return f"ResearcherAgent Error for {active_research_id}: {llm_response}"

            tool_call_match = re.search(r'{\s*"tool_to_use":\s*".*?",\s*"tool_arguments":\s*{.*?}\s*}', llm_response, re.DOTALL)
            if tool_call_match:
                json_str = tool_call_match.group(0)
                try:
                    tool_request = json.loads(json_str)
                    tool_name = tool_request.get("tool_to_use")
                    tool_args = tool_request.get("tool_arguments", {})
                    reasoning_before_tool = llm_response[:tool_call_match.start()].strip()
                    if reasoning_before_tool:
                         self.append_to_notebook_section(active_research_id, current_stage.capitalize(), f"LLM Reasoning before tool '{tool_name}': {reasoning_before_tool}\n")
                    tool_output = self._execute_tool(tool_name, tool_args, active_research_id)
                    accumulated_context += f"\n# Observation from Tool '{tool_name}' (Args: {json.dumps(tool_args)}):\n{tool_output}\n"
                    continue
                except (json.JSONDecodeError, ToolException, SentinelException) as e_tool_react:
                    error_msg = f"Error processing tool request ('{tool_name if 'tool_name' in locals() else 'unknown tool'}'): {str(e_tool_react)}"
                    print(f"[ResearcherAgent_ERROR] {error_msg}")
                    accumulated_context += f"\n# Error with tool: {error_msg}\nLLM should reconsider or try a different approach."
                    self.append_to_notebook_section(active_research_id, current_stage.capitalize(), f"Tool Error: {error_msg}\n")
                    continue
                except Exception as e_unexpected_tool_react:
                    error_msg = f"Unexpected error during tool handling: {str(e_unexpected_tool_react)}"
                    print(f"[ResearcherAgent_ERROR] {error_msg}")
                    import traceback; traceback.print_exc()
                    accumulated_context += f"\n# System Error with tool: {error_msg}\nLLM should reconsider."
                    self.append_to_notebook_section(active_research_id, current_stage.capitalize(), f"System Error with Tool: {error_msg}\n")
                    continue
            else: # No tool call
                self.append_to_notebook_section(active_research_id, current_stage.capitalize(), llm_response)
                accumulated_context += f"\n# LLM Output (Stage: {current_stage}):\n{llm_response}\n"
                advance_stage = False
                if current_stage == "planning": # If planning stage just outputted text (the plan), advance.
                    advance_stage = True
                    print(f"[ResearcherAgent] Planning stage output received for {active_research_id}. Advancing to information gathering.")
                elif current_stage == "information_gathering" and ("sufficient information gathered" in llm_response.lower() or "ready for analysis" in llm_response.lower() or "proceed to analysis" in llm_response.lower() or "enough information for now" in llm_response.lower()):
                    advance_stage = True
                elif current_stage == "analysis" and ("analysis complete" in llm_response.lower() or "ready for synthesis" in llm_response.lower() or "proceed to report" in llm_response.lower()):
                    advance_stage = True
                elif current_stage == "synthesis_report":
                    if len(llm_response) > 300 or "conclusion" in llm_response.lower() and ("executive summary" in llm_response.lower() or "introduction" in llm_response.lower()): # Heuristic for a report
                        saved_report_file = self.complete_research(active_research_id, llm_response)
                        return f"Research project {active_research_id} completed.\nFinal Report (preview):\n{llm_response[:1000]}...\n(Full report saved to: {saved_report_file})"

                if advance_stage:
                    current_research_data["stages_completed"].append(current_stage)
                    current_stage_idx += 1
                    if current_stage_idx >= len(research_stages):
                        saved_report_file = self.complete_research(active_research_id, llm_response) # Treat last output as report
                        return f"Research project {active_research_id} completed all stages.\nFinal Output (treated as report):\n{llm_response[:1000]}...\n(Saved to: {saved_report_file})"
                    else:
                        current_research_data["current_stage_index"] = current_stage_idx
                        print(f"[ResearcherAgent] Advanced research {active_research_id} to stage: {research_stages[current_stage_idx]}")
                        accumulated_context = f"Advanced to stage: {research_stages[current_stage_idx]}. Previous stage '{current_stage}' output is now in notebook.\n" # Reset context for new stage
                self._save_research_history()
                if current_research_data["status"] != "completed":
                    continue
                else:
                    break

        final_status_msg = f"ResearcherAgent reached max iterations ({max_iterations}) for research ID '{active_research_id}'. The research is still '{self.citation_index[active_research_id]['status']}'. You can continue it using 'Researcher, continue research {active_research_id}' or review the notebook."
        self.append_to_notebook_section(active_research_id, research_stages[current_stage_idx].capitalize(), f"Note: Max iterations reached. Current findings and context:\n{accumulated_context}")
        return final_status_msg

