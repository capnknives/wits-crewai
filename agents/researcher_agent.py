# agents/researcher_agent.py
import ollama
import re
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from .tools.base_tool import Tool, ToolException
from .sentinel_agent import SentinelException


class ResearcherAgent:
    """
    A specialized agent that focuses on in-depth research tasks, synthesizing information 
    from multiple sources, and creating comprehensive research reports.
    The Researcher is capable of methodical exploration of topics, citation tracking,
    and maintaining a research notebook for multi-stage inquiries.
    """
    
    def __init__(self, config, memory, quartermaster, sentinel, tools=None):
        self.config = config
        self.memory = memory
        self.qm = quartermaster
        self.sentinel = sentinel
        self.tools = tools if tools else []
        model_conf = config.get("models", {})
        # Default to using the same model as the analyst if no specific researcher model
        self.model_name = model_conf.get("researcher", model_conf.get("analyst", model_conf.get("default", "llama2")))
        
        # Research notebook directory
        self.notebook_dir = "research_notes"
        self.qm.list_files(self.notebook_dir)  # This will create the directory if it doesn't exist
        
        # Track citations and sources
        self.citation_index = {}
        self.current_research_id = None
        
        # Load research history if available
        self._load_research_history()
    
    def _load_research_history(self):
        """Load previous research history if available"""
        try:
            research_index_file = f"{self.notebook_dir}/research_index.json"
            content = self.qm.read_file(research_index_file)
            self.citation_index = json.loads(content)
            print(f"[ResearcherAgent] Loaded {len(self.citation_index)} previous research entries")
        except Exception as e:
            print(f"[ResearcherAgent] No previous research index found or error loading: {e}")
            self.citation_index = {}
    
    def _save_research_history(self):
        """Save research history to file"""
        try:
            research_index_file = f"{self.notebook_dir}/research_index.json"
            self.qm.write_file(research_index_file, json.dumps(self.citation_index, indent=2))
        except Exception as e:
            print(f"[ResearcherAgent] Error saving research index: {e}")
    
    def _get_tool_descriptions_for_llm(self):
        """Generate descriptions of available tools for the LLM prompt"""
        if not self.tools:
            return "No tools are currently available to you for direct use."
        
        descriptions = ["\nAVAILABLE TOOLS (use these to gather information):"]
        for tool in self.tools:
            descriptions.append(f"- {tool.get_full_description_for_llm()}")
        return "\n".join(descriptions)
    
    def _generate_research_id(self, topic):
        """Generate a unique ID for this research topic"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Create a simplified slug from the topic
        topic_slug = re.sub(r'[^a-zA-Z0-9]', '_', topic.lower())[:30]
        return f"{timestamp}_{topic_slug}"
    
    def _call_llm(self, prompt: str):
        """Call the LLM with the given prompt"""
        print(f"\n[ResearcherAgent] Sending prompt to LLM (model: {self.model_name}):\n{prompt[:500]}...\n-----------------------------")
        ollama_response = ollama.generate(model=self.model_name, prompt=prompt)
        llm_output = ollama_response['response'].strip()
        print(f"[ResearcherAgent] LLM Output (truncated):\n{llm_output[:500]}...\n-----------------------------")
        return llm_output
    
    def _generate_research_prompt(self, query, context="", stage="initial"):
        """Generate a prompt for the LLM based on the research stage"""
        tool_descriptions = self._get_tool_descriptions_for_llm()
        
        # Base prompt components
        system_prompt = (
            "You are a Researcher Agent (NOT an Analyst Agent), specialized in conducting thorough, methodical research "
            "on complex topics. Your expertise is in breaking down research questions, "
            "gathering relevant information from multiple sources, analyzing findings, "
            "and synthesizing comprehensive answers with proper citation.\n\n"
        )
        
        # Context includes previous research steps if any
        research_context = f"\nRESEARCH CONTEXT:\n{context}\n" if context else ""
        
        # Tool usage instructions
        tool_instructions = (
            "\nTOOL USAGE INSTRUCTIONS:\n"
            "To use a tool to gather information, respond with a JSON object in the following format:\n"
            "```json\n{\"tool_to_use\": \"tool_name\", \"tool_arguments\": {\"arg_name\": \"value\"}}\n```\n"
            "Only use tools when you need specific information that you don't already have.\n"
        )
        
        # Stage-specific instructions
        if stage == "initial":
            stage_instructions = (
                "\nINITIAL RESEARCH PLANNING:\n"
                "1. Analyze the research query to understand its scope and requirements\n"
                "2. Break down the query into key questions or areas to investigate\n"
                "3. Determine what information you need to gather first\n"
                "4. Use appropriate tools to gather initial information\n"
                "5. After gathering information, create a structured research plan\n"
            )
        elif stage == "information_gathering":
            stage_instructions = (
                "\nINFORMATION GATHERING PHASE:\n"
                "1. Use tools to gather information based on your research plan\n"
                "2. For each piece of information, track the source\n"
                "3. Evaluate the credibility and relevance of information\n"
                "4. Identify any gaps in the information gathered\n"
                "5. Continue gathering information until you have a comprehensive understanding\n"
            )
        elif stage == "analysis":
            stage_instructions = (
                "\nANALYSIS PHASE:\n"
                "1. Analyze the information gathered from all sources\n"
                "2. Identify patterns, contradictions, or consensus across sources\n"
                "3. Evaluate the strength of evidence for different viewpoints\n"
                "4. Consider alternative interpretations of the information\n"
                "5. Develop preliminary conclusions based on your analysis\n"
            )
        elif stage == "synthesis":
            stage_instructions = (
                "\nSYNTHESIS PHASE:\n"
                "1. Integrate all findings into a coherent narrative\n"
                "2. Address the original research query comprehensively\n"
                "3. Cite all sources properly in your response\n"
                "4. Acknowledge limitations or areas for further research\n"
                "5. Organize your response in a clear, logical structure\n"
                "6. Create a final research report with sections including: Executive Summary, Methodology, Findings, Analysis, Conclusions, and References\n"
            )
        else:
            stage_instructions = (
                "\nRESEARCH CONTINUATION:\n"
                "1. Review what you know so far and identify information gaps\n"
                "2. Determine what additional information would strengthen your research\n"
                "3. Use tools as needed to gather more information\n"
                "4. Integrate new findings with previous research\n"
                "5. Move toward a comprehensive answer to the research query\n"
            )
        
        # Combine all components
        full_prompt = (
            f"{system_prompt}{research_context}{tool_descriptions}\n{tool_instructions}\n{stage_instructions}\n\n"
            f"RESEARCH QUERY: {query}\n\n"
            "Based on the above, either use a tool to gather information by outputting a JSON object, "
            "or provide your research findings in a clear, structured format with proper citation of sources."
        )
        
        return full_prompt
    
    def start_new_research(self, topic):
        """Begin a new research project on the given topic"""
        research_id = self._generate_research_id(topic)
        self.current_research_id = research_id
        
        # Create entry in citation index
        self.citation_index[research_id] = {
            "topic": topic,
            "created": datetime.now().isoformat(),
            "updated": datetime.now().isoformat(),
            "status": "in_progress",
            "stages": ["planning"],
            "sources": [],
            "notebook_file": f"{self.notebook_dir}/{research_id}_notebook.md"
        }
        
        # Initialize research notebook
        notebook_content = (
            f"# Research Notebook: {topic}\n\n"
            f"## Research ID: {research_id}\n"
            f"## Created: {datetime.now().isoformat()}\n\n"
            f"## Original Query\n{topic}\n\n"
            "## Research Plan\n\n"
            "## Information Gathering\n\n"
            "## Analysis\n\n"
            "## Conclusions\n\n"
            "## References\n\n"
        )
        
        self.qm.write_file(self.citation_index[research_id]["notebook_file"], notebook_content)
        self._save_research_history()
        
        return research_id
    
    def update_notebook(self, research_id, section, content):
        """Update a section of the research notebook"""
        if research_id not in self.citation_index:
            print(f"[ResearcherAgent] Research ID {research_id} not found")
            return False
        
        try:
            notebook_file = self.citation_index[research_id]["notebook_file"]
            notebook_content = self.qm.read_file(notebook_file)
            
            # Find the section and update it
            if f"## {section}" in notebook_content:
                # Split by sections
                sections = re.split(r'## ', notebook_content)
                updated_sections = []
                
                for s in sections:
                    if s.startswith(section):
                        # Keep the section name and replace the content
                        section_name = s.split('\n')[0]
                        updated_sections.append(f"{section_name}\n\n{content}\n\n")
                    else:
                        updated_sections.append(s)
                
                # Rejoin with section markers
                updated_content = "## ".join(updated_sections)
                if not updated_content.startswith("## "):
                    # First section doesn't have ## prefix in the split result
                    updated_content = sections[0] + "## ".join(updated_sections[1:])
                
                self.qm.write_file(notebook_file, updated_content)
                
                # Update the timestamp
                self.citation_index[research_id]["updated"] = datetime.now().isoformat()
                self._save_research_history()
                return True
            else:
                print(f"[ResearcherAgent] Section {section} not found in notebook")
                return False
        except Exception as e:
            print(f"[ResearcherAgent] Error updating notebook: {e}")
            return False
    
    def add_source(self, research_id, source_info):
        """Add a source to the research project"""
        if research_id not in self.citation_index:
            print(f"[ResearcherAgent] Research ID {research_id} not found")
            return False
        
        # Add source to the index
        source_id = len(self.citation_index[research_id]["sources"]) + 1
        source_entry = {
            "id": source_id,
            "title": source_info.get("title", "Untitled Source"),
            "url": source_info.get("url", ""),
            "type": source_info.get("type", "web"),
            "added": datetime.now().isoformat(),
            "summary": source_info.get("summary", "")
        }
        
        self.citation_index[research_id]["sources"].append(source_entry)
        self._save_research_history()
        
        # Update the references section in the notebook
        self._update_references(research_id)
        
        return source_id
    
    def _update_references(self, research_id):
        """Update the references section in the notebook"""
        if research_id not in self.citation_index:
            return False
        
        references = []
        for source in self.citation_index[research_id]["sources"]:
            ref = f"[{source['id']}] {source['title']}"
            if source.get("url"):
                ref += f" - {source['url']}"
            references.append(ref)
        
        references_text = "\n".join(references)
        self.update_notebook(research_id, "References", references_text)
        return True
    
    def complete_research(self, research_id, final_report):
        """Mark research as complete and save the final report"""
        if research_id not in self.citation_index:
            print(f"[ResearcherAgent] Research ID {research_id} not found")
            return False
        
        # Update the index
        self.citation_index[research_id]["status"] = "completed"
        self.citation_index[research_id]["completed"] = datetime.now().isoformat()
        
        # Create a final report file
        report_file = f"{self.notebook_dir}/{research_id}_final_report.md"
        self.qm.write_file(report_file, final_report)
        self.citation_index[research_id]["report_file"] = report_file
        
        self._save_research_history()
        return report_file
    
    def list_research_projects(self):
        """List all research projects"""
        if not self.citation_index:
            return "No research projects found."
        
        projects = []
        for rid, data in self.citation_index.items():
            status = data.get("status", "unknown")
            status_emoji = "✅" if status == "completed" else "🔍"
            projects.append(f"{status_emoji} [{rid}] {data.get('topic', 'Untitled Research')}")
        
        return "\n".join(projects)
    
    def get_research_details(self, research_id):
        """Get details about a specific research project"""
        if research_id not in self.citation_index:
            return f"Research ID {research_id} not found."
        
        data = self.citation_index[research_id]
        
        details = [
            f"# Research Project: {data.get('topic', 'Untitled')}",
            f"Research ID: {research_id}",
            f"Status: {data.get('status', 'unknown')}",
            f"Created: {data.get('created', 'unknown')}",
            f"Last Updated: {data.get('updated', 'unknown')}",
            f"Notebook: {data.get('notebook_file', 'not available')}",
        ]
        
        if data.get("report_file"):
            details.append(f"Final Report: {data.get('report_file')}")
        
        if data.get("sources"):
            details.append("\n## Sources:")
            for source in data["sources"]:
                details.append(f"[{source['id']}] {source.get('title', 'Untitled')} ({source.get('type', 'unknown')})")
        
        return "\n".join(details)
    
    def run(self, command):
        """Main entry point for the Researcher Agent"""
        print(f"[ResearcherAgent] Processing command: {command[:100]}...")
        
        # Clean up the command - remove any unwanted prefixes
        if command.startswith("er,"):
            command = command[3:].strip()
        
        try:
            # Check for specific researcher commands
            cmd_lower = command.lower()
            
            # List research projects
            if "list research" in cmd_lower or "list projects" in cmd_lower or "show research" in cmd_lower:
                return self.list_research_projects()
            
            # Get details about a specific research project
            if "details for" in cmd_lower or "research details" in cmd_lower:
                # Extract the research ID
                research_id_match = re.search(r'(?:details for|research details) (.+)', command, re.IGNORECASE)
                if research_id_match:
                    research_id = research_id_match.group(1).strip()
                    return self.get_research_details(research_id)
            
            # Handle generic research requests like "build a report on..."
            if any(keyword in cmd_lower for keyword in ["research", "build a report", "compile a report", "analyze", "investigate"]):
                # Initialize a new research project
                research_id = self.start_new_research(command)
                self.current_research_id = research_id
                
                # First, let's do a web search to gather information
                search_terms = command
                for prefix in ["research on", "build a report on", "compile a report on", "analyze", "investigate"]:
                    if search_terms.lower().startswith(prefix):
                        search_terms = search_terms[len(prefix):].strip()
                
                # Execute the web search directly
                search_results = ""
                web_search_tool = next((t for t in self.tools if t.name == "internet_search"), None)
                if web_search_tool:
                    try:
                        self.sentinel.approve_action(
                            agent_name="ResearcherAgent",
                            action_type="tool_use:internet_search",
                            detail=f"Search query: {search_terms}"
                        )
                        search_results = web_search_tool.execute(query=search_terms)
                        
                        # Add as a source
                        source_info = {
                            "title": f"Web search for: {search_terms}",
                            "type": "web_search",
                            "summary": search_results
                        }
                        self.add_source(research_id, source_info)
                        
                        # Update the notebook with search results
                        self.update_notebook(
                            research_id, 
                            "Information Gathering", 
                            f"### Web Search Results\n**Query:** {search_terms}\n\n**Results:**\n{search_results}\n\n"
                        )
                    except Exception as e:
                        search_results = f"Error performing web search: {str(e)}"
                        self.update_notebook(
                            research_id, 
                            "Information Gathering", 
                            f"### Web Search Error\n**Query:** {search_terms}\n\n**Error:**\n{search_results}\n\n"
                        )
                
                # Now generate research plan using the search results as context
                research_context = f"Initial search results:\n{search_results}\n\n" if search_results else ""
                research_prompt = self._generate_research_prompt(command, context=research_context, stage="initial")
                
                # Modify the prompt to explicitly identify as a Researcher Agent (not Analyst)
                research_prompt = research_prompt.replace(
                    "You are a Research Agent", 
                    "You are a Researcher Agent (NOT an Analyst Agent)"
                )
                
                plan_output = self._call_llm(research_prompt)
                
                # Update the research notebook with the plan
                self.update_notebook(research_id, "Research Plan", plan_output)
                
                # Now generate an initial analysis based on the search results
                if search_results:
                    notebook_file = self.citation_index[research_id]["notebook_file"]
                    notebook_content = self.qm.read_file(notebook_file)
                    analysis_prompt = self._generate_research_prompt(
                        command,
                        context=notebook_content,
                        stage="analysis"
                    )
                    
                    # Ensure we identify as a Researcher Agent
                    analysis_prompt = analysis_prompt.replace(
                        "You are a Research Agent", 
                        "You are a Researcher Agent (NOT an Analyst Agent)"
                    )
                    
                    analysis_output = self._call_llm(analysis_prompt)
                    
                    # Update the analysis section
                    self.update_notebook(research_id, "Analysis", analysis_output)
                    
                    # Generate a draft report
                    synthesis_prompt = self._generate_research_prompt(
                        command,
                        context=notebook_content + "\n\n" + analysis_output,
                        stage="synthesis"
                    )
                    
                    # Ensure we identify as a Researcher Agent
                    synthesis_prompt = synthesis_prompt.replace(
                        "You are a Research Agent", 
                        "You are a Researcher Agent (NOT an Analyst Agent)"
                    )
                    
                    report_draft = self._call_llm(synthesis_prompt)
                    
                    # Use Sentinel to check the content
                    if self.config.get("ethics_enabled", True) and self.sentinel.ethics_filter:
                        self.sentinel.ethics_filter.check_text(report_draft)
                    
                    return report_draft
                
                return f"Research project initialized with ID: {research_id}\n\nInitial Research Plan:\n{plan_output}\n\nI'll continue gathering information to build a comprehensive report on this topic."
            
            # Continue with existing research project
            elif self.current_research_id:
                # Continue the current research project
                research_id = self.current_research_id
                if research_id in self.citation_index:
                    # Get the current state of the research
                    notebook_file = self.citation_index[research_id]["notebook_file"]
                    notebook_content = self.qm.read_file(notebook_file)
                    
                    # Generate a continuation prompt
                    research_prompt = self._generate_research_prompt(
                        self.citation_index[research_id]["topic"],
                        context=notebook_content,
                        stage="continuation"
                    )
                    
                    # Ensure we identify as a Researcher Agent
                    research_prompt = research_prompt.replace(
                        "You are a Research Agent", 
                        "You are a Researcher Agent (NOT an Analyst Agent)"
                    )
                    
                    # Add the user's specific question or direction
                    research_prompt += f"\n\nUser's additional direction: {command}\n"
                    
                    # Get the continuation response
                    continuation_output = self._call_llm(research_prompt)
                    
                    # Check if the output is a tool request
                    if continuation_output.strip().startswith("{") and "tool_to_use" in continuation_output:
                        try:
                            tool_request = json.loads(continuation_output)
                            tool_name = tool_request.get("tool_to_use")
                            tool_args = tool_request.get("tool_arguments", {})
                            
                            # Find the requested tool
                            tool = next((t for t in self.tools if t.name == tool_name), None)
                            if tool:
                                # Execute the tool
                                self.sentinel.approve_action(
                                    agent_name="ResearcherAgent",
                                    action_type=f"tool_use:{tool.name}",
                                    detail=str(tool_args)
                                )
                                tool_output = tool.execute(**tool_args)
                                
                                # If it's a web search, add it as a source
                                if tool_name == "internet_search":
                                    source_info = {
                                        "title": f"Web search for: {tool_args.get('query', 'unknown query')}",
                                        "type": "web_search",
                                        "summary": tool_output
                                    }
                                    self.add_source(research_id, source_info)
                                
                                # Update research with tool output
                                info_update = f"### Tool: {tool_name}\n**Query:** {tool_args}\n\n**Results:**\n{tool_output}\n\n"
                                
                                # Update the information gathering section
                                current_info = re.search(r'## Information Gathering\n\n(.*?)(?=\n## |$)', notebook_content, re.DOTALL)
                                if current_info:
                                    updated_info = current_info.group(1) + info_update
                                else:
                                    updated_info = info_update
                                
                                self.update_notebook(research_id, "Information Gathering", updated_info)
                                
                                # Now get analysis based on the new information
                                updated_notebook = self.qm.read_file(notebook_file)
                                analysis_prompt = self._generate_research_prompt(
                                    self.citation_index[research_id]["topic"],
                                    context=updated_notebook,
                                    stage="analysis"
                                )
                                
                                # Ensure we identify as a Researcher Agent
                                analysis_prompt = analysis_prompt.replace(
                                    "You are a Research Agent", 
                                    "You are a Researcher Agent (NOT an Analyst Agent)"
                                )
                                
                                analysis_output = self._call_llm(analysis_prompt)
                                
                                # Update the analysis section
                                self.update_notebook(research_id, "Analysis", analysis_output)
                                
                                return f"Research Update for ID {research_id}:\n\nNew Information Gathered:\n{tool_output}\n\nAnalysis:\n{analysis_output}"
                            else:
                                return f"Error: Tool '{tool_name}' not found. Please use one of the available tools."
                        except json.JSONDecodeError:
                            # Not valid JSON, assume it's a research continuation
                            pass
                    
                    # If we get here, it's a direct response rather than a tool request
                    # Check if it looks like a final report
                    if "conclusion" in continuation_output.lower() and len(continuation_output) > 1000:
                        # This might be a final report, update the conclusions
                        self.update_notebook(research_id, "Conclusions", continuation_output)
                        
                        # Ask if this should be marked as the final report
                        return f"Research Update for ID {research_id}:\n\n{continuation_output}\n\n(This looks like it might be a final report. To complete this research, use the command 'complete research {research_id}')"
                    else:
                        # Update the analysis section
                        current_analysis = re.search(r'## Analysis\n\n(.*?)(?=\n## |$)', notebook_content, re.DOTALL)
                        if current_analysis:
                            updated_analysis = current_analysis.group(1) + "\n\n### Additional Analysis\n" + continuation_output
                        else:
                            updated_analysis = continuation_output
                        
                        self.update_notebook(research_id, "Analysis", updated_analysis)
                        
                        return f"Research Update for ID {research_id}:\n\n{continuation_output}"
            
            # Complete research command
            if "complete research" in cmd_lower:
                # Extract the research ID
                research_id_match = re.search(r'complete research\s+(.+)', command, re.IGNORECASE)
                if research_id_match:
                    research_id = research_id_match.group(1).strip()
                    if research_id in self.citation_index:
                        # Generate a final synthesis prompt
                        notebook_file = self.citation_index[research_id]["notebook_file"]
                        notebook_content = self.qm.read_file(notebook_file)
                        
                        synthesis_prompt = self._generate_research_prompt(
                            self.citation_index[research_id]["topic"],
                            context=notebook_content,
                            stage="synthesis"
                        )
                        
                        # Ensure we identify as a Researcher Agent
                        synthesis_prompt = synthesis_prompt.replace(
                            "You are a Research Agent", 
                            "You are a Researcher Agent (NOT an Analyst Agent)"
                        )
                        
                        final_report = self._call_llm(synthesis_prompt)
                        
                        # Save the final report
                        report_file = self.complete_research(research_id, final_report)
                        
                        return f"Research project {research_id} completed. Final report saved to {report_file}.\n\n{final_report}"
                    else:
                        return f"Research ID {research_id} not found."
            
            # No specific command recognized, treat it as a new research topic
            research_id = self.start_new_research(command)
            self.current_research_id = research_id
            
            # Generate initial research plan
            research_prompt = self._generate_research_prompt(command, stage="initial")
            
            # Ensure we identify as a Researcher Agent
            research_prompt = research_prompt.replace(
                "You are a Research Agent", 
                "You are a Researcher Agent (NOT an Analyst Agent)"
            )
            
            plan_output = self._call_llm(research_prompt)
            
            # Update the research notebook with the plan
            self.update_notebook(research_id, "Research Plan", plan_output)
            
            return f"New research project initialized with ID: {research_id}\n\nInitial Research Plan:\n{plan_output}"
        
        except SentinelException as se:
            print(f"[ResearcherAgent] Action blocked by Sentinel: {se}")
            return f"Research action blocked: {se}"
        except ToolException as te:
            print(f"[ResearcherAgent] Tool error: {te}")
            return f"Research tool error: {te}"
        except Exception as e:
            print(f"[ResearcherAgent] Error processing command: {e}")
            import traceback
            print(traceback.format_exc())
            return f"Research agent error: {str(e)}"