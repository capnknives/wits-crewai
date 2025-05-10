# agents/quartermaster_agent.py
import os
import json
import requests # For internet_search method
import re
from datetime import datetime
from typing import Optional

class QuartermasterException(Exception):
    """Custom exception for Quartermaster-specific errors."""
    pass

class QuartermasterAgent:
    def __init__(self, config, memory, sentinel):
        self.config = config
        self.memory = memory
        self.sentinel = sentinel
        # Ensure output_dir is an absolute path or correctly relative to project root
        # If config provides a relative path, os.path.join with current working directory
        # or a known project root is safer. For now, assuming it's handled or relative to where main.py runs.
        self.output_dir = config.get("output_directory", "output")
        os.makedirs(self.output_dir, exist_ok=True)

    def read_file(self, file_path: str) -> str:
        """
        Reads the content of a specified file.
        Handles path resolution to be relative to output_dir if path is not absolute
        and doesn't exist elsewhere.
        """
        # Path normalization and security check
        # If file_path is not absolute, try to resolve it within output_dir primarily
        # or project root for certain readable files (like configs, agent's own source for analysis).
        # Sentinel should be the final gatekeeper for path safety.
        
        # Simplified logic: if not absolute, assume it's relative to output_dir or needs to be found
        # Quartermaster's original logic for finding candidate in output_dir is kept.
        if not os.path.isabs(file_path):
            # Check if it's directly in output_dir if no path separator is present
            if os.sep not in file_path and not file_path.startswith(".."): # Avoid path traversal attempts
                candidate_path = os.path.join(self.output_dir, file_path)
                if os.path.exists(candidate_path):
                    file_path = candidate_path
                # If not in output_dir, it might be a path relative to project root (e.g. "ethics/overlay.md")
                # Sentinel will make the final decision on access.
        
        # Sentinel approves the action based on the resolved or provided file_path
        self.sentinel.approve_action("Quartermaster", "file_read", detail=file_path)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            raise QuartermasterException(f"File not found: {file_path}")
        except Exception as e:
            raise QuartermasterException(f"Error reading file '{file_path}': {str(e)}")

    def write_file(self, file_path: str, content: str) -> bool:
        """
        Writes or overwrites content to a specified file, ensuring it's within the output_dir.
        """
        # Security: Ensure file_path is constrained to the output directory
        # Normalize path to prevent escape (e.g. ../../system_file)
        # os.path.abspath will resolve, then we check if it's within output_dir
        
        # If file_path is relative, join it with self.output_dir
        if not os.path.isabs(file_path):
            # Sanitize to prevent path traversal before joining
            # Basic sanitization: remove leading slashes/dots that might attempt to escape
            # A more robust solution might involve checking each component of the path.
            safe_relative_path = os.path.normpath(file_path.lstrip(os.sep + '.'))
            if ".." in safe_relative_path.split(os.sep): # Disallow ".." in path components
                 raise QuartermasterException(f"Invalid file path for writing (contains '..'): {file_path}")
            full_path = os.path.join(self.output_dir, safe_relative_path)
        else: # If absolute, it must already be within output_dir (checked by Sentinel)
            full_path = file_path 
            # This case needs careful handling by Sentinel to ensure it's a permitted absolute path.
            # For Quartermaster's own logic, we primarily expect relative paths to output_dir.

        # Sentinel makes the final approval, which should check if full_path is within output_dir
        self.sentinel.approve_action("Quartermaster", "file_write", detail=full_path)
        
        try:
            # Ensure parent directory exists
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"[Quartermaster] Content written to: {full_path}")
            return True
        except Exception as e:
            raise QuartermasterException(f"Error writing file '{full_path}': {str(e)}")

    def delete_file(self, file_path: str) -> bool:
        """
        Deletes a specified file, ensuring it's within the output_dir.
        """
        # Similar path handling as write_file
        if not os.path.isabs(file_path):
            safe_relative_path = os.path.normpath(file_path.lstrip(os.sep + '.'))
            if ".." in safe_relative_path.split(os.sep):
                 raise QuartermasterException(f"Invalid file path for deletion (contains '..'): {file_path}")
            full_path = os.path.join(self.output_dir, safe_relative_path)
        else:
            full_path = file_path

        self.sentinel.approve_action("Quartermaster", "file_delete", detail=full_path)
        
        try:
            if not os.path.exists(full_path):
                 raise QuartermasterException(f"File not found for deletion: {full_path}")
            os.remove(full_path)
            print(f"[Quartermaster] File deleted: {full_path}")
            return True
        except FileNotFoundError: # Should be caught by os.path.exists, but good to have
            raise QuartermasterException(f"File not found for deletion: {full_path}")
        except Exception as e:
            raise QuartermasterException(f"Error deleting file '{full_path}': {str(e)}")

    def list_files(self, directory: Optional[str] = None) -> str:
        """
        Lists files and subdirectories within a specified directory.
        Defaults to self.output_dir. Allows listing other project-relative dirs if Sentinel approves.
        """
        target_directory = directory
        if target_directory is None:
            target_directory = self.output_dir
        
        # If relative path, make it absolute for Sentinel's check (usually relative to project root)
        # For Quartermaster, if 'directory' is given, it's often a sub-path within output_dir or a specific other allowed path.
        if not os.path.isabs(target_directory):
            # Check if it's intended as a sub-directory of output_dir
            potential_output_subdir = os.path.join(self.output_dir, target_directory)
            if os.path.isdir(potential_output_subdir): # If it exists as a subdir of output
                resolved_target_dir = potential_output_subdir
            else: # Assume it's relative to project root (e.g. "agents/tools")
                resolved_target_dir = os.path.abspath(target_directory)
        else:
            resolved_target_dir = target_directory

        self.sentinel.approve_action("Quartermaster", "file_read", detail=resolved_target_dir) # Sentinel checks the resolved path
        
        try:
            if not os.path.isdir(resolved_target_dir): # Check if it's actually a directory after approval
                # If list_files was called on a path that isn't a directory (e.g., a file path)
                # after Sentinel approval, it means Sentinel allowed reading that path, but it's not listable.
                # This case might indicate a misuse of list_files or an unexpected path.
                # For now, let's treat it as "directory not found" for listing purposes.
                raise QuartermasterException(f"Path is not a directory: {resolved_target_dir}")

            entries = os.listdir(resolved_target_dir)
        except FileNotFoundError: # This would catch if resolved_target_dir doesn't exist at all
            raise QuartermasterException(f"Directory not found: {resolved_target_dir}")
        except NotADirectoryError: # Explicitly catch if it's a file
             raise QuartermasterException(f"Path is not a directory: {resolved_target_dir}")
        except Exception as e:
            raise QuartermasterException(f"Error listing directory '{resolved_target_dir}': {str(e)}")
        
        if not entries:
            return f"No files or subdirectories found in {resolved_target_dir}."
        
        entries.sort() # Sort for consistent output
        # Indicate if an entry is a directory
        formatted_entries = []
        for entry in entries:
            entry_path = os.path.join(resolved_target_dir, entry)
            if os.path.isdir(entry_path):
                formatted_entries.append(f"- {entry}/ (directory)")
            else:
                formatted_entries.append(f"- {entry}")

        return f"Contents of {resolved_target_dir}:\n" + "\n".join(formatted_entries)

    def internet_search(self, query: str) -> str:
        """Performs an internet search using DuckDuckGo."""
        self.sentinel.approve_action("Quartermaster", "internet", detail=query)
        try:
            print(f"[Quartermaster] Performing internet search for: {query}")
            response = requests.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
                timeout=10 # Added timeout
            )
            response.raise_for_status() # Raises an HTTPError for bad responses (4XX or 5XX)
            data = response.json()
        except requests.exceptions.RequestException as e:
            print(f"[Quartermaster] Internet search network error: {e}")
            raise QuartermasterException(f"Internet search failed due to a network error: {str(e)}")
        except json.JSONDecodeError:
            print(f"[Quartermaster] Internet search error: Could not decode JSON response.")
            raise QuartermasterException("Internet search failed: Invalid response format from search API.")
        except Exception as e: # Catch any other unexpected errors
            print(f"[Quartermaster] Unexpected internet search error: {e}")
            raise QuartermasterException(f"An unexpected error occurred during internet search: {str(e)}")

        summary = ""
        if data.get("AbstractText"):
            summary = data["AbstractText"]
        elif data.get("RelatedTopics"):
            # Iterate through related topics to find the first one with text, preferring non-category entries
            for topic_group in data["RelatedTopics"]:
                if topic_group.get("Topics"): # Handle nested topics if present
                    for sub_topic in topic_group["Topics"]:
                        if isinstance(sub_topic, dict) and sub_topic.get("Text") and not sub_topic.get("Name"): # Prefer entries with Text but no "Name" (which are often categories)
                            summary = sub_topic["Text"]
                            break
                elif isinstance(topic_group, dict) and topic_group.get("Text"): # Handle flat related topics
                    summary = topic_group["Text"]
                    break
                if summary: break # Found a summary
        
        if not summary and data.get("Results"): # Fallback to first result title/text if no abstract
            first_result = data["Results"][0] if data["Results"] else {}
            summary = first_result.get("Text", first_result.get("FirstURL", "No specific text found in first result."))


        return summary.strip() if summary else "No relevant information found online."

    def handle_command(self, command: str) -> str:
        """Handles direct commands given to the Quartermaster."""
        clower = command.lower().strip()
        print(f"[Quartermaster] Handling command: {command[:100]}")

        try:
            if clower.startswith("read file") or clower.startswith("open file"):
                # Extract file path more robustly, handling potential quotes
                match = re.search(r'(?:read file|open file)\s+(?:["\']?([^"\']+)["\']?|(\S+))', command, re.IGNORECASE)
                file_path = ""
                if match:
                    file_path = match.group(1) or match.group(2) # group(1) for quoted, group(2) for unquoted
                
                if not file_path: return "Please specify the file path to read."
                content = self.read_file(file_path.strip())
                # Truncate long content for display, but indicate truncation
                max_len = 2000 
                if len(content) > max_len:
                    return f"Content of '{file_path}' (first {max_len} chars):\n{content[:max_len]}\n\n... (File content truncated due to length)"
                return f"Content of '{file_path}':\n{content}"

            elif clower.startswith("list files") or clower.startswith("show files") or clower.startswith("ls"):
                dir_name = None
                # More robust extraction of directory path, handling "in" or direct path
                # e.g., "list files in subfolder" or "list files subfolder" or "ls output/reports"
                match = re.search(r'(?:list files|show files|ls)(?:\s+in)?\s+(.+)', command, re.IGNORECASE)
                if match:
                    dir_name = match.group(1).strip().strip('"\'')
                elif clower == "list files" or clower == "show files" or clower == "ls": # Just the command itself
                    dir_name = None # Defaults to output_dir in list_files method
                else: # This case should not be hit if previous regex is good
                    return "Invalid 'list files' format. Try 'list files [directory_path]' or 'list files'."
                
                return self.list_files(dir_name)

            elif "list goals" in clower or "show goals" in clower or "current goals" in clower:
                return self.memory.list_goals()

            elif "list completed" in clower or "show completed" in clower:
                return self.memory.list_completed()

            elif clower.startswith("add goal") or clower.startswith("new goal"):
                # Extract task description, handling "for [agent]" part
                key_phrase_match = re.match(r'(add goal|new goal)\s+(.+)', command, re.IGNORECASE)
                if not key_phrase_match:
                    return "Please specify the goal description. E.g., 'add goal Research new AI models'"
                
                full_task_part = key_phrase_match.group(2).strip()
                task_description = full_task_part
                agent_name = None

                # Check for "for [AgentName]" at the end of the task description
                agent_for_match = re.search(r'(.*)\s+for\s+([a-zA-Z]+)$', full_task_part)
                if agent_for_match:
                    potential_task = agent_for_match.group(1).strip()
                    potential_agent = agent_for_match.group(2).strip()
                    # Check if potential_agent is a known agent type (case-insensitive)
                    known_agents = ["scribe", "analyst", "engineer", "quartermaster", "sentinel", "planner", "researcher"]
                    if potential_agent.lower() in known_agents:
                        agent_name = potential_agent.capitalize() # Store with consistent casing
                        task_description = potential_task
                
                if not task_description: # If stripping "for agent" left nothing
                    return "Please provide a valid goal description before specifying an agent."

                self.memory.add_goal(task_description, agent=agent_name)
                return f"Added new goal{' for ' + agent_name if agent_name else ''}: {task_description}"

            elif clower.startswith("complete goal") or clower.startswith("mark goal") or clower.startswith("done goal"):
                # Try to match goal by number first, e.g., "complete goal 3"
                num_match = re.search(r'(?:complete|mark|done) goal\s+(\d+)', command, re.IGNORECASE)
                if num_match:
                    goal_num = int(num_match.group(1))
                    if self.memory.complete_goal(goal_num):
                        return f"Goal #{goal_num} marked as completed."
                    else:
                        return f"Goal #{goal_num} not found or already completed."
                else: # Try to match by text
                    text_match = re.search(r'(?:complete|mark|done) goal\s+(.+)', command, re.IGNORECASE)
                    if text_match:
                        goal_text = text_match.group(1).strip()
                        if self.memory.complete_goal(goal_text):
                            return f"Goal matching \"{goal_text}\" marked as completed."
                        else:
                            return f"No pending goal found matching \"{goal_text}\"."
                    else:
                        return "Please specify which goal to complete (by number or text)."
            
            elif clower.startswith("delete file") or clower.startswith("remove file"): # Made more specific
                match = re.search(r'(?:delete file|remove file)\s+(?:["\']?([^"\']+)["\']?|(\S+))', command, re.IGNORECASE)
                file_path_del = ""
                if match:
                    file_path_del = match.group(1) or match.group(2)
                
                if not file_path_del: return "Please specify the file to delete."
                self.delete_file(file_path_del.strip())
                return f"File '{file_path_del.strip()}' has been deleted."

            elif "save" in clower or "store" in clower or "write last output" in clower: # "save last output as X"
                # Regex to capture filename, allowing for paths with directories
                file_match_save = re.search(r'(?:save|store|write last output)(?:\s+last output)?\s+as\s+([\w\.\-\s\/\\]+)', command, re.IGNORECASE)
                custom_name = file_match_save.group(1).strip() if file_match_save else None

                # Determine which agent's output to save (prioritize Scribe, Engineer, Analyst, Researcher, Planner)
                content_to_save = (self.memory.recall_agent_output("scribe") or
                                   self.memory.recall_agent_output("engineer") or
                                   self.memory.recall_agent_output("analyst") or
                                   self.memory.recall_agent_output("researcher") or
                                   self.memory.recall_agent_output("planner") or # Added Planner
                                   self.memory.recall_agent_output(self.memory.get_last_agent() or "unknown")) # Fallback to very last agent

                if not content_to_save:
                    return "There is no recent agent output available to save."

                # Determine filename
                agent_source = self.memory.get_last_agent() or "output" # Agent who produced the content
                default_ext = ".txt"
                if agent_source == "engineer": default_ext = ".py"
                elif agent_source == "scribe":
                    if "report" in (custom_name or "").lower() or "document" in (custom_name or "").lower(): default_ext = ".md" # Or .pdf if Scribe can do PDF
                    elif "blog" in (custom_name or "").lower(): default_ext = ".md"
                elif agent_source == "researcher": default_ext = ".md" # Research reports often in Markdown
                elif agent_source == "planner": default_ext = ".json" # Plans are JSON

                filename_to_save_with = ""
                if custom_name:
                    base, ext = os.path.splitext(custom_name)
                    # If user provides an extension, respect it. Otherwise, use default.
                    filename_to_save_with = custom_name if ext else base + default_ext
                else:
                    # Generate a default filename if not provided
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename_to_save_with = f"{agent_source}_output_{timestamp}{default_ext}"
                
                self.write_file(filename_to_save_with, content_to_save)
                return f"Last agent output (from {agent_source}) saved to file: {os.path.join(self.output_dir, filename_to_save_with)}"

            # Fallback for unrecognized Quartermaster commands
            return ("Quartermaster: Command not fully recognized. I can manage files ('read file X', 'list files', 'delete file X', 'save last output as Y'), "
                    "goals ('add goal Z', 'list goals', 'complete goal N'), and perform internet searches via tools if Analyst/Researcher request it.")

        except QuartermasterException as qe:
            print(f"[Quartermaster] Error: {qe}")
            return f"Quartermaster Error: {str(qe)}"
        except Exception as e: # Catch any other unexpected error within handle_command
            import traceback
            print(f"[Quartermaster] Unexpected error in handle_command: {e}\n{traceback.format_exc()}")
            return f"Quartermaster encountered an unexpected internal error: {str(e)}"

    def run(self, command: str) -> str:
        """
        Alias for handle_command to maintain compatibility with the main loop
        which calls .run() on agent instances.
        """
        return self.handle_command(command)