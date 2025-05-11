# agents/quartermaster_agent.py
import os
import json
import requests # For internet_search method
import re
from datetime import datetime
from typing import Optional, Dict, Any, List # Added List, Dict, Any

# Assuming SentinelException and QuartermasterException are defined
# If they are in a base agent file or utils, adjust import path
# For now, defining them here if not imported from elsewhere.
class QuartermasterException(Exception):
    """Custom exception for Quartermaster-specific errors."""
    pass

# Assuming SentinelAgent is imported if its exception is used, or define SentinelException
# from .sentinel_agent import SentinelException # If you have this structure

class QuartermasterAgent:
    def __init__(self, config: Dict[str, Any], memory, sentinel): # Added type hints
        self.config = config
        self.memory = memory # Should be an instance of EnhancedMemory or VectorMemory
        self.sentinel = sentinel # Should be an instance of SentinelAgent
        
        self.output_dir = config.get("output_directory", "output")
        # Ensure output_dir is absolute or correctly relative to project root.
        # os.makedirs should ideally be handled by QM itself if it's responsible for this dir.
        if not os.path.isabs(self.output_dir):
            self.output_dir = os.path.abspath(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"[Quartermaster] Output directory set to: {self.output_dir}")


    def read_file(self, file_path: str) -> str:
        """
        Reads the content of a specified file.
        Handles path resolution to be relative to output_dir if path is not absolute.
        Sentinel approval is sought before reading.
        """
        # Determine the absolute path for Sentinel and file operations
        if os.path.isabs(file_path):
            abs_path_to_check = os.path.normpath(file_path)
        else:
            # Try resolving relative to output_dir first
            candidate_path = os.path.normpath(os.path.join(self.output_dir, file_path))
            if os.path.exists(candidate_path): # Check if it exists within output_dir
                abs_path_to_check = candidate_path
            else:
                # If not in output_dir, assume it might be relative to project root for Sentinel check
                # This allows reading config files or agent source if permitted by Sentinel.
                abs_path_to_check = os.path.abspath(file_path)

        print(f"[Quartermaster] Attempting to read file. Original path: '{file_path}', Resolved absolute for check: '{abs_path_to_check}'")
        self.sentinel.approve_action("Quartermaster", "file_read", detail=abs_path_to_check)
        
        try:
            # Use the path that Sentinel approved (which should be absolute and validated)
            with open(abs_path_to_check, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            raise QuartermasterException(f"File not found: {abs_path_to_check}")
        except Exception as e:
            raise QuartermasterException(f"Error reading file '{abs_path_to_check}': {str(e)}")

    def write_file(self, file_path: str, content: str) -> bool:
        """
        Writes or overwrites content to a specified file, ensuring it's within the output_dir.
        Sentinel approval is sought before writing.
        """
        # Normalize path and ensure it's within output_dir
        # os.path.abspath will resolve, then we check if it's within output_dir
        if os.path.isabs(file_path):
            # If an absolute path is given, Sentinel must ensure it's a permitted write location.
            # For Quartermaster's primary role, we usually expect relative paths to output_dir.
            # Sentinel's approval on abs_path will be the safeguard.
            abs_path_to_write = os.path.normpath(file_path)
        else:
            # Sanitize to prevent path traversal before joining with output_dir
            # Remove leading slashes/dots. A more robust solution might check each component.
            safe_relative_path = os.path.normpath(file_path.lstrip(os.sep + '.'))
            if ".." in safe_relative_path.split(os.sep): # Disallow ".." in path components
                 raise QuartermasterException(f"Invalid file path for writing (contains '..'): {file_path}")
            abs_path_to_write = os.path.join(self.output_dir, safe_relative_path)
            abs_path_to_write = os.path.normpath(abs_path_to_write) # Normalize again after join

        print(f"[Quartermaster] Attempting to write file. Original path: '{file_path}', Resolved absolute for write: '{abs_path_to_write}'")
        # Sentinel makes the final approval, which should check if abs_path_to_write is within output_dir
        self.sentinel.approve_action("Quartermaster", "file_write", detail=abs_path_to_write)
        
        try:
            # Ensure parent directory exists within the output_dir structure
            parent_dir = os.path.dirname(abs_path_to_write)
            if not parent_dir.startswith(self.output_dir): # Double check parent is within output_dir
                # This check is a bit redundant if Sentinel is configured correctly, but adds safety
                raise QuartermasterException(f"Cannot create parent directory '{parent_dir}' outside of output directory '{self.output_dir}'.")
            os.makedirs(parent_dir, exist_ok=True)

            with open(abs_path_to_write, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"[Quartermaster] Content written to: {abs_path_to_write}")
            return True
        except Exception as e:
            raise QuartermasterException(f"Error writing file '{abs_path_to_write}': {str(e)}")

    def delete_file(self, file_path: str) -> bool:
        """
        Deletes a specified file, ensuring it's within the output_dir.
        Sentinel approval is sought before deletion.
        """
        if os.path.isabs(file_path):
            abs_path_to_delete = os.path.normpath(file_path)
        else:
            safe_relative_path = os.path.normpath(file_path.lstrip(os.sep + '.'))
            if ".." in safe_relative_path.split(os.sep):
                 raise QuartermasterException(f"Invalid file path for deletion (contains '..'): {file_path}")
            abs_path_to_delete = os.path.join(self.output_dir, safe_relative_path)
            abs_path_to_delete = os.path.normpath(abs_path_to_delete)

        print(f"[Quartermaster] Attempting to delete file. Original path: '{file_path}', Resolved absolute for delete: '{abs_path_to_delete}'")
        self.sentinel.approve_action("Quartermaster", "file_delete", detail=abs_path_to_delete)
        
        try:
            if not os.path.exists(abs_path_to_delete): # Check if file exists before attempting delete
                 raise QuartermasterException(f"File not found for deletion: {abs_path_to_delete}")
            if not abs_path_to_delete.startswith(self.output_dir): # Ensure it's within output_dir
                raise QuartermasterException(f"Deletion attempt outside output directory: {abs_path_to_delete}")

            os.remove(abs_path_to_delete)
            print(f"[Quartermaster] File deleted: {abs_path_to_delete}")
            return True
        except FileNotFoundError: # Should be caught by os.path.exists, but good to have
            raise QuartermasterException(f"File not found for deletion: {abs_path_to_delete}")
        except Exception as e:
            raise QuartermasterException(f"Error deleting file '{abs_path_to_delete}': {str(e)}")

    def list_files(self, directory: Optional[str] = None) -> str:
        """
        Lists files and subdirectories within a specified directory.
        Defaults to self.output_dir. Allows listing other project-relative dirs if Sentinel approves.
        """
        target_directory_for_listing: str
        if directory is None or directory == ".": # Treat "." as the main output directory
            target_directory_for_listing = self.output_dir
        elif os.path.isabs(directory):
            target_directory_for_listing = os.path.normpath(directory)
        else: # Relative path
            # Assume relative to output_dir unless it's clearly an attempt to go outside (e.g. "../")
            # Sentinel will be the ultimate judge of path safety.
            # For Quartermaster's own logic, if a relative path is given, it's typically a sub-path within output_dir.
            potential_output_subdir = os.path.normpath(os.path.join(self.output_dir, directory))
            # A more direct approach: let Sentinel validate the path as given by user if not absolute.
            # Sentinel should resolve based on project root or output dir context.
            target_directory_for_listing = os.path.abspath(os.path.join(self.output_dir, directory))


        print(f"[Quartermaster] Listing directory. Original input: '{directory}', Resolved for approval: '{target_directory_for_listing}'")
        self.sentinel.approve_action("Quartermaster", "file_read", detail=target_directory_for_listing)
        
        try:
            if not os.path.isdir(target_directory_for_listing):
                # If list_files was called on a path that isn't a directory after Sentinel approval,
                # it means Sentinel allowed reading that path, but it's not listable by os.listdir.
                raise QuartermasterException(f"Path is not a directory or does not exist: {target_directory_for_listing}")

            entries = os.listdir(target_directory_for_listing)
        except FileNotFoundError:
            raise QuartermasterException(f"Directory not found: {target_directory_for_listing}")
        except NotADirectoryError:
             raise QuartermasterException(f"Path is not a directory: {target_directory_for_listing}")
        except Exception as e:
            raise QuartermasterException(f"Error listing directory '{target_directory_for_listing}': {str(e)}")
        
        if not entries:
            return f"No files or subdirectories found in {target_directory_for_listing}."
        
        entries.sort()
        formatted_entries = []
        for entry in entries:
            entry_path = os.path.join(target_directory_for_listing, entry) # Use target_directory_for_listing for consistency
            if os.path.isdir(entry_path):
                formatted_entries.append(f"- {entry}/ (directory)")
            else:
                formatted_entries.append(f"- {entry}")

        return f"Contents of {target_directory_for_listing}:\n" + "\n".join(formatted_entries)

    def internet_search(self, query: str) -> str:
        """Performs an internet search using DuckDuckGo."""
        self.sentinel.approve_action("Quartermaster", "internet", detail=query)
        try:
            print(f"[Quartermaster] Performing internet search for: {query}")
            response = requests.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            print(f"[Quartermaster] Internet search network error: {e}")
            raise QuartermasterException(f"Internet search failed due to a network error: {str(e)}")
        except json.JSONDecodeError:
            print(f"[Quartermaster] Internet search error: Could not decode JSON response.")
            raise QuartermasterException("Internet search failed: Invalid response format from search API.")
        except Exception as e:
            print(f"[Quartermaster] Unexpected internet search error: {e}")
            raise QuartermasterException(f"An unexpected error occurred during internet search: {str(e)}")

        summary = ""
        if data.get("AbstractText"):
            summary = data["AbstractText"]
        elif data.get("RelatedTopics"):
            for topic_group in data["RelatedTopics"]:
                if topic_group.get("Topics"):
                    for sub_topic in topic_group["Topics"]:
                        if isinstance(sub_topic, dict) and sub_topic.get("Text") and not sub_topic.get("Name"):
                            summary = sub_topic["Text"]
                            break
                elif isinstance(topic_group, dict) and topic_group.get("Text"):
                    summary = topic_group["Text"]
                    break
                if summary: break
        
        if not summary and data.get("Results"):
            first_result = data["Results"][0] if data["Results"] else {}
            summary = first_result.get("Text", first_result.get("FirstURL", "No specific text found in first result."))

        return summary.strip() if summary else "No relevant information found online."

    def handle_command(self, command: str) -> str:
        """Handles direct commands given to the Quartermaster."""
        # Normalize whitespace in the input command, including non-breaking spaces
        command_cleaned_nbsp = command.replace('\xa0', ' ') 
        command_normalized = ' '.join(command_cleaned_nbsp.split())
        clower = command_normalized.lower() # Use this for all matching

        print(f"[Quartermaster] Handling command (normalized): '{clower[:100]}'")

        try:
            if clower.startswith("read file") or clower.startswith("open file"):
                match = re.search(r'(?:read file|open file)\s+(?:["\']?([^"\']+)["\']?|(\S+))', command_normalized, re.IGNORECASE)
                file_path = ""
                if match:
                    file_path = match.group(1) or match.group(2)
                if not file_path: return "Please specify the file path to read."
                content = self.read_file(file_path.strip())
                max_len = 2000
                if len(content) > max_len:
                    return f"Content of '{file_path}' (first {max_len} chars):\n{content[:max_len]}\n\n... (File content truncated due to length)"
                return f"Content of '{file_path}':\n{content}"

            elif clower.startswith("list files") or clower.startswith("show files") or clower.startswith("ls"):
                dir_name = None
                match = re.search(r'(?:list files|show files|ls)(?:\s+in)?\s+(.+)', command_normalized, re.IGNORECASE)
                if match:
                    dir_name = match.group(1).strip().strip('"\'')
                elif clower in ["list files", "show files", "ls"]:
                    dir_name = None 
                else:
                    return "Invalid 'list files' format. Try 'list files [directory_path]' or 'list files'."
                return self.list_files(dir_name)

            elif "list goals" in clower or "show goals" in clower or "current goals" in clower:
                return self.memory.list_goals()

            elif "list completed" in clower or "show completed" in clower:
                return self.memory.list_completed()

            elif clower.startswith("add goal") or clower.startswith("new goal"):
                key_phrase_match = re.match(r'(add goal|new goal)\s+(.+)', command_normalized, re.IGNORECASE)
                if not key_phrase_match:
                    return "Please specify the goal description. E.g., 'add goal Research new AI models'"
                
                full_task_part = key_phrase_match.group(2).strip()
                task_description = full_task_part
                agent_name = None
                agent_for_match = re.search(r'(.*)\s+for\s+([a-zA-Z]+)$', full_task_part)
                if agent_for_match:
                    potential_task = agent_for_match.group(1).strip()
                    potential_agent = agent_for_match.group(2).strip()
                    known_agents = ["scribe", "analyst", "engineer", "quartermaster", "sentinel", "planner", "researcher"]
                    if potential_agent.lower() in known_agents:
                        agent_name = potential_agent.capitalize()
                        task_description = potential_task
                if not task_description:
                    return "Please provide a valid goal description before specifying an agent."
                self.memory.add_goal(task_description, agent=agent_name)
                return f"Added new goal{' for ' + agent_name if agent_name else ''}: {task_description}"

            elif clower.startswith("complete goal") or clower.startswith("mark goal") or clower.startswith("done goal"):
                # Try to match "complete goal by id XXXXX" first
                id_match = re.search(r'(?:complete|mark|done)\s+goal\s+by\s+id\s+([a-f0-9\-]+)(?:\s+with\s+result:(.*))?', clower, re.IGNORECASE)
                if id_match:
                    goal_id_to_complete = id_match.group(1).strip()
                    result_text = id_match.group(2).strip() if id_match.group(2) else None
                    if self.memory.complete_goal(goal_id_to_complete, result=result_text):
                        return f"Goal ID '{goal_id_to_complete}' marked as completed." + (f" With result: {result_text}" if result_text else "")
                    else:
                        return f"Goal ID '{goal_id_to_complete}' not found or already completed."
                
                # Fallback to match by number or text if "by id" is not present
                num_match = re.search(r'(?:complete|mark|done)\s+goal\s+(\d+)(?:\s+with\s+result:(.*))?', clower, re.IGNORECASE)
                if num_match:
                    goal_num = int(num_match.group(1))
                    result_text = num_match.group(2).strip() if num_match.group(2) else None
                    if self.memory.complete_goal(goal_num, result=result_text): # Assumes complete_goal can take index
                        return f"Goal #{goal_num} marked as completed." + (f" With result: {result_text}" if result_text else "")
                    else:
                        return f"Goal #{goal_num} not found or already completed."
                else:
                    text_match = re.search(r'(?:complete|mark|done)\s+goal\s+([^with]+?)(?:\s+with\s+result:(.*))?$', clower, re.IGNORECASE)
                    if text_match:
                        goal_text = text_match.group(1).strip()
                        result_text = text_match.group(2).strip() if text_match.group(2) else None
                        if self.memory.complete_goal(goal_text, result=result_text): # Assumes complete_goal can take text
                            return f"Goal matching \"{goal_text}\" marked as completed." + (f" With result: {result_text}" if result_text else "")
                        else:
                            return f"No pending goal found matching \"{goal_text}\"."
                    else:
                        return "Please specify which goal to complete (by ID, number, or text)."

            # --- NEW: Delete Goal Permanently Command ---
            elif clower.startswith("delete goal permanently by id") or clower.startswith("permanently delete goal by id"):
                match = re.search(r"(?:delete goal permanently by id|permanently delete goal by id)\s+([a-f0-9\-]+)", clower, re.IGNORECASE)
                if match:
                    goal_id_to_delete = match.group(1).strip()
                    if self.memory.delete_goal_permanently(goal_id_to_delete):
                        return f"Goal ID '{goal_id_to_delete}' and its related memory segments permanently deleted."
                    else:
                        return f"Could not permanently delete goal ID '{goal_id_to_delete}'. It might not exist."
                else:
                    return "Please specify the Goal ID to delete permanently (e.g., 'delete goal permanently by id xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx')."
            # --- END NEW ---

            elif clower.startswith("delete file") or clower.startswith("remove file"):
                match = re.search(r'(?:delete file|remove file)\s+(?:["\']?([^"\']+)["\']?|(\S+))', command_normalized, re.IGNORECASE)
                file_path_del = ""
                if match:
                    file_path_del = match.group(1) or match.group(2)
                if not file_path_del: return "Please specify the file to delete."
                self.delete_file(file_path_del.strip())
                return f"File '{file_path_del.strip()}' has been deleted."

            elif "save" in clower or "store" in clower or "write last output" in clower:
                file_match_save = re.search(r'(?:save|store|write last output)(?:\s+last output)?\s+as\s+([\w\.\-\s\/\\]+)', command_normalized, re.IGNORECASE)
                custom_name = file_match_save.group(1).strip() if file_match_save else None
                content_to_save = (self.memory.recall_agent_output("scribe") or
                                   self.memory.recall_agent_output("engineer") or
                                   self.memory.recall_agent_output("analyst") or
                                   self.memory.recall_agent_output("researcher") or
                                   self.memory.recall_agent_output("planner") or
                                   self.memory.recall_agent_output(self.memory.get_last_agent() or "unknown"))
                if not content_to_save:
                    return "There is no recent agent output available to save."
                agent_source = self.memory.get_last_agent() or "output"
                default_ext = ".txt"
                if agent_source == "engineer": default_ext = ".py"
                elif agent_source == "scribe":
                    if "report" in (custom_name or "").lower() or "document" in (custom_name or "").lower(): default_ext = ".md"
                    elif "blog" in (custom_name or "").lower(): default_ext = ".md"
                elif agent_source == "researcher": default_ext = ".md"
                elif agent_source == "planner": default_ext = ".json"
                filename_to_save_with = ""
                if custom_name:
                    base, ext = os.path.splitext(custom_name)
                    filename_to_save_with = custom_name if ext else base + default_ext
                else:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename_to_save_with = f"{agent_source}_output_{timestamp}{default_ext}"
                self.write_file(filename_to_save_with, content_to_save)
                # Construct full path for confirmation, assuming write_file places it in output_dir
                full_saved_path = os.path.join(self.output_dir, filename_to_save_with)
                # Normalize for consistent display
                full_saved_path = os.path.normpath(full_saved_path)
                return f"Last agent output (from {agent_source}) saved to file: {full_saved_path}"

            return ("Quartermaster: Command not fully recognized. I can manage files ('read file X', 'list files', 'delete file X', 'save last output as Y'), "
                    "goals ('add goal Z', 'list goals', 'complete goal N/ID', 'delete goal permanently by id ID'), and perform internet searches via tools if Analyst/Researcher request it.")

        except QuartermasterException as qe:
            print(f"[Quartermaster] Error: {qe}")
            return f"Quartermaster Error: {str(qe)}"
        except Exception as e:
            import traceback
            print(f"[Quartermaster] Unexpected error in handle_command: {e}\n{traceback.format_exc()}")
            return f"Quartermaster encountered an unexpected internal error: {str(e)}"

    def run(self, command: str, **kwargs) -> str: # Added **kwargs to accept unexpected args but ignore them
        """
        Alias for handle_command to maintain compatibility with the main loop
        which calls .run() on agent instances.
        Ignores unexpected keyword arguments.
        """
        # The `associated_goal_id_for_new_plan` is handled by PlannerAgent.
        # Quartermaster's run doesn't use it.
        return self.handle_command(command)
