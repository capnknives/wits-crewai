# agents/tools/file_list_tool.py
from .base_tool import Tool, ToolException
from ..quartermaster_agent import QuartermasterAgent, QuartermasterException
import os # For potential path manipulation if needed, though QM handles most

class ListFilesTool(Tool):
    name = "list_files_in_directory"
    description = ("Lists files and subdirectories within a specified directory in the project's accessible areas "
                   "(defaults to the 'output' directory). Can optionally filter by file extension. "
                   "Useful for discovering available resources or project structure.")
    argument_schema = {
        "directory_path": "str (optional): The relative path of the directory to list (e.g., '.', 'data_folder'). Defaults to the main output directory if not provided or empty.",
        "extension_filter": "str (optional): A file extension to filter by (e.g., '.py', '.txt'). If provided, only files matching the extension are listed."
    }

    def __init__(self, quartermaster: QuartermasterAgent):
        super().__init__()
        if not isinstance(quartermaster, QuartermasterAgent):
            raise ValueError("ListFilesTool requires an instance of QuartermasterAgent.")
        self.qm = quartermaster

    def execute(self, **kwargs) -> str:
        directory_path = kwargs.get("directory_path") # QM's list_files handles None as output_dir
        extension_filter = kwargs.get("extension_filter")

        if directory_path and not isinstance(directory_path, str):
            raise ToolException("ListFilesTool: 'directory_path' must be a string if provided.")
        if extension_filter and not isinstance(extension_filter, str):
            raise ToolException("ListFilesTool: 'extension_filter' must be a string if provided.")
        
        if extension_filter and not extension_filter.startswith('.'):
            extension_filter = '.' + extension_filter # Ensure it has a leading dot

        try:
            print(f"[ListFilesTool] Listing directory: '{directory_path if directory_path else 'output_dir'}' with filter: '{extension_filter}'")
            # Quartermaster's list_files already handles Sentinel approval.
            # We will perform filtering here if QM doesn't support it directly.
            
            # Assuming self.qm.list_files(directory_path) returns a string like:
            # "Files in <resolved_path>:\n- file1.txt\n- folderA\n- file2.py"
            # or "No files found in <resolved_path>."
            
            raw_listing_str = self.qm.list_files(directory_path)

            if "No files found" in raw_listing_str or not raw_listing_str:
                return f"No files or directories found in '{directory_path if directory_path else self.qm.output_dir}'."

            lines = raw_listing_str.split('\n')
            header = lines[0] # "Files in <resolved_path>:"
            entries = [line.lstrip("- ") for line in lines[1:] if line.strip() and line.startswith("- ")]

            if extension_filter:
                filtered_entries = [entry for entry in entries if entry.endswith(extension_filter) and not os.path.isdir(os.path.join(directory_path or self.qm.output_dir, entry))]
                # Also include directories in the listing if no extension filter or if we want to show all structure
                # For now, if extension_filter is on, only list matching files.
                if not filtered_entries:
                    return f"No files matching extension '{extension_filter}' found in '{directory_path if directory_path else self.qm.output_dir}'.\nOriginal listing (all entries):\n{raw_listing_str}"
                return f"{header} (filtered by {extension_filter})\n" + "\n".join([f"- {e}" for e in filtered_entries])
            else:
                # Return the original unfiltered listing from Quartermaster
                return raw_listing_str

        except QuartermasterException as qe:
            print(f"[ListFilesTool] Quartermaster error listing directory '{directory_path}': {qe}")
            return f"Error listing files in '{directory_path if directory_path else 'output_dir'}': {str(qe)}"
        except Exception as e:
            print(f"[ListFilesTool] Unexpected error listing directory '{directory_path}': {e}")
            raise ToolException(f"ListFilesTool: Failed to list files for '{directory_path}'. Error: {str(e)}")

