# agents/tools/file_read_tool.py
from .base_tool import Tool, ToolException
from ..quartermaster_agent import QuartermasterAgent, QuartermasterException

class ReadFileTool(Tool):
    name = "read_file_content"
    description = ("Reads the content of a specified file from the project's accessible directories "
                   "(primarily the 'output' directory, but can also read from project root if allowed by Sentinel). "
                   "Useful for analyzing existing data, code, or notes.")
    argument_schema = {
        "file_path": "str: The name or relative path of the file to read (e.g., 'my_notes.txt', 'data/report.csv')."
    }

    def __init__(self, quartermaster: QuartermasterAgent):
        super().__init__()
        if not isinstance(quartermaster, QuartermasterAgent):
            raise ValueError("ReadFileTool requires an instance of QuartermasterAgent.")
        self.qm = quartermaster

    def execute(self, **kwargs) -> str:
        file_path = kwargs.get("file_path")
        if not file_path or not isinstance(file_path, str):
            raise ToolException("ReadFileTool: 'file_path' argument is required and must be a non-empty string.")

        try:
            print(f"[ReadFileTool] Attempting to read file: {file_path}")
            # Quartermaster's read_file method handles path resolution (relative to output_dir)
            # and Sentinel approval.
            content = self.qm.read_file(file_path)
            # Truncate very long files for LLM context, but indicate truncation.
            max_len = 2000 # Max characters to return to LLM to avoid overly long context
            if len(content) > max_len:
                return f"Content of '{file_path}' (truncated to {max_len} chars):\n{content[:max_len]}\n... (file truncated)"
            return f"Content of '{file_path}':\n{content}"
        except QuartermasterException as qe: # Catch specific QM errors like FileNotFoundError
            print(f"[ReadFileTool] Quartermaster error reading '{file_path}': {qe}")
            # Return a clear error message that the LLM can understand
            return f"Error reading file '{file_path}': {str(qe)}" 
        except Exception as e:
            print(f"[ReadFileTool] Unexpected error reading '{file_path}': {e}")
            raise ToolException(f"ReadFileTool: Failed to read file '{file_path}'. Error: {str(e)}")
