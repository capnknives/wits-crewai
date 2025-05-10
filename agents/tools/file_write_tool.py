# agents/tools/file_write_tool.py
from .base_tool import Tool, ToolException
from ..quartermaster_agent import QuartermasterAgent, QuartermasterException

class WriteFileTool(Tool):
    name = "write_file_content"
    description = ("Writes or overwrites the given content to a specified file within the project's 'output' directory. "
                   "Useful for saving generated text, code, analysis results, or notes. "
                   "Use with caution, as it will overwrite existing files with the same name.")
    argument_schema = {
        "file_path": "str: The name or relative path of the file to write within the 'output' directory (e.g., 'my_analysis.txt', 'code/script.py'). Subdirectories will be created if they don't exist.",
        "content": "str: The text content to write to the file."
    }

    def __init__(self, quartermaster: QuartermasterAgent):
        super().__init__()
        if not isinstance(quartermaster, QuartermasterAgent):
            raise ValueError("WriteFileTool requires an instance of QuartermasterAgent.")
        self.qm = quartermaster

    def execute(self, **kwargs) -> str:
        file_path = kwargs.get("file_path")
        content = kwargs.get("content")

        if not file_path or not isinstance(file_path, str):
            raise ToolException("WriteFileTool: 'file_path' argument is required and must be a non-empty string.")
        if content is None or not isinstance(content, str): # Allow empty string content
            raise ToolException("WriteFileTool: 'content' argument is required and must be a string.")

        try:
            print(f"[WriteFileTool] Attempting to write to file: {file_path}")
            # Quartermaster's write_file handles path resolution (ensuring it's within output_dir)
            # and Sentinel approval.
            self.qm.write_file(file_path, content)
            # Construct the full path for the confirmation message
            full_saved_path = os.path.join(self.qm.output_dir, file_path) # QM ensures it's in output_dir
            return f"Content successfully written to file: '{full_saved_path}'."
        except QuartermasterException as qe:
            print(f"[WriteFileTool] Quartermaster error writing to '{file_path}': {qe}")
            return f"Error writing to file '{file_path}': {str(qe)}"
        except Exception as e:
            print(f"[WriteFileTool] Unexpected error writing to '{file_path}': {e}")
            raise ToolException(f"WriteFileTool: Failed to write to file '{file_path}'. Error: {str(e)}")

