# agents/tools/base_tool.py
import inspect

class ToolException(Exception):
    """Custom exception for tool-related errors."""
    pass

class Tool:
    name: str = "base_tool" # Must be overridden by subclasses
    description: str = "This is a base tool and does nothing." # Must be overridden
    # Defines the arguments the tool's 'execute' method expects.
    # Keys are argument names, values are descriptive strings (e.g., type, purpose).
    argument_schema: dict = {}

    def __init__(self):
        if self.name == "base_tool" or not self.description or self.name == "" :
            raise NotImplementedError(
                "Tool subclasses must define a unique 'name' and a 'description'."
            )

    def get_signature_string(self) -> str:
        """Returns a string representation of the tool's arguments for the LLM."""
        if not self.argument_schema:
            return "This tool takes no arguments."
        
        args_parts = []
        for arg_name, arg_desc in self.argument_schema.items():
            args_parts.append(f"'{arg_name}': ({arg_desc})")
        return f"Arguments (as a JSON object): {{ {', '.join(args_parts)} }}"

    def execute(self, **kwargs) -> str:
        """
        Executes the tool with the given arguments.
        Subclasses MUST override this method.
        The return value should be a string, as this will be fed back to the LLM.
        """
        raise NotImplementedError("Tool subclasses must implement the 'execute' method.")

    def get_full_description_for_llm(self) -> str:
        """Provides a full description of the tool for the LLM, including name, description, and arguments."""
        return (
            f"Tool Name: \"{self.name}\"\n"
            f"Description: {self.description}\n"
            f"{self.get_signature_string()}"
        )