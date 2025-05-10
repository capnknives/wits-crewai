# agents/tools/calculator_tool.py
from .base_tool import Tool, ToolException
import re
import ast 

class CalculatorTool(Tool):
    name = "calculator"
    description = ("Evaluates a simple mathematical arithmetic expression string involving numbers, "
                   "addition (+), subtraction (-), multiplication (*), division (/), and parentheses. "
                   "It does not support functions, variables, or complex algebra.")
    argument_schema = {
        "expression": "str: The simple arithmetic expression to evaluate (e.g., '2 + 2', '100 / (5 * 2)')."
    }

    def execute(self, **kwargs) -> str:
        expression = kwargs.get("expression")
        if expression is None: 
            raise ToolException("CalculatorTool: 'expression' argument is required and was not provided.")
        if not isinstance(expression, str):
            raise ToolException(f"CalculatorTool: 'expression' argument must be a string, got {type(expression)}.")

        allowed_chars_pattern = r"^[0-9\s\.\+\-\*\/\(\)]+$"
        if not re.match(allowed_chars_pattern, expression):
            raise ToolException(
                f"CalculatorTool: Invalid characters in expression: '{expression}'. "
                "Only numbers, spaces, decimal points, +, -, *, /, (, ) are allowed."
            )
        
        if any(char.isalpha() for char in expression): 
             raise ToolException("CalculatorTool: Expression contains alphabetical characters, which is not allowed for safety.")

        try:
            # WARNING: eval() is dangerous. Using ast.literal_eval for basic safety,
            # but it only handles literals, not arithmetic.
            # For actual arithmetic, a proper parser or safer eval mechanism is needed.
            # This is a placeholder for a safer arithmetic evaluation.
            # For now, we'll stick to the more functional (but risky) eval for the example,
            # assuming LLM provides simple arithmetic.
            
            # A slightly safer approach for very simple arithmetic might be:
            # Sanitize further or use a dedicated math library.
            # For this example, we'll proceed with eval, highlighting the risk.
            result = eval(expression) 
            return f"The result of the expression '{expression}' is {result}."
        except ZeroDivisionError:
            return f"Error: Cannot divide by zero in the expression '{expression}'."
        except SyntaxError:
            raise ToolException(f"CalculatorTool: Syntax error in expression: '{expression}'. Please provide a valid mathematical expression.")
        except Exception as e:
            print(f"[CalculatorTool] Unexpected error evaluating expression '{expression}': {e}") 
            raise ToolException(f"CalculatorTool: Could not evaluate the expression '{expression}'. Ensure it is a valid, simple arithmetic expression.")