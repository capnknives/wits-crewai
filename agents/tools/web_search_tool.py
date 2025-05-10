# agents/tools/web_search_tool.py
from .base_tool import Tool, ToolException
from ..quartermaster_agent import QuartermasterAgent # To call QM's methods

class WebSearchTool(Tool):
    name = "internet_search"
    description = ("Performs an internet search for a given query using DuckDuckGo (via Quartermaster) "
                   "and returns a summary of the findings. Useful for finding current information, "
                   "researching topics, or getting quick facts.")
    argument_schema = {
        "query": "str: The search query string."
    }

    def __init__(self, quartermaster: QuartermasterAgent):
        super().__init__() # Initialize the base Tool class
        if not isinstance(quartermaster, QuartermasterAgent):
            raise ValueError("WebSearchTool requires an instance of QuartermasterAgent.")
        self.qm = quartermaster

    def execute(self, **kwargs) -> str:
        query = kwargs.get("query")
        if not query or not isinstance(query, str):
            raise ToolException("WebSearchTool: 'query' argument is required and must be a non-empty string.")

        try:
            print(f"[WebSearchTool] Executing search for query: {query}")
            # The Quartermaster's internet_search method already includes Sentinel approval
            search_result = self.qm.internet_search(query)
            if not search_result or "No relevant information found online." in search_result:
                return f"Internet search for '{query}' yielded no specific results or summary."
            return f"Internet search results for '{query}':\n{search_result}"
        except Exception as e:
            # Catch exceptions from Quartermaster or requests if they occur
            print(f"[WebSearchTool] Error during internet search for '{query}': {e}")
            raise ToolException(f"WebSearchTool: Failed to execute internet search for '{query}'. Error: {str(e)}")
