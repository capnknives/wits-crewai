# core/router.py
# Routes a given user command to the most relevant agent

# No config import needed here if main.py passes the fallback agent name directly.
# However, if the router itself were to load the config, it would be:
# import yaml
# import os
# config = {}
# config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml") # Path relative to this file
# if os.path.exists(config_path):
#     with open(config_path, 'r', encoding='utf-8') as f:
#         config = yaml.safe_load(f)
# FALLBACK_AGENT = config.get("router", {}).get("fallback_agent", "analyst")

# core/router.py
# Routes a given user command to the most relevant agent

def route_task(user_input, fallback_agent="analyst"):
    """
    Routes a given user command to the most relevant agent.
    Args:
        user_input (str): The raw input from the user.
        fallback_agent (str): The name of the agent to use if no other agent is matched.
    Returns:
        tuple: (agent_name_key, task_content)
    """
    # Clean up the input - remove any unwanted prefixes
    if user_input.startswith("er,"):
        user_input = user_input[3:].strip()
    
    text_lower = user_input.lower()
    task_content = user_input  # Default task content is the full input

    # Agent-specific parsing (stripping agent name from task)
    if text_lower.startswith("scribe"):
        agent_name = "scribe"
        task_content = user_input[len("scribe"):].lstrip(" :,")
    elif text_lower.startswith("analyst"):
        agent_name = "analyst"
        task_content = user_input[len("analyst"):].lstrip(" :,")
    elif text_lower.startswith("engineer"):
        agent_name = "engineer"
        task_content = user_input[len("engineer"):].lstrip(" :,")
    elif text_lower.startswith("researcher"):
        agent_name = "researcher"
        task_content = user_input[len("researcher"):].lstrip(" :,")
    elif text_lower.startswith("quartermaster"):
        agent_name = "quartermaster"
        # Quartermaster typically handles its own sub-command parsing from the content
        task_content = user_input[len("quartermaster"):].lstrip(" :,")
        # If content is empty, it implies a general query to Quartermaster
        if not task_content and user_input.lower() == "quartermaster":
            task_content = "help"  # or some default general status query
    elif text_lower.startswith("sentinel"):
        agent_name = "sentinel"
        task_content = user_input[len("sentinel"):].lstrip(" :,")
    elif text_lower.startswith("planner"):
        agent_name = "planner"
        task_content = user_input[len("planner"):].lstrip(" :,")
    else:
        # Fallback keyword-based inference (if no explicit agent prefix)
        agent_name = None
        
        # Research-specific keywords (should check these before analyst keywords)
        if any(kw in text_lower for kw in ["research on", "build a report", "compile a report", "investigate", "research project"]):
            agent_name = "researcher"
        # Writing-specific keywords
        elif any(kw in text_lower for kw in ["write", "draft", "chapter", "book", "poem", "article"]):
            agent_name = "scribe"
        # Analysis-specific keywords
        elif any(kw in text_lower for kw in ["analyze", "summary", "research", "report", "find", "what is", "explain"]):
            agent_name = "analyst"
        # Engineering-specific keywords
        elif any(kw in text_lower for kw in ["code", "fix", "script", "generate code", "develop", "debug", "function"]):
            agent_name = "engineer"
        # Planning-specific keywords
        elif any(kw in text_lower for kw in ["plan", "strategy", "break down", "organize", "schedule"]):
            agent_name = "planner"
        # File and goal management keywords
        elif any(kw in text_lower for kw in ["goal", "file", "inventory", "save", "list", "read", "delete", "add goal", "complete goal"]):
            agent_name = "quartermaster"
        # Ethics and guideline keywords
        elif any(kw in text_lower for kw in ["ethic", "rule", "audit", "violation", "guideline", "policy"]):
            agent_name = "sentinel"

    if agent_name:
        # If task_content wasn't set by a prefix, and we inferred an agent,
        # the task_content remains the full user_input for that agent.
        return agent_name, task_content
    else:
        # If no agent identified by prefix or keywords, use the fallback.
        # The task for the fallback agent is the original user input.
        return fallback_agent, user_input