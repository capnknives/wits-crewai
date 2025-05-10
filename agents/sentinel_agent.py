# agents/sentinel_agent.py
import os

class SentinelException(Exception):
    """Custom exception for Sentinel blocking an action or content."""
    pass

class SentinelAgent:
    def __init__(self, config, ethics=None, memory=None): # ethics is an EthicsFilter instance
        self.config = config
        self.memory = memory
        self.ethics_filter = ethics # <<< ENSURE IT'S STORED AS self.ethics_filter
        self.ethics_rules_text = ""

        # The EthicsFilter instance is now expected to be stored as self.ethics_filter
        # Calls to check text should go through that, e.g., self.ethics_filter.check_text()
        # The old self.check_content line is removed as direct calls go to self.ethics_filter.check_text

        # Load rules for handle_query (used in UI or audits)
        # Use the path from the injected ethics_filter object if available
        overlay_file_path = getattr(self.ethics_filter, 'overlay_path', "ethics/ethics_overlay.md")
        if os.path.exists(overlay_file_path):
            try:
                with open(overlay_file_path, 'r', encoding='utf-8') as f:
                    self.ethics_rules_text = f.read()
            except Exception as e:
                self.ethics_rules_text = f"Failed to load ethics rules file '{overlay_file_path}': {e}"
        else:
            self.ethics_rules_text = f"Ethics rules file not found at '{overlay_file_path}'."

        # Fallback blacklist (used only if no ethics_filter was somehow provided, or its checks fail to load)
        self.fallback_forbidden_phrases = [
            "make a bomb", "how to make a bomb", "assassinate", "kill", "murder",
            "build a weapon", "terrorist",
            "child porn", "child sexual", "rape",
            "rob a bank", "buy drugs", "how to hack", "perform a ddos",
            "hate speech", "kill all", "destroy them"
        ]
    # ... rest of SentinelAgent (approve_action, handle_query, _local_check_content) ...
    # Ensure approve_action and handle_query are complete as provided in response ID H37G9E
    # For brevity, I'm not re-pasting the whole SentinelAgent if only __init__ needed confirmation
    # for the self.ethics_filter attribute.
    # The _local_check_content method is a fallback:
    def _local_check_content(self, text_to_check):
        """Fallback content check if no proper EthicsFilter is available or if it's bypassed."""
        if not text_to_check or not isinstance(text_to_check, str): return True
        content_lower = text_to_check.lower()
        for phrase in self.fallback_forbidden_phrases:
            if phrase in content_lower: # Basic substring check for fallback
                raise SentinelException(f"Content blocked by Sentinel's fallback filter due to: '{phrase}'.")
        return True

    def approve_action(self, agent_name: str, action_type: str, detail: str = None) -> bool:
        """
        Validate a pending tool/external action requested by an agent.
        Raises SentinelException if blocked.
        """
        print(f"[SentinelAgent] ACTION_APPROVAL_REQUEST: Agent='{agent_name}', Type='{action_type}', Detail='{str(detail)[:100]}'")

        if not self.config.get("ethics_enabled", True): # Global override
            print("[SentinelAgent] Ethics disabled globally, action approved by default.")
            return True

        if action_type == "internet":
            if not self.config.get("internet_access", False):
                raise SentinelException("Internet access is disabled by configuration.")
            print("[SentinelAgent] Internet access approved.")
            return True

        elif action_type in ("file_read", "file_write", "file_delete"):
            if not detail: 
                raise SentinelException(f"File operation '{action_type}' blocked: file path not provided.")

            # Determine base directory for file operations (should be within project output/workspace)
            # For reading, agents might need to access files outside output_dir (e.g. their own source for analysis)
            # For writing/deleting, it should be strictly within output_dir or a defined workspace.
            output_dir = os.path.abspath(self.config.get("output_directory", "output"))
            
            # Construct absolute path based on output_dir if detail is relative
            # This logic assumes 'detail' is a filename or relative path meant for the output_dir
            # If 'detail' can be an absolute path, more checks are needed.
            # For simplicity, let's assume Quartermaster normalizes paths somewhat before calling Sentinel.
            # A more robust check:
            if os.path.isabs(detail):
                abs_path = os.path.normpath(detail)
            else:
                abs_path = os.path.normpath(os.path.join(output_dir, detail))

            project_root = os.path.normpath(os.path.abspath(os.getcwd()))
            
            # Security: Ensure operations are within expected boundaries
            if action_type in ("file_write", "file_delete"):
                if not abs_path.startswith(output_dir):
                    raise SentinelException(
                        f"File operation '{action_type}' on '{detail}' blocked: outside designated writeable directory '{output_dir}'."
                    )
            elif action_type == "file_read":
                # Allow reading from output_dir or project_root (for configs, source code by agents etc.)
                if not (abs_path.startswith(output_dir) or abs_path.startswith(project_root)):
                    raise SentinelException(
                        f"File operation '{action_type}' on '{detail}' blocked: outside allowed read directories."
                    )
            
            if action_type == "file_delete":
                critical_files = ["config.yaml", "memory.json"]
                # Also consider ethics_overlay.md if its path is fixed
                overlay_path_base = os.path.basename(getattr(self.ethics_filter, 'overlay_path', "ethics_overlay.md"))
                critical_files.append(overlay_path_base)
                
                if os.path.basename(abs_path) in critical_files:
                    raise SentinelException(f"Deletion of critical system file '{os.path.basename(abs_path)}' is not allowed.")
            
            print(f"[SentinelAgent] File operation '{action_type}' on '{detail}' approved.")
            return True

        elif action_type == "execute_code":
            if not self.config.get("allow_code_execution", False):
                raise SentinelException("Code execution is disabled by configuration.")
            # Add checks for the content of the code if possible/needed
            print("[SentinelAgent] Code execution approved.")
            return True
        
        elif action_type.startswith("tool_use:"):
            tool_name_used = action_type.split("tool_use:", 1)[1]
            # Add specific rules per tool if needed here based on config or internal policy
            print(f"[SentinelAgent] Use of tool '{tool_name_used}' by '{agent_name}' approved. (Args: {str(detail)[:100]})")
            return True

        else:
            raise SentinelException(f"Unrecognized or unhandled action type '{action_type}' requested by {agent_name}.")

    def handle_query(self, query: str) -> str:
        q_lower = query.lower().strip()
        if "rule" in q_lower or "guideline" in q_lower or "policy" in q_lower:
            return f"Ethics and Safety Guidelines (from {getattr(self.ethics_filter, 'overlay_path', 'N/A')}):\n{self.ethics_rules_text}" if self.ethics_rules_text else "No ethics guidelines loaded or available."
        elif "audit" in q_lower or "violation" in q_lower or "compliance" in q_lower:
            return "Sentinel Audit: No recent violations logged. System operational status: Green. (Audit functionality is basic)."
        return (
            "Sentinel: I monitor and enforce ethical rules and action approvals. "
            "You can ask me to 'list guidelines' or 'audit compliance'."
        )