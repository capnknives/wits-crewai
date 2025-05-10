# ethics/ethics_rules.py
# Sentinel's ethical filter engine

import os
import re

class EthicsViolation(Exception):
    pass

class EthicsFilter:
    def __init__(self, overlay_path="ethics/ethics_overlay.md", config=None):
        self.config = config or {}
        self.overlay_path = overlay_path
        self.parsed_overlay_rules = [] # Store parsed rules from the overlay
        self.static_blacklist = [ # Renamed for clarity
            # Keywords indicative of severe harm or illegal activities
            "kill", "assassinate", "murder", "bomb", "explosive",
            "terrorist", "terrorism",
            "child porn", "child sexual abuse", "non-consensual sexual", "rape",
            "incite violence", "promote hate",
            "self-harm", "suicide", # Added based on common safety guidelines
            # Keywords for potentially illegal activities (use with context)
            "hack", "phish", "malware", "ddos",
            "illicit drugs", "meth", "fentanyl", "cocaine", # Be specific if needed
            # Keywords for hate speech (often need context, but can be indicative)
            "racist", "sexist", "homophobic", "bigoted"
        ]
        self._load_and_parse_rules()

    def _load_and_parse_rules(self):
        """
        Loads rules from the overlay_path and parses them.
        Currently, it focuses on extracting disallowed phrases from bullet points
        under a 'Disallowed Content' heading.
        """
        self.parsed_overlay_rules = []
        if os.path.exists(self.overlay_path):
            try:
                with open(self.overlay_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Simple parsing: find "## Disallowed Content" and extract bullet points
                disallowed_section_match = re.search(r"##\s*Disallowed Content\s*([\s\S]*?)(?=\n##|\Z)", content, re.IGNORECASE)
                if disallowed_section_match:
                    section_text = disallowed_section_match.group(1)
                    # Extract bullet points (lines starting with '-' or '*')
                    bullet_points = re.findall(r"^\s*[-*]\s*(.+)", section_text, re.MULTILINE)
                    for point in bullet_points:
                        # Clean up the extracted point - remove examples in parentheses, convert to lower
                        # e.g., "Instructions or encouragement of violent or criminal activities (e.g., bomb making, assassination)."
                        # becomes "instructions or encouragement of violent or criminal activities"
                        cleaned_point = re.sub(r'\(e\.g\..,.*?\)', '', point, flags=re.IGNORECASE).strip().lower()
                        if cleaned_point:
                            self.parsed_overlay_rules.append(cleaned_point)
                if self.parsed_overlay_rules:
                    print(f"[ETHICS] Loaded {len(self.parsed_overlay_rules)} rules from overlay: {self.overlay_path}")
                else:
                    print(f"[ETHICS] Overlay found, but no 'Disallowed Content' rules parsed or section not found: {self.overlay_path}")

            except Exception as e:
                print(f"[ETHICS] Error loading or parsing overlay file {self.overlay_path}: {e}")
        else:
            print(f"[ETHICS] Overlay file not found: {self.overlay_path}. Running with static blacklist only.")

    def check_text(self, text_to_check):
        if not self.config.get("ethics_enabled", True):
            return True  # Bypass ethics filter when disabled

        if not text_to_check or not isinstance(text_to_check, str): # Handle None or non-string input
            return True

        text_lower = text_to_check.lower()

        # Check against static blacklist (more severe/direct terms)
        for phrase in self.static_blacklist:
            # Use word boundaries for more precise matching of some terms
            if re.search(r'\b' + re.escape(phrase) + r'\b', text_lower):
                raise EthicsViolation(f"Blocked: Potentially harmful content related to '{phrase}' found in output.")

        # Check against rules parsed from ethics_overlay.md
        for rule_phrase in self.parsed_overlay_rules:
            if rule_phrase in text_lower:
                # Making this check less strict to avoid too many false positives from general phrases
                # For example, if a rule is "personal data", we don't want to block "analyze my personal data usage".
                # This needs more sophisticated NLP to be truly effective.
                # For now, if a direct phrase from the overlay is found, we raise.
                # More advanced: check if the LLM is *promoting* or *instructing* the disallowed content.
                raise EthicsViolation(f"Blocked: Content matches disallowed phrase from ethics guidelines: '{rule_phrase}'.")
        return True

    def approve_action(self, agent_name, action_type, detail=None):
        """
        Validates whether an agent is allowed to perform a specific action.
        This method is largely superseded by SentinelAgent.approve_action for system actions.
        It can be kept for specific content-related pre-checks if needed, or simplified/removed.
        For now, keeping it simple and focused on config flags.
        """
        if not self.config.get("ethics_enabled", True):
            return True

        if action_type == "execute_code" and not self.config.get("allow_code_execution", False):
            raise EthicsViolation(f"Code execution by {agent_name} is disabled by configuration.")

        if action_type == "internet" and not self.config.get("internet_access", False):
            raise EthicsViolation(f"Internet access by {agent_name} is disabled by configuration.")

        # More detailed file operation checks (like path traversal, specific file blocks)
        # are better handled in SentinelAgent.approve_action which has more context.
        # This approve_action might be called by agents *before* Sentinel if there's a two-step approval.
        # If Sentinel is the sole gatekeeper for actions, this method here might become redundant
        # or only check very generic things.

        # Example for delete_file, though Sentinel's is more comprehensive:
        # if action_type == "delete_file":
        #     target = detail or ""
        #     if "config.yaml" in target.lower() or "memory.json" in target.lower(): # Very basic
        #         raise EthicsViolation("Attempt to delete critical system file blocked by EthicsFilter.")
        return True