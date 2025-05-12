# agents/analyst_agent.py
import ollama
import re
import json
from typing import Optional # Added for type hinting in class

from .tools.base_tool import Tool, ToolException
from .sentinel_agent import SentinelException # Assuming SentinelException is defined here

class AnalystAgent:
    def __init__(self, config, memory, quartermaster, sentinel, engineer=None, tools=None):
        self.config = config
        self.memory = memory
        self.qm = quartermaster
        self.sentinel = sentinel
        self.engineer = engineer # Instance of EngineerAgent for delegation
        self.tools = tools if tools else []
        model_conf = config.get("models", {})
        self.model_name = model_conf.get("analyst", model_conf.get("default", "llama2"))
        self.delegation_keywords = {
            # Map internal keys to the Agent Class Name strings the LLM might use
            "engineer": "EngineerAgent", 
            "scribe": "ScribeAgent" 
        }
        # Add other agents if Analyst can delegate to them, e.g.
        # "researcher": "ResearcherAgent"

    def _get_tool_descriptions_for_llm(self):
        """
        Generates a string describing available tools for the LLM prompt.
        """
        if not self.tools:
            return "No tools are currently available to you for direct use."

        descriptions = ["\nAVAILABLE TOOLS (use only if necessary and appropriate for the task; prefer direct analysis if possible):"]
        for tool in self.tools:
            descriptions.append(f"- {tool.get_full_description_for_llm()}")
        return "\n".join(descriptions)

    def _get_prompt_for_analyst(self, task_description: str, previous_context: str = ""):
        """
        Generates the full prompt for the Analyst LLM, including context, tool descriptions,
        and instructions for tool use or delegation.
        """
        tool_info = self._get_tool_descriptions_for_llm()

        delegation_instructions = ""
        if self.engineer: # Check if an engineer instance is available
            delegation_instructions += (
                "\nDELEGATION TO ENGINEER:\n"
                "If the user's request, after your analysis or tool use, requires Python code generation "
                "(e.g., a script, a function to process data, or a visualization), "
                "you MUST delegate this coding task to the EngineerAgent. "
                "To do this, clearly state on a new line: "
                # Using the class name as the LLM might be trained on that.
                # Make sure self.delegation_keywords maps this back if needed.
                f"'DELEGATE_TO: {self.delegation_keywords.get('engineer', 'EngineerAgent')}, TASK: [Provide a very clear and concise description of the "
                "Python code or function to be created. Include necessary details like function purpose, "
                "expected inputs, outputs, and any specific libraries if known or relevant from your analysis or tool output. "
                "Provide any data summaries or parameters the Engineer will need directly in this task description.].'\n"
            )
        # Example for Scribe delegation (if Analyst can delegate to Scribe)
        # scribe_class_name = self.delegation_keywords.get('scribe', 'ScribeAgent')
        # if any(agent_key for agent_key, agent_class in self.delegation_keywords.items() if agent_class == scribe_class_name): # Check if Scribe is a delegation target
        #     delegation_instructions += (
        #         f"\nDELEGATION TO SCRIBE:\n"
        #         f"If the task requires drafting extensive text, a report, or creative writing, "
        #         f"delegate to the ScribeAgent: 'DELEGATE_TO: {scribe_class_name}, TASK: [Detailed writing instructions].'\n"
        #     )


        prompt = (
            f"{previous_context}" # This will contain observations from previous tool uses or pre-searches
            "You are an Analyst Agent. Your primary role is to research information, analyze data (text, summaries), "
            "and provide clear, concise insights. You can read files and search the internet (often by using a specific tool if available).\n"
            f"{tool_info}\n"
            "\nTOOL USAGE INSTRUCTIONS (Use Sparingly and Precisely):\n"
            "1. Analyze the current task and context. Determine if a specific piece of information is missing that a tool can provide, or if an action (like calculation or saving a file) is explicitly needed.\n"
            "2. If AND ONLY IF a tool is essential for the NEXT STEP of your analysis or to fulfill an explicit part of the request, "
            "respond *ONLY* with a single JSON object on a new line, formatted exactly as follows:\n" # Emphasize "ONLY"
            "   {\"tool_to_use\": \"exact_tool_name_from_list\", \"tool_arguments\": {\"arg_name1\": \"value1\", \"arg_name2\": \"value2\"}}\n"
            "3. Ensure 'tool_name' exactly matches one from the AVAILABLE TOOLS list. "
            "4. Provide all required arguments for the chosen tool as key-value pairs in 'tool_arguments'.\n"
            "5. After the tool executes, I will provide you with its output (as an 'Observation'). You MUST then use that observation to continue your analysis or formulate the final answer. If a tool fails or provides no useful information (e.g. 'File not found', 'No search results'), acknowledge this observation and try a different approach or conclude if necessary.\n\n" # Added instruction for tool failure
            f"{delegation_instructions}"
            "FINAL RESPONSE OR NEXT ACTION:\n"
            "If no tool is needed at the current step, and no delegation is required, perform your analysis and provide your direct, comprehensive answer to the user's request. "
            "If you have used a tool and received its output (even if it's an error or no data), use that observation to formulate your final comprehensive answer or decide on the next step (which could be another tool use if essential and different from the failed one, or delegation, or a final answer).\n\n"
            f"User's current request/task to address: \"{task_description}\""
        )
        return prompt

    def _call_llm(self, prompt: str):
        """
        Sends a prompt to the LLM and returns the stripped response.
        """
        print(f"\n[AnalystAgent] Sending prompt to LLM (model: {self.model_name}):\n{prompt[:1000]}...\n-----------------------------") # Log more of the prompt
        ollama_response = ollama.generate(model=self.model_name, prompt=prompt)
        llm_output = ollama_response['response'].strip()
        print(f"[AnalystAgent] LLM Raw Output:\n{llm_output}\n-----------------------------")
        return llm_output

    def _handle_delegation(self, llm_output_with_delegation_signal: str, original_command_context: str=""):
        """
        Handles delegation to another agent if the DELEGATE_TO signal is found.
        """
        # Regex to find "DELEGATE_TO: AgentName, TASK: details"
        # It captures the agent name (e.g., EngineerAgent) and the task details.
        # Making agent name matching more flexible for class names or keys.
        delegation_match = re.search(
            r"DELEGATE_TO:\s*([\w_]+(?:Agent)?),\s*TASK:\s*([\s\S]+)", # Allows "EngineerAgent" or "engineer"
            llm_output_with_delegation_signal,
            re.IGNORECASE 
        )

        if not delegation_match:
            # No valid delegation signal found, return the original LLM output
            if self.config.get("ethics_enabled", True) and self.sentinel and hasattr(self.sentinel, 'ethics_filter') and self.sentinel.ethics_filter:
                self.sentinel.ethics_filter.check_text(llm_output_with_delegation_signal) # Check before returning
            return llm_output_with_delegation_signal

        target_agent_name_from_llm = delegation_match.group(1).strip() # e.g., "EngineerAgent" or "engineer"
        delegated_task_detail = delegation_match.group(2).strip()

        # Extract any text from the LLM that came *before* the delegation signal
        analyst_lead_in = llm_output_with_delegation_signal.split("DELEGATE_TO:")[0].strip()
        if analyst_lead_in:
            analyst_lead_in += "\n\n" # Add spacing if there was introductory text

        print(f"[AnalystAgent] Attempting to delegate task to '{target_agent_name_from_llm}'. Task details: {delegated_task_detail[:150]}...")
        delegated_result_text = ""

        # Find the actual agent instance to delegate to
        target_agent_instance = None
        resolved_agent_key = None

        # Try matching target_agent_name_from_llm (case-insensitive) with keys in self.delegation_keywords
        for key, class_name_str in self.delegation_keywords.items():
            if target_agent_name_from_llm.lower() == class_name_str.lower() or target_agent_name_from_llm.lower() == key.lower():
                if key == "engineer" and self.engineer:
                    target_agent_instance = self.engineer
                    resolved_agent_key = "engineer" # Use the internal key
                    break
                # Add elif for "scribe" if Analyst can delegate to Scribe and self.scribe (Scribe instance) exists
                # elif key == "scribe" and self.scribe: # Assuming self.scribe = ScribeAgent()
                #     target_agent_instance = self.scribe
                #     resolved_agent_key = "scribe"
                #     break
        
        if not target_agent_instance:
             print(f"[AnalystAgent] Could not find a configured agent instance for '{target_agent_name_from_llm}'. Check delegation_keywords and __init__.")
             delegated_result_text = f"[AnalystAgent] Error: Could not delegate. Target agent '{target_agent_name_from_llm}' is not configured for delegation or not available."


        if target_agent_instance and resolved_agent_key:
            try:
                # Sentinel approval for delegation action
                if self.sentinel:
                    self.sentinel.approve_action(
                        agent_name="AnalystAgent",
                        action_type=f"delegate_to:{resolved_agent_key}", # Use the resolved key
                        detail=f"Task: {delegated_task_detail[:100]}..."
                    )

                # Call the target agent's run method (assuming all agents have a .run() method)
                # The task for the delegated agent is the `delegated_task_detail`
                agent_response = target_agent_instance.run(delegated_task_detail)

                delegated_result_text = f"The {self.delegation_keywords.get(resolved_agent_key, resolved_agent_key.capitalize()+'Agent')} responded:\n{agent_response}"
                
            except SentinelException as se:
                print(f"[AnalystAgent] Delegation to {resolved_agent_key} blocked by Sentinel: {se}")
                delegated_result_text = f"[AnalystAgent] Delegation to {resolved_agent_key} was blocked: {str(se)}"
            except Exception as e_delegate:
                print(f"[AnalystAgent] Error during delegation call to {resolved_agent_key}: {e_delegate}")
                import traceback; print(traceback.format_exc())
                delegated_result_text = f"[AnalystAgent] Error encountered while delegating to {resolved_agent_key}: {str(e_delegate)}"
        
        # Combine analyst's lead-in (if any) with the delegation result
        final_response = f"{analyst_lead_in}{delegated_result_text}"
        if self.config.get("ethics_enabled", True) and self.sentinel and hasattr(self.sentinel, 'ethics_filter') and self.sentinel.ethics_filter:
            self.sentinel.ethics_filter.check_text(final_response)
        return final_response

    def run(self, command: str, max_iterations=3): # Reduced max_iterations to break loops faster
        """
        Main execution loop for the AnalystAgent.
        Uses a ReAct-style loop: Reason, Act (tool/delegate), Observe.
        Includes logic to prevent getting stuck on persistently failing tool calls.
        """
        original_user_command = command.strip()
        accumulated_context = "" 
        failed_tool_attempts = {} # Tracks: {(tool_name, frozenset(tool_args.items())): count}
        MAX_TOOL_RETRIES_SAME_ARGS = 2 # Max times to retry the *exact same* tool call

        # --- Optional Pre-computation/Pre-analysis based on command ---
        # (Your existing pre-computation logic can remain here)
        file_match = re.search(r'\b([\w\-]+\.(?:txt|md|csv|json|py))\b', original_user_command, flags=re.IGNORECASE)
        if file_match and self.qm:
            file_name = file_match.group(1)
            try:
                print(f"[AnalystAgent] Pre-reading mentioned file: {file_name}")
                content = self.qm.read_file(file_name)
                accumulated_context += f"Context from pre-read file '{file_name}':\n{content}\n\n"
            except Exception as e:
                accumulated_context += f"[Warning] Could not pre-read file '{file_name}': {e}\n\n"

        search_keywords = ["search for", "research", "find information on", "what is", "latest news on"]
        if any(original_user_command.lower().startswith(kw) for kw in search_keywords) and self.qm:
            temp_query = original_user_command
            for kw in search_keywords:
                if temp_query.lower().startswith(kw):
                    temp_query = temp_query[len(kw):].strip(); break
            query_part = temp_query.split('.')[0].split(' and then')[0].split(' also ')[0].strip()
            if query_part and 3 < len(query_part) < 100:
                try:
                    print(f"[AnalystAgent] Pre-searching for: {query_part}")
                    search_results = self.qm.internet_search(query_part)
                    accumulated_context += f"Initial internet search results for '{query_part}':\n{search_results}\n\n"
                except Exception as e:
                    accumulated_context += f"[Initial Search Error for '{query_part}'] {e}\n\n"
        # --- End of Pre-computation ---

        for iteration in range(max_iterations):
            print(f"\n[AnalystAgent] --- Iteration {iteration + 1}/{max_iterations} ---")
            current_task_for_llm = original_user_command
            print(f"[AnalystAgent] Task for LLM (original command): {current_task_for_llm[:150]}")
            if accumulated_context:
                context_preview = accumulated_context[-600:] if len(accumulated_context) > 600 else accumulated_context # Show more context
                print(f"[AnalystAgent] Accumulated Context (preview):\n...{context_preview}")

            prompt_for_llm = self._get_prompt_for_analyst(current_task_for_llm, accumulated_context)
            llm_output = self._call_llm(prompt_for_llm)

            tool_call_data = None
            json_tool_match = re.search(r'{\s*"tool_to_use":\s*".*?",\s*"tool_arguments":\s*{.*?}\s*}', llm_output, re.DOTALL)

            if json_tool_match:
                json_str = json_tool_match.group(0)
                try:
                    parsed_json = json.loads(json_str)
                    if "tool_to_use" in parsed_json and "tool_arguments" in parsed_json:
                        tool_call_data = parsed_json
                        pre_json_text = llm_output[:json_tool_match.start()].strip()
                        post_json_text = llm_output[json_tool_match.end():].strip()
                        if pre_json_text or post_json_text:
                            print(f"[AnalystAgent] LLM included text around tool call: Pre='{pre_json_text}', Post='{post_json_text}'")
                            accumulated_context += f"\n# LLM commentary around tool call:\n{pre_json_text}\n{post_json_text}\n"
                except json.JSONDecodeError:
                    print(f"[AnalystAgent] Failed to parse extracted JSON string as tool call: {json_str}")
            
            if tool_call_data:
                tool_name = tool_call_data.get("tool_to_use")
                tool_args = tool_call_data.get("tool_arguments", {})
                tool_signature = (tool_name, frozenset(tool_args.items())) # Use frozenset for dict args

                # Check if this exact tool call has failed too many times
                if failed_tool_attempts.get(tool_signature, 0) >= MAX_TOOL_RETRIES_SAME_ARGS:
                    print(f"[AnalystAgent] Tool call {tool_name} with args {tool_args} has failed {MAX_TOOL_RETRIES_SAME_ARGS} times. Aborting this tool attempt.")
                    accumulated_context += (
                        f"\n# Observation: The tool call '{tool_name}' with arguments {json.dumps(tool_args)} "
                        f"has failed repeatedly. I will not attempt it again. I should try a different approach or conclude.\n"
                    )
                    continue # Go to next LLM iteration for it to reconsider

                print(f"[AnalystAgent] LLM requested tool: '{tool_name}' with args: {tool_args}")
                chosen_tool_instance: Optional[Tool] = next((t for t in self.tools if t.name == tool_name), None)

                if chosen_tool_instance:
                    try:
                        if self.sentinel:
                            self.sentinel.approve_action("AnalystAgent", f"tool_use:{chosen_tool_instance.name}", str(tool_args))
                        
                        tool_output_str = chosen_tool_instance.execute(**tool_args)
                        print(f"[AnalystAgent] Output from tool '{tool_name}': {tool_output_str[:200]}...")
                        
                        # Check for known "no useful data" responses to increment failed_tool_attempts
                        no_data_indicators = [
                            "file not found", 
                            "yielded no specific results", 
                            "could not be found",
                            "no relevant information found" # From Quartermaster internet search
                        ]
                        tool_output_lower = tool_output_str.lower()
                        if any(indicator in tool_output_lower for indicator in no_data_indicators):
                            failed_tool_attempts[tool_signature] = failed_tool_attempts.get(tool_signature, 0) + 1
                            print(f"[AnalystAgent] Tool '{tool_name}' call for {tool_args} did not yield useful data. Attempt {failed_tool_attempts[tool_signature]}/{MAX_TOOL_RETRIES_SAME_ARGS}.")
                            accumulated_context += (
                                f"\n# Observation: Tool '{tool_name}' with arguments {json.dumps(tool_args)} "
                                f"returned: '{tool_output_str}'. This seems like no specific data was found. "
                                f"Attempt {failed_tool_attempts[tool_signature]}.\n"
                            )
                        else: # Successful tool use with data
                             failed_tool_attempts.pop(tool_signature, None) # Reset counter on success
                             accumulated_context += (
                                f"\n# Observation from successfully using tool '{tool_name}' "
                                f"with arguments {json.dumps(tool_args)}:\n{tool_output_str}\n"
                            )
                        continue 
                    except (ToolException, SentinelException) as e_tool_blocked:
                        failed_tool_attempts[tool_signature] = failed_tool_attempts.get(tool_signature, 0) + 1
                        print(f"[AnalystAgent] Error/Block using tool '{tool_name}': {e_tool_blocked}. Attempt {failed_tool_attempts[tool_signature]}/{MAX_TOOL_RETRIES_SAME_ARGS}.")
                        accumulated_context += (
                            f"\n# Error/Block when trying to use tool '{tool_name}' "
                            f"with arguments {json.dumps(tool_args)}: {str(e_tool_blocked)}. Attempt {failed_tool_attempts[tool_signature]}. "
                            "You should analyze this error and proceed without the tool if possible, or try a different approach.\n"
                        )
                        continue
                    except Exception as e_tool_unexpected:
                        failed_tool_attempts[tool_signature] = failed_tool_attempts.get(tool_signature, 0) + 1
                        print(f"[AnalystAgent] Unexpected error with tool '{tool_name}': {e_tool_unexpected}. Attempt {failed_tool_attempts[tool_signature]}/{MAX_TOOL_RETRIES_SAME_ARGS}.")
                        import traceback; print(traceback.format_exc())
                        accumulated_context += (
                            f"\n# Unexpected system error with tool '{tool_name}': {str(e_tool_unexpected)}. Attempt {failed_tool_attempts[tool_signature]}. "
                            "Please proceed based on other available information.\n"
                        )
                        continue
                else: # Tool not found
                    print(f"[AnalystAgent] LLM requested an unknown or unavailable tool: '{tool_name}'")
                    accumulated_context += (
                        f"\n# Attempted to use an unknown tool: '{tool_name}'. This tool is not available. "
                        "Please choose from the AVAILABLE TOOLS list or answer directly.\n"
                    )
                    continue 
            
            elif "DELEGATE_TO:" in llm_output.upper():
                return self._handle_delegation(llm_output, original_command_context=accumulated_context)
            
            else: # Final Answer
                print("[AnalystAgent] LLM provided a direct final answer.")
                final_answer_from_llm = llm_output
                if self.config.get("ethics_enabled", True) and self.sentinel and hasattr(self.sentinel, 'ethics_filter') and self.sentinel.ethics_filter:
                    self.sentinel.ethics_filter.check_text(final_answer_from_llm)
                return final_answer_from_llm

        # Fallback if max_iterations is reached
        print(f"[AnalystAgent] Reached maximum iterations ({max_iterations}). Returning last accumulated context or a message.")
        final_fallback_response = (
            f"[AnalystAgent] Reached maximum processing iterations for the command: '{original_user_command}'. "
            "Unable to provide a conclusive answer with the current approach.\n"
            "Current accumulated information (if any):\n" + accumulated_context
        )
        if self.config.get("ethics_enabled", True) and self.sentinel and hasattr(self.sentinel, 'ethics_filter') and self.sentinel.ethics_filter:
            self.sentinel.ethics_filter.check_text(final_fallback_response)
        return final_fallback_response
