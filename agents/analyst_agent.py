# agents/analyst_agent.py
import ollama
import re
import json

from .tools.base_tool import Tool, ToolException
from .sentinel_agent import SentinelException # Assuming SentinelException is defined here

class AnalystAgent:
    def __init__(self, config, memory, quartermaster, sentinel, engineer=None, tools=None):
        self.config = config
        self.memory = memory
        self.qm = quartermaster
        self.sentinel = sentinel
        self.engineer = engineer
        self.tools = tools if tools else []
        model_conf = config.get("models", {})
        self.model_name = model_conf.get("analyst", model_conf.get("default", "llama2"))
        self.delegation_keywords = {
            "engineer": "EngineerAgent", # Key used in code: Class name as string
            "scribe": "ScribeAgent"
        }

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
                "'DELEGATE_TO: EngineerAgent, TASK: [Provide a very clear and concise description of the "
                "Python code or function to be created. Include necessary details like function purpose, "
                "expected inputs, outputs, and any specific libraries if known or relevant from your analysis or tool output. "
                "Provide any data summaries or parameters the Engineer will need directly in this task description.].'\n"
            )
        # Add Scribe delegation if needed, similar to Engineer
        # if self.scribe: # Assuming self.scribe would be an instance of ScribeAgent
        #     delegation_instructions += (
        #         "\nDELEGATION TO SCRIBE:\n"
        #         "If the task requires drafting extensive text, a report, or creative writing, "
        #         "delegate to the ScribeAgent: 'DELEGATE_TO: ScribeAgent, TASK: [Detailed writing instructions].'\n"
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
            "5. After the tool executes, I will provide you with its output (as an 'Observation'). You MUST then use that observation to continue your analysis or formulate the final answer.\n\n"
            f"{delegation_instructions}"
            "FINAL RESPONSE OR NEXT ACTION:\n"
            "If no tool is needed at the current step, and no delegation is required, perform your analysis and provide your direct, comprehensive answer to the user's request. "
            "If you have used a tool and received its output, use that observation to formulate your final comprehensive answer or decide on the next step (which could be another tool use if essential, or delegation, or a final answer).\n\n"
            f"User's current request/task to address: \"{task_description}\""
        )
        return prompt

    def _call_llm(self, prompt: str):
        """
        Sends a prompt to the LLM and returns the stripped response.
        """
        print(f"\n[AnalystAgent] Sending prompt to LLM (model: {self.model_name}):\n{prompt[:1000]}...\n-----------------------------")
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
        delegation_match = re.search(
            r"DELEGATE_TO:\s*(" + "|".join(re.escape(cn) for cn in self.delegation_keywords.values()) + r"),\s*TASK:\s*([\s\S]+)",
            llm_output_with_delegation_signal,
            re.IGNORECASE # Make agent name matching case-insensitive
        )

        if not delegation_match:
            # No valid delegation signal found, return the original LLM output
            if self.config.get("ethics_enabled", True) and self.sentinel and hasattr(self.sentinel, 'ethics_filter') and self.sentinel.ethics_filter:
                self.sentinel.ethics_filter.check_text(llm_output_with_delegation_signal)
            return llm_output_with_delegation_signal

        target_agent_class_name = delegation_match.group(1).strip() # e.g., "EngineerAgent"
        delegated_task_detail = delegation_match.group(2).strip()

        # Extract any text from the LLM that came *before* the delegation signal
        analyst_lead_in = llm_output_with_delegation_signal.split("DELEGATE_TO:")[0].strip()
        if analyst_lead_in:
            analyst_lead_in += "\n\n" # Add spacing if there was introductory text

        print(f"[AnalystAgent] Delegating task to {target_agent_class_name}. Task details: {delegated_task_detail[:150]}...")
        delegated_result_text = ""

        # Find the actual agent instance to delegate to
        target_agent_instance = None
        if target_agent_class_name == self.delegation_keywords.get("engineer") and self.engineer:
            target_agent_instance = self.engineer
        # Add similar elif for Scribe if Scribe delegation is implemented
        # elif target_agent_class_name == self.delegation_keywords.get("scribe") and self.scribe_instance: # Assuming self.scribe_instance
        #     target_agent_instance = self.scribe_instance

        if target_agent_instance:
            try:
                # Sentinel approval for delegation action
                if self.sentinel:
                    self.sentinel.approve_action(
                        agent_name="AnalystAgent",
                        action_type=f"delegate_to:{target_agent_class_name}",
                        detail=f"Task: {delegated_task_detail[:100]}..."
                    )

                # Call the target agent's appropriate method.
                # For Engineer, it might be create_python_function or a general run.
                # For Scribe, it would likely be its run method.
                if target_agent_class_name == self.delegation_keywords.get("engineer"):
                    # Assuming EngineerAgent has a method like `create_python_function` or a general `run`
                    # If it's a specific method:
                    # code_script = target_agent_instance.create_python_function(delegated_task_detail)
                    # If it's a general run method:
                    code_script = target_agent_instance.run(f"Create Python code for: {delegated_task_detail}")

                    delegated_result_text = f"The {target_agent_class_name} has provided the following based on the analysis:\n```python\n{code_script}\n```"
                # Add Scribe logic here if Scribe delegation is enabled
                # elif target_agent_class_name == self.delegation_keywords.get("scribe"):
                #     written_text = target_agent_instance.run(delegated_task_detail)
                #     delegated_result_text = f"The {target_agent_class_name} has drafted:\n{written_text}"

            except SentinelException as se:
                print(f"[AnalystAgent] Delegation to {target_agent_class_name} blocked by Sentinel: {se}")
                delegated_result_text = f"[AnalystAgent] Delegation to {target_agent_class_name} was blocked: {str(se)}"
            except Exception as e_delegate:
                print(f"[AnalystAgent] Error during delegation call to {target_agent_class_name}: {e_delegate}")
                delegated_result_text = f"[AnalystAgent] Error encountered while delegating to {target_agent_class_name}: {str(e_delegate)}"
        else:
            final_response_on_error = f"{analyst_lead_in}[AnalystAgent] Error: Could not delegate. Target agent '{target_agent_class_name}' unknown or not available."
            if self.config.get("ethics_enabled", True) and self.sentinel and hasattr(self.sentinel, 'ethics_filter') and self.sentinel.ethics_filter:
                self.sentinel.ethics_filter.check_text(final_response_on_error)
            return final_response_on_error

        final_response = f"{analyst_lead_in}{delegated_result_text}"
        if self.config.get("ethics_enabled", True) and self.sentinel and hasattr(self.sentinel, 'ethics_filter') and self.sentinel.ethics_filter:
            self.sentinel.ethics_filter.check_text(final_response)
        return final_response

    def run(self, command: str, max_iterations=5): # Max iterations for ReAct loop
        """
        Main execution loop for the AnalystAgent.
        Uses a ReAct-style loop: Reason, Act (tool/delegate), Observe.
        """
        original_user_command = command.strip()
        accumulated_context = "" # Stores observations from file reads, web searches, tool outputs

        # --- Optional Pre-computation/Pre-analysis based on command ---
        # Example: If command mentions a file, pre-read it.
        file_match = re.search(r'\b([\w\-]+\.(?:txt|md|csv|json|py))\b', original_user_command, flags=re.IGNORECASE)
        if file_match and self.qm:
            file_name = file_match.group(1)
            try:
                print(f"[AnalystAgent] Pre-reading mentioned file: {file_name}")
                # Using Quartermaster directly for pre-fetch. LLM will use ReadFileTool for explicit requests.
                content = self.qm.read_file(file_name)
                accumulated_context += f"Context from pre-read file '{file_name}':\n{content}\n\n"
            except Exception as e:
                accumulated_context += f"[Warning] Could not pre-read file '{file_name}': {e}\n\n"

        # Example: Basic pre-search if command implies research (more robustly handled by LLM choosing WebSearchTool)
        search_keywords = ["search for", "research", "find information on", "what is", "latest news on"]
        # Make pre-searching more conservative or rely on LLM's decision to use a tool.
        # For "Research the latest ai news", this pre-search will trigger.
        if any(original_user_command.lower().startswith(kw) for kw in search_keywords) and self.qm:
            temp_query = original_user_command
            for kw in search_keywords:
                if temp_query.lower().startswith(kw):
                    temp_query = temp_query[len(kw):].strip()
                    break
            # Extract a reasonable query part, avoid overly long or complex pre-searches
            query_part = temp_query.split('.')[0].split(' and then')[0].split(' also ')[0].strip()
            if query_part and 3 < len(query_part) < 100: # Basic query sanity
                try:
                    print(f"[AnalystAgent] Pre-searching for: {query_part}")
                    search_results = self.qm.internet_search(query_part) # Uses Quartermaster
                    accumulated_context += f"Initial internet search results for '{query_part}':\n{search_results}\n\n"
                except Exception as e:
                    accumulated_context += f"[Initial Search Error for '{query_part}'] {e}\n\n"
        # --- End of Pre-computation ---

        for iteration in range(max_iterations):
            print(f"\n[AnalystAgent] --- Iteration {iteration + 1}/{max_iterations} ---")
            # The task for the LLM is always the original command, but context grows.
            current_task_for_llm = original_user_command
            print(f"[AnalystAgent] Task for LLM (original command): {current_task_for_llm[:150]}")
            if accumulated_context:
                # Show only the last part of the context if it's too long for logs
                context_preview = accumulated_context[-500:] if len(accumulated_context) > 500 else accumulated_context
                print(f"[AnalystAgent] Accumulated Context (preview):\n...{context_preview}")

            prompt_for_llm = self._get_prompt_for_analyst(current_task_for_llm, accumulated_context)
            llm_output = self._call_llm(prompt_for_llm)

            tool_call_data = None
            # Robust JSON extraction for tool calls
            # This regex looks for a structure like: {"tool_to_use": "name", "tool_arguments": {...}}
            # It allows for whitespace and newlines within the JSON structure.
            json_tool_match = re.search(r'{\s*"tool_to_use":\s*".*?",\s*"tool_arguments":\s*{.*?}\s*}', llm_output, re.DOTALL)

            if json_tool_match:
                json_str = json_tool_match.group(0)
                try:
                    parsed_json = json.loads(json_str)
                    if "tool_to_use" in parsed_json and "tool_arguments" in parsed_json:
                        tool_call_data = parsed_json
                        # Add any text surrounding the JSON to context if LLM didn't follow "ONLY JSON" rule
                        pre_json_text = llm_output[:json_tool_match.start()].strip()
                        post_json_text = llm_output[json_tool_match.end():].strip()
                        if pre_json_text or post_json_text:
                            print(f"[AnalystAgent] LLM included text around tool call: Pre='{pre_json_text}', Post='{post_json_text}'")
                            accumulated_context += f"\n# LLM commentary around tool call:\n{pre_json_text}\n{post_json_text}\n"
                    else:
                        print("[AnalystAgent] Extracted JSON, but it's missing required 'tool_to_use' or 'tool_arguments' keys.")
                except json.JSONDecodeError:
                    print(f"[AnalystAgent] Failed to parse extracted JSON string as tool call: {json_str}")
            # If no JSON tool call is found by regex, tool_call_data remains None.

            if tool_call_data:
                tool_name = tool_call_data.get("tool_to_use")
                tool_args = tool_call_data.get("tool_arguments", {})
                print(f"[AnalystAgent] LLM requested tool: '{tool_name}' with args: {tool_args}")

                chosen_tool_instance: Optional[Tool] = None # Type hint
                for t_instance in self.tools:
                    if t_instance.name == tool_name:
                        chosen_tool_instance = t_instance
                        break

                if chosen_tool_instance:
                    try:
                        if self.sentinel:
                            self.sentinel.approve_action(
                                agent_name="AnalystAgent",
                                action_type=f"tool_use:{chosen_tool_instance.name}",
                                detail=str(tool_args)
                            )
                        tool_output_str = chosen_tool_instance.execute(**tool_args)
                        print(f"[AnalystAgent] Output from tool '{tool_name}': {tool_output_str[:200]}...") # Log snippet
                        accumulated_context += (
                            f"\n# Observation from successfully using tool '{tool_name}' "
                            f"with arguments {json.dumps(tool_args)}:\n{tool_output_str}\n"
                        )
                        continue # Go to next iteration with updated context

                    except (ToolException, SentinelException) as e_tool_blocked:
                        print(f"[AnalystAgent] Error or block using tool '{tool_name}': {e_tool_blocked}")
                        accumulated_context += (
                            f"\n# Error/Block when trying to use tool '{tool_name}' "
                            f"with arguments {json.dumps(tool_args)}: {str(e_tool_blocked)}\n"
                            "You should analyze this error and proceed without the tool if possible, or try a different approach.\n"
                        )
                        continue
                    except Exception as e_tool_unexpected:
                        print(f"[AnalystAgent] Unexpected error with tool '{tool_name}': {e_tool_unexpected}")
                        import traceback; print(traceback.format_exc()) # Log full traceback for unexpected
                        accumulated_context += (
                            f"\n# Unexpected system error with tool '{tool_name}': {str(e_tool_unexpected)}\n"
                            "Please proceed based on other available information.\n"
                        )
                        continue
                else:
                    print(f"[AnalystAgent] LLM requested an unknown or unavailable tool: '{tool_name}'")
                    accumulated_context += (
                        f"\n# Attempted to use an unknown tool: '{tool_name}'. This tool is not available. "
                        "Please choose from the AVAILABLE TOOLS list or answer directly.\n"
                    )
                    continue # Go to next iteration for LLM to reconsider

            # If not a tool call, check for Delegation
            # The DELEGATE_TO signal should be distinct and not part of a normal sentence.
            if "DELEGATE_TO:" in llm_output.upper(): # Check case-insensitively for the signal
                return self._handle_delegation(llm_output, original_command_context=accumulated_context)

            # If not a tool call and not delegation, assume it's the Final Answer
            else:
                print("[AnalystAgent] LLM provided a direct final answer.")
                # The LLM's final answer should ideally incorporate the accumulated_context.
                # If it doesn't, we might prepend it, but it's better if LLM handles synthesis.
                final_answer_from_llm = llm_output
                if self.config.get("ethics_enabled", True) and self.sentinel and hasattr(self.sentinel, 'ethics_filter') and self.sentinel.ethics_filter:
                    self.sentinel.ethics_filter.check_text(final_answer_from_llm)

                # Optionally, prepend context if LLM seems to have ignored it (heuristic)
                # if not original_user_command.lower() in final_answer_from_llm.lower() and accumulated_context:
                #    final_answer_from_llm = accumulated_context + "\nAnalyst's Summary:\n" + final_answer_from_llm

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
