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
            "engineer": "EngineerAgent",
            "scribe": "ScribeAgent" 
        }

    def _get_tool_descriptions_for_llm(self):
        if not self.tools:
            return "No tools are currently available to you for direct use."
        
        descriptions = ["\nAVAILABLE TOOLS (use only if necessary and appropriate for the task; prefer direct analysis if possible):"]
        for tool in self.tools:
            descriptions.append(f"- {tool.get_full_description_for_llm()}") 
        return "\n".join(descriptions)

    def _get_prompt_for_analyst(self, task_description: str, previous_context: str = ""):
        tool_info = self._get_tool_descriptions_for_llm()
        
        delegation_instructions = ""
        if self.engineer:
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
        
        prompt = (
            f"{previous_context}" 
            "You are an Analyst Agent. Your primary role is to research information, analyze data (text, summaries), "
            "and provide clear, concise insights. You can read files and search the internet via Quartermaster (often by using a specific tool if available).\n"
            f"{tool_info}\n" 
            "\nTOOL USAGE INSTRUCTIONS (Use Sparingly and Precisely):\n"
            "1. Analyze the current task and context. Determine if a specific piece of information is missing that a tool can provide, or if an action (like calculation or saving a file) is explicitly needed.\n"
            "2. If AND ONLY IF a tool is essential for the NEXT STEP of your analysis or to fulfill an explicit part of the request, "
            "respond *ONLY* with a single JSON object on a new line, formatted exactly as follows:\n"
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
        print(f"\n[AnalystAgent] Sending prompt to LLM (model: {self.model_name}):\n{prompt[:1000]}...\n-----------------------------") # Log more
        ollama_response = ollama.generate(model=self.model_name, prompt=prompt)
        llm_output = ollama_response['response'].strip()
        print(f"[AnalystAgent] LLM Raw Output:\n{llm_output}\n-----------------------------")
        return llm_output

    def _handle_delegation(self, llm_output_with_delegation_signal: str, original_command_context: str=""):
        delegation_match = re.search(
            r"DELEGATE_TO:\s*(" + "|".join(self.delegation_keywords.values()) + r"),\s*TASK:\s*([\s\S]+)",
            llm_output_with_delegation_signal
        )
        if not delegation_match:
            if self.config.get("ethics_enabled", True) and self.sentinel.ethics_filter: 
                self.sentinel.ethics_filter.check_text(llm_output_with_delegation_signal)
            return llm_output_with_delegation_signal

        target_agent_key = delegation_match.group(1).strip()
        delegated_task_detail = delegation_match.group(2).strip()
        
        analyst_lead_in = llm_output_with_delegation_signal.split("DELEGATE_TO:")[0].strip()
        if analyst_lead_in: 
            analyst_lead_in += "\n\n" 

        print(f"[AnalystAgent] Delegating task to {target_agent_key}. Task details: {delegated_task_detail[:150]}...")
        delegated_result_text = ""

        if target_agent_key == self.delegation_keywords.get("engineer") and self.engineer:
            try:
                print(f"[AnalystAgent] Calling EngineerAgent.create_python_function...")
                # Ensure EngineerAgent has create_python_function or adapt the call
                code_script = self.engineer.create_python_function(delegated_task_detail) 
                delegated_result_text = f"The EngineerAgent has provided the following Python code based on the analysis:\n```python\n{code_script}\n```"
            except Exception as e_delegate:
                print(f"[AnalystAgent] Error during delegation call to EngineerAgent: {e_delegate}")
                delegated_result_text = f"[AnalystAgent] Error encountered while delegating to EngineerAgent: {str(e_delegate)}"
        else:
            final_response_on_error = f"{analyst_lead_in}[AnalystAgent] Error: Could not delegate. Target agent '{target_agent_key}' unknown or not available."
            if self.config.get("ethics_enabled", True) and self.sentinel.ethics_filter: 
                self.sentinel.ethics_filter.check_text(final_response_on_error) 
            return final_response_on_error

        final_response = f"{analyst_lead_in}{delegated_result_text}"
        if self.config.get("ethics_enabled", True) and self.sentinel.ethics_filter:
            self.sentinel.ethics_filter.check_text(final_response)
        return final_response

    def run(self, command: str, max_iterations=4): # Max iterations for ReAct loop
        current_task_for_llm = command.strip() 
        original_user_command = command.strip() 
        accumulated_context = "" # Stores observations from file reads, web searches, tool outputs

        # Initial Information Gathering (Optional, based on command keywords)
        # This section can be made more sophisticated, or rely on LLM to use tools for info gathering.
        # For now, let's include the pre-read for files mentioned in the original command.
        file_match = re.search(r'\b([\w\-]+\.(?:txt|md|csv|json|py))\b', original_user_command, flags=re.IGNORECASE)
        if file_match:
            file_name = file_match.group(1)
            try:
                print(f"[AnalystAgent] Pre-reading mentioned file: {file_name}")
                # Use the ReadFileTool for consistency, or qm directly if preferred for pre-fetch
                # For simplicity, using qm directly for pre-fetch. LLM will use the tool for explicit requests.
                content = self.qm.read_file(file_name) 
                accumulated_context += f"Context from file '{file_name}':\n{content}\n\n"
            except Exception as e:
                accumulated_context += f"[Warning] Could not pre-read file '{file_name}': {e}\n\n"
        
        # Basic pre-search if keywords are present (LLM should ideally use search tool)
        search_keywords = ["search for", "research", "find information on", "what is"]
        # Check if the command *starts* with search keywords or contains them generally
        # To avoid overly aggressive pre-searching, make this more specific or remove it
        # and rely on the LLM to use the WebSearchTool.
        # For now, keeping a simplified version:
        if any(original_user_command.lower().startswith(kw) for kw in search_keywords):
            temp_query = original_user_command
            for kw in search_keywords: 
                if temp_query.lower().startswith(kw): # only if it starts with the keyword
                    temp_query = temp_query[len(kw):].strip()
                    break
            query_part = temp_query.split('.')[0].split(' and then')[0].strip() 
            if query_part and len(query_part) > 3 and len(query_part) < 100: # Basic query sanity
                try:
                    print(f"[AnalystAgent] Pre-searching for: {query_part}")
                    search_results = self.qm.internet_search(query_part)
                    accumulated_context += f"Initial internet search results for '{query_part}':\n{search_results}\n\n"
                except Exception as e:
                    accumulated_context += f"[Initial Search Error for '{query_part}'] {e}\n\n"


        for iteration in range(max_iterations):
            print(f"\n[AnalystAgent] --- Iteration {iteration + 1}/{max_iterations} ---")
            # The task for the LLM is now the original command, with context building up
            current_task_for_llm = original_user_command 
            print(f"[AnalystAgent] Task for LLM (original command): {current_task_for_llm[:150]}")
            if accumulated_context:
                print(f"[AnalystAgent] Accumulated Context (last 300 chars): ...{accumulated_context[-300:]}")

            prompt_for_llm = self._get_prompt_for_analyst(current_task_for_llm, accumulated_context)
            llm_output = self._call_llm(prompt_for_llm)

            tool_call_data = None
            if llm_output.startswith("{") and llm_output.endswith("}"):
                try:
                    parsed_json = json.loads(llm_output)
                    if "tool_to_use" in parsed_json and "tool_arguments" in parsed_json:
                        tool_call_data = parsed_json
                except json.JSONDecodeError:
                    print("[AnalystAgent] LLM output is JSON-like but not a valid tool call structure or failed to parse.")
                except Exception as e_json: 
                    print(f"[AnalystAgent] Error processing potential JSON tool call: {e_json}")

            if tool_call_data:
                tool_name = tool_call_data.get("tool_to_use")
                tool_args = tool_call_data.get("tool_arguments", {})
                print(f"[AnalystAgent] LLM requested tool: '{tool_name}' with args: {tool_args}")

                chosen_tool_instance: Tool = None
                for t_instance in self.tools:
                    if t_instance.name == tool_name:
                        chosen_tool_instance = t_instance
                        break
                
                if chosen_tool_instance:
                    try:
                        self.sentinel.approve_action(
                            agent_name="AnalystAgent",
                            action_type=f"tool_use:{chosen_tool_instance.name}",
                            detail=str(tool_args) 
                        )
                        tool_output_str = chosen_tool_instance.execute(**tool_args)
                        print(f"[AnalystAgent] Output from tool '{tool_name}': {tool_output_str}")
                        accumulated_context += (
                            f"\n# Observation from successfully using tool '{tool_name}' "
                            f"with arguments {json.dumps(tool_args)}:\n{tool_output_str}\n" # Log args as JSON string
                        )
                        # Task remains the original command; LLM uses new observation to refine its approach
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
                    continue 
            
            # If not a tool call, check for Delegation
            if "DELEGATE_TO:" in llm_output:
                # Pass accumulated context to _handle_delegation so it can be part of the final combined response
                return self._handle_delegation(llm_output, original_command_context=accumulated_context)
            
            # If not a tool call and not delegation, assume it's the Final Answer
            else:
                print("[AnalystAgent] LLM provided a direct final answer.")
                final_answer = accumulated_context + llm_output # Prepend context for completeness if LLM didn't use it
                if self.config.get("ethics_enabled", True) and self.sentinel.ethics_filter:
                    self.sentinel.ethics_filter.check_text(final_answer)
                return final_answer

        print(f"[AnalystAgent] Reached maximum iterations ({max_iterations}). Returning last accumulated context or a message.")
        final_fallback_response = (
            f"[AnalystAgent] Reached maximum processing iterations for the command: '{original_user_command}'. "
            "Unable to provide a conclusive answer with the current approach.\n"
            "Current accumulated information:\n" + accumulated_context
        )
        if self.config.get("ethics_enabled", True) and self.sentinel.ethics_filter:
            self.sentinel.ethics_filter.check_text(final_fallback_response)
        return final_fallback_response
