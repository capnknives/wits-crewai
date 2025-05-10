# agents/engineer_agent.py
import ollama
import re
import os 

class EngineerAgent:
    def __init__(self, config, memory, quartermaster, sentinel, tools=None): # Added tools
        self.config = config
        self.memory = memory
        self.qm = quartermaster
        self.sentinel = sentinel 
        self.tools = tools if tools else [] # Store tools, though not actively used by run() yet
        model_conf = config.get("models", {})
        self.model_name = model_conf.get("engineer", model_conf.get("default", "codellama:7b"))

    def _cleanup_code_response(self, raw_code_text: str) -> str:
        code_block_match = re.search(r"```(?:python)?\n(.*?)\n```", raw_code_text, re.DOTALL | re.IGNORECASE)
        if code_block_match:
            return code_block_match.group(1).strip()
        
        lines = raw_code_text.strip().split('\n')
        if lines and ("here is the code" in lines[0].lower() or "sure, here's" in lines[0].lower() or lines[0].lower().startswith("certainly")):
            lines.pop(0)
        if lines and (lines[-1].lower().startswith("let me know") or lines[-1].lower().startswith("this function will") or lines[-1].lower().startswith("i hope this helps")):
            lines.pop(-1)
        
        return "\n".join(lines).strip()

    def create_python_function(self, function_description: str) -> str:
        print(f"[EngineerAgent] Service call: create_python_function. Description: {function_description[:200]}...")
        
        prompt = (
            f"You are a Python code generation specialist. Your sole task is to generate a single, complete Python function based on the following description. "
            f"Do NOT include any explanatory text, comments outside the function, or markdown formatting like ```python. Only output the raw Python code for the function.\n\n"
            f"Function Description:\n{function_description}"
        )
        
        try:
            print(f"[EngineerAgent] Sending prompt to LLM (model: {self.model_name}) for function generation.")
            ollama_response = ollama.generate(model=self.model_name, prompt=prompt)
            raw_code_text = ollama_response['response'].strip()
            
            print(f"[EngineerAgent] LLM Raw Output for function:\n{raw_code_text}")
            
            cleaned_code = self._cleanup_code_response(raw_code_text)
            
            if self.config.get("ethics_enabled", True) and self.sentinel.ethics_filter: 
                self.sentinel.ethics_filter.check_text(cleaned_code)
            
            return cleaned_code
        except Exception as e:
            print(f"[EngineerAgent] Error in create_python_function: {e}")
            return f"# Error generating function: {str(e)}"

    def run(self, command):
        print(f"[EngineerAgent] Handling direct command: {command[:100]}...")
        try:
            file_match = re.search(r'\b([\w\-]+\.(?:py|js|java|cpp|c|cs|sh|bat|txt|json|html|ts|jsx|tsx))\b', command, flags=re.IGNORECASE)
            file_name_ref = file_match.group(1) if file_match else None
            need_save = False
            content_before = ""
            main_task = command

            save_match = re.search(r'save (?:to|as)\s+([\w\.\-\/\\]+)', command, flags=re.IGNORECASE)
            file_name_to_save = None 
            if save_match:
                file_name_to_save = save_match.group(1)
                if not file_name_ref: file_name_ref = file_name_to_save 
                save_idx = main_task.lower().find('save')
                if save_idx != -1: 
                    main_task = main_task[:save_idx].strip() 
                need_save = True
            elif file_name_ref: 
                file_name_to_save = file_name_ref

            if file_name_ref:
                try:
                    existing_content = self.qm.read_file(file_name_ref) 
                    content_before = f"The current content of '{file_name_ref}' is:\n```\n{existing_content}\n```\n"
                    main_task = re.sub(r'\b' + re.escape(file_name_ref) + r'\b', '', main_task, flags=re.IGNORECASE).strip()
                    main_task = main_task.replace("  ", " ").strip(" ,.") 
                except Exception as e: 
                    content_before = f"(Referenced file '{file_name_ref}' could not be loaded or does not exist: {e}. If creating a new file, this is fine.)\n"
            else: 
                recent_code = self.memory.recall_output("engineer")
                if recent_code:
                    content_before = f"Consider this previous code you generated (it may or may not be relevant to the current task):\n```python\n{recent_code}\n```\n"

            prompt = (
                f"{content_before}"
                f"Based on the above context (if any), and the user's request, complete the following coding task:\n"
                f"User Request: \"{main_task}\"\n\n"
                f"Provide only the complete, updated, or new code block. "
                f"If modifying existing code, provide the full modified code. "
                f"Do not include explanations or chatter before or after the code block unless explicitly asked to explain."
            )
            print(f"[EngineerAgent] Sending prompt to LLM (model: {self.model_name}) for general task.")

            ollama_response = ollama.generate(model=self.model_name, prompt=prompt)
            raw_code_text = ollama_response['response'].strip()
            print(f"[EngineerAgent] LLM Raw Output for general task:\n{raw_code_text}")
            
            cleaned_code_text = self._cleanup_code_response(raw_code_text)

            if self.config.get("ethics_enabled", True) and self.sentinel.ethics_filter:
                self.sentinel.ethics_filter.check_text(cleaned_code_text)

            if need_save and file_name_to_save: 
                try:
                    self.qm.write_file(file_name_to_save, cleaned_code_text)
                    self.memory.remember_output("engineer", cleaned_code_text)
                    return f"Code saved to file: {file_name_to_save}\n```python\n{cleaned_code_text}\n```"
                except Exception as e:
                    print(f"[EngineerAgent] Error saving file '{file_name_to_save}': {e}")
                    return (f"Error saving file {file_name_to_save}: {e}\n\n"
                            f"However, here is the generated code:\n```python\n{cleaned_code_text}\n```")
            else:
                self.memory.remember_output("engineer", cleaned_code_text)
                return cleaned_code_text

        except Exception as e:
            print(f"[EngineerAgent] Error in run method: {e}")
            return f"Engineer Error: {str(e)}"
