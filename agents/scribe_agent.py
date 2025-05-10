# agents/scribe_agent.py
import ollama
import re
import os 
from datetime import datetime # For default filenames

class ScribeAgent:
    def __init__(self, config, memory, quartermaster, sentinel, tools=None): 
        self.config = config
        self.memory = memory
        self.qm = quartermaster
        self.sentinel = sentinel 
        self.tools = tools if tools else [] 
        model_conf = config.get("models", {})
        self.model_name = model_conf.get("scribe", model_conf.get("default", "llama2"))
        
        # Keywords that might indicate a follow-up to the Scribe's previous output
        self.follow_up_keywords = [
            "continue", "expand on that", "elaborate", "refine this", 
            "use that outline", "based on the previous", "using that draft",
            "add to that", "the outline", "that book", "the draft", "expand on it",
            "make it longer", "more detail", 
            "add more chapter", "add chapters", "add a chapter", "add 5 more chapters", # Made more flexible
            "next chapter", "revise that", "improve that", "edit that", "another chapter",
            "the previous work", "that story"
        ]
        # Keywords that indicate a new, distinct task, overriding follow-up
        self.new_task_keywords = [
            "draft a new book about", "write a new article on", "create a new blog post about",
            "start a new story", "draft a book about", "write an article on", "compose a poem about",
            "a new poem", "a new book", "a new story", "generate a new"
        ]
        # Keywords that imply saving, even without a filename
        self.save_intent_keywords = ["save this", "save draft", "save the draft", "save it"]

    def run(self, command: str):
        print(f"[ScribeAgent] Received command: {command[:100]}...")
        try:
            file_path_to_save = None
            task_description_original = command.strip()
            task_description_for_llm = task_description_original 
            
            previous_scribe_output = None
            prompt_context = "" 
            command_lower_original = task_description_original.lower()

            # --- "Save as" handling (done early) ---
            # Regex to capture filename, allowing for paths with directories
            save_match_with_filename = re.search(r'save (?:to|as)\s+([\w\.\-\s\/\\]+)', task_description_original, flags=re.IGNORECASE)
            command_implies_save = any(keyword in command_lower_original for keyword in self.save_intent_keywords)

            if save_match_with_filename:
                file_path_to_save = save_match_with_filename.group(1).strip()
                # Update task_description_for_llm to remove the save instruction for the LLM prompt
                task_description_for_llm = task_description_original[:save_match_with_filename.start()].strip()
                print(f"[ScribeAgent] Will attempt to save output to specified file: '{file_path_to_save}'")
            elif command_implies_save: # No filename given, but save intent is clear
                # Generate a default filename. The Scribe LLM might also suggest one.
                # For now, let's make a generic one.
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_path_to_save = f"scribe_draft_{timestamp}.txt"
                # The task for the LLM is still the original command, it might just be "save this"
                # We don't want to strip "save this" from task_description_for_llm if it's the *whole* command.
                # If "save this" is part of a larger instruction, it's okay for LLM to see it.
                print(f"[ScribeAgent] Save intent detected. Will attempt to save to default file: '{file_path_to_save}'")


            # --- Context Handling for Follow-up Commands ---
            is_follow_up = False
            is_explicit_new_task = any(keyword in command_lower_original for keyword in self.new_task_keywords)
            
            if is_explicit_new_task:
                print("[ScribeAgent] Explicit new task keyword detected. Starting fresh.")
            else:
                # Check for follow-up keywords. Using a more flexible check for "add X chapters".
                if any(keyword in command_lower_original for keyword in self.follow_up_keywords) or \
                   re.search(r"add \d+ more chapter", command_lower_original): # More specific regex for "add N chapters"
                    is_follow_up = True
                    print(f"[ScribeAgent] Follow-up keyword/pattern detected in command: '{task_description_original}'")
                elif len(task_description_original.split()) < 7 and not file_path_to_save: 
                    # Short commands are follow-ups *unless* they are just a save command for previous work
                    # If file_path_to_save was set by "save this draft", it's a follow-up action on previous text.
                    if command_implies_save and not save_match_with_filename: # e.g. "save this draft"
                         is_follow_up = True # It's a follow-up action (saving) on previous content
                         print("[ScribeAgent] Short save command detected, treating as follow-up action on previous content.")
                    elif not command_implies_save: # Short command, not about saving
                         is_follow_up = True
                         print("[ScribeAgent] Short command (not save-related) detected, considering it a potential follow-up instruction.")


            if is_follow_up: 
                previous_scribe_output = self.memory.recall_output("scribe")
                if previous_scribe_output:
                    print("[ScribeAgent] Using previous Scribe output as context for follow-up.")
                    prompt_context = (
                        "You are continuing, refining, or acting upon a previous piece of writing. "
                        "Here is the PREVIOUS WORK you (ScribeAgent) generated:\n"
                        "--- BEGIN PREVIOUS WORK ---\n"
                        f"{previous_scribe_output}\n"
                        "--- END PREVIOUS WORK ---\n\n"
                        "Now, based on that PREVIOUS WORK, please address the following new instruction carefully:\n"
                    )
                else: # Follow-up indicated, but no previous Scribe output found
                    print("[ScribeAgent] Follow-up indicated, but no previous Scribe output found in memory. Treating as a new task.")
                    is_follow_up = False 
            
            # Determine prompt prefix
            current_task_lower_for_prefix = task_description_for_llm.lower() # Use the one potentially stripped of "save as filename"
            prompt_prefix = "Write a comprehensive piece on the following topic: " 

            if 'blog post' in current_task_lower_for_prefix or 'blog article' in current_task_lower_for_prefix:
                prompt_prefix = "Draft a well-structured and engaging blog post about: "
            elif 'book chapter' in current_task_lower_for_prefix:
                prompt_prefix = "Write a detailed book chapter focusing on: "
            elif ('book' in current_task_lower_for_prefix and \
                  ('draft' in current_task_lower_for_prefix or \
                   'write' in current_task_lower_for_prefix or \
                   'create' in current_task_lower_for_prefix or \
                   'develop' in current_task_lower_for_prefix)) or \
                 ('turn it into a book' in current_task_lower_for_prefix and prompt_context):
                if prompt_context: 
                    prompt_prefix = "Using the PREVIOUS WORK (likely an outline or earlier draft) as your primary guide, expand or continue writing the book, focusing on the current request: "
                else: 
                    prompt_prefix = "You are tasked with drafting a new book. Begin with the first chapter or an outline if appropriate, focusing on the core request: "
            elif 'poem' in current_task_lower_for_prefix:
                prompt_prefix = "Compose a creative and expressive poem about: "
            elif (any(kw in command_lower_original for kw in ["add more chapter", "add chapters", "add a chapter"]) or \
                  re.search(r"add \d+ more chapter", command_lower_original)) and prompt_context:
                prompt_prefix = "Based on the PREVIOUS WORK provided, " 
            elif is_follow_up and prompt_context: # Generic follow-up with context
                 # If the command is just "save this as a draft", the task for LLM is minimal.
                 if command_implies_save and not save_match_with_filename and len(task_description_for_llm.split()) < 4:
                     prompt_prefix = "The user wants to save the PREVIOUS WORK. Your task is simply to acknowledge this if no other instruction is given. If there are other instructions, address them. The file will be saved as specified or with a default name. Current instruction: "
                 else:
                    prompt_prefix = "Based on the PREVIOUS WORK provided, " 

            full_prompt = f"{prompt_context}{prompt_prefix}{task_description_for_llm}"

            print(f"[ScribeAgent] Sending prompt to LLM (model: {self.model_name}): {full_prompt[:600]}...")
            ollama_response = ollama.generate(model=self.model_name, prompt=full_prompt)
            output_text = ollama_response['response'].strip()
            print(f"[ScribeAgent] LLM Raw Output received (length: {len(output_text)}).")

            # If the task was primarily just to save, and LLM gives a confirmation,
            # we should use the *previous_scribe_output* for saving, not the LLM's confirmation message.
            text_to_save = output_text
            if command_implies_save and not save_match_with_filename and \
               len(task_description_for_llm.split()) < 4 and previous_scribe_output and \
               ("save" in output_text.lower() or "will save" in output_text.lower() or "acknowledg" in output_text.lower()):
                print("[ScribeAgent] Save command detected for previous work. Using previous output for saving.")
                text_to_save = previous_scribe_output


            if self.config.get("ethics_enabled", True) and self.sentinel.ethics_filter:
                self.sentinel.ethics_filter.check_text(text_to_save) # Check the text that will be saved/returned

            # Always remember the latest *generated or acted upon* output
            # If it was just a save command, the content of the book (previous_scribe_output) is what's relevant.
            if text_to_save == previous_scribe_output and previous_scribe_output is not None:
                 self.memory.remember_output("scribe", previous_scribe_output)
            else:
                 self.memory.remember_output("scribe", output_text) # LLM generated new content or modified

            if file_path_to_save: # This is now set if "save as filename" OR if "save this draft" (default name)
                try:
                    self.qm.write_file(file_path_to_save, text_to_save) # Save the correct text
                    saved_path_confirmation = os.path.normpath(os.path.join(self.qm.output_dir, file_path_to_save))
                    
                    # If the LLM just gave a confirmation for saving, don't return that as the main output.
                    if text_to_save == previous_scribe_output and previous_scribe_output is not None:
                         return f"Previous draft successfully saved to '{saved_path_confirmation}'."
                    else:
                         return f"Draft successfully generated/updated and saved to '{saved_path_confirmation}'.\n\nContent (first 500 chars):\n{text_to_save[:500]}..."

                except Exception as e_save:
                    print(f"[ScribeAgent] Error saving file '{file_path_to_save}': {e_save}")
                    return (f"Draft generated/updated, but failed to save to '{file_path_to_save}': {e_save}\n\n"
                            f"Content (first 500 chars):\n{text_to_save[:500]}...")
            else: # No save instruction
                return output_text

        except Exception as e:
            print(f"[ScribeAgent] Error in run method: {e}")
            # import traceback
            # print(traceback.format_exc())
            return f"Scribe Error: {str(e)}"

