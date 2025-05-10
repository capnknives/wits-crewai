import os
import json
import requests
import re
from datetime import datetime

class QuartermasterException(Exception):
    pass

class QuartermasterAgent:
    def __init__(self, config, memory, sentinel):
        self.config = config
        self.memory = memory
        self.sentinel = sentinel
        self.output_dir = config.get("output_directory", "output")
        os.makedirs(self.output_dir, exist_ok=True)

    def read_file(self, file_path):
        if not os.path.isabs(file_path):
            if os.sep not in file_path:
                candidate = os.path.join(self.output_dir, file_path)
                if os.path.exists(candidate):
                    file_path = candidate
        self.sentinel.approve_action("Quartermaster", "file_read", detail=file_path)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            raise QuartermasterException(f"File not found: {file_path}")
        except Exception as e:
            raise QuartermasterException(f"Error reading file: {e}")

    def write_file(self, file_path, content):
        if not os.path.isabs(file_path):
            file_path = os.path.join(self.output_dir, file_path)
        self.sentinel.approve_action("Quartermaster", "file_write", detail=file_path)
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception as e:
            raise QuartermasterException(f"Error writing file {file_path}: {e}")
        return True

    def delete_file(self, file_path):
        if not os.path.isabs(file_path):
            file_path = os.path.join(self.output_dir, file_path)
        self.sentinel.approve_action("Quartermaster", "file_delete", detail=file_path)
        try:
            os.remove(file_path)
        except FileNotFoundError:
            raise QuartermasterException(f"File not found: {file_path}")
        except Exception as e:
            raise QuartermasterException(f"Error deleting file: {e}")
        return True

    def list_files(self, directory=None):
        if directory is None:
            directory = self.output_dir
        if not os.path.isabs(directory):
            directory = os.path.join(os.getcwd(), directory)
        self.sentinel.approve_action("Quartermaster", "file_read", detail=directory)
        try:
            entries = os.listdir(directory)
        except FileNotFoundError:
            raise QuartermasterException(f"Directory not found: {directory}")
        except Exception as e:
            raise QuartermasterException(f"Error listing directory: {e}")
        if not entries:
            return f"No files found in {directory}."
        entries.sort()
        return "\n".join([f"Files in {directory}:"] + [f"- {e}" for e in entries])

    def internet_search(self, query):
        self.sentinel.approve_action("Quartermaster", "internet", detail=query)
        try:
            res = requests.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
                timeout=10
            )
            if res.status_code != 200:
                raise QuartermasterException(f"Search request failed with status {res.status_code}")
            data = res.json()
        except Exception as e:
            raise QuartermasterException(f"Internet search error: {e}")

        summary = ""
        if data.get("AbstractText"):
            summary = data["AbstractText"]
        elif data.get("RelatedTopics"):
            for topic in data["RelatedTopics"]:
                if isinstance(topic, dict) and topic.get("Text"):
                    summary = topic["Text"]
                    break
        return summary or "No relevant information found online."

    def handle_command(self, command):
        clower = command.lower().strip()

        if clower.startswith("read file") or clower.startswith("open file"):
            file_path = command.split("file", 1)[-1].strip(" :\"")
            if not file_path:
                return "Please specify the file to read."
            try:
                content = self.read_file(file_path)
                return content[:1000] + "\n. (output truncated)" if len(content) > 1000 else content
            except Exception as e:
                return f"Error reading file: {e}"

        if clower.startswith("list files") or clower.startswith("show files") or clower.startswith("ls"):
            dir_name = None
            if " in " in clower:
                parts = clower.split(" in ", 1)
                if len(parts) > 1:
                    dir_name = parts[1].strip()
            try:
                return self.list_files(dir_name) if dir_name else self.list_files()
            except Exception as e:
                return f"Error listing files: {e}"

        if "list goals" in clower or "show goals" in clower or "current goals" in clower:
            return self.memory.list_goals()

        if "list completed" in clower or "show completed" in clower:
            return self.memory.list_completed()

        if "add goal" in clower or "new goal" in clower:
            key_phrase = "add goal" if "add goal" in clower else "new goal"
            idx = clower.find(key_phrase)
            insert_text = command[idx + len(key_phrase):].strip(" :")
            if not insert_text:
                return "Please specify the goal description to add."
            agent_name = None
            if " for " in insert_text.lower():
                parts = insert_text.split(" for ")
                if len(parts) >= 2:
                    possible_agent = parts[-1].strip().strip('.').replace(":", "").strip()
                    if possible_agent.lower() in ["scribe", "analyst", "engineer", "quartermaster", "sentinel"]:
                        agent_name = possible_agent.capitalize()
                        insert_text = parts[0].strip()
            self.memory.add_goal(insert_text, agent=agent_name)
            return f"Added new goal{' for ' + agent_name if agent_name else ''}: {insert_text}"

        if "complete goal" in clower or "mark goal" in clower or "done goal" in clower:
            match = re.search(r'goal\s+(\d+)', clower)
            if match:
                goal_num = int(match.group(1))
                success = self.memory.complete_goal(goal_num)
                if success:
                    return f"Goal #{goal_num} marked as completed."
            else:
                goal_text = clower.split("goal", 1)[-1].strip(" :")
                success = self.memory.complete_goal(goal_text)
                if success:
                    return f"Goal matching \"{goal_text}\" marked as completed."
            return "Goal not found or could not be completed."

        if clower.startswith("delete") or clower.startswith("remove"):
            file_path = command.split("file", 1)[-1].strip(" :\"") if "file" in clower else command[len("delete"):].strip(" :\"")
            if not file_path:
                return "Please specify the file to delete."
            try:
                self.delete_file(file_path)
                return f"File '{file_path}' has been deleted."
            except Exception as e:
                return f"Error deleting file: {e}"

        if "save" in clower or "store" in clower or "write last output" in clower:
            file_match = re.search(r'(?:save|store).*?as\s+([\w\-\\\/\.:]+)', clower)
            custom_name = file_match.group(1).strip() if file_match else None

            content = (self.memory.recall_output("scribe")
                       or self.memory.recall_output("engineer")
                       or self.memory.recall_output("analyst"))
            if not content:
                return "There is no recent agent output to save."

            agent = self.memory.get_last_agent() or "output"
            default_ext = ".txt"
            if agent == "engineer":
                default_ext = ".py"
            elif agent == "scribe" and "blog" in (custom_name or "").lower():
                default_ext = ".md"

            if custom_name:
                base, ext = os.path.splitext(custom_name)
                filename = base + default_ext if not ext else custom_name
            else:
                filename = f"{agent}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{default_ext}"

            try:
                self.write_file(filename, content)
                return f"Saved output to file: {filename}"
            except Exception as e:
                return f"Failed to save file: {e}"

        return ("Quartermaster: I can manage files and goals. Try commands like 'list files', 'add goal ...', "
                "'complete goal ...', or 'delete file ...'.")
    