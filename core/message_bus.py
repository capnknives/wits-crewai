# core/message_bus.py
"""
Message Bus System for inter-agent communication in WITS CrewAI.
Provides a centralized message passing system that enables agents to communicate
with each other through structured messages.
"""

import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
import re
import threading
import time
from queue import Queue, Empty


class Message:
    """
    Represents a message sent between agents in the system.
    """
    def __init__(self, 
                 sender: str, 
                 recipient: str, 
                 content: str, 
                 message_type: str = "information",
                 context: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        self.id = self._generate_id()
        self.sender = sender
        self.recipient = recipient
        self.content = content
        self.message_type = message_type  # information, request, response, broadcast
        self.context = context  # Optional context (e.g., task ID, conversation ID)
        self.metadata = metadata or {}
        self.timestamp = datetime.now().isoformat()
        self.read = False
        self.replied_to = False
        self.parent_id = None  # For threaded conversations
    
    def _generate_id(self) -> str:
        """Generate a unique message ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        return f"msg_{timestamp}"
    
    def mark_as_read(self) -> None:
        """Mark the message as read"""
        self.read = True
    
    def mark_as_replied(self) -> None:
        """Mark the message as replied to"""
        self.replied_to = True
    
    def create_reply(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> 'Message':
        """Create a reply to this message"""
        reply = Message(
            sender=self.recipient,
            recipient=self.sender,
            content=content,
            message_type="response",
            context=self.context,
            metadata=metadata or {}
        )
        reply.parent_id = self.id
        return reply
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the message to a dictionary"""
        return {
            "id": self.id,
            "sender": self.sender,
            "recipient": self.recipient,
            "content": self.content,
            "message_type": self.message_type,
            "context": self.context,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "read": self.read,
            "replied_to": self.replied_to,
            "parent_id": self.parent_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create a message from a dictionary"""
        msg = cls(
            sender=data["sender"],
            recipient=data["recipient"],
            content=data["content"],
            message_type=data["message_type"],
            context=data.get("context"),
            metadata=data.get("metadata", {})
        )
        msg.id = data["id"]
        msg.timestamp = data["timestamp"]
        msg.read = data["read"]
        msg.replied_to = data["replied_to"]
        msg.parent_id = data.get("parent_id")
        return msg
    
    def __str__(self) -> str:
        return f"Message(id={self.id}, sender={self.sender}, recipient={self.recipient}, type={self.message_type})"


class MessageBus:
    """
    Central message bus for routing messages between agents.
    """
    def __init__(self, save_path: Optional[str] = None):
        self.messages: List[Message] = []
        self.subscribers: Dict[str, List[Callable[[Message], None]]] = {}
        self.save_path = save_path
        self.messaging_lock = threading.Lock()
        self.message_queues: Dict[str, Queue] = {}  # Queue for each agent
        
        # Start processing thread
        self.running = True
        self.processing_thread = threading.Thread(target=self._process_messages)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        if save_path:
            self._load_messages()
    
    def _process_messages(self) -> None:
        """Background thread to process message queues"""
        while self.running:
            time.sleep(0.1)  # Small delay to reduce CPU usage
            # Nothing to do here for now - future expansion for delayed delivery
    
    def shutdown(self) -> None:
        """Shutdown the message bus"""
        self.running = False
        if self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)
        if self.save_path:
            self._save_messages()
    
    def _load_messages(self) -> None:
        """Load messages from disk"""
        try:
            import os
            if os.path.exists(self.save_path):
                with open(self.save_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.messages = [Message.from_dict(msg_data) for msg_data in data]
                    print(f"[MessageBus] Loaded {len(self.messages)} messages from {self.save_path}")
        except Exception as e:
            print(f"[MessageBus] Error loading messages: {e}")
    
    def _save_messages(self) -> None:
        """Save messages to disk"""
        try:
            with open(self.save_path, 'w', encoding='utf-8') as f:
                data = [msg.to_dict() for msg in self.messages]
                json.dump(data, f, indent=2)
                print(f"[MessageBus] Saved {len(self.messages)} messages to {self.save_path}")
        except Exception as e:
            print(f"[MessageBus] Error saving messages: {e}")
    
    def register_agent(self, agent_name: str) -> None:
        """Register an agent with the message bus"""
        agent_name = agent_name.lower()
        if agent_name not in self.message_queues:
            self.message_queues[agent_name] = Queue()
            print(f"[MessageBus] Registered agent: {agent_name}")
    
    def send_message(self, message: Message) -> str:
        """Send a message to the specified recipient"""
        with self.messaging_lock:
            # Add to central message log
            self.messages.append(message)
            
            # If recipient is "all", broadcast to all registered agents
            if message.recipient.lower() == "all":
                print(f"[MessageBus] Broadcasting message from {message.sender}")
                for agent_name in self.message_queues.keys():
                    if agent_name != message.sender.lower():
                        self.message_queues[agent_name].put(message)
                        
                # Call all broadcast subscribers
                if "broadcast" in self.subscribers:
                    for callback in self.subscribers["broadcast"]:
                        try:
                            callback(message)
                        except Exception as e:
                            print(f"[MessageBus] Error in broadcast subscriber callback: {e}")
            else:
                # Regular message to single recipient
                recipient = message.recipient.lower()
                if recipient in self.message_queues:
                    self.message_queues[recipient].put(message)
                    print(f"[MessageBus] Message sent: {message.sender} -> {message.recipient}")
                else:
                    print(f"[MessageBus] Warning: Recipient '{message.recipient}' not registered")
            
            # Call any subscribers for this sender or recipient
            for key in [message.sender.lower(), message.recipient.lower()]:
                if key in self.subscribers:
                    for callback in self.subscribers[key]:
                        try:
                            callback(message)
                        except Exception as e:
                            print(f"[MessageBus] Error in subscriber callback for {key}: {e}")
            
            # Auto-save if path is configured
            if self.save_path:
                self._save_messages()
            
            return message.id
    
    def get_message(self, agent_name: str, block: bool = False, timeout: Optional[float] = None) -> Optional[Message]:
        """Get the next message for an agent"""
        agent_name = agent_name.lower()
        if agent_name not in self.message_queues:
            print(f"[MessageBus] Warning: Agent '{agent_name}' not registered")
            return None
        
        try:
            message = self.message_queues[agent_name].get(block=block, timeout=timeout)
            message.mark_as_read()
            return message
        except Empty:
            return None
    
    def get_all_messages(self, agent_name: str) -> List[Message]:
        """Get all queued messages for an agent"""
        agent_name = agent_name.lower()
        messages = []
        
        if agent_name not in self.message_queues:
            print(f"[MessageBus] Warning: Agent '{agent_name}' not registered")
            return messages
        
        # Get all messages without blocking
        while not self.message_queues[agent_name].empty():
            try:
                message = self.message_queues[agent_name].get(block=False)
                message.mark_as_read()
                messages.append(message)
            except Empty:
                break
        
        return messages
    
    def subscribe(self, agent_name: str, callback: Callable[[Message], None]) -> None:
        """Subscribe to messages for a specific agent"""
        agent_name = agent_name.lower()
        if agent_name not in self.subscribers:
            self.subscribers[agent_name] = []
        self.subscribers[agent_name].append(callback)
    
    def unsubscribe(self, agent_name: str, callback: Callable[[Message], None]) -> None:
        """Unsubscribe from messages for a specific agent"""
        agent_name = agent_name.lower()
        if agent_name in self.subscribers and callback in self.subscribers[agent_name]:
            self.subscribers[agent_name].remove(callback)
    
    def get_conversation(self, message_id: str) -> List[Message]:
        """Get a conversation thread starting from a specific message"""
        # Find the root message
        root_message = None
        for message in self.messages:
            if message.id == message_id:
                # If this message has a parent, find the root
                current = message
                while current.parent_id:
                    for parent in self.messages:
                        if parent.id == current.parent_id:
                            current = parent
                            break
                root_message = current
                break
        
        if not root_message:
            return []
        
        # Now get all messages in this thread
        thread = [root_message]
        self._get_replies(root_message.id, thread)
        
        # Sort by timestamp
        thread.sort(key=lambda m: m.timestamp)
        
        return thread
    
    def _get_replies(self, parent_id: str, thread: List[Message]) -> None:
        """Recursively get all replies to a message"""
        for message in self.messages:
            if message.parent_id == parent_id and message not in thread:
                thread.append(message)
                self._get_replies(message.id, thread)
    
    def get_messages_by_context(self, context: str) -> List[Message]:
        """Get all messages with a specific context"""
        return [msg for msg in self.messages if msg.context == context]
    
    def get_unread_count(self, agent_name: str) -> int:
        """Get the number of unread messages for an agent"""
        agent_name = agent_name.lower()
        if agent_name not in self.message_queues:
            return 0
        return self.message_queues[agent_name].qsize()
    
    def search_messages(self, query: str, agent_name: Optional[str] = None) -> List[Message]:
        """Search messages by content"""
        results = []
        for message in self.messages:
            if (agent_name is None or 
                message.sender.lower() == agent_name.lower() or 
                message.recipient.lower() == agent_name.lower()):
                if re.search(query, message.content, re.IGNORECASE):
                    results.append(message)
        return results
    
    def clear_agent_queue(self, agent_name: str) -> int:
        """Clear all queued messages for an agent, returns count of cleared messages"""
        agent_name = agent_name.lower()
        if agent_name not in self.message_queues:
            return 0
        
        count = 0
        while not self.message_queues[agent_name].empty():
            try:
                self.message_queues[agent_name].get(block=False)
                count += 1
            except Empty:
                break
        
        return count


# Decorator for agent methods to handle incoming messages
def message_handler(message_types=None):
    """
    Decorator to mark a method as a message handler.
    
    Usage:
        @message_handler(message_types=["request", "information"])
        def handle_message(self, message):
            # Process message
            return response_message
    """
    if message_types is None:
        message_types = ["information", "request"]
    
    def decorator(func):
        func.is_message_handler = True
        func.message_types = message_types
        return func
    
    return decorator


class MessageBusClient:
    """
    Client interface for agents to interact with the message bus.
    Provides a simplified API for sending and receiving messages.
    """
    def __init__(self, agent_name: str, message_bus: MessageBus):
        self.agent_name = agent_name.lower()
        self.message_bus = message_bus
        self.message_bus.register_agent(self.agent_name)
        
        # Register message handlers
        self._register_message_handlers()
    
    def _register_message_handlers(self):
        """Register any methods marked with @message_handler decorator"""
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if hasattr(attr, 'is_message_handler') and attr.is_message_handler:
                def make_callback(method):
                    def callback(message):
                        if message.message_type in method.message_types:
                            method(message)
                    return callback
                
                self.message_bus.subscribe(self.agent_name, make_callback(attr))
    
    def send_message(self, recipient: str, content: str, message_type: str = "information", 
                     context: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Send a message to another agent"""
        message = Message(
            sender=self.agent_name,
            recipient=recipient,
            content=content,
            message_type=message_type,
            context=context,
            metadata=metadata
        )
        return self.message_bus.send_message(message)
    
    def broadcast(self, content: str, context: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None) -> str:
        """Broadcast a message to all agents"""
        message = Message(
            sender=self.agent_name,
            recipient="all",
            content=content,
            message_type="broadcast",
            context=context,
            metadata=metadata
        )
        return self.message_bus.send_message(message)
    
    def reply_to(self, original_message: Message, content: str, 
                metadata: Optional[Dict[str, Any]] = None) -> str:
        """Reply to a specific message"""
        reply = original_message.create_reply(content, metadata)
        # Override the sender to be this agent
        reply.sender = self.agent_name
        return self.message_bus.send_message(reply)
    
    def get_next_message(self, block: bool = False, timeout: Optional[float] = None) -> Optional[Message]:
        """Get the next message from the queue"""
        return self.message_bus.get_message(self.agent_name, block, timeout)
    
    def get_all_messages(self) -> List[Message]:
        """Get all queued messages"""
        return self.message_bus.get_all_messages(self.agent_name)
    
    def get_unread_count(self) -> int:
        """Get number of unread messages"""
        return self.message_bus.get_unread_count(self.agent_name)
    
    def search_messages(self, query: str) -> List[Message]:
        """Search messages by content"""
        return self.message_bus.search_messages(query, self.agent_name)
    
    def clear_queue(self) -> int:
        """Clear all queued messages, returns count of cleared messages"""
        return self.message_bus.clear_agent_queue(self.agent_name)
    
    def process_messages(self, max_messages: int = 10) -> List[Message]:
        """
        Process up to max_messages from the queue, looking for handlers to handle them.
        Returns the list of processed messages.
        """
        processed = []
        for _ in range(max_messages):
            message = self.get_next_message(block=False)
            if not message:
                break
                
            processed.append(message)
            # Find handlers for this message type
            handled = False
            for attr_name in dir(self):
                attr = getattr(self, attr_name)
                if (hasattr(attr, 'is_message_handler') and 
                    attr.is_message_handler and 
                    message.message_type in attr.message_types):
                    try:
                        attr(message)
                        handled = True
                        break
                    except Exception as e:
                        print(f"[MessageBusClient:{self.agent_name}] Error in message handler {attr_name}: {e}")
            
            if not handled:
                print(f"[MessageBusClient:{self.agent_name}] No handler found for message type: {message.message_type}")
                
        return processed


# Example usage:
"""
# Initialize the message bus
message_bus = MessageBus(save_path="message_history.json")

# Create clients for each agent
analyst_client = MessageBusClient("analyst", message_bus)
engineer_client = MessageBusClient("engineer", message_bus)

# Send messages
analyst_client.send_message("engineer", "Could you create a function to parse CSV files?", 
                          message_type="request", context="data_processing_task")

# Handle incoming messages in the Engineer agent
class EngineerMessageHandler(MessageBusClient):
    def __init__(self, message_bus):
        super().__init__("engineer", message_bus)
    
    @message_handler(message_types=["request"])
    def handle_requests(self, message):
        print(f"Received request from {message.sender}: {message.content}")
        # Process the request
        # ...
        # Send a reply
        self.reply_to(message, "Here's the function you requested: [code snippet]")

# Process messages
engineer_client.process_messages()
"""