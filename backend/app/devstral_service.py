"""
Devstral Small 2 24B Service for Eloquent Code Editor

A dedicated service for the Devstral Small 2 model with:
- Proper Mistral-format tool calling
- Vision support for code screenshots
- Optimized for single-GPU local inference via llama-cpp-python
"""

import logging
import json
import uuid
import re
import os
import base64
import httpx
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# External LLM API configuration (koboldcpp, ollama, or any OpenAI-compatible server)
EXTERNAL_LLM_URL = os.getenv("DEVSTRAL_API_URL", "http://localhost:5001/v1")  # koboldcpp default
EXTERNAL_LLM_ENABLED = os.getenv("DEVSTRAL_EXTERNAL", "true").lower() == "true"

# Devstral Small 2 system prompt
# Note: Devstral Small 2 uses this prompt format: <s>[SYSTEM_PROMPT]{system_prompt}[/SYSTEM_PROMPT][INST]{prompt}[/INST]
# llama-cpp-python should handle this automatically if chat_format is set correctly
DEVSTRAL_SYSTEM_PROMPT = """You are Devstral, an expert AI coding assistant. You help with software engineering tasks by reading, writing, and modifying code.

Current date: {today}

You have access to tools for file operations. Use them to accomplish tasks.

IMPORTANT GUIDELINES:
1. Use tools for ALL file operations - don't just describe what you would do
2. Read files before modifying them
3. Write complete file content, not placeholders
4. Execute ONE tool at a time, then observe the result before continuing
5. Be precise and thorough

When you need to use a tool, call it directly. When done or communicating, respond with text.

You are a DOER, not a DESCRIBER."""


class DevstralService:
    """
    Service for Devstral Small 2 24B model interactions.
    Handles tool calling, vision, and code editing tasks.
    """
    
    def __init__(self, model_manager=None):
        self.model_manager = model_manager
        self.base_dir = os.getcwd()
        
    def get_system_prompt(self, working_dir: str = None) -> str:
        """Generate the system prompt with current context"""
        today = datetime.today().strftime("%Y-%m-%d")
        
        prompt = DEVSTRAL_SYSTEM_PROMPT.format(
            name="Devstral Small 2",
            today=today
        )
        
        if working_dir:
            prompt += f"\n\nCurrent working directory: {working_dir}"
            
        return prompt
    
    def get_tools_definition(self) -> List[Dict]:
        """Return the tool definitions for Devstral"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read the contents of a file at the specified path",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filepath": {
                                "type": "string",
                                "description": "Path to the file to read (relative to working directory)"
                            }
                        },
                        "required": ["filepath"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write content to a file, creating it if it doesn't exist",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filepath": {
                                "type": "string",
                                "description": "Path to the file to write"
                            },
                            "content": {
                                "type": "string",
                                "description": "The complete content to write to the file"
                            }
                        },
                        "required": ["filepath", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_directory",
                    "description": "List files and folders in a directory",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Directory path to list (default: current directory)"
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_files",
                    "description": "Search for text content within files (grep-like)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Text to search for"
                            },
                            "path": {
                                "type": "string",
                                "description": "Directory to search in (default: current directory)"
                            },
                            "file_pattern": {
                                "type": "string",
                                "description": "File pattern to match (e.g., '*.py', '*.js')"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "run_command",
                    "description": "Execute a shell command",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "The shell command to execute"
                            },
                            "working_dir": {
                                "type": "string",
                                "description": "Working directory for the command"
                            }
                        },
                        "required": ["command"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_directory",
                    "description": "Create a new directory",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Path of the directory to create"
                            }
                        },
                        "required": ["path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "delete_file",
                    "description": "Delete a file (use with caution)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filepath": {
                                "type": "string",
                                "description": "Path to the file to delete"
                            }
                        },
                        "required": ["filepath"]
                    }
                }
            }
        ]
    
    def parse_tool_calls(self, content: str) -> Tuple[List[Dict], str]:
        """
        Parse tool calls from Devstral's response.
        
        Devstral Small 2 can output tool calls in multiple formats:
        1. Structured tool_calls in the response (preferred)
        2. Text-based: function_name{"arg": "value"}
        3. XML-like: <tool_call>...</tool_call>
        
        Returns: (list of parsed tool calls, remaining content)
        """
        if not content:
            return [], ""
        
        tool_calls = []
        remaining_content = content
        
        # Pattern 1: function_name{json} format
        # Match known tool names followed by JSON
        known_tools = ['read_file', 'write_file', 'list_directory', 'search_files', 
                       'run_command', 'create_directory', 'delete_file']
        
        for tool_name in known_tools:
            # Look for tool_name{ or tool_name { patterns
            pattern = rf'{tool_name}\s*(\{{[^}}]*\}})'
            matches = re.finditer(pattern, content, re.DOTALL)
            
            for match in matches:
                try:
                    json_str = match.group(1)
                    # Handle potential nested braces by finding matching close
                    args = self._parse_json_robust(json_str, content, match.start(1))
                    
                    if args is not None:
                        tool_call = {
                            "id": f"call_{uuid.uuid4().hex[:8]}",
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(args) if isinstance(args, dict) else args
                            }
                        }
                        tool_calls.append(tool_call)
                        # Remove the matched tool call from content
                        remaining_content = remaining_content.replace(match.group(0), '', 1)
                        
                except Exception as e:
                    logger.warning(f"Failed to parse tool call for {tool_name}: {e}")
                    continue
        
        # Pattern 2: Look for [TOOL_CALL] or similar markers
        tool_call_pattern = r'\[TOOL_CALL\]\s*(\w+)\s*(\{.*?\})\s*\[/TOOL_CALL\]'
        for match in re.finditer(tool_call_pattern, content, re.DOTALL):
            try:
                func_name = match.group(1)
                args = json.loads(match.group(2))
                tool_call = {
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function", 
                    "function": {
                        "name": func_name,
                        "arguments": json.dumps(args)
                    }
                }
                tool_calls.append(tool_call)
                remaining_content = remaining_content.replace(match.group(0), '', 1)
            except:
                continue
        
        return tool_calls, remaining_content.strip()
    
    def _parse_json_robust(self, json_str: str, full_content: str, start_pos: int) -> Optional[Dict]:
        """Robustly parse JSON, handling nested braces"""
        # First try direct parse
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        # Find matching brace by counting
        brace_count = 0
        json_end = -1
        
        for i in range(start_pos, len(full_content)):
            if full_content[i] == '{':
                brace_count += 1
            elif full_content[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    json_end = i + 1
                    break
        
        if json_end > 0:
            try:
                return json.loads(full_content[start_pos:json_end])
            except:
                pass
        
        return None
    
    def format_tool_result(self, tool_name: str, result: Any, success: bool = True) -> str:
        """Format tool execution result for the model"""
        if success:
            if isinstance(result, dict):
                return json.dumps(result, indent=2)
            return str(result)
        else:
            return f"Error executing {tool_name}: {result}"
    
    def is_devstral_model(self, model_path: str) -> bool:
        """Check if the model is a Devstral model"""
        model_name = os.path.basename(model_path).lower()
        return "devstral" in model_name
    
    def is_devstral_2(self, model_path: str) -> bool:
        """
        Check if specifically Devstral Small 2 24B
        
        Note from bartowski's GGUF release:
        - Uses llama.cpp b7335+
        - Tool calls work but chained tool calls may break
        - Prompt format: <s>[SYSTEM_PROMPT]{system}[/SYSTEM_PROMPT][INST]{prompt}[/INST]
        """
        model_name = os.path.basename(model_path).lower()
        # Match patterns like devstral-small-2, devstral_small_2, 2-24b, 2512, etc.
        is_devstral = "devstral" in model_name
        is_v2 = any(x in model_name for x in ["small-2", "small_2", "2-24b", "24b", "2512"])
        return is_devstral and is_v2
    
    async def chat_with_tools(
        self,
        messages: List[Dict],
        model_instance=None,
        working_dir: str = None,
        temperature: float = 0.15,
        max_tokens: int = 4096,
        image_base64: str = None
    ) -> Dict:
        """
        Send a chat request to Devstral with tool support.
        
        Supports two modes:
        1. External API (koboldcpp, ollama, etc.) - when DEVSTRAL_EXTERNAL=true
        2. Direct llama-cpp-python - when model_instance is provided
        
        Args:
            messages: Conversation history
            model_instance: The loaded llama-cpp model (optional if using external API)
            working_dir: Current working directory for file operations
            temperature: Sampling temperature (low for coding tasks)
            max_tokens: Maximum response tokens
            image_base64: Optional base64 image for vision
            
        Returns:
            Response dict with message and optional tool_calls
        """
        tools = self.get_tools_definition()
        
        # Ensure system prompt is included
        if not messages or messages[0].get('role') != 'system':
            messages.insert(0, {
                'role': 'system',
                'content': self.get_system_prompt(working_dir)
            })
        
        # Handle vision - add image to the last user message
        if image_base64:
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].get('role') == 'user':
                    content = messages[i].get('content', '')
                    messages[i]['content'] = [
                        {"type": "text", "text": content},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                    ]
                    break
        
        try:
            # Use external API if enabled (koboldcpp, ollama, etc.)
            if EXTERNAL_LLM_ENABLED:
                response = await self._call_external_api(messages, tools, temperature, max_tokens)
            elif model_instance:
                # Direct llama-cpp-python call
                response = model_instance.create_chat_completion(
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=False
                )
            else:
                raise ValueError("No model available - either enable external API or load a model")
            
            # Process response for tool calls
            if response and 'choices' in response and response['choices']:
                choice = response['choices'][0]
                message = choice.get('message', {})
                
                # If structured tool calls exist, return as-is
                if message.get('tool_calls'):
                    logger.info(f"🔧 Got {len(message['tool_calls'])} structured tool calls")
                    return response
                
                # Otherwise, try to parse from content
                content = message.get('content', '')
                if content:
                    parsed_calls, remaining = self.parse_tool_calls(content)
                    if parsed_calls:
                        logger.info(f"🔧 Parsed {len(parsed_calls)} tool calls from content")
                        message['tool_calls'] = parsed_calls
                        message['content'] = remaining if remaining else None
                        
            return response
            
        except Exception as e:
            logger.error(f"❌ Error in chat_with_tools: {e}")
            raise
    
    async def _call_external_api(
        self,
        messages: List[Dict],
        tools: List[Dict],
        temperature: float,
        max_tokens: int
    ) -> Dict:
        """
        Call an external OpenAI-compatible API (koboldcpp, ollama, etc.)
        """
        url = f"{EXTERNAL_LLM_URL}/chat/completions"
        
        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        # Add tools if supported (some backends don't support tool calling)
        # koboldcpp might not support tools parameter, so we inject into system prompt
        if tools:
            # Inject tool descriptions into system prompt for backends without native tool support
            tool_text = self._format_tools_for_prompt(tools)
            for msg in payload["messages"]:
                if msg.get("role") == "system":
                    msg["content"] += f"\n\n{tool_text}"
                    break
        
        logger.info(f"🌐 Calling external LLM API: {url}")
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            result = response.json()
            
        logger.info(f"✅ External API response received")
        return result
    
    def _format_tools_for_prompt(self, tools: List[Dict]) -> str:
        """Format tools as text for injection into system prompt"""
        lines = ["## Available Tools", "Call tools using this format: tool_name{\"arg\": \"value\"}", ""]
        
        for tool in tools:
            func = tool.get("function", {})
            name = func.get("name", "")
            desc = func.get("description", "")
            params = func.get("parameters", {}).get("properties", {})
            
            param_list = ", ".join([f'{k}: {v.get("type", "string")}' for k, v in params.items()])
            lines.append(f"- **{name}**({param_list}): {desc}")
        
        lines.append("")
        lines.append("Execute ONE tool at a time. After seeing the result, decide your next action.")
        
        return "\n".join(lines)
    
    async def execute_tool(self, tool_name: str, arguments: Dict, base_dir: str) -> Tuple[bool, Any]:
        """
        Execute a tool and return the result.
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments
            base_dir: Base directory for file operations (security boundary)
            
        Returns:
            Tuple of (success: bool, result: Any)
        """
        import subprocess
        import shutil
        import fnmatch
        
        def get_safe_path(base: str, path: str) -> Optional[str]:
            """Ensure path is within base directory"""
            if not path:
                return base
            # Handle both absolute and relative paths
            if os.path.isabs(path):
                full_path = os.path.normpath(path)
            else:
                full_path = os.path.normpath(os.path.join(base, path))
            # Check that the resolved path is within base
            if not full_path.startswith(os.path.normpath(base)):
                return None
            return full_path
        
        try:
            if tool_name == "read_file":
                filepath = arguments.get('filepath', '')
                safe_path = get_safe_path(base_dir, filepath)
                if not safe_path:
                    return False, "Invalid file path (outside working directory)"
                
                if not os.path.exists(safe_path):
                    return False, f"File not found: {filepath}"
                
                with open(safe_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                    
                # Limit content size for very large files
                if len(content) > 100000:
                    content = content[:100000] + f"\n\n... [Truncated - file has {len(content)} characters total]"
                    
                return True, content
            
            elif tool_name == "write_file":
                filepath = arguments.get('filepath', '')
                content = arguments.get('content', '')
                
                safe_path = get_safe_path(base_dir, filepath)
                if not safe_path:
                    return False, "Invalid file path (outside working directory)"
                
                # Create directories if needed
                os.makedirs(os.path.dirname(safe_path), exist_ok=True)
                
                # Backup existing file
                if os.path.exists(safe_path):
                    backup_path = f"{safe_path}.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    shutil.copy2(safe_path, backup_path)
                
                with open(safe_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                    
                return True, f"Successfully wrote {len(content)} bytes to {filepath}"
            
            elif tool_name == "list_directory":
                path = arguments.get('path', '.')
                safe_path = get_safe_path(base_dir, path)
                if not safe_path:
                    return False, "Invalid directory path"
                
                if not os.path.isdir(safe_path):
                    return False, f"Not a directory: {path}"
                
                items = []
                for item in os.listdir(safe_path):
                    item_path = os.path.join(safe_path, item)
                    item_type = 'folder' if os.path.isdir(item_path) else 'file'
                    size = os.path.getsize(item_path) if os.path.isfile(item_path) else None
                    items.append({
                        'name': item,
                        'type': item_type,
                        'size': size
                    })
                
                # Sort: folders first, then files
                items.sort(key=lambda x: (x['type'] == 'file', x['name'].lower()))
                
                result = f"Contents of {path}:\n"
                for item in items:
                    icon = "📁" if item['type'] == 'folder' else "📄"
                    size_str = f" ({item['size']} bytes)" if item['size'] else ""
                    result += f"{icon} {item['name']}{size_str}\n"
                    
                return True, result
            
            elif tool_name == "search_files":
                query = arguments.get('query', '')
                path = arguments.get('path', '.')
                pattern = arguments.get('file_pattern', '*')
                
                safe_path = get_safe_path(base_dir, path)
                if not safe_path:
                    return False, "Invalid search path"
                
                results = []
                max_results = 50
                
                for root, dirs, files in os.walk(safe_path):
                    # Skip common non-code directories
                    dirs[:] = [d for d in dirs if d not in ['.git', 'node_modules', '__pycache__', 'venv', '.venv']]
                    
                    for file in files:
                        if not fnmatch.fnmatch(file, pattern):
                            continue
                            
                        file_path = os.path.join(root, file)
                        rel_path = os.path.relpath(file_path, base_dir)
                        
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                for line_num, line in enumerate(f, 1):
                                    if query.lower() in line.lower():
                                        results.append({
                                            'file': rel_path,
                                            'line': line_num,
                                            'content': line.strip()[:200]
                                        })
                                        if len(results) >= max_results:
                                            break
                        except:
                            continue
                            
                        if len(results) >= max_results:
                            break
                    if len(results) >= max_results:
                        break
                
                if not results:
                    return True, f"No matches found for '{query}'"
                
                result = f"Found {len(results)} matches for '{query}':\n\n"
                for r in results:
                    result += f"{r['file']}:{r['line']}: {r['content']}\n"
                    
                return True, result
            
            elif tool_name == "run_command":
                command = arguments.get('command', '')
                working_dir = arguments.get('working_dir', base_dir)
                
                # Security: block dangerous commands
                dangerous = ['rm -rf /', 'format', 'mkfs', 'dd if=', ':(){', 'del /f /s /q c:']
                if any(d in command.lower() for d in dangerous):
                    return False, "Command blocked for safety"
                
                safe_dir = get_safe_path(base_dir, working_dir)
                if not safe_dir:
                    safe_dir = base_dir
                
                try:
                    result = subprocess.run(
                        command,
                        shell=True,
                        cwd=safe_dir,
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    
                    output = f"Exit code: {result.returncode}\n"
                    if result.stdout:
                        output += f"Output:\n{result.stdout}\n"
                    if result.stderr:
                        output += f"Errors:\n{result.stderr}\n"
                        
                    return result.returncode == 0, output
                    
                except subprocess.TimeoutExpired:
                    return False, "Command timed out after 30 seconds"
            
            elif tool_name == "create_directory":
                path = arguments.get('path', '')
                safe_path = get_safe_path(base_dir, path)
                if not safe_path:
                    return False, "Invalid directory path"
                
                os.makedirs(safe_path, exist_ok=True)
                return True, f"Created directory: {path}"
            
            elif tool_name == "delete_file":
                filepath = arguments.get('filepath', '')
                safe_path = get_safe_path(base_dir, filepath)
                if not safe_path:
                    return False, "Invalid file path"
                
                if not os.path.exists(safe_path):
                    return False, f"File not found: {filepath}"
                
                # Create backup before deleting
                backup_path = f"{safe_path}.deleted.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                shutil.copy2(safe_path, backup_path)
                os.remove(safe_path)
                
                return True, f"Deleted {filepath} (backup saved as {os.path.basename(backup_path)})"
            
            else:
                return False, f"Unknown tool: {tool_name}"
                
        except Exception as e:
            logger.error(f"❌ Tool execution error ({tool_name}): {e}")
            return False, str(e)


# Global service instance
devstral_service = DevstralService()

