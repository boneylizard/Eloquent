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
DEVSTRAL_SYSTEM_PROMPT = """You are Devstral, an expert AI coding assistant. You help with software engineering tasks by reading, writing, and modifying code.

Current date: {today}

You have access to tools for file operations. Use them to accomplish tasks.

IMPORTANT GUIDELINES:
1. Use tools for ALL file operations - don't just describe what you would do
2. Read files before modifying them
3. Write complete file content, not placeholders
4. Execute ONE tool at a time, then observe the result before continuing
5. Be precise and thorough

CRITICAL RULES:
- DO NOT simulate tool execution.
- DO NOT make up tool results or output "Tool Results:".
- You MUST invoke the tool function directly to perform an action.
- Wait for the system to provide the real tool output.
- IF YOU OUTPUT CODE BUT DO NOT CALL `write_file`, YOU HAVE FAILED.
- ALWAYS use `write_file` to save your changes to the disk.
- DO NOT put the code in a markdown block in your text response. Put it ONLY in the `write_file` `content` parameter.
- Providing the code in text WITHOUT the tool call is a critical error."""


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
        
        Updated to robustly handle nested braces (critical for write_file code content).
        """
        if not content:
            return [], ""
        
        tool_calls = []
        remaining_content = content
        
        known_tools = ['read_file', 'write_file', 'list_directory', 'search_files', 
                       'run_command', 'create_directory', 'delete_file']
        
        # Helper to find tool calls by scanning for "ToolName{" or "ToolName {"
        # We iterate to find all potential starts
        for tool_name in known_tools:
            # Simple regex to find the start index of potential calls
            # We don't try to capture the JSON with regex anymore
            start_pattern = re.compile(rf'{re.escape(tool_name)}\s*\{{')
            
            # We loop to find multiple occurrences
            search_start_pos = 0
            while True:
                match = start_pattern.search(content, search_start_pos)
                if not match:
                    break
                
                # Start of the JSON object (the '{' character)
                json_start_index = match.end() - 1 
                
                # Use robust brace counting to find the full object
                args_dict, end_index = self._extract_balanced_json(content, json_start_index)
                
                if args_dict is not None:
                    # Construct the tool call
                    tool_call = {
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(args_dict)
                        }
                    }
                    tool_calls.append(tool_call)
                    
                    # Remove the text from remaining_content
                    # We grab the full matched string: "tool_name { ... }"
                    full_match_text = content[match.start():end_index]
                    remaining_content = remaining_content.replace(full_match_text, "", 1)
                    
                    # Move search forward past this match
                    search_start_pos = end_index
                else:
                    # If we couldn't parse valid JSON, just skip past the start to avoid infinite loop
                    search_start_pos = match.end()

        # Fallback: Look for [TOOL_CALL] markers (legacy format)
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

    def _extract_balanced_json(self, text: str, start_index: int) -> Tuple[Optional[Dict], int]:
        """
        Extract a JSON object starting at start_index by counting braces.
        Returns (parsed_dict, end_index) or (None, -1).
        """
        if start_index >= len(text) or text[start_index] != '{':
            return None, -1
            
        brace_count = 0
        in_string = False
        escape = False
        
        for i in range(start_index, len(text)):
            char = text[i]
            
            if in_string:
                if escape:
                    escape = False
                elif char == '\\':
                    escape = True
                elif char == '"':
                    in_string = False
            else:
                if char == '"':
                    in_string = True
                elif char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        # Found the closing brace
                        json_str = text[start_index:i+1]
                        try:
                            return json.loads(json_str), i+1
                        except json.JSONDecodeError:
                            return None, -1
                            
        return None, -1
    
    # Kept for backward compatibility if needed, but _extract_balanced_json is preferred
    def _parse_json_robust(self, json_str: str, full_content: str, start_pos: int) -> Optional[Dict]:
        return self._extract_balanced_json(full_content, start_pos)[0]
    
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
        """Check if specifically Devstral Small 2 24B"""
        model_name = os.path.basename(model_path).lower()
        is_devstral = "devstral" in model_name
        is_v2 = any(x in model_name for x in ["small-2", "small_2", "2-24b", "24b", "2512", "123b"])
        return is_devstral and is_v2
    
    async def chat_with_tools(
        self,
        messages: List[Dict],
        model_instance=None,
        working_dir: str = None,
        temperature: float = 0.15,
        max_tokens: int = 4096,
        image_base64: str = None,
        api_config: Optional[Dict] = None
    ) -> Dict:
        """
        Send a chat request to Devstral with tool support.
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
            # Mode 1: Specific API Configuration (e.g. OpenRouter)
            if api_config:
                response = await self._call_external_api(messages, tools, temperature, max_tokens, api_config)
            
            # Mode 2: Legacy External API (Env vars)
            elif EXTERNAL_LLM_ENABLED:
                response = await self._call_external_api(messages, tools, temperature, max_tokens)
            
            # Mode 3: Local Model
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
                
                # Otherwise, try to parse from content (Fixes Mistral/Devstral text-based fallback)
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
        max_tokens: int,
        api_config: Optional[Dict] = None
    ) -> Dict:
        """
        Call an external OpenAI-compatible API (koboldcpp, ollama, OpenRouter, etc.)
        """
        if api_config:
            # Use provided configuration
            base_url = api_config.get('url', EXTERNAL_LLM_URL).rstrip('/')
            
            if base_url.endswith('/v1'):
                url = f"{base_url}/chat/completions"
            else:
                url = f"{base_url}/v1/chat/completions"
                
            api_key = api_config.get('api_key')
            model_name = api_config.get('model', 'default')
            
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "Eloquent/Devstral",
                "Authorization": f"Bearer {api_key}" if api_key else ""
            }
            
            # Additional headers (e.g. for OpenRouter)
            if 'openrouter.ai' in url:
                headers["HTTP-Referer"] = "http://localhost:3000"
                headers["X-Title"] = "Eloquent Code Editor"
                
            payload = {
                "model": model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False
            }
            
            # NATIVE TOOL CALLING
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
            
            logger.info(f"🌐 Calling API: {url} (Model: {model_name}) with native tools")
            
        else:
            # Legacy/Env-var configuration
            url = f"{EXTERNAL_LLM_URL}/chat/completions"
            headers = {"Content-Type": "application/json"}
            
            payload = {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False
            }
        
            # Prompt Injection for backends without native tool support
            if tools:
                tool_text = self._format_tools_for_prompt(tools)
                for msg in payload["messages"]:
                    if msg.get("role") == "system":
                        msg["content"] += f"\n\n{tool_text}"
                        break
            
            logger.info(f"🌐 Calling legacy external LLM API: {url}")
        
        try:
            async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
                response = await client.post(url, json=payload, headers=headers)
                if response.status_code != 200:
                    logger.error(f"API Error ({response.status_code}) from {url}: {response.text}")
                    response.raise_for_status()
                result = response.json()
        except Exception as e:
            logger.error(f"Failed to connect to {url}: {e}")
            raise Exception(f"Failed to connect to {url}: {e}")
            
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
        """
        import subprocess
        import shutil
        import fnmatch
        
        def get_safe_path(base: str, path: str) -> Optional[str]:
            """Ensure path is within base directory"""
            if not path:
                return base
            if os.path.isabs(path):
                full_path = os.path.normpath(path)
            else:
                full_path = os.path.normpath(os.path.join(base, path))
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
                    
                if len(content) > 100000:
                    content = content[:100000] + f"\n\n... [Truncated - file has {len(content)} characters total]"
                    
                return True, content
            
            elif tool_name == "write_file":
                filepath = arguments.get('filepath', '')
                content = arguments.get('content', '')
                
                safe_path = get_safe_path(base_dir, filepath)
                if not safe_path:
                    return False, "Invalid file path (outside working directory)"
                
                os.makedirs(os.path.dirname(safe_path), exist_ok=True)
                
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