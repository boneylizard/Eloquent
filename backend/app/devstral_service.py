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
import base64
import httpx
import asyncio
import random
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

You are equipped with a set of native tools for file operations. You must use these tools to perform actions.

GUIDELINES:
1. To read a file, you MUST use the `read_file` tool and you MUST provide the `filepath` argument.
2. To write a file, you MUST use the `write_file` tool with both `filepath` and `content` arguments.
3. To list a directory, use `list_directory` with the `path` argument.
4. Execute tools sequentially: call a tool, wait for the result, then decide the next step.
5. Be precise. Do not guess file paths. Use `list_directory` or `search_files` if unsure.

INPUT/OUTPUT FORMAT:
- You are interacting via a structured tool-calling API.
- Do not make up tool names. Use only the tools provided in your schema.
- Do not output empty arguments. If a tool requires a parameter, you must provide it.
- To read the next part of a file, you MUST explicitly calculate and provide the new `start_line` and `end_line`.
- DO NOT send empty arguments to "page" through a file.
- WHEN EDITING FILES: Use `replace_lines` for modifying existing code. Use `write_file` ONLY for creating new files or replacing small files completely.
"""


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
                            },
                            "start_line": {
                                "type": "integer",
                                "description": "Optional 1-based start line to read"
                            },
                            "end_line": {
                                "type": "integer",
                                "description": "Optional 1-based end line to read (inclusive)"
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
                    "name": "replace_lines",
                    "description": "Replace a specific range of lines in a file with new content. use this for EDITING existing files.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filepath": {
                                "type": "string",
                                "description": "Path to the file to edit"
                            },
                            "start_line": {
                                "type": "integer",
                                "description": "The 1-based line number to start replacing from"
                            },
                            "end_line": {
                                "type": "integer",
                                "description": "The 1-based line number to stop replacing at (inclusive)"
                            },
                            "content": {
                                "type": "string",
                                "description": "The new content to insert in place of the specified lines"
                            }
                        },
                        "required": ["filepath", "start_line", "end_line", "content"]
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
    
    async def run_agent_loop(
        self,
        messages: List[Dict],
        model_instance: Any = None,
        working_dir: str = None,
        temperature: float = 0.15,
        max_tokens: int = 4096,
        image_base64: str = None,
        api_config: Optional[Dict] = None,
        max_loops: int = 10
    ) -> Dict:
        """
        Execute an agentic loop where the model can call tools, observe results, and continue.
        """
        loop_count = 0
        tool_steps = []
        final_response = None
        consecutive_blocked_calls = 0
        consecutive_same_file_reads = 0  # Track sequential reads of same file
        last_read_file = {
            "filepath": None,
            "end_line": None
        }
        
        # Clone messages to avoid modifying the original list in place during the loop
        # (Though we will eventually return the full history)
        conversation_history = list(messages)
        
        # Inject system prompt if missing (handled in chat_with_tools, but safe to check)
        if not conversation_history or conversation_history[0].get('role') != 'system':
             conversation_history.insert(0, {
                'role': 'system',
                'content': self.get_system_prompt(working_dir)
            })

        logger.info(f"🤖 Starting Agent Loop (Max: {max_loops})")

        try:
            # Handle vision - add image to the first user message if present
            if image_base64:
                for i in range(len(conversation_history) - 1, -1, -1):
                    if conversation_history[i].get('role') == 'user':
                        content = conversation_history[i].get('content', '')
                        conversation_history[i]['content'] = [
                            {"type": "text", "text": content},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                        ]
                        break

            for loop_count in range(max_loops):
                logger.info(f"🔄 Agent Loop Step {loop_count+1}/{max_loops}")
                
                # 1. Call Model
                if loop_count > 0:
                    # random short delay to be nice to the API
                    await asyncio.sleep(0.5)
                    
                response = await self.chat_with_tools(
                    messages=conversation_history,
                    model_instance=model_instance,
                    working_dir=working_dir,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    api_config=api_config  # Reuse config (native tools enabled)
                )
                
                final_response = response
                
                if not response or 'choices' not in response or not response['choices']:
                    logger.warning("Agent loop: Empty response from model")
                    break
                    
                choice = response['choices'][0]
                message = choice.get('message', {})
                tool_calls = message.get('tool_calls', [])
                content = message.get('content')
                
                # Log raw tool call payloads when names are missing (diagnostic)
                missing_name_indexes = []
                if tool_calls:
                    for i, tc in enumerate(tool_calls):
                        func = tc.get('function', {})
                        if not func.get('name'):
                            missing_name_indexes.append(i)
                    if missing_name_indexes:
                        try:
                            raw_tool_calls = json.dumps(tool_calls, ensure_ascii=False)
                            raw_message = json.dumps(message, ensure_ascii=False)
                        except Exception:
                            raw_tool_calls = str(tool_calls)
                            raw_message = str(message)
                        logger.warning(f"🧾 Tool call(s) missing name at indexes: {missing_name_indexes}")
                        logger.warning(f"🧾 Raw tool_calls: {raw_tool_calls[:8000]}")
                        if content:
                            logger.warning(f"🧾 Raw message content: {str(content)[:2000]}")
                        logger.warning(f"🧾 Raw message object: {raw_message[:8000]}")
                        logger.warning(f"🧾 Response meta: id={response.get('id')} model={response.get('model')}")

                # Sanitize tool calls to prevent empty names causing downstream API errors
                # Some providers crash (503) if we feed back a tool call with no name
                if tool_calls:
                    for tc in tool_calls:
                        if 'function' in tc and not tc['function'].get('name'):
                            # Just patch silently - we'll infer the real name later
                            tc['function']['name'] = 'unknown_tool'

                # Append model response to history
                conversation_history.append(message)
                
                # 2. Check termination (No tools)
                if not tool_calls:
                    logger.info("🛑 No tool calls. Agent loop finished.")
                    break
                    
                # 3. Execute Tools
                halt_loop = False
                for tool_call in tool_calls:
                    func = tool_call.get('function', {})
                    tool_name = func.get('name')
                    tool_call_id = tool_call.get('id')
                    
                    try:
                        arguments = json.loads(func.get('arguments', '{}'))
                    except json.JSONDecodeError:
                        arguments = {}
                    
                    # --- FIX MALFORMED JSON FROM NANOGPT ---
                    # NanoGPT sometimes returns corrupted JSON like:
                    # '{"filepath": "main.py, "start_line": 1, "end_line": 50{"filepath": "main.py", ...}'
                    # We need to extract values using regex
                    if not arguments:
                        raw_args = func.get('arguments', '')
                        if raw_args and '{' in raw_args:
                            logger.warning(f"🔧 Attempting to parse malformed JSON: {raw_args[:200]}...")
                            extracted = {}
                            
                            # Extract filepath - handle both quoted and unquoted
                            fp_match = re.search(r'"filepath"\s*:\s*"([^"]+?)(?:"|,|\s|{)', raw_args)
                            if fp_match:
                                extracted['filepath'] = fp_match.group(1).rstrip(',').strip()
                            
                            # Extract start_line
                            sl_match = re.search(r'"start_line"\s*:\s*(\d+)', raw_args)
                            if sl_match:
                                extracted['start_line'] = int(sl_match.group(1))
                            
                            # Extract end_line
                            el_match = re.search(r'"end_line"\s*:\s*(\d+)', raw_args)
                            if el_match:
                                extracted['end_line'] = int(el_match.group(1))
                            
                            # Extract path (for list_directory)
                            path_match = re.search(r'"path"\s*:\s*"([^"]+)"', raw_args)
                            if path_match:
                                extracted['path'] = path_match.group(1)
                            
                            # Extract query (for search_files)
                            query_match = re.search(r'"query"\s*:\s*"([^"]+)"', raw_args)
                            if query_match:
                                extracted['query'] = query_match.group(1)
                            
                            # Extract command (for run_command)
                            cmd_match = re.search(r'"command"\s*:\s*"([^"]+)"', raw_args)
                            if cmd_match:
                                extracted['command'] = cmd_match.group(1)
                            
                            if extracted:
                                logger.info(f"✅ Rescued from malformed JSON: {extracted}")
                                arguments = extracted
                                # Update tool call so history is correct
                                if 'function' in tool_call:
                                    tool_call['function']['arguments'] = json.dumps(arguments)
                                
                                # ALSO infer tool name if it was empty/unknown
                                if not tool_name or tool_name == 'unknown_tool':
                                    inferred_name = None
                                    if 'content' in extracted and 'filepath' in extracted:
                                        inferred_name = 'write_file'
                                    elif 'query' in extracted:
                                        inferred_name = 'search_files'
                                    elif 'filepath' in extracted:
                                        inferred_name = 'read_file'
                                    elif 'path' in extracted:
                                        inferred_name = 'list_directory'
                                    elif 'command' in extracted:
                                        inferred_name = 'run_command'
                                    
                                    if inferred_name:
                                        logger.info(f"✅ Also inferred tool name: {inferred_name}")
                                        tool_name = inferred_name
                                        if 'function' in tool_call:
                                            tool_call['function']['name'] = inferred_name

                    # --- RESCUE LOGIC START ---
                    # If arguments are empty, the API might have failed to parse them into the structured field.
                    # Look for them in the message content instead.
                    if not arguments:
                        # Log diagnosis info
                        logger.warning(f"🧾 Empty args detected for {tool_name}. Content present: {bool(content)}, Content: {str(content)[:300] if content else 'None'}")
                        logger.warning(f"🧾 Raw func obj: {func}")
                        
                    if not arguments and content:
                        logger.warning(f"🛟 Attempting to rescue empty args for {tool_name} from content...")
                        logger.warning(f"🧾 Content to parse: {content[:500]}...")
                        
                        # Reuse the existing parser logic
                        parsed_calls, _ = self.parse_tool_calls(content)
                        rescued = False
                        for pc in parsed_calls:
                            pc_name = pc['function']['name']
                            pc_args_str = pc['function']['arguments']
                            try:
                                rescued_args = json.loads(pc_args_str)
                                if rescued_args:
                                    # Match by name if we have one, otherwise take first valid parsed call
                                    if (tool_name and pc_name == tool_name) or (not tool_name or tool_name == 'unknown_tool'):
                                        arguments = rescued_args
                                        logger.info(f"✅ Rescued arguments from text: {arguments}")
                                        if 'function' in tool_call:
                                            tool_call['function']['arguments'] = json.dumps(arguments)
                                        # Also rescue tool name if we didn't have one
                                        if not tool_name or tool_name == 'unknown_tool':
                                            tool_name = pc_name
                                            if 'function' in tool_call:
                                                tool_call['function']['name'] = pc_name
                                            logger.info(f"✅ Also rescued tool name: {tool_name}")
                                        rescued = True
                                        break
                            except:
                                pass
                        
                        # FALLBACK: If rescue failed but tool is read_file, try to extract filepath directly
                        if not rescued and (tool_name == "read_file" or not tool_name):
                            # Define helper inline since we're in the loop
                            path_match = re.search(r'`([^`]+\.(?:py|js|jsx|ts|tsx|json|md|yml|yaml|txt|html|css))`', content, re.IGNORECASE)
                            if not path_match:
                                path_match = re.search(r'([\w./\\-]+\.(?:py|js|jsx|ts|tsx|json|md|yml|yaml|txt|html|css))', content, re.IGNORECASE)
                            if path_match:
                                extracted_path = path_match.group(1)
                                arguments = {"filepath": extracted_path, "start_line": 1, "end_line": 200}
                                tool_name = "read_file"
                                if 'function' in tool_call:
                                    tool_call['function']['arguments'] = json.dumps(arguments)
                                    tool_call['function']['name'] = tool_name
                                logger.info(f"✅ Fallback: Extracted filepath from content: {extracted_path}")
                                rescued = True
                        
                        if not rescued:
                            logger.warning("⚠️ Rescue from parse_tool_calls failed, no valid calls found.")
                    # --- RESCUE LOGIC END ---                    

                    def _is_nonempty_str(value: Any) -> bool:
                        return isinstance(value, str) and value.strip() != ""
                    def _extract_filepath_from_text(text: str) -> Optional[str]:
                        if not text:
                            return None
                        # Prefer backticked paths
                        m = re.search(r'`([^`]+\.(?:py|js|jsx|ts|tsx|json|md|yml|yaml|txt|html|css|scss|rs|go|java|cs|cpp|c|h|hpp))`', text, re.IGNORECASE)
                        if m:
                            return m.group(1)
                        # Fallback to any path-like token with an extension
                        m = re.search(r'([\w./\\-]+\.(?:py|js|jsx|ts|tsx|json|md|yml|yaml|txt|html|css|scss|rs|go|java|cs|cpp|c|h|hpp))', text, re.IGNORECASE)
                        if m:
                            return m.group(1)
                        return None
                    logger.info(f"🔍 Raw arguments for {tool_name}: {arguments}")

                    # ALIAS MAPPING: Fix common model mistakes (e.g. filename vs filepath)
                    if tool_name == "read_file":
                        if 'filepath' not in arguments or not arguments['filepath']:
                            for alias in ['filename', 'path', 'file', 'name']:
                                if alias in arguments and arguments[alias]:
                                    logger.warning(f"🩹 Remapping alias '{alias}' to 'filepath' for read_file")
                                    arguments['filepath'] = arguments[alias]
                                    # Critical: Update the original tool_call object
                                    if 'function' in tool_call:
                                        tool_call['function']['arguments'] = json.dumps(arguments)
                                    break
                    elif tool_name == "list_directory":
                        if 'path' not in arguments or not arguments['path']:
                            for alias in ['dir', 'directory', 'folder', 'filepath']:
                                if alias in arguments and arguments[alias]:
                                    logger.warning(f"🩹 Remapping alias '{alias}' to 'path' for list_directory")
                                    arguments['path'] = arguments[alias]
                                    if 'function' in tool_call:
                                        tool_call['function']['arguments'] = json.dumps(arguments)
                                    break

                    # If read_file args are empty, try to infer filepath from message content
                    if tool_name == "read_file" and not _is_nonempty_str(arguments.get("filepath", "")) and content:
                        inferred_path = _extract_filepath_from_text(content)
                        if inferred_path:
                            arguments["filepath"] = inferred_path
                            if 'function' in tool_call:
                                tool_call['function']['arguments'] = json.dumps(arguments)
                            logger.warning(f"🩹 Inferred filepath from message content: {inferred_path}")
                                    
                    # HEURISTIC RECOVERY: Attempt to infer tool name from arguments if missing
                    # Note: We must update the `tool_call` object itself so the history is corrected.
                    if (not tool_name or tool_name == 'unknown_tool') and arguments:
                        inferred_name = None
                        if 'content' in arguments and 'filepath' in arguments:
                            inferred_name = 'write_file'
                        elif 'query' in arguments:
                            inferred_name = 'search_files'
                        elif 'filepath' in arguments:
                            inferred_name = 'read_file'
                        elif 'path' in arguments:
                            inferred_name = 'list_directory'
                        elif 'command' in arguments:
                            inferred_name = 'run_command'
                        
                        if inferred_name:
                            logger.warning(f"🩹 Inferring tool '{inferred_name}' from arguments (was '{tool_name}')")
                            tool_name = inferred_name
                            # Critical: Update the original tool_call object so history is fixed for next turn
                            if 'function' in tool_call:
                                tool_call['function']['name'] = inferred_name

                    # Auto-fill + enforce read_file ranges to avoid huge file dumps
                    if tool_name == "read_file":
                        if not _is_nonempty_str(arguments.get("filepath", "")):
                            if last_read_file.get("filepath"):
                                # LOOP BRAKE: Prevent endless paging on large files
                                current_end = last_read_file.get("end_line", 0) or 0
                                if current_end > 3000:
                                    logger.warning(f"🛑 Blocking excessive paging on {last_read_file['filepath']} (Line {current_end})")
                                    success = False
                                    result = (f"SYSTEM MONITOR: You have read {current_end} lines of this file via paging. "
                                              f"Stop reading linearly. You MUST use 'search_files' to find the specific code section.")
                                    tool_steps.append({
                                        'step': loop_count + 1,
                                        'tool': tool_name,
                                        'args': arguments,
                                        'result': result,
                                        'success': False
                                    })
                                    conversation_history.append({
                                        "role": "tool",
                                        "tool_call_id": tool_call_id,
                                        "name": tool_name,
                                        "content": result
                                    })
                                    continue

                                logger.info(f"🩹 Autofill: Using context '{last_read_file['filepath']}' for empty read_file call")
                                arguments["filepath"] = last_read_file["filepath"]

                                # Auto-increment lines for pagination
                                if arguments.get("start_line") is None:
                                    arguments["start_line"] = current_end + 1 if current_end else 1
                                if arguments.get("end_line") is None:
                                    arguments["end_line"] = arguments["start_line"] + 199
                                logger.info(f"   -> Auto-paging to lines {arguments['start_line']}-{arguments['end_line']}")

                                if 'function' in tool_call:
                                    tool_call['function']['arguments'] = json.dumps(arguments)
                            else:
                                logger.warning("⚠️ Autofill failed: No previous file read context available.")
                        else:
                            # If no range provided, enforce a small window
                            if arguments.get("start_line") is None and arguments.get("end_line") is None:
                                start_line = 1
                                if last_read_file.get("filepath") == arguments.get("filepath") and last_read_file.get("end_line"):
                                    start_line = last_read_file["end_line"] + 1
                                end_line = start_line + 199
                                arguments["start_line"] = start_line
                                arguments["end_line"] = end_line
                                if 'function' in tool_call:
                                    tool_call['function']['arguments'] = json.dumps(arguments)
                                logger.warning(f"🩹 Enforcing read_file range: {arguments['filepath']} lines {start_line}-{end_line}")
                        
                        # EARLY LOOP BREAK: Detect consecutive reads of same file (3+ in a row)
                        current_fp = arguments.get("filepath", "")
                        if current_fp and current_fp == last_read_file.get("filepath"):
                            consecutive_same_file_reads += 1
                            if consecutive_same_file_reads >= 3:
                                logger.warning(f"🛑 Breaking read loop: {consecutive_same_file_reads} consecutive reads of {current_fp}")
                                result = (f"SYSTEM MONITOR: You've made {consecutive_same_file_reads} sequential reads of '{current_fp}'. "
                                          f"STOP reading linearly. Use 'search_files' with query=<what you're looking for> path=<directory> "
                                          f"to find the exact line numbers, then read_file with those specific line ranges.")
                                tool_steps.append({
                                    'step': loop_count + 1,
                                    'tool': tool_name,
                                    'args': arguments,
                                    'result': result,
                                    'success': False
                                })
                                conversation_history.append({
                                    "role": "tool",
                                    "tool_call_id": tool_call_id,
                                    "name": tool_name,
                                    "content": result
                                })
                                consecutive_same_file_reads = 0  # Reset counter
                                continue  # Skip execution, force model to rethink
                        else:
                            consecutive_same_file_reads = 1  # New file, reset to 1


                    # Check for empty tool name (Model Malfunction)
                    if not tool_name:
                        logger.warning("🛑 Blocking empty tool name and patching to 'unknown_tool'")
                        
                        result = "SYSTEM ERROR: You generated a tool call with NO FUNCTION NAME. I could not infer it from arguments. You must specify 'name' in your tool call."
                        
                        # Patch to 'unknown_tool' so history is valid JSON for next API call
                        if 'function' in tool_call:
                            tool_call['function']['name'] = 'unknown_tool'

                        success = False
                        
                        # Manually handle logging for this edge case since we skip normal execution
                        tool_steps.append({
                            'step': loop_count + 1,
                            'tool': 'UNKNOWN',
                            'args': arguments,
                            'result': result,
                            'success': False
                        })
                        conversation_history.append({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "name": "unknown_tool",
                            "content": result
                        })
                        consecutive_blocked_calls += 1
                        if consecutive_blocked_calls >= 2:
                            halt_loop = True
                        continue

                    # Validate arguments before executing
                    invalid_reason = None
                    if tool_name == "read_file":
                        if not _is_nonempty_str(arguments.get('filepath', '')):
                            invalid_reason = "read_file requires a non-empty 'filepath' string."
                    elif tool_name == "write_file":
                        if not _is_nonempty_str(arguments.get('filepath', '')):
                            invalid_reason = "write_file requires a non-empty 'filepath' string."
                        elif not isinstance(arguments.get('content', None), str):
                            invalid_reason = "write_file requires a string 'content' value."
                    elif tool_name == "list_directory":
                        if 'path' in arguments and not isinstance(arguments.get('path'), str):
                            invalid_reason = "list_directory 'path' must be a string."
                    elif tool_name == "search_files":
                        if not _is_nonempty_str(arguments.get('query', '')):
                            invalid_reason = "search_files requires a non-empty 'query' string."
                    elif tool_name == "run_command":
                        if not _is_nonempty_str(arguments.get('command', '')):
                            invalid_reason = "run_command requires a non-empty 'command' string."
                    elif tool_name == "create_directory":
                        if not _is_nonempty_str(arguments.get('path', '')):
                            invalid_reason = "create_directory requires a non-empty 'path' string."
                    elif tool_name == "delete_file":
                        if not _is_nonempty_str(arguments.get('filepath', '')):
                            invalid_reason = "delete_file requires a non-empty 'filepath' string."

                    if invalid_reason:
                        logger.warning(f"🛑 Blocking invalid tool args for {tool_name}: {invalid_reason}")
                        success = False
                        result = (
                            f"SYSTEM ERROR: Invalid arguments for {tool_name}. {invalid_reason} "
                            f"Provide valid arguments and try a DIFFERENT tool call."
                        )
                        tool_steps.append({
                            'step': loop_count + 1,
                            'tool': tool_name,
                            'args': arguments,
                            'result': result,
                            'success': False
                        })
                        conversation_history.append({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "name": tool_name,
                            "content": result
                        })
                        consecutive_blocked_calls += 1
                        if consecutive_blocked_calls >= 2:
                            halt_loop = True
                        continue
                    
                    # REPETITION GUARD: Check for ANY identical tool call in history
                    # This prevents loops where the agent retries the same thing later or ignores previous results
                    is_repetition = False
                    previous_step_index = -1
                    
                    current_args_json = json.dumps(arguments, sort_keys=True)
                    
                    for i, step in enumerate(tool_steps):
                        if (step['tool'] == tool_name and 
                            json.dumps(step['args'], sort_keys=True) == current_args_json):
                            is_repetition = True
                            previous_step_index = step['step']
                            break
                    
                    if is_repetition:
                        logger.warning(f"🛑 Blocking repetitive tool call: {tool_name} (Same as Step {previous_step_index})")
                        success = False
                        result = (f"SYSTEM MONITOR: You ALREADY executed this exact tool call in Step {previous_step_index}. "
                                  f"Do not repeat actions. Check the history for the result. "
                                  f"You must try a DIFFERENT tool, path, or argument.")
                        consecutive_blocked_calls += 1
                        if consecutive_blocked_calls >= 2:
                            halt_loop = True
                    else:
                        logger.info(f"🔨 Agent Executing: {tool_name}")
                        success, result = await self.execute_tool(
                            tool_name=tool_name,
                            arguments=arguments,
                            base_dir=working_dir or self.base_dir
                        )
                        consecutive_blocked_calls = 0

                        if tool_name == "read_file" and success:
                            last_read_file["filepath"] = arguments.get("filepath")
                            end_line_val = None
                            try:
                                end_line_arg = arguments.get("end_line")
                                start_line_arg = arguments.get("start_line")
                                if end_line_arg is not None:
                                    end_line_val = int(end_line_arg)
                                else:
                                    line_count = len(str(result).splitlines())
                                    if line_count > 0:
                                        if start_line_arg is not None:
                                            end_line_val = int(start_line_arg) + line_count - 1
                                        else:
                                            end_line_val = line_count
                            except Exception:
                                end_line_val = None

                            last_read_file["end_line"] = end_line_val
                    
                    # Truncate result for history if too long to prevent bloating context
                    result_str = str(result)
                    if len(result_str) > 20000:
                        history_result = result_str[:20000] + "... [Truncated for Context]"
                    else:
                        history_result = result_str

                    # Append execution to tool_steps log (for UI)
                    tool_steps.append({
                        'step': loop_count + 1,
                        'tool': tool_name,
                        'args': arguments,
                        'result': result_str[:500], # Keep UI log concise
                        'success': success
                    })
                    
                    # Append result to history
                    conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": tool_name,
                        "content": history_result
                    })

                    if halt_loop:
                        break
                
                if halt_loop:
                    logger.warning("🛑 Stopping agent loop due to repeated invalid/repetitive tool calls.")
                    break
        
        except Exception as e:
            logger.error(f"❌ Agent Loop Crashed: {e}")
            # Add a final error step so the user sees what happened
            tool_steps.append({
                'step': -1,
                'tool': 'SYSTEM_CRASH',
                'args': {},
                'result': f"Agent Loop Terminated Unexpectedly: {str(e)}",
                'success': False
            })
            # Make sure we return what we have so far
            if not final_response:
                 final_response = {'choices': [{'message': {'role': 'assistant', 'content': f"Agent crashed: {e}"}}]}
            
        return {
            "final_response": final_response,
            "conversation_history": conversation_history,
            "tool_steps": tool_steps,
            "loops_run": loop_count
        }

    async def run_agent_loop_streaming(
        self,
        messages: List[Dict],
        model_instance: Any = None,
        working_dir: str = None,
        temperature: float = 0.15,
        max_tokens: int = 4096,
        image_base64: str = None,
        api_config: Optional[Dict] = None,
        max_loops: int = 10
    ):
        """
        Streaming version of run_agent_loop that YIELDS events for real-time UI updates.
        Events: {"type": "content/tool_call/tool_result/done", ...}
        """
        import json as json_lib
        
        loop_count = 0
        tool_steps = []
        final_response = None
        consecutive_blocked_calls = 0
        consecutive_same_file_reads = 0
        last_read_file = {"filepath": None, "end_line": None}
        
        conversation_history = list(messages)
        
        if not conversation_history or conversation_history[0].get('role') != 'system':
            conversation_history.insert(0, {
                'role': 'system',
                'content': self.get_system_prompt(working_dir)
            })

        logger.info(f"🤖 Starting Streaming Agent Loop (Max: {max_loops})")
        
        # Yield start event
        yield {"type": "start", "max_loops": max_loops}

        try:
            if image_base64:
                for i in range(len(conversation_history) - 1, -1, -1):
                    if conversation_history[i].get('role') == 'user':
                        content = conversation_history[i].get('content', '')
                        conversation_history[i]['content'] = [
                            {"type": "text", "text": content},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                        ]
                        break

            for loop_count in range(max_loops):
                yield {"type": "step_start", "step": loop_count + 1, "max_loops": max_loops}
                
                # Prepare messages with CoT injection if needed
                messages_to_send = list(conversation_history)
                if loop_count > 0:
                    messages_to_send.append({
                        "role": "user", 
                        "content": "Proceed. Briefly explain your next step, then use a tool."
                    })

                # Call model
                response = await self.chat_with_tools(
                    messages=messages_to_send,
                    model_instance=model_instance,
                    working_dir=working_dir,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    image_base64=image_base64 if loop_count == 0 else None,
                    api_config=api_config
                )
                
                final_response = response
                
                logger.info(f"🔍 FULL API RESPONSE: {json_lib.dumps(response, default=str)[:5000]}...") # Log first 5000 chars
                
                message = response.get('choices', [{}])[0].get('message', {})
                content = message.get('content', '')
                tool_calls = message.get('tool_calls', [])
                
                # --- HALLUCINATION RESCUE START ---
                # If model wrote a tool call in text content instead of using the function, rescue it!
                if not tool_calls and content and "🔧" in content:
                    logger.warning("🕵️ Detected potential tool hallucination in text. Attempting rescue...")
                    # Regex to match: 🔧 **tool_name**: {json_args}
                    import re
                    # Look for the pattern user showed in logs
                    matches = re.finditer(r"🔧 \*\*(?P<name>[\w_]+)\*\*: (?P<args>\{.*?\})", content, re.DOTALL)
                    rescued_tools = []
                    for match in matches:
                        t_name = match.group("name")
                        t_args = match.group("args")
                        # Try to clean up args if they have trailing ... or similar
                        if "..." in t_args:
                            t_args = t_args.split("...")[0] + "}" # Very naive fix, might need better json repair
                        
                        # Validate JSON
                        try:
                            json_lib.loads(t_args) # Just check if valid
                            rescued_tools.append({
                                "id": f"rescued_{loop_count}_{len(rescued_tools)}",
                                "type": "function",
                                "function": {
                                    "name": t_name,
                                    "arguments": t_args
                                }
                            })
                            logger.info(f"✅ Rescued tool call from text: {t_name}({t_args})")
                        except:
                            logger.warning(f"❌ Failed to parse rescued JSON for {t_name}")
                    
                    if rescued_tools:
                        tool_calls = rescued_tools
                        # Update the message object so history is correct-ish
                        message['tool_calls'] = tool_calls
                # --- HALLUCINATION RESCUE END ---

                # Patch empty tool names - log this!
                if tool_calls:
                    empty_name_indexes = [i for i, tc in enumerate(tool_calls) 
                                          if 'function' in tc and not tc['function'].get('name')]
                    if empty_name_indexes:
                        logger.warning(f"🧾 Tool call(s) missing name at indexes: {empty_name_indexes}")
                    for tc in tool_calls:
                        if 'function' in tc and not tc['function'].get('name'):
                            tc['function']['name'] = 'unknown_tool'

                # Yield content if present
                if content:
                    logger.info(f"📝 Model Content (Step {loop_count+1}): {content}")
                    yield {"type": "content", "step": loop_count + 1, "text": content}
                else:
                    logger.info(f"📝 Model Content (Step {loop_count+1}): [No text content]")
                
                logger.info(f"🔧 Got {len(tool_calls)} structured tool calls")
                
                conversation_history.append(message)
                
                if not tool_calls:
                    logger.info("🛑 No tool calls. Streaming agent loop finished.")
                    break
                
                # Process tool calls
                for tool_call in tool_calls:
                    func = tool_call.get('function', {})
                    tool_name = func.get('name')
                    tool_call_id = tool_call.get('id')
                    
                    try:
                        arguments = json_lib.loads(func.get('arguments', '{}'))
                    except json_lib.JSONDecodeError:
                        arguments = {}
                    
                    # Malformed JSON rescue
                    if not arguments:
                        raw_args = func.get('arguments', '')
                        if raw_args and '{' in raw_args:
                            logger.warning(f"🔧 Attempting to parse malformed JSON: {raw_args[:200]}...")
                            extracted = {}
                            fp_match = re.search(r'"filepath"\s*:\s*"([^"]+?)(?:"|,|\s|{)', raw_args)
                            if fp_match:
                                extracted['filepath'] = fp_match.group(1).rstrip(',').strip()
                            sl_match = re.search(r'"start_line"\s*:\s*(\d+)', raw_args)
                            if sl_match:
                                extracted['start_line'] = int(sl_match.group(1))
                            el_match = re.search(r'"end_line"\s*:\s*(\d+)', raw_args)
                            if el_match:
                                extracted['end_line'] = int(el_match.group(1))
                            path_match = re.search(r'"path"\s*:\s*"([^"]+)"', raw_args)
                            if path_match:
                                extracted['path'] = path_match.group(1)
                            query_match = re.search(r'"query"\s*:\s*"([^"]+)"', raw_args)
                            if query_match:
                                extracted['query'] = query_match.group(1)
                            if extracted:
                                logger.info(f"✅ Rescued from malformed JSON: {extracted}")
                                arguments = extracted
                                # Infer tool name
                                if not tool_name or tool_name == 'unknown_tool':
                                    if 'query' in extracted:
                                        tool_name = 'search_files'
                                    elif 'filepath' in extracted:
                                        tool_name = 'read_file'
                                    elif 'path' in extracted:
                                        tool_name = 'list_directory'
                                    if tool_name and tool_name != 'unknown_tool':
                                        logger.info(f"✅ Also inferred tool name: {tool_name}")
                                        func['name'] = tool_name
                    
                    # Infer tool name from arguments if missing
                    if (not tool_name or tool_name == 'unknown_tool') and arguments:
                        inferred_name = None
                        if 'content' in arguments and 'filepath' in arguments:
                            inferred_name = 'write_file'
                        elif 'query' in arguments:
                            inferred_name = 'search_files'
                        elif 'filepath' in arguments:
                            inferred_name = 'read_file'
                        elif 'path' in arguments:
                            inferred_name = 'list_directory'
                        elif 'command' in arguments:
                            inferred_name = 'run_command'
                        if inferred_name:
                            logger.warning(f"🩹 Inferring tool '{inferred_name}' from arguments (was '{tool_name}')")
                            tool_name = inferred_name
                            func['name'] = tool_name
                    
                    # Log raw arguments
                    logger.info(f"🔍 Raw arguments for {tool_name}: {arguments}")
                    
                    # Yield tool call event BEFORE execution
                    yield {
                        "type": "tool_call",
                        "step": loop_count + 1,
                        "tool": tool_name,
                        "args": arguments
                    }
                    
                    # Execute tool
                    logger.info(f"🔨 Agent Executing: {tool_name}")
                    success, result = await self.execute_tool(
                        tool_name=tool_name,
                        arguments=arguments,
                        base_dir=working_dir or self.base_dir
                    )
                    
                    # Track read_file progress
                    if tool_name == "read_file" and success:
                        last_read_file["filepath"] = arguments.get("filepath")
                        last_read_file["end_line"] = arguments.get("end_line")
                    
                    result_str = str(result)
                    
                    # Yield tool result event
                    yield {
                        "type": "tool_result",
                        "step": loop_count + 1,
                        "tool": tool_name,
                        "success": success,
                        "result": result_str[:500]  # Truncate for UI
                    }
                    
                    tool_steps.append({
                        'step': loop_count + 1,
                        'tool': tool_name,
                        'args': arguments,
                        'result': result_str[:500],
                        'success': success
                    })
                    
                    # Add to conversation history
                    history_result = result_str[:20000] + "..." if len(result_str) > 20000 else result_str
                    conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": tool_name,
                        "content": history_result
                    })

        except Exception as e:
            logger.error(f"❌ Streaming Agent Loop Crashed: {e}")
            yield {"type": "error", "error": str(e)}
            tool_steps.append({
                'step': -1,
                'tool': 'SYSTEM_CRASH',
                'args': {},
                'result': f"Agent crashed: {e}",
                'success': False
            })
        
        # Yield final done event
        yield {
            "type": "done",
            "final_response": final_response,
            "tool_steps": tool_steps,
            "loops_run": loop_count + 1
        }

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
            
            # Deep copy messages to avoid mutating the original history or accumulating tool prompts across retries/loops
            import copy
            safe_messages = copy.deepcopy(messages)
            
            payload = {
                "messages": safe_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False
            }
            
            # Use NATIVE tool calling for OpenAI-compatible APIs (like NanoGPT/Devstral)
            # This ensures the model receives the schema correctly and outputs structured tool_calls
            if tools:
                payload["tools"] = tools
                payload["tool_choice"] = "auto"
                logger.info(f"🌐 Calling external LLM API with NATIVE tools: {url}")
            else:
                logger.info(f"🌐 Calling external LLM API (No tools): {url}")
        
        # Retry Config
        max_retries = 5
        base_delay = 2.0
        
        try:
            # Increase timeout for large models (Devstral 123b can take time)
            timeout = httpx.Timeout(300.0, connect=60.0)
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                response = await client.post(url, json=payload, headers=headers)
                
                if response.status_code != 200:
                    logger.error(f"API Error ({response.status_code}) from {url}: {response.text}")
                    response.raise_for_status()
                
                result = response.json()
                logger.info(f"✅ External API response received")
                return result

        except Exception as e:
            logger.error(f"Failed to connect to {url}: {e}")
            raise Exception(f"Failed to connect to {url}: {e}")
    
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
                start_line = arguments.get('start_line')
                end_line = arguments.get('end_line')
                safe_path = get_safe_path(base_dir, filepath)
                if not safe_path:
                    return False, "Invalid file path (outside working directory)"
                
                if not os.path.exists(safe_path):
                    return False, f"File not found: {filepath}"
                
                if os.path.isdir(safe_path):
                    return False, f"Path is a directory, not a file. Use 'list_directory' to see contents: {filepath}"
                
                with open(safe_path, 'r', encoding='utf-8', errors='replace') as f:
                    if start_line is not None or end_line is not None:
                        lines = f.readlines()
                        total_lines = len(lines)
                        try:
                            start = int(start_line) if start_line is not None else 1
                            end = int(end_line) if end_line is not None else total_lines
                        except (TypeError, ValueError):
                            start = 1
                            end = total_lines

                        if start < 1:
                            start = 1
                        if end < start:
                            end = start
                        if end > total_lines:
                            end = total_lines

                        start_idx = start - 1
                        end_idx = end
                        sliced = lines[start_idx:end_idx]
                        numbered = [f"{i+1}: {line}" for i, line in enumerate(lines[start_idx:end_idx], start=start_idx)]
                        content = "".join(numbered)
                        header = f"[read_file] {filepath} (lines {start}-{end} of {total_lines})\n"
                        return True, header + content

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

            elif tool_name == "replace_lines":
                filepath = arguments.get('filepath', '')
                try:
                    start_line = int(arguments.get('start_line'))
                    end_line = int(arguments.get('end_line'))
                except (ValueError, TypeError):
                    return False, "start_line and end_line must be integers"
                    
                content = arguments.get('content', '')
                
                safe_path = get_safe_path(base_dir, filepath)
                if not safe_path:
                    return False, "Invalid file path (outside working directory)"
                
                if not os.path.exists(safe_path):
                    return False, f"File not found: {filepath}. Use write_file to create new files."
                
                # Create backup
                backup_path = f"{safe_path}.bak.replace.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                shutil.copy2(safe_path, backup_path)
                
                try:
                    with open(safe_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        
                    total_lines = len(lines)
                    
                    # Validate range
                    if start_line < 1: start_line = 1
                    if end_line > total_lines: end_line = total_lines
                    if start_line > end_line:
                        return False, f"Invalid range: start_line ({start_line}) > end_line ({end_line})"

                    # Prepare replacement content (handle creating newlines if needed)
                    # The model might give 'line1\nline2'. We need to make sure it fits into the list structure.
                    # Simplest way: Replace the slice with the new string (as a single string or split lines?)
                    # Splitting into lines is safer for reconstruction.
                    new_lines = content.splitlines(keepends=True)
                    # content.splitlines might lose the last newline if it doesn't exist.
                    # Let's ensure consistency: 
                    new_lines = [l if l.endswith('\n') else l + '\n' for l in content.splitlines()]
                    if not content and not new_lines: # Empty content means delete
                        new_lines = []

                    # Perform replacement (0-indexed)
                    start_idx = start_line - 1
                    end_idx = end_line
                    
                    final_lines = lines[:start_idx] + new_lines + lines[end_idx:]
                    
                    with open(safe_path, 'w', encoding='utf-8') as f:
                        f.writelines(final_lines)
                        
                    return True, f"Successfully replaced lines {start_line}-{end_line} in {filepath} (Backup: {os.path.basename(backup_path)})"
                    
                except Exception as e:
                    return False, f"Failed to replace lines: {str(e)}"
            
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
                
                if os.path.isfile(safe_path):
                    # Search single file
                    files_to_search = [safe_path]
                    root_dir = os.path.dirname(safe_path)
                else:
                    # Walk directory
                    files_to_search = []
                    for root, dirs, files in os.walk(safe_path):
                        dirs[:] = [d for d in dirs if d not in ['.git', 'node_modules', '__pycache__', 'venv', '.venv']]
                        for file in files:
                            if fnmatch.fnmatch(file, pattern):
                                files_to_search.append(os.path.join(root, file))
                                
                    root_dir = safe_path

                for file_path in files_to_search:
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
