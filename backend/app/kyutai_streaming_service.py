# kyutai_streaming_service.py
import asyncio
import json
import logging
import websockets
import torch
from typing import AsyncGenerator, Optional
import io
import soundfile as sf

logger = logging.getLogger(__name__)

class KyutaiStreamingService:
    def __init__(self, server_url: str = "ws://localhost:7007"):
        """
        Kyutai streaming TTS service that connects to the Rust server
        
        Args:
            server_url: WebSocket URL of the Kyutai Rust server
        """
        self.server_url = server_url
        self.websocket = None
        self.is_connected = False
        
    async def connect(self):
        """Connect to the Kyutai Rust server"""
        try:
            self.websocket = await websockets.connect(self.server_url)
            self.is_connected = True
            logger.info(f"🗣️ [Kyutai Streaming] Connected to server at {self.server_url}")
        except Exception as e:
            logger.error(f"🗣️ [Kyutai Streaming] Failed to connect: {e}")
            self.is_connected = False
            raise
    
    async def disconnect(self):
        """Disconnect from the server"""
        if self.websocket:
            await self.websocket.close()
            self.is_connected = False
            logger.info("🗣️ [Kyutai Streaming] Disconnected from server")
    
    async def start_stream(self, voice_reference: Optional[str] = None) -> str:
        """
        Start a new streaming session
        
        Args:
            voice_reference: Path to 10-second audio file for voice cloning
            
        Returns:
            session_id: Unique identifier for this streaming session
        """
        if not self.is_connected:
            await self.connect()
        
        # Initialize streaming session
        init_message = {
            "type": "start_stream",
            "voice_reference": voice_reference
        }
        
        await self.websocket.send(json.dumps(init_message))
        response = await self.websocket.recv()
        session_data = json.loads(response)
        
        if session_data.get("status") == "ready":
            session_id = session_data.get("session_id")
            logger.info(f"🗣️ [Kyutai Streaming] Started session {session_id}")
            return session_id
        else:
            raise RuntimeError(f"Failed to start stream: {session_data}")
    
    async def stream_text_and_get_audio(
        self, 
        text_stream: AsyncGenerator[str, None],
        session_id: str
    ) -> AsyncGenerator[bytes, None]:
        """
        Stream text tokens and yield audio chunks as they're generated
        
        Args:
            text_stream: Async generator yielding text chunks
            session_id: Session ID from start_stream()
            
        Yields:
            Audio chunks as bytes (WAV format)
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to Kyutai server")
        
        # Start text streaming task
        text_task = asyncio.create_task(
            self._send_text_stream(text_stream, session_id)
        )
        
        # Start audio receiving task
        audio_task = asyncio.create_task(
            self._receive_audio_stream(session_id)
        )
        
        try:
            # Yield audio chunks as they arrive
            async for audio_chunk in audio_task:
                yield audio_chunk
        finally:
            # Clean up tasks
            text_task.cancel()
            audio_task.cancel()
            
            # End the session
            await self._end_session(session_id)
    
    async def _send_text_stream(self, text_stream: AsyncGenerator[str, None], session_id: str):
        """Send text chunks to the server"""
        try:
            async for text_chunk in text_stream:
                if text_chunk:  # Only send non-empty chunks
                    message = {
                        "type": "text_chunk",
                        "session_id": session_id,
                        "text": text_chunk
                    }
                    await self.websocket.send(json.dumps(message))
                    logger.debug(f"🗣️ [Kyutai Streaming] Sent text: '{text_chunk[:50]}...'")
            
            # Signal end of text
            end_message = {
                "type": "text_end",
                "session_id": session_id
            }
            await self.websocket.send(json.dumps(end_message))
            logger.info(f"🗣️ [Kyutai Streaming] Text stream ended for session {session_id}")
            
        except Exception as e:
            logger.error(f"🗣️ [Kyutai Streaming] Error sending text: {e}")
    
    async def _receive_audio_stream(self, session_id: str) -> AsyncGenerator[bytes, None]:
        """Receive audio chunks from the server"""
        try:
            while True:
                response = await self.websocket.recv()
                
                # Handle binary audio data
                if isinstance(response, bytes):
                    yield response
                    continue
                
                # Handle JSON messages
                try:
                    message = json.loads(response)
                    
                    if message.get("type") == "audio_chunk":
                        # Audio data is base64 encoded in JSON
                        import base64
                        audio_data = base64.b64decode(message["audio_data"])
                        yield audio_data
                        
                    elif message.get("type") == "audio_end":
                        logger.info(f"🗣️ [Kyutai Streaming] Audio stream ended for session {session_id}")
                        break
                        
                    elif message.get("type") == "word_timestamp":
                        # Handle word-level timestamps if needed
                        logger.debug(f"🗣️ [Kyutai Streaming] Word timestamp: {message}")
                        
                except json.JSONDecodeError:
                    # Handle raw binary data
                    yield response.encode() if isinstance(response, str) else response
                    
        except Exception as e:
            logger.error(f"🗣️ [Kyutai Streaming] Error receiving audio: {e}")
    
    async def _end_session(self, session_id: str):
        """End a streaming session"""
        try:
            end_message = {
                "type": "end_session",
                "session_id": session_id
            }
            await self.websocket.send(json.dumps(end_message))
            logger.info(f"🗣️ [Kyutai Streaming] Ended session {session_id}")
        except Exception as e:
            logger.error(f"🗣️ [Kyutai Streaming] Error ending session: {e}")

# Singleton instance
kyutai_streaming = KyutaiStreamingService()

# Text stream generator that connects to your LLM
async def create_text_stream_from_llm(prompt: str, model_manager, model_name: str, **llm_kwargs) -> AsyncGenerator[str, None]:
    """
    Create a text stream from your LLM that Kyutai can consume
    This connects to your existing LLM streaming logic
    """
    # Call your actual streaming function directly
    async for token in generate_text_streaming(
        prompt=prompt,
        model_manager=model_manager,
        model_name=model_name,
        **llm_kwargs
    ):
        if token:  # Only yield non-empty tokens
            yield token

# Alternative: Simple sentence-based streaming
async def create_sentence_stream(text: str) -> AsyncGenerator[str, None]:
    """
    Convert complete text into sentence chunks for streaming
    Useful for testing or non-streaming LLMs
    """
    import re
    
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    for sentence in sentences:
        if sentence.strip():
            yield sentence.strip() + " "
            # Small delay to simulate streaming
            await asyncio.sleep(0.1)

async def generate_text_streaming(prompt: str, model_manager, model_name: str, **kwargs):
    """
    Adapter that connects to your existing LLM streaming implementation
    """
    from .inference import generate_text_streaming as inference_stream
    from . import dual_chat_utils as dcu
    
    # Extract parameters that your inference function expects
    max_tokens = kwargs.get('max_tokens', 1024)
    temperature = kwargs.get('temperature', 0.7)
    top_p = kwargs.get('top_p', 0.9)
    top_k = kwargs.get('top_k', 40)
    repetition_penalty = kwargs.get('repetition_penalty', 1.1)
    stop_sequences = kwargs.get('stop_sequences', [])
    gpu_id = kwargs.get('gpu_id', 0)
    echo = kwargs.get('echo', False)
    request_purpose = kwargs.get('request_purpose', 'kyutai_streaming')
    
    # Call your actual LLM streaming function
    async for token in inference_stream(
        model_manager=model_manager,
        model_name=model_name,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        stop_sequences=stop_sequences,
        gpu_id=gpu_id,
        echo=echo,
        request_purpose=request_purpose
    ):
        yield token