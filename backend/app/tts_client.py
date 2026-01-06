# tts_client.py - Client for communicating with the TTS service

import httpx
import logging
import asyncio
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class TTSClient:
    """Client for communicating with the TTS service"""
    
    def __init__(self, base_url: str = "http://localhost:8002"):
        self.base_url = base_url
        self.client = None
        self._is_available = False
        
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client"""
        if self.client is None:
            # No timeout - TTS can take a long time (especially Maya1 first load)
            self.client = httpx.AsyncClient(timeout=None)
        return self.client
    
    async def check_health(self) -> bool:
        """Check if TTS service is available"""
        try:
            client = await self._get_client()
            response = await client.get(f"{self.base_url}/health")
            if response.status_code == 200:
                self._is_available = True
                logger.info("✅ TTS service is healthy")
                return True
            else:
                logger.warning(f"⚠️ TTS service health check failed: {response.status_code}")
                self._is_available = False
                return False
        except Exception as e:
            logger.warning(f"⚠️ TTS service health check failed: {e}")
            self._is_available = False
            return False
    
    async def synthesize_speech(
        self,
        text: str,
        engine: str = "chatterbox",
        voice: str = "default",
        audio_prompt_path: Optional[str] = None,
        exaggeration: float = 0.5,
        cfg: float = 0.5
    ) -> Optional[bytes]:
        """Synthesize speech using the TTS service"""
        try:
            if not self._is_available:
                # Try to check health first
                if not await self.check_health():
                    logger.error("❌ TTS service is not available")
                    return None
            
            client = await self._get_client()
            
            # Prepare request payload
            payload = {
                "text": text,
                "engine": engine,
                "voice": voice,
                "exaggeration": exaggeration,
                "cfg": cfg
            }
            
            if audio_prompt_path:
                payload["audio_prompt_path"] = audio_prompt_path
            
            logger.info(f"🎤 Sending TTS request to service: engine={engine}, text='{text[:50]}...'")
            
            # Send request to TTS service
            response = await client.post(
                f"{self.base_url}/tts/synthesize",
                json=payload
            )
            
            if response.status_code == 200:
                audio_bytes = response.content
                logger.info(f"✅ TTS synthesis successful: {len(audio_bytes)} bytes")
                return audio_bytes
            else:
                logger.error(f"❌ TTS synthesis failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"❌ TTS synthesis request failed: {e}", exc_info=True)
            return None
    
    async def stream_speech(
        self,
        text: str,
        engine: str = "chatterbox",
        voice: str = "default",
        audio_prompt_path: Optional[str] = None,
        exaggeration: float = 0.5,
        cfg: float = 0.5
    ) -> Optional[bytes]:
        """Stream speech using the TTS service"""
        try:
            if not self._is_available:
                if not await self.check_health():
                    logger.error("❌ TTS service is not available")
                    return None
            
            client = await self._get_client()
            
            # Prepare request payload
            payload = {
                "text": text,
                "engine": engine,
                "voice": voice,
                "exaggeration": exaggeration,
                "cfg": cfg
            }
            
            if audio_prompt_path:
                payload["audio_prompt_path"] = audio_prompt_path
            
            logger.info(f"🌊 Sending TTS stream request: engine={engine}, text='{text[:50]}...'")
            
            # Send request to TTS service
            response = await client.post(
                f"{self.base_url}/tts/stream",
                json=payload
            )
            
            if response.status_code == 200:
                audio_bytes = response.content
                logger.info(f"✅ TTS stream successful: {len(audio_bytes)} bytes")
                return audio_bytes
            else:
                logger.error(f"❌ TTS stream failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"❌ TTS stream request failed: {e}", exc_info=True)
            return None
    
    async def warmup_models(self) -> bool:
        """Warm up TTS models"""
        try:
            if not self._is_available:
                if not await self.check_health():
                    logger.error("❌ TTS service is not available")
                    return False
            
            client = await self._get_client()
            
            logger.info("🔥 Requesting TTS model warmup...")
            
            response = await client.post(f"{self.base_url}/tts/warmup")
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"✅ TTS warmup successful: {result}")
                return True
            else:
                logger.error(f"❌ TTS warmup failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"❌ TTS warmup request failed: {e}", exc_info=True)
            return False
    
    async def get_models(self) -> Optional[Dict[str, Any]]:
        """Get available TTS models"""
        try:
            if not self._is_available:
                if not await self.check_health():
                    logger.error("❌ TTS service is not available")
                    return None
            
            client = await self._get_client()
            
            response = await client.get(f"{self.base_url}/tts/models")
            
            if response.status_code == 200:
                models = response.json()
                logger.info(f"✅ Retrieved TTS models: {models}")
                return models
            else:
                logger.error(f"❌ Failed to get TTS models: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to get TTS models: {e}", exc_info=True)
            return None
    
    async def close(self):
        """Close the HTTP client"""
        if self.client:
            await self.client.aclose()
            self.client = None
    
    @property
    def is_available(self) -> bool:
        """Check if TTS service is available"""
        return self._is_available

# Global TTS client instance
tts_client = TTSClient()

async def get_tts_client() -> TTSClient:
    """Get the global TTS client instance"""
    return tts_client

async def synthesize_speech_via_service(
    text: str,
    engine: str = "chatterbox",
    voice: str = "default",
    audio_prompt_path: Optional[str] = None,
    exaggeration: float = 0.5,
    cfg: float = 0.5
) -> Optional[bytes]:
    """Convenience function to synthesize speech via TTS service"""
    client = await get_tts_client()
    return await client.synthesize_speech(
        text=text,
        engine=engine,
        voice=voice,
        audio_prompt_path=audio_prompt_path,
        exaggeration=exaggeration,
        cfg=cfg
    )
