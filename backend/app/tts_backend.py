# tts_backend.py - Dedicated TTS Service Backend
# This runs independently from the main backend to avoid resource conflicts

import os
# Disable problematic Torch optimizations for Python 3.12+ (MUST BE AT TOP)
os.environ["TORCH_DYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

# MONKEYPATCH: Disable torch.compile to avoid Dynamo error on Python 3.12+
try:
    import torch
    if not hasattr(torch, '_original_compile'):
        torch._original_compile = torch.compile
        def dummy_compile(f, *args, **kwargs): return f
        torch.compile = dummy_compile
except Exception:
    pass

import sys
import logging
import asyncio
import json
import time
import tempfile
import uuid
import gc
from pathlib import Path
from typing import Optional

# Add the backend directory to the path
backend_dir = Path(__file__).parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

# FastAPI imports
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn

# Import TTS service functions
from tts_service import (
    load_chatterbox_model, 
    synthesize_speech, 
    ChatterboxTTS,
    TTSStreamer  # Add the TTSStreamer class import
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="LiangLocal TTS Service", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for loaded models
chatterbox_model = None
tts_initialized = False

@app.on_event("startup")
async def startup_event():
    """Initialize TTS service - model loading is deferred to prevent CUDA conflicts"""
    global tts_initialized
    
    try:
        logger.info("🚀 Starting TTS Backend Service...")
        
        # Set environment variables for later CUDA optimization
        # These don't initialize CUDA, just configure it for when it's used
        os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
        os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
        os.environ["CUDA_CACHE_DISABLE"] = "0"
        os.environ["CUDA_CACHE_PATH"] = "/tmp/cuda_cache"
        
        # CRITICAL: Do NOT initialize PyTorch CUDA here!
        # This would conflict with stable-diffusion.cpp's ggml CUDA backend.
        # CUDA will be initialized when Chatterbox is first loaded on TTS request.
        
        logger.info("📌 TTS Backend ready - CUDA/model loading deferred")
        logger.info("📌 Chatterbox will load on first TTS request")
        logger.info("📌 This prevents CUDA conflicts with Stable Diffusion (ggml)")
        
        tts_initialized = True
        logger.info("🎉 TTS Backend Service ready!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize TTS service: {e}", exc_info=True)
        raise

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "tts_backend",
        "initialized": tts_initialized,
        "models": {
            "chatterbox": chatterbox_model is not None
        }
    }

@app.post("/tts/synthesize")
async def synthesize_endpoint(request: Request):
    """Synthesize speech using the specified engine"""
    try:
        if not tts_initialized:
            raise HTTPException(status_code=503, detail="TTS service not initialized")
        
        # Parse request body
        body = await request.json()
        text = body.get("text", "")
        engine = body.get("engine", "chatterbox")
        voice = body.get("voice", "default")
        audio_prompt_path = body.get("audio_prompt_path")
        exaggeration = body.get("exaggeration", 0.5)
        cfg = body.get("cfg", 0.5)
        
        if not text:
            raise HTTPException(status_code=400, detail="Text is required")
        
        logger.info(f"🎤 TTS request: engine={engine}, text='{text[:50]}...'")
        
        # Synthesize speech
        start_time = time.perf_counter()
        audio_bytes = await synthesize_speech(
            text=text,
            voice=voice,
            engine=engine,
            audio_prompt_path=audio_prompt_path,
            exaggeration=exaggeration,
            cfg=cfg
        )
        end_time = time.perf_counter()
        
        duration_ms = (end_time - start_time) * 1000
        logger.info(f"✅ TTS completed in {duration_ms:.2f}ms, {len(audio_bytes)} bytes")
        
        # Return audio as streaming response
        return StreamingResponse(
            iter([audio_bytes]),
            media_type="audio/wav",
            headers={
                "Content-Length": str(len(audio_bytes)),
                "X-TTS-Duration": f"{duration_ms:.2f}ms"
            }
        )
        
    except Exception as e:
        logger.error(f"❌ TTS synthesis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"TTS synthesis failed: {str(e)}")

@app.post("/tts/stream")
async def tts_stream_endpoint(request: Request):
    """Streaming TTS endpoint for real-time synthesis"""
    try:
        if not tts_initialized:
            raise HTTPException(status_code=503, detail="TTS service not initialized")
        
        # Parse request body
        body = await request.json()
        text = body.get("text", "")
        engine = body.get("engine", "chatterbox")
        voice = body.get("voice", "default")
        audio_prompt_path = body.get("audio_prompt_path")
        exaggeration = body.get("exaggeration", 0.5)
        cfg = body.get("cfg", 0.5)
        
        if not text:
            raise HTTPException(status_code=400, detail="Text is required")
        
        logger.info(f"🌊 TTS Stream request: engine={engine}, text='{text[:50]}...'")
        
        # For now, return the full audio (can be enhanced for true streaming later)
        start_time = time.perf_counter()
        audio_bytes = await synthesize_speech(
            text=text,
            voice=voice,
            engine=engine,
            audio_prompt_path=audio_prompt_path,
            exaggeration=exaggeration,
            cfg=cfg
        )
        end_time = time.perf_counter()
        
        duration_ms = (end_time - start_time) * 1000
        logger.info(f"✅ TTS Stream completed in {duration_ms:.2f}ms, {len(audio_bytes)} bytes")
        
        return StreamingResponse(
            iter([audio_bytes]),
            media_type="audio/wav",
            headers={
                "Content-Length": str(len(audio_bytes)),
                "X-TTS-Duration": f"{duration_ms:.2f}ms"
            }
        )
        
    except Exception as e:
        logger.error(f"❌ TTS Stream failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"TTS Stream failed: {str(e)}")

@app.get("/tts/models")
async def list_models():
    """List available TTS models"""
    return {
        "models": {
            "chatterbox": {
                "available": chatterbox_model is not None,
                "name": "Chatterbox TTS",
                "description": "High-quality voice cloning TTS"
            }
        }
    }

@app.post("/tts/warmup")
async def warmup_endpoint():
    """Warm up TTS models for better performance"""
    try:
        if not tts_initialized:
            raise HTTPException(status_code=503, detail="TTS service not initialized")
        
        logger.info("🔥 Starting TTS warmup...")
        
        # Warm up Chatterbox
        if chatterbox_model:
            logger.info("🔥 Warming up Chatterbox model...")
            import torch
            with torch.inference_mode():
                # Generate a short test audio
                test_text = "This is a warmup test for optimal performance."
                if hasattr(chatterbox_model, 'generate'):
                    dummy_audio = chatterbox_model.generate(test_text, language_id="en")
                    logger.info("✅ Chatterbox warmup complete")
                else:
                    logger.info("⚠️ Chatterbox model doesn't support generate method")
        
        logger.info("🎉 TTS warmup complete!")
        return {"status": "success", "message": "TTS warmup complete"}
        
    except Exception as e:
        logger.error(f"❌ TTS warmup failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"TTS warmup failed: {str(e)}")

@app.post("/tts/upload-voice")
async def upload_voice_reference(file: UploadFile = File(...)):
    """Upload a reference audio file for Chatterbox voice cloning."""
    try:
        # Validate file type
        allowed_extensions = {'.wav', '.mp3', '.flac', '.m4a'}
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Define the voice references directory
        voices_dir = Path(__file__).parent / "static" / "voice_references"
        voices_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a clean filename based on original name
        original_name = Path(file.filename).stem  # Remove extension
        clean_name = "".join(c for c in original_name if c.isalnum() or c in (' ', '-', '_')).strip()
        clean_name = clean_name.replace(' ', '_')  # Replace spaces with underscores
        
        # Ensure filename is not empty
        if not clean_name:
            clean_name = "uploaded_voice"
        
        # Create the final filename with original extension
        final_filename = f"{clean_name}{file_extension}"
        save_path = voices_dir / final_filename
        
        # Handle duplicates by adding a number suffix
        counter = 1
        while save_path.exists():
            final_filename = f"{clean_name}_{counter}{file_extension}"
            save_path = voices_dir / final_filename
            counter += 1
        
        # Save the uploaded file
        with open(save_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"📁 Voice reference uploaded: {save_path}")
        
        return {
            "status": "success",
            "voice_id": final_filename,  # Return the clean filename instead of UUID
            "file_path": str(save_path),
            "message": f"Voice reference '{file.filename}' uploaded successfully"
        }
        
    except Exception as e:
        logger.error(f"❌ Error uploading voice reference: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to upload voice reference: {str(e)}")

@app.post("/tts/unload-chatterbox")
async def unload_chatterbox():
    """Unload Chatterbox model from VRAM to free up memory"""
    global chatterbox_model, tts_initialized
    
    try:
        if chatterbox_model is None:
            logger.info("🔓 [Chatterbox] Model is already unloaded")
            return {
                "status": "success", 
                "message": "Chatterbox model was already unloaded",
                "vram_freed": 0
            }
        
        logger.info("🔓 [Chatterbox] Unloading model to free VRAM...")
        
        import torch
        
        vram_freed = 0.0
        model_device = None
        
        # Detect which GPU device the model is on
        if hasattr(chatterbox_model, 'device'):
            model_device = chatterbox_model.device
            logger.info(f"🔓 Model is on device: {model_device}")
        elif hasattr(chatterbox_model, 'model') and hasattr(chatterbox_model.model, 'device'):
            model_device = chatterbox_model.model.device
            logger.info(f"🔓 Model submodule is on device: {model_device}")
        
        # Log VRAM before unload for the correct device
        if torch.cuda.is_available():
            # Check all GPU devices
            for i in range(torch.cuda.device_count()):
                vram_on_device = torch.cuda.memory_allocated(i) / 1024**3
                logger.info(f"🔓 GPU {i} VRAM before unload: {vram_on_device:.2f} GB")
            
            # Use device 0 as baseline (current device)
            current_device = torch.cuda.current_device()
            vram_before = torch.cuda.memory_allocated(current_device) / 1024**3
            logger.info(f"🔓 Current device: {current_device}, VRAM: {vram_before:.2f} GB")
        else:
            vram_before = 0.0
            current_device = 0
        
        # Move model to CPU first if it has a .to() method
        try:
            if hasattr(chatterbox_model, 'to'):
                logger.info("🔓 Moving model to CPU...")
                chatterbox_model = chatterbox_model.to('cpu')
            
            # If model has submodules, move them too
            if hasattr(chatterbox_model, 'model'):
                if hasattr(chatterbox_model.model, 'to'):
                    chatterbox_model.model = chatterbox_model.model.to('cpu')
            
            if hasattr(chatterbox_model, 'tts'):
                if hasattr(chatterbox_model.tts, 'to'):
                    chatterbox_model.tts = chatterbox_model.tts.to('cpu')
        except Exception as e:
            logger.warning(f"⚠️ Could not move model to CPU: {e}")
        
        # Clear any cached voice embeddings
        if hasattr(chatterbox_model, 'clear_cache'):
            try:
                chatterbox_model.clear_cache()
                logger.info("🔓 Cleared model cache")
            except Exception as e:
                logger.warning(f"⚠️ Could not clear model cache: {e}")
        
        # Delete the model
        del chatterbox_model
        chatterbox_model = None
        
        # AGGRESSIVE garbage collection (multiple passes)
        logger.info("🔓 Running aggressive garbage collection...")
        for i in range(3):
            collected = gc.collect()
            logger.info(f"🔓 GC pass {i+1}: collected {collected} objects")
        
        # Clear CUDA cache if available (multiple times)
        if torch.cuda.is_available():
            logger.info("🔓 Clearing CUDA cache...")
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Wait for all operations to complete
            torch.cuda.empty_cache()  # Clear again after sync
            torch.cuda.ipc_collect()  # Collect IPC memory
            
            # Log VRAM after unload for all devices
            logger.info("🔓 VRAM after unload:")
            for i in range(torch.cuda.device_count()):
                vram_on_device = torch.cuda.memory_allocated(i) / 1024**3
                logger.info(f"🔓 GPU {i} VRAM after unload: {vram_on_device:.2f} GB")
            
            # Calculate freed VRAM on current device
            vram_after = torch.cuda.memory_allocated(current_device) / 1024**3
            vram_freed = vram_before - vram_after
            logger.info(f"✅ VRAM freed on GPU {current_device}: {vram_freed:.2f} GB")
        
        logger.info("✅ [Chatterbox] Model unloaded successfully, VRAM freed")
        return {
            "status": "success",
            "message": "Chatterbox model unloaded successfully",
            "vram_freed": f"{vram_freed:.2f}GB" if torch.cuda.is_available() else "Unknown"
        }
        
    except Exception as e:
        logger.error(f"❌ [Chatterbox] Error unloading model: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}


@app.post("/tts/reload-chatterbox")
async def reload_chatterbox():
    """Reload Chatterbox model for use"""
    global chatterbox_model, tts_initialized
    
    try:
        if chatterbox_model is not None:
            logger.info("🔄 [Chatterbox] Model is already loaded")
            return {
                "status": "success",
                "message": "Chatterbox model is already loaded",
                "already_loaded": True
            }
        
        logger.info("🔄 [Chatterbox] Reloading model...")
        
        # Load the model (will trigger full loading + warmup)
        chatterbox_model = load_chatterbox_model()
        
        if chatterbox_model is None:
            raise RuntimeError("Failed to load Chatterbox model")
        
        # Force warmup after reload
        logger.info("🔥 Forcing comprehensive warmup after reload...")
        try:
            import torch
            with torch.inference_mode():
                # Force T3 compilation with a test generation
                test_text = "This is a warmup test after reload for optimal performance."
                if hasattr(chatterbox_model, 'generate'):
                    logger.info("🔥 Warming up T3 compilation...")
                    dummy_audio = chatterbox_model.generate(test_text, language_id="en")
                    logger.info("✅ T3 warmup complete")
                    del dummy_audio
                
                # Clear warmup artifacts
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                logger.info("🎉 Reload warmup complete!")
                
        except Exception as e:
            logger.warning(f"⚠️ Reload warmup failed: {e}")
        
        logger.info("✅ [Chatterbox] Model reloaded successfully")
        return {
            "status": "success",
            "message": "Chatterbox model loaded and ready for use",
            "already_loaded": False
        }
        
    except Exception as e:
        logger.error(f"❌ [Chatterbox] Error reloading model: {e}")
        return {"status": "error", "message": str(e)}

@app.websocket("/tts-stream")
async def websocket_streaming_tts(websocket: WebSocket):
    """
    WebSocket endpoint for real-time streaming TTS
    Handles MULTIPLE message streams over a single connection
    
    Protocol for each message:
    1. Client sends settings: {"engine": "...", "voice": "...", ...}
    2. Client sends text chunks as they arrive
    3. Client sends "--END--" when done
    4. Server sends audio chunks as WAV bytes
    5. Loop back to step 1 for next message
    """
    await websocket.accept()
    logger.info("✅ [WebSocket] Connection accepted. Ready for multiple message streams.")
    streamer = None
    active_streamers = set() # Track all active streamers for this connection
    prefetched_message = None # buffer for message read during wait

    async def cancel_all_streamers():
        """Helper to forcefully cancel all known active streamers"""
        if not active_streamers:
            logger.info("🛑 [WebSocket] No active streamers in registry to cancel.")
            return
            
        logger.critical(f"🛑 [WebSocket] Force cancelling {len(active_streamers)} active streamers...")
        # Iterate over copy
        for s in list(active_streamers):
            try:
                if hasattr(s, 'cancel'):
                    await s.cancel() # This should be awaited!
            except Exception as e:
                logger.error(f"Error cancelling streamer: {e}")
        active_streamers.clear()
        
    try:
        # Primary loop - handles multiple message streams over single connection
        while True:
            # 1. Wait for settings for new message stream
            logger.info("👂 [WebSocket] Waiting for new message stream (expecting settings)...")
            
            try:
                if prefetched_message:
                    logger.info("📦 [WebSocket] Using prefetched message.")
                    settings_data = prefetched_message
                    prefetched_message = None
                else:
                    settings_data = await websocket.receive_text()
            except WebSocketDisconnect:
                logger.info("🔌 [WebSocket] Client disconnected.")
                break
                
            # Parse settings
            try:
                data = json.loads(settings_data)
                
                # CHECK FOR INTERRUPT FIRST - before treating as settings!
                if isinstance(data, dict) and data.get('type') == 'interrupt':
                    logger.critical(f"🛑 [WebSocket] INTERRUPT RECEIVED (Global Stop) - Data: {data}")
                    
                    # Cancel CURRENT streamer if any
                    if streamer: await streamer.cancel()
                    streamer = None
                    
                    # Cancel ALL tracked streamers (Zombies)
                    await cancel_all_streamers()
                    
                    # Don't start a new stream, just continue waiting
                    continue
                
                # Check if this is settings or a text chunk
                if isinstance(data, dict) and not data.get('text'):
                    tts_settings = data
                    logger.info(f"🔧 [WebSocket] Received settings for new stream: {tts_settings}")
                else:
                    logger.warning(f"⚠️ [WebSocket] Expected settings but got text. Skipping.")
                    continue
            except json.JSONDecodeError:
                logger.error(f"❌ [WebSocket] Invalid JSON: {settings_data[:100]}")
                continue
            
            # 2. Clean up any previous streamer ref (redundant safety)
            if streamer:
                await streamer.cancel()
            
            # 3. Create new streamer for this message
            session_id = str(uuid.uuid4())
            streamer = TTSStreamer(websocket, tts_settings)
            
            # CRITICAL: Add to registry immediately
            active_streamers.add(streamer) 
            logger.info(f"✅ [WebSocket] Created streamer for message {session_id} (Registry size: {len(active_streamers)})")
            
            # 4. Process text chunks for this message
            while True:
                try:
                    text = await websocket.receive_text()
                except WebSocketDisconnect:
                    logger.info("🔌 [WebSocket] Client disconnected during message.")
                    await cancel_all_streamers()
                    return
                
                # Check for control messages (JSON)
                try:
                    if text.startswith('{'):
                        control_msg = json.loads(text)
                        if control_msg.get('type') == 'interrupt':
                            logger.critical(f"🛑 [WebSocket] INTERRUPT RECEIVED inside message {session_id}")
                            await cancel_all_streamers()
                            streamer = None
                            break # Return to outer loop for next message
                except json.JSONDecodeError:
                    pass # Not JSON, treat as text
                    
                if text == "--END--":
                    logger.info(f"🏁 [WebSocket] End signal received for message {session_id}")
                    if streamer:
                        streamer.finish()
                    
                    # Wait for synthesis to complete BUT keep listening for interrupts/next messages
                    if streamer and hasattr(streamer, '_synthesis_task') and streamer._synthesis_task:
                        logger.info(f"⏳ [WebSocket] Waiting for synthesis completion or new message...")
                        
                        synthesis_task = streamer._synthesis_task
                        # Create receive task to listen for Interrupts/Settings while synthesizing
                        recv_task = asyncio.create_task(websocket.receive_text())
                        
                        try:
                            done, pending = await asyncio.wait(
                                [synthesis_task, recv_task], 
                                return_when=asyncio.FIRST_COMPLETED
                            )
                            
                            if recv_task in done:
                                # We received a message WHILE synthesizing
                                try:
                                    res_text = recv_task.result()
                                    
                                    # Check for interrupt
                                    try:
                                        msg_data = json.loads(res_text)
                                        if isinstance(msg_data, dict) and msg_data.get('type') == 'interrupt':
                                            logger.critical("🛑 [WebSocket] INTERRUPT received during synthesis wait!")
                                            await cancel_all_streamers()
                                            streamer = None
                                            break # Break inner loop
                                    except json.JSONDecodeError:
                                        pass
                                        
                                    # If not interrupt, it's the next message (Settings or Text? Protocol says Settings next)
                                    logger.info(f"⚡ [WebSocket] Received next message early: {res_text[:50]}...")
                                    
                                    # We must buffer this message for the start of the next outer loop
                                    # Hack: Push it back? Or use a standard variable.
                                    # We can't push back to websocket. Creates 'prefetched_message'
                                    prefetched_message = res_text
                                    
                                    # Since new message arrived, we assume we should Finish/Cancel current?
                                    # Standard behavior: If next message starts, current one finishes naturally? 
                                    # Or do we enforce sequentiality?
                                    # If we want to process next message, we must break inner loop.
                                    
                                    # BUT, if synthesis task is still running, do we cancel it?
                                    # If it's a new message, usually we want to finish speaking the old one?
                                    # But we are here because 'first_completed' returned recv_task. 
                                    # So synthesis is NOT done.
                                    
                                    # If it is NOT an interrupt, we probably should wait for synthesis to finish?
                                    # But then we turn into blocking wait.
                                    # Unless we check if it IS an interrupt.
                                    
                                    # Re-eval: Only INTERRUPT commands should stop us. 
                                    # Settings (new message) should probably wait?
                                    # But we can't un-read the socket.
                                    
                                    # Decision: If it's settings, we CANCEL current synthesis (fast switching) or Queue it?
                                    # Fast switching seems better for a chat app. "I changed my mind."
                                    # BUT usually user expects full answer.
                                    # Let's assume Cancel allows for "Interruption by new prompt".
                                    
                                    # However, simpler: Just break, and let outer loop handle 'prefetched_message'.
                                    # And ensure we cancel previous streamer if it's still running.
                                    
                                    # For now, just break.
                                    logger.info("⚡ [WebSocket] Breaking wait to process new message.")
                                    break
                                    
                                except Exception as e:
                                    logger.error(f"Error processing early message: {e}")
                                    break
                                    
                            if synthesis_task in done:
                                # Synthesis finished naturally.
                                # Check potential errors
                                try:
                                    synthesis_task.result() # Will raise if failed
                                except asyncio.CancelledError:
                                    logger.info("Synthesis task was cancelled.")
                                except Exception as e:
                                    logger.error(f"Synthesis task failed: {e}")
                                    
                                # Cancel the unused recv_task
                                recv_task.cancel()
                                try:
                                    await recv_task
                                except asyncio.CancelledError:
                                    pass
                        
                        except Exception as e:
                             logger.error(f"Error in dual-wait: {e}")

                    logger.info(f"✅ [WebSocket] Message {session_id} completed.")
                    # DO NOT REMOVE from active_streamers here. Let it fade or be removed by cancel_all.
                    # Or remove safely.
                    if streamer in active_streamers: active_streamers.remove(streamer)
                    streamer = None
                    break  # Break inner loop, continue outer loop for next message
                else:
                    # Process text chunk
                    if streamer:
                        await streamer.add_text(text)
                    
    except WebSocketDisconnect:
        logger.info("🔌 [WebSocket] Client disconnected.")
    except Exception as e:
        logger.error(f"❌ [WebSocket] Error in connection handler: {e}", exc_info=True)
    finally:
        await cancel_all_streamers()
        logger.info("👋 [WebSocket] Connection handler exiting.")

@app.post("/tts/upload-voice-reference")
async def upload_voice_reference(file: UploadFile = File(...)):
    """Upload a voice reference file for voice cloning"""
    try:
        # Validate file type
        allowed_extensions = {'.wav', '.mp3', '.flac', '.m4a'}
        file_extension = Path(file.filename).suffix.lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Create voice references directory if it doesn't exist
        voice_refs_dir = Path(__file__).parent.parent / "static" / "voice_references"
        voice_refs_dir.mkdir(parents=True, exist_ok=True)
        
        # Use original filename (like Chatterbox does)
        filename = file.filename
        file_path = voice_refs_dir / filename
        
        # Handle duplicate filenames by adding a number suffix
        counter = 1
        original_file_path = file_path
        while file_path.exists():
            name_parts = original_file_path.stem, counter, original_file_path.suffix
            filename = f"{name_parts[0]}_{name_parts[1]}{name_parts[2]}"
            file_path = voice_refs_dir / filename
            counter += 1
        
        # Save the file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"📁 Voice reference uploaded: {file.filename} -> {file_path}")
        
        return {
            "status": "success",
            "message": "Voice reference uploaded successfully",
            "file_path": str(file_path),
            "filename": filename,
            "original_name": file.filename
        }
        
    except Exception as e:
        logger.error(f"❌ Error uploading voice reference: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

if __name__ == "__main__":
    # Run the TTS service
    port = int(os.environ.get("TTS_PORT", 8002))
    host = os.environ.get("TTS_HOST", "0.0.0.0")
    
    logger.info(f"🚀 Starting TTS Backend on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")
