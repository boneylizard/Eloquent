// streaming-audio.js - Add this to your frontend

class StreamingAudioPlayer {
  constructor() {
    this.audioContext = null;
    this.audioQueue = [];
    this.isPlaying = false;
    this.currentSource = null;
    this.nextPlayTime = 0;
    this.websocket = null;
    this.sessionId = null;
  }

  async initialize() {
    // Initialize Web Audio API
    if (!this.audioContext) {
      this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }
    
    // Resume context if suspended (Chrome autoplay policy)
    if (this.audioContext.state === 'suspended') {
      await this.audioContext.resume();
    }
  }

  async connectStreaming(serverUrl = 'ws://localhost:8000/ws/stream-tts') {
    if (this.websocket) {
      this.websocket.close();
    }

    return new Promise((resolve, reject) => {
      this.websocket = new WebSocket(serverUrl);
      
      this.websocket.onopen = () => {
        console.log('🔊 Connected to streaming TTS server');
        resolve();
      };
      
      this.websocket.onmessage = async (event) => {
        const message = JSON.parse(event.data);
        await this.handleStreamingMessage(message);
      };
      
      this.websocket.onerror = (error) => {
        console.error('🔊 WebSocket error:', error);
        reject(error);
      };
      
      this.websocket.onclose = () => {
        console.log('🔊 Streaming TTS connection closed');
        this.cleanup();
      };
    });
  }

  async handleStreamingMessage(message) {
    switch (message.type) {
      case 'started':
        this.sessionId = message.session_id;
        console.log(`🔊 Streaming session started: ${this.sessionId}`);
        break;
        
      case 'audio_chunk':
        await this.queueAudioChunk(message.data);
        break;
        
      case 'complete':
        console.log(`🔊 Streaming complete for session: ${this.sessionId}`);
        this.sessionId = null;
        break;
        
      case 'error':
        console.error('🔊 Streaming error:', message.message);
        break;
    }
  }

  async queueAudioChunk(base64AudioData) {
    try {
      // Decode base64 audio data
      const audioData = this.base64ToArrayBuffer(base64AudioData);
      
      // Decode audio
      const audioBuffer = await this.audioContext.decodeAudioData(audioData);
      
      // Add to queue
      this.audioQueue.push(audioBuffer);
      
      // Start playback if not already playing
      if (!this.isPlaying) {
        this.startPlayback();
      }
      
    } catch (error) {
      console.error('🔊 Error processing audio chunk:', error);
    }
  }

  base64ToArrayBuffer(base64) {
    const binaryString = window.atob(base64);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }
    return bytes.buffer;
  }

  startPlayback() {
    if (this.isPlaying || this.audioQueue.length === 0) return;
    
    this.isPlaying = true;
    this.nextPlayTime = this.audioContext.currentTime;
    this.playNextChunk();
  }

  playNextChunk() {
    if (this.audioQueue.length === 0) {
      this.isPlaying = false;
      return;
    }

    const audioBuffer = this.audioQueue.shift();
    const source = this.audioContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(this.audioContext.destination);
    
    // Schedule playback
    source.start(this.nextPlayTime);
    this.nextPlayTime += audioBuffer.duration;
    
    // Set up next chunk
    source.onended = () => {
      this.playNextChunk();
    };
    
    this.currentSource = source;
  }

  async startStreamingTTS(prompt, options = {}) {
    await this.initialize();
    
    if (!this.websocket || this.websocket.readyState !== WebSocket.OPEN) {
      throw new Error('WebSocket not connected');
    }

    const message = {
      type: 'start',
      prompt: prompt,
      voice_reference: options.voiceReference,
      stream_llm: options.streamLLM !== false // Default to true
    };

    this.websocket.send(JSON.stringify(message));
  }

  stop() {
    // Stop current playback
    if (this.currentSource) {
      this.currentSource.stop();
      this.currentSource = null;
    }
    
    // Clear queue
    this.audioQueue = [];
    this.isPlaying = false;
    
    // Send interrupt to server
    if (this.websocket && this.sessionId) {
      this.websocket.send(JSON.stringify({
        type: 'interrupt',
        session_id: this.sessionId
      }));
    }
  }

  cleanup() {
    this.stop();
    if (this.websocket) {
      this.websocket.close();
      this.websocket = null;
    }
    this.sessionId = null;
  }
}

// Enhanced Chat Component for Streaming TTS
class StreamingChatHandler {
  constructor() {
    this.audioPlayer = new StreamingAudioPlayer();
    this.isStreaming = false;
    this.currentMessageElement = null;
  }

  async initialize() {
    await this.audioPlayer.initialize();
    await this.audioPlayer.connectStreaming();
  }

  async sendMessage(message, options = {}) {
    if (this.isStreaming) {
      this.audioPlayer.stop();
    }

    this.isStreaming = true;
    
    try {
      // Create message element in chat
      this.currentMessageElement = this.createMessageElement();
      
      // Start streaming TTS
      await this.audioPlayer.startStreamingTTS(message, {
        voiceReference: options.voiceReference,
        streamLLM: options.enableStreaming
      });
      
    } catch (error) {
      console.error('🔊 Error sending streaming message:', error);
      this.isStreaming = false;
    }
  }

  createMessageElement() {
    // Create new message bubble in your chat interface
    const messageElement = document.createElement('div');
    messageElement.className = 'message assistant-message streaming';
    messageElement.innerHTML = `
      <div class="message-content">
        <div class="streaming-indicator">🔊 Generating response...</div>
        <div class="message-text"></div>
      </div>
      <button class="stop-streaming" onclick="this.stopStreaming()">Stop</button>
    `;
    
    // Add to chat container (adjust selector for your UI)
    const chatContainer = document.querySelector('.chat-messages');
    chatContainer.appendChild(messageElement);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    
    return messageElement;
  }

  stopStreaming() {
    this.audioPlayer.stop();
    this.isStreaming = false;
    
    if (this.currentMessageElement) {
      this.currentMessageElement.classList.remove('streaming');
      const indicator = this.currentMessageElement.querySelector('.streaming-indicator');
      if (indicator) indicator.remove();
    }
  }
}

// Integration with your existing chat interface
const streamingChat = new StreamingChatHandler();

// Initialize when page loads
document.addEventListener('DOMContentLoaded', async () => {
  try {
    await streamingChat.initialize();
    console.log('🔊 Streaming TTS initialized successfully');
  } catch (error) {
    console.error('🔊 Failed to initialize streaming TTS:', error);
    // Fall back to your existing TTS method
  }
});

// Enhanced version of your existing sendMessage function
async function sendMessageWithStreaming(message, useStreaming = true) {
  const ttsEnabled = document.getElementById('tts-enabled')?.checked || false;
  const voiceReference = document.getElementById('voice-reference-file')?.files[0];
  
  if (ttsEnabled && useStreaming) {
    // Use streaming TTS
    await streamingChat.sendMessage(message, {
      voiceReference: voiceReference,
      enableStreaming: true
    });
  } else {
    // Fall back to your existing method
    await sendMessageOriginal(message);
  }
}

// Alternative: Replace your existing WebSocket chat with streaming version
class EnhancedWebSocketChat {
  constructor() {
    this.websocket = null;
    this.audioPlayer = new StreamingAudioPlayer();
  }

  async connect() {
    await this.audioPlayer.initialize();
    
    this.websocket = new WebSocket('ws://localhost:8000/ws/chat-stream');
    
    this.websocket.onopen = () => {
      console.log('🔊 Connected to enhanced chat stream');
    };
    
    this.websocket.onmessage = async (event) => {
      const message = JSON.parse(event.data);
      await this.handleChatMessage(message);
    };
    
    this.websocket.onerror = (error) => {
      console.error('🔊 Enhanced chat error:', error);
    };
  }

  async handleChatMessage(message) {
    switch (message.type) {
      case 'response_chunk':
        // Update text display
        this.updateMessageText(message.text);
        
        // Play audio chunk
        if (message.audio) {
          await this.audioPlayer.queueAudioChunk(message.audio);
        }
        break;
        
      case 'response_complete':
        // Handle complete text-only response
        this.displayCompleteMessage(message.text);
        break;
        
      case 'interrupted':
        console.log('🔊 Response interrupted');
        break;
    }
  }

  async sendMessage(message, ttsEnabled = true, voiceReference = null) {
    if (!this.websocket || this.websocket.readyState !== WebSocket.OPEN) {
      await this.connect();
    }

    const chatMessage = {
      type: 'chat',
      message: message,
      tts_enabled: ttsEnabled,
      voice_reference: voiceReference
    };

    this.websocket.send(JSON.stringify(chatMessage));
  }

  interrupt() {
    if (this.websocket) {
      this.websocket.send(JSON.stringify({ type: 'interrupt' }));
    }
    this.audioPlayer.stop();
  }

  updateMessageText(text) {
    // Update the current message element with new text
    if (this.currentMessageElement) {
      const textElement = this.currentMessageElement.querySelector('.message-text');
      if (textElement) {
        textElement.textContent = text;
      }
    }
  }

  displayCompleteMessage(text) {
    // Display complete message (for text-only mode)
    const messageElement = this.createMessageElement();
    messageElement.querySelector('.message-text').textContent = text;
  }
}

// Export for use in other files
window.StreamingAudioPlayer = StreamingAudioPlayer;
window.StreamingChatHandler = StreamingChatHandler;
window.EnhancedWebSocketChat = EnhancedWebSocketChat;