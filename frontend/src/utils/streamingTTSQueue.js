// streamingTTSQueue.js - A completely new, stable, and correct implementation
import { synthesizeSpeech } from './apiCall.js';

class StreamingTTSQueue {
  constructor() {
    this.textBuffer = '';
    this.audioQueue = [];
    this.isPlaying = false;
    this.sessionActive = false;
    this.chunkTimer = null;
    this.currentAudioContext = null;
    this.settings = null;
    this.CHUNK_INTERVAL = 1000; // Check for sentences every 1 second
  }

  initialize(ttsSettings) {
    this.settings = ttsSettings;
  }

  start() {
    if (this.sessionActive) return;
    this.resetState();
    this.sessionActive = true;
    this.startChunkTimer();
  }

  add(newText) {
    if (!this.sessionActive) return;
    this.textBuffer += newText;
  }

  finish() {
    if (!this.sessionActive) return;
    clearTimeout(this.chunkTimer);
    if (this.textBuffer.trim().length > 0) {
      this.synthesizeAndQueue(this.textBuffer.trim());
    }
    this.sessionActive = false;
  }

  stop() {
    this.sessionActive = false;
    clearTimeout(this.chunkTimer);
    if (this.currentAudioContext && this.currentAudioContext.source) {
      try { this.currentAudioContext.source.stop(); } catch (e) {}
      try { this.currentAudioContext.ctx.close(); } catch (e) {}
    }
    this.resetState();
  }

  resetState() {
    this.textBuffer = '';
    this.audioQueue = [];
    this.isPlaying = false;
    this.currentAudioContext = null;
  }

  startChunkTimer() {
    this.chunkTimer = setTimeout(() => {
      if (!this.sessionActive) return;
      this.processBuffer();
      this.startChunkTimer();
    }, this.CHUNK_INTERVAL);
  }

  // --- THIS IS THE NEW, CORRECTED CHUNKING LOGIC ---
  processBuffer() {
    // Find the position of the FIRST sentence-ending punctuation mark.
    const sentenceEndRegex = /[.!?]/;
    const match = this.textBuffer.match(sentenceEndRegex);

    // If we found a punctuation mark
    if (match && match.index > 0) {
      const splitPoint = match.index + 1;
      
      // The chunk to process is everything from the beginning to that point.
      const chunkToProcess = this.textBuffer.substring(0, splitPoint).trim();
      
      // CRITICALLY: Remove the processed chunk from the start of the buffer.
      this.textBuffer = this.textBuffer.substring(splitPoint);

      if (chunkToProcess) {
        this.synthesizeAndQueue(chunkToProcess);
      }
    }
  }

  async synthesizeAndQueue(text) {
    try {
      const audioUrl = await synthesizeSpeech(text, {
        voice: this.settings.ttsVoice || 'af_heart',
        engine: this.settings.ttsEngine || 'kokoro',
      });
      if (audioUrl) {
        this.audioQueue.push(audioUrl);
        if (!this.isPlaying) {
          this.playNextInQueue();
        }
      }
    } catch (error) {
      console.error('❌ [Queue] Synthesis Error:', error);
    }
  }

  async playNextInQueue() {
    if (this.audioQueue.length === 0) {
      this.isPlaying = false;
      return;
    }
    this.isPlaying = true;
    const audioUrl = this.audioQueue.shift();
    try {
      const resp = await fetch(audioUrl);
      const arrayBuf = await resp.arrayBuffer();
      const ctx = new AudioContext();
      const audioBuffer = await ctx.decodeAudioData(arrayBuf);
      const source = ctx.createBufferSource();
      source.buffer = audioBuffer;
      source.playbackRate.value = this.settings.ttsSpeed || 1.0;
      source.detune.value = (this.settings.ttsPitch || 0) * 100;
      source.connect(ctx.destination);
      this.currentAudioContext = { ctx, source };
      source.onended = () => {
        ctx.close().catch(e => {});
        URL.revokeObjectURL(audioUrl);
        this.playNextInQueue();
      };
      source.start();
    } catch (error) {
      console.error("❌ [Queue] Playback Error:", error);
      URL.revokeObjectURL(audioUrl);
      this.isPlaying = false;
      this.playNextInQueue();
    }
  }
}

// --- Exports ---
export const ttsQueue = new StreamingTTSQueue();
export function startStreamingTTS(ttsSettings) { ttsQueue.initialize(ttsSettings); ttsQueue.start(); }
export function addStreamingText(text) { ttsQueue.add(text); }
export function endStreamingTTS() { ttsQueue.finish(); }
export function stopStreamingTTS() { ttsQueue.stop(); }