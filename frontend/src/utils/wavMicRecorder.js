/**
 * Capture mic audio as 16 kHz mono WAV in the browser.
 * Avoids server-side WebM decode (librosa/audioread/ffmpeg) for instant STT.
 */

function mergeFloat32Arrays(arrays) {
  const total = arrays.reduce((n, a) => n + a.length, 0);
  const out = new Float32Array(total);
  let offset = 0;
  for (const arr of arrays) {
    out.set(arr, offset);
    offset += arr.length;
  }
  return out;
}

async function resampleTo16k(samples, fromRate) {
  if (fromRate === 16000) return samples;
  const duration = samples.length / fromRate;
  const length = Math.max(1, Math.ceil(duration * 16000));
  const ctx = new OfflineAudioContext(1, length, 16000);
  const buffer = ctx.createBuffer(1, samples.length, fromRate);
  buffer.copyToChannel(samples, 0);
  const src = ctx.createBufferSource();
  src.buffer = buffer;
  src.connect(ctx.destination);
  src.start(0);
  const rendered = await ctx.startRendering();
  return rendered.getChannelData(0);
}

function encodeWavPcm16(float32, sampleRate) {
  const numSamples = float32.length;
  const buffer = new ArrayBuffer(44 + numSamples * 2);
  const view = new DataView(buffer);

  const writeString = (offset, str) => {
    for (let i = 0; i < str.length; i += 1) {
      view.setUint8(offset + i, str.charCodeAt(i));
    }
  };

  writeString(0, 'RIFF');
  view.setUint32(4, 36 + numSamples * 2, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, 'data');
  view.setUint32(40, numSamples * 2, true);

  let offset = 44;
  for (let i = 0; i < numSamples; i += 1) {
    const s = Math.max(-1, Math.min(1, float32[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    offset += 2;
  }
  return buffer;
}

export function createWavMicRecorder() {
  let audioContext = null;
  let mediaStream = null;
  let processor = null;
  let source = null;
  const chunks = [];

  const teardownTracks = () => {
    if (mediaStream) {
      mediaStream.getTracks().forEach((t) => t.stop());
      mediaStream = null;
    }
  };

  return {
    async start() {
      teardownTracks();
      chunks.length = 0;
      mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const Ctx = window.AudioContext || window.webkitAudioContext;
      audioContext = new Ctx();
      source = audioContext.createMediaStreamSource(mediaStream);
      processor = audioContext.createScriptProcessor(4096, 1, 1);
      processor.onaudioprocess = (event) => {
        chunks.push(new Float32Array(event.inputBuffer.getChannelData(0)));
      };
      source.connect(processor);
      processor.connect(audioContext.destination);
    },

    async stop() {
      if (processor) {
        processor.onaudioprocess = null;
        processor.disconnect();
        processor = null;
      }
      if (source) {
        source.disconnect();
        source = null;
      }
      const inputRate = audioContext?.sampleRate || 48000;
      teardownTracks();
      const merged = mergeFloat32Arrays(chunks);
      chunks.length = 0;
      const samples16k = await resampleTo16k(merged, inputRate);
      if (audioContext) {
        await audioContext.close();
        audioContext = null;
      }
      return new Blob([encodeWavPcm16(samples16k, 16000)], { type: 'audio/wav' });
    },

    cancel() {
      if (processor) {
        processor.onaudioprocess = null;
        processor.disconnect();
        processor = null;
      }
      if (source) {
        source.disconnect();
        source = null;
      }
      chunks.length = 0;
      teardownTracks();
      if (audioContext) {
        audioContext.close().catch(() => {});
        audioContext = null;
      }
    },
  };
}
