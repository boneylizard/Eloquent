import { eventSource, event_types, saveSettingsDebounced } from '/script.js';
import { extension_settings, getContext } from '/scripts/extensions.js';

const MODULE_NAME = 'mirid_bridge';
const DEFAULT_SETTINGS = {
    baseUrl: 'http://127.0.0.1:8000',
    password: '',
    autoSpeak: false,
    ttsEngine: 'kokoro',
    ttsVoice: 'af_heart',
    sttEngine: 'whisper',
};

let activeAudio = null;
let recorder = null;
let recordingChunks = [];

function settings() {
    extension_settings[MODULE_NAME] ??= structuredClone(DEFAULT_SETTINGS);
    return extension_settings[MODULE_NAME];
}

function baseUrl() {
    return String(settings().baseUrl || DEFAULT_SETTINGS.baseUrl).trim().replace(/\/+$/, '');
}

function authHeaders(extra = {}) {
    const password = String(settings().password || '').trim();
    return password ? { ...extra, Authorization: `Bearer ${password}` } : extra;
}

async function miridFetch(path, init = {}) {
    const response = await fetch(`${baseUrl()}${path}`, {
        ...init,
        headers: authHeaders(init.headers || {}),
    });
    if (!response.ok) {
        let message = `${response.status} ${response.statusText}`;
        try {
            const body = await response.json();
            message = body.detail || body.message || message;
        } catch {
            // Keep the HTTP status when the body is not JSON.
        }
        throw new Error(message);
    }
    return response;
}

function notify(kind, message) {
    const toast = globalThis.toastr?.[kind];
    if (typeof toast === 'function') toast(message, 'Mirid');
}

function cleanSpeechText(text) {
    return String(text || '')
        .replace(/```[\s\S]*?```/g, ' ')
        .replace(/<[^>]+>/g, ' ')
        .replace(/!\[[^\]]*\]\([^)]*\)/g, ' ')
        .replace(/\[([^\]]+)\]\([^)]*\)/g, '$1')
        .replace(/[*_~#>`]/g, '')
        .replace(/\s+/g, ' ')
        .trim();
}

async function speak(text) {
    const input = cleanSpeechText(text);
    if (!input) return;

    const response = await miridFetch('/v1/audio/speech', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            input,
            model: settings().ttsEngine,
            engine: settings().ttsEngine,
            voice: settings().ttsVoice,
            response_format: 'wav',
        }),
    });
    const url = URL.createObjectURL(await response.blob());
    if (activeAudio) {
        activeAudio.pause();
        URL.revokeObjectURL(activeAudio.src);
    }
    activeAudio = new Audio(url);
    activeAudio.addEventListener('ended', () => URL.revokeObjectURL(url), { once: true });
    await activeAudio.play();
}

async function handleAssistantMessage(messageId) {
    if (!settings().autoSpeak) return;
    const message = getContext().chat?.[messageId];
    if (!message || message.is_user || message.is_system) return;
    try {
        await speak(message.extra?.display_text || message.mes);
    } catch (error) {
        notify('error', `Mirid could not play this reply. ${error.message}`);
    }
}

async function refreshVoices() {
    const select = document.querySelector('#mirid_bridge_voice');
    if (!select) return;
    const [voiceResponse, nanoResponse, sttResponse] = await Promise.all([
        miridFetch('/tts/voices'),
        miridFetch('/tts/nanogpt-models').catch(() => null),
        miridFetch('/stt/available-engines').catch(() => null),
    ]);
    const data = await voiceResponse.json();
    const nanoData = nanoResponse ? await nanoResponse.json() : {};
    const voices = [...(data.kokoro_voices || []), ...(data.chatterbox_voices || [])];
    for (const model of nanoData.models || []) {
        for (const voice of model.voices || []) {
            voices.push({
                id: voice,
                name: `${voice} — NanoGPT ${model.name || model.id}`,
                engine: `nanogpt-${model.id}`,
            });
        }
    }
    select.replaceChildren(...voices.map((voice) => {
        const option = document.createElement('option');
        option.value = `${voice.engine}::${voice.id}`;
        option.textContent = `${voice.name || voice.id} — ${voice.engine}`;
        option.dataset.engine = voice.engine;
        option.dataset.voice = voice.id;
        return option;
    }));
    const configured = voices.find((voice) => (
        voice.id === settings().ttsVoice && voice.engine === settings().ttsEngine
    ));
    if (configured) select.value = `${configured.engine}::${configured.id}`;
    if (sttResponse) {
        const sttData = await sttResponse.json();
        const sttSelect = document.querySelector('#mirid_bridge_stt');
        const engines = sttData.available_engines || [];
        sttSelect.replaceChildren(...engines.map((engine) => {
            const option = document.createElement('option');
            option.value = engine;
            option.textContent = engine.startsWith('nanogpt-') ? `${engine.slice(8)} — NanoGPT` : engine;
            return option;
        }));
        if (engines.includes(settings().sttEngine)) sttSelect.value = settings().sttEngine;
    }
    notify('success', `Found ${voices.length} voices across Mirid's local and NanoGPT engines.`);
}

async function testConnection() {
    const status = document.querySelector('#mirid_bridge_status');
    status.textContent = 'Listening for Mirid…';
    status.dataset.state = 'checking';
    try {
        const response = await miridFetch('/integrations/sillytavern/capabilities');
        const data = await response.json();
        status.textContent = data.images?.available
            ? 'Connected. Text, voice, transcription, and local images are available.'
            : 'Connected. Text, voice, and transcription are available; the local image engine is offline.';
        status.dataset.state = 'ready';
    } catch (error) {
        status.textContent = `No answer from Mirid. ${error.message}`;
        status.dataset.state = 'error';
    }
}

function writeTranscript(text) {
    const textarea = document.querySelector('#send_textarea');
    if (!textarea) return;
    const spacer = textarea.value && !/\s$/.test(textarea.value) ? ' ' : '';
    textarea.value = `${textarea.value}${spacer}${text}`;
    textarea.dispatchEvent(new Event('input', { bubbles: true }));
    textarea.focus();
}

async function transcribeRecording(blob) {
    const form = new FormData();
    form.append('file', blob, 'sillytavern-recording.webm');
    form.append('model', settings().sttEngine || 'whisper');
    form.append('engine', settings().sttEngine || 'whisper');
    const response = await miridFetch('/v1/audio/transcriptions', { method: 'POST', body: form });
    const data = await response.json();
    writeTranscript(data.text || data.transcript || '');
}

async function toggleRecording(button) {
    if (recorder?.state === 'recording') {
        recorder.stop();
        button.classList.remove('mirid-recording');
        button.title = 'Dictate with Mirid';
        return;
    }
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        recordingChunks = [];
        recorder = new MediaRecorder(stream);
        recorder.addEventListener('dataavailable', (event) => {
            if (event.data.size) recordingChunks.push(event.data);
        });
        recorder.addEventListener('stop', async () => {
            stream.getTracks().forEach((track) => track.stop());
            try {
                await transcribeRecording(new Blob(recordingChunks, { type: recorder.mimeType || 'audio/webm' }));
            } catch (error) {
                notify('error', `Mirid could not transcribe that recording. ${error.message}`);
            }
        });
        recorder.start();
        button.classList.add('mirid-recording');
        button.title = 'Stop and transcribe';
    } catch (error) {
        notify('error', `The microphone could not be opened. ${error.message}`);
    }
}

function addMicrophoneButton() {
    if (document.querySelector('#mirid_bridge_mic') || !navigator.mediaDevices?.getUserMedia) return;
    const button = document.createElement('button');
    button.id = 'mirid_bridge_mic';
    button.type = 'button';
    button.className = 'menu_button fa-solid fa-microphone';
    button.title = 'Dictate with Mirid';
    button.setAttribute('aria-label', 'Dictate with Mirid');
    button.addEventListener('click', () => toggleRecording(button));
    document.querySelector('#send_but')?.before(button);
}

function settingRow(label, control, description) {
    return `<label class="mirid-bridge-row"><span>${label}</span>${control}<small>${description}</small></label>`;
}

function updateConnectionAddresses(panel) {
    panel.querySelector('#mirid_bridge_text_url').textContent = `${baseUrl()}/v1`;
    panel.querySelector('#mirid_bridge_image_url').textContent = baseUrl();
}

function applySettingsToPanel(panel) {
    const current = settings();
    panel.querySelector('#mirid_bridge_url').value = current.baseUrl;
    panel.querySelector('#mirid_bridge_password').value = current.password;
    panel.querySelector('#mirid_bridge_engine').value = current.ttsEngine;
    panel.querySelector('#mirid_bridge_voice').value = `${current.ttsEngine}::${current.ttsVoice}`;
    panel.querySelector('#mirid_bridge_auto_speak').checked = current.autoSpeak;
    panel.querySelector('#mirid_bridge_stt').value = current.sttEngine;
    updateConnectionAddresses(panel);
}

function renderSettings() {
    const panel = document.createElement('div');
    panel.id = 'mirid_bridge_settings';
    panel.className = 'mirid-bridge-panel';
    panel.innerHTML = `
        <div class="inline-drawer">
            <div class="inline-drawer-toggle inline-drawer-header">
                <b>Mirid AI Backend</b>
                <div class="inline-drawer-icon fa-solid fa-circle-chevron-down down"></div>
            </div>
            <div class="inline-drawer-content">
                <p><a href="https://mirid.ai" target="_blank" rel="noopener noreferrer">Mirid</a> is a Windows desktop app for downloading, running, and talking to AI models. It can run GGUF models locally or connect to hosted AI providers.</p>
                <p>This extension lets SillyTavern use a running Mirid app for chat, character voices, speech recognition, and image generation. It does not install Mirid or models.</p>
                <p><a href="https://github.com/boneylizard/Eloquent/releases/latest" target="_blank" rel="noopener noreferrer">Download Mirid for Windows</a>, open it, then connect below.</p>
                ${settingRow('Mirid address', '<input id="mirid_bridge_url" class="text_pole" type="url">', 'Use the address shown by Mirid. Local installs usually use port 8000.')}
                ${settingRow('Remote-access password', '<input id="mirid_bridge_password" class="text_pole" type="password" autocomplete="off">', 'Leave blank on localhost unless Mirid has a password.')}
                <div class="mirid-bridge-actions">
                    <button id="mirid_bridge_test" class="menu_button">Test connection</button>
                    <button id="mirid_bridge_refresh_voices" class="menu_button">Find voices</button>
                </div>
                <p id="mirid_bridge_status" class="mirid-bridge-status">Connection not tested.</p>
                ${settingRow('Voice engine', '<input id="mirid_bridge_engine" class="text_pole" type="text">', 'Kokoro is the lightest local default. Other installed Mirid engines also work.')}
                ${settingRow('Voice', '<select id="mirid_bridge_voice" class="text_pole"><option value="kokoro::af_heart" data-engine="kokoro" data-voice="af_heart">af_heart — kokoro</option></select>', 'Refresh the list after Mirid connects.')}
                <label class="checkbox_label"><input id="mirid_bridge_auto_speak" type="checkbox"><span>Speak new character replies automatically</span></label>
                ${settingRow('Transcription engine', '<select id="mirid_bridge_stt" class="text_pole"><option value="whisper">whisper</option></select>', 'The microphone button places Mirid\'s transcript in the message box before you send it. NanoGPT engines appear when configured in Mirid.')}
                <details class="mirid-bridge-setup">
                    <summary>Connect SillyTavern's text and image tools</summary>
                    <p><b>Text:</b> choose a Custom OpenAI-compatible Chat Completion source. Set its API URL to <code id="mirid_bridge_text_url"></code>, then choose a model returned by <code>/v1/models</code>.</p>
                    <p><b>Images:</b> choose the Automatic1111 source in Image Generation and set its URL to <code id="mirid_bridge_image_url"></code>. Mirid translates that protocol to its local stable-diffusion.cpp engine.</p>
                    <p>If Mirid remote access has a password, use the same value as the API key or HTTP password.</p>
                </details>
            </div>
        </div>`;
    document.querySelector('#extensions_settings')?.append(panel);

    applySettingsToPanel(panel);

    const save = () => {
        const current = settings();
        current.baseUrl = panel.querySelector('#mirid_bridge_url').value.trim();
        current.password = panel.querySelector('#mirid_bridge_password').value;
        current.ttsEngine = panel.querySelector('#mirid_bridge_engine').value.trim() || 'kokoro';
        current.ttsVoice = panel.querySelector('#mirid_bridge_voice').selectedOptions[0]?.dataset.voice || 'af_heart';
        current.autoSpeak = panel.querySelector('#mirid_bridge_auto_speak').checked;
        current.sttEngine = panel.querySelector('#mirid_bridge_stt').value.trim() || 'whisper';
        updateConnectionAddresses(panel);
        saveSettingsDebounced();
    };
    panel.querySelectorAll('input, select').forEach((control) => control.addEventListener('change', save));
    panel.querySelector('#mirid_bridge_url').addEventListener('input', save);
    panel.querySelector('#mirid_bridge_test').addEventListener('click', testConnection);
    panel.querySelector('#mirid_bridge_refresh_voices').addEventListener('click', () => refreshVoices().catch((error) => notify('error', error.message)));
    panel.querySelector('#mirid_bridge_voice').addEventListener('change', (event) => {
        const engine = event.target.selectedOptions[0]?.dataset.engine;
        if (engine) panel.querySelector('#mirid_bridge_engine').value = engine;
        save();
    });
    eventSource.on(event_types.EXTENSION_SETTINGS_LOADED, () => applySettingsToPanel(panel));
}

jQuery(async () => {
    settings();
    renderSettings();
    addMicrophoneButton();
    eventSource.makeLast(event_types.CHARACTER_MESSAGE_RENDERED, handleAssistantMessage);
});
