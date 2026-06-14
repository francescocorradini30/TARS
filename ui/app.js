// ── State ─────────────────────────────────────────────────────────────────────
const BARS = 12;
let isProcessing = false;
let isRecording = false;

let streamingContent = null;
const audioQueue = [];
let isPlayingQueue = false;
let streamComplete = false;

// ── Init ──────────────────────────────────────────────────────────────────────
function initBars() {
    const wf = document.getElementById('waveform');
    for (let i = 0; i < BARS; i++) {
        const b = document.createElement('div');
        b.className = 'bar';
        wf.appendChild(b);
    }
}

// ── Status ────────────────────────────────────────────────────────────────────
const STATUS_LABELS = {
    idle:         'STANDBY',
    listening:    'LISTENING',
    transcribing: 'TRANSCRIBING',
    thinking:     'PROCESSING',
    speaking:     'RESPONDING',
};

function setStatus(key) {
    document.getElementById('tarsBody').className = `tars-body ${key === 'idle' ? '' : key}`.trim();
    document.getElementById('statusText').textContent = STATUS_LABELS[key] ?? key.toUpperCase();
}

// ── Chat log ──────────────────────────────────────────────────────────────────
function addMessage(role, text) {
    const log = document.getElementById('chatLog');
    const msg = document.createElement('div');
    msg.className = `message ${role}`;
    msg.innerHTML = `<div class="sender">${role === 'user' ? 'YOU' : 'TARS'}</div>
                     <div class="content">${escHtml(text)}</div>`;
    log.appendChild(msg);
    log.scrollTop = log.scrollHeight;
}

function escHtml(s) {
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// ── Streaming text (called from Python via evaluate_js) ───────────────────────
function startTarsStream() {
    const log = document.getElementById('chatLog');
    const msg = document.createElement('div');
    msg.className = 'message tars';
    const sender = document.createElement('div');
    sender.className = 'sender';
    sender.textContent = 'TARS';
    streamingContent = document.createElement('div');
    streamingContent.className = 'content';
    msg.appendChild(sender);
    msg.appendChild(streamingContent);
    log.appendChild(msg);
    log.scrollTop = log.scrollHeight;
}

function appendTarsText(chunk) {
    if (!streamingContent) return;
    streamingContent.textContent += chunk;
    document.getElementById('chatLog').scrollTop = document.getElementById('chatLog').scrollHeight;
}

function endTarsStream() {
    streamingContent = null;
    streamComplete = true;
    if (!isPlayingQueue && audioQueue.length === 0) {
        finishProcessing();
    }
}

// ── Called from Python after successful transcription ─────────────────────────
function onTranscript(text) {
    addMessage('user', text);
    setStatus('thinking');
}

// ── Called from Python when transcription is empty ────────────────────────────
function resetProcessing() {
    setStatus('idle');
    isProcessing = false;
}

// ── Audio queue (called from Python via evaluate_js) ──────────────────────────
function queueAudio(b64) {
    audioQueue.push(b64);
    if (!isPlayingQueue) {
        processQueue();
    }
}

async function processQueue() {
    isPlayingQueue = true;
    while (audioQueue.length > 0) {
        setStatus('speaking');
        await playAudio(audioQueue.shift());
    }
    isPlayingQueue = false;
    if (streamComplete) {
        finishProcessing();
    }
}

function finishProcessing() {
    streamComplete = false;
    setStatus('idle');
    isProcessing = false;
}

// ── Audio playback ────────────────────────────────────────────────────────────
async function playAudio(b64) {
    const binary = atob(b64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    const blob = new Blob([bytes], { type: 'audio/mp3' });
    const url = URL.createObjectURL(blob);
    const audio = new Audio(url);
    return new Promise(resolve => {
        audio.onended = () => { URL.revokeObjectURL(url); resolve(); };
        audio.onerror = () => { URL.revokeObjectURL(url); resolve(); };
        audio.play().catch(resolve);
    });
}

// ── Microphone (native via Python — no browser permission needed) ─────────────
async function startVoice() {
    if (isProcessing) return;
    isProcessing = true;
    isRecording = true;
    setStatus('listening');
    document.getElementById('micBtn').classList.add('active');
    await window.pywebview.api.start_recording();
}

async function stopVoice() {
    if (!isRecording) return;
    isRecording = false;
    document.getElementById('micBtn').classList.remove('active');
    setStatus('transcribing');
    await window.pywebview.api.stop_and_process();
}

// ── Text input ────────────────────────────────────────────────────────────────
async function sendText() {
    if (isProcessing) return;
    const input = document.getElementById('textInput');
    const text = input.value.trim();
    if (!text) return;
    input.value = '';
    isProcessing = true;
    addMessage('user', text);
    setStatus('thinking');
    await window.pywebview.api.send_message(text);
}

// ── Reset ─────────────────────────────────────────────────────────────────────
async function resetConversation() {
    if (isProcessing) return;
    audioQueue.length = 0;
    isPlayingQueue = false;
    streamComplete = false;
    streamingContent = null;
    await window.pywebview.api.reset();
    document.getElementById('chatLog').innerHTML = '';
    addMessage('tars', 'Memory wiped. Systems nominal.');
    setStatus('idle');
}

// ── Bootstrap ─────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initBars();

    const mic = document.getElementById('micBtn');
    mic.addEventListener('mousedown',  startVoice);
    mic.addEventListener('mouseup',    stopVoice);
    mic.addEventListener('mouseleave', stopVoice);
    mic.addEventListener('touchstart', e => { e.preventDefault(); startVoice(); }, { passive: false });
    mic.addEventListener('touchend',   stopVoice);

    document.getElementById('sendBtn').addEventListener('click', sendText);
    document.getElementById('resetBtn').addEventListener('click', resetConversation);

    document.getElementById('textInput').addEventListener('keydown', e => {
        if (e.key === 'Enter') sendText();
    });

    document.addEventListener('keydown', e => {
        if (e.code === 'Space' && e.target.tagName !== 'INPUT') {
            e.preventDefault();
            startVoice();
        }
    });
    document.addEventListener('keyup', e => {
        if (e.code === 'Space' && e.target.tagName !== 'INPUT') stopVoice();
    });
});
