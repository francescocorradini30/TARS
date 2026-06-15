// ── State ─────────────────────────────────────────────────────────────────────
const BARS = 12;
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

// ── Stream lifecycle (called from Python via evaluate_js) ─────────────────────
function startTarsStream() {
    streamComplete = false;
    setStatus('thinking');
}

function endTarsStream() {
    streamComplete = true;
    if (!isPlayingQueue && audioQueue.length === 0) {
        finishProcessing();
    }
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
}

// ── Audio playback ────────────────────────────────────────────────────────────
let currentAudio = null;
let interruptAudioResolve = null;

async function playAudio(b64) {
    const binary = atob(b64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    const blob = new Blob([bytes], { type: 'audio/mp3' });
    const url = URL.createObjectURL(blob);
    const audio = new Audio(url);
    currentAudio = audio;
    return new Promise(resolve => {
        interruptAudioResolve = resolve;
        audio.onended = () => { URL.revokeObjectURL(url); currentAudio = null; interruptAudioResolve = null; resolve(); };
        audio.onerror = () => { URL.revokeObjectURL(url); currentAudio = null; interruptAudioResolve = null; resolve(); };
        audio.play().catch(resolve);
    });
}

// Called from Python when user speaks over TARS
function interruptStream() {
    if (currentAudio) { currentAudio.pause(); currentAudio = null; }
    if (interruptAudioResolve) { interruptAudioResolve(); interruptAudioResolve = null; }
    audioQueue.length = 0;
    isPlayingQueue = false;
    streamComplete = false;
    setStatus('idle');
}

// ── Reset ─────────────────────────────────────────────────────────────────────
async function resetConversation() {
    audioQueue.length = 0;
    isPlayingQueue = false;
    streamComplete = false;
    await window.pywebview.api.reset();
}

// ── Bootstrap ─────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initBars();
    document.getElementById('resetBtn').addEventListener('click', resetConversation);
});
