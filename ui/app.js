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

// ── Readout levels (called from Python via evaluate_js) ───────────────────────
// The HUMOR bar is driven by the real config value, not hardcoded. Python pushes it
// on load and can call setLevels() again whenever the value changes, so the readout
// stays connected in real time (e.g. once self-modification can move the dial).
const BAR_CELLS = 16;

function renderBar(pct) {
    const filled = Math.max(0, Math.min(BAR_CELLS, Math.round(pct / 100 * BAR_CELLS)));
    return '█'.repeat(filled) + '░'.repeat(BAR_CELLS - filled);
}

function setLevels(humor) {
    document.getElementById('humorBar').textContent = renderBar(humor);
    document.getElementById('humorPct').textContent = `${Math.round(humor)}%`;
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
    const blob = new Blob([bytes], { type: 'audio/wav' });
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

// ── Power switch ──────────────────────────────────────────────────────────────
// Turning TARS off stops the mic listener (Python releases the input stream);
// turning it on restarts it. The visual goes dormant/OFFLINE while powered down.
let powered = true;

function setPowerUI(on) {
    powered = on;
    const sw = document.getElementById('powerSwitch');
    sw.classList.toggle('on', on);
    sw.classList.toggle('off', !on);
    sw.setAttribute('aria-checked', String(on));
    if (on) {
        setStatus('idle');
    } else {
        // Killing the listener also means no audio should keep playing.
        interruptStream();
        document.getElementById('tarsBody').className = 'tars-body dormant';
        document.getElementById('statusText').textContent = 'OFFLINE';
    }
}

async function togglePower() {
    const next = !powered;
    setPowerUI(next);                       // optimistic — snappy click feedback
    await window.pywebview.api.set_power(next);
}

// ── Bootstrap ─────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initBars();
    document.getElementById('resetBtn').addEventListener('click', resetConversation);
    document.getElementById('powerSwitch').addEventListener('click', togglePower);
});
