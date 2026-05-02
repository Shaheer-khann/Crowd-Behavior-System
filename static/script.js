/* ===================================================================
   Crowd Behavior Analysis Dashboard — script.js
   Part 2.5a: DOM References & File Handling
   =================================================================== */

'use strict';

/* ─── DOM References ─── */
const dropZone      = document.getElementById('drop-zone');
const videoInput    = document.getElementById('video-input');
const uploadForm    = document.getElementById('upload-form');
const analyzeBtn    = document.getElementById('analyze-btn');

const filePreview   = document.getElementById('file-preview');
const fileName      = document.getElementById('file-name');
const fileSize      = document.getElementById('file-size');
const fileRemoveBtn = document.getElementById('file-remove-btn');

const idlePanel     = document.getElementById('idle-panel');
const loadingPanel  = document.getElementById('loading-panel');
const resultsPanel  = document.getElementById('results-panel');

/* ─── State ─── */
let selectedFile = null;

const ALLOWED_TYPES = ['video/mp4', 'video/avi', 'video/quicktime',
                       'video/x-msvideo', 'video/x-matroska', 'video/webm'];
const MAX_SIZE_MB   = 100;
const MAX_SIZE_B    = MAX_SIZE_MB * 1024 * 1024;


/* ─── Helpers ─── */
function formatBytes(bytes) {
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
}

function isValidFile(file) {
  const extOk  = /\.(mp4|avi|mov|mkv|webm)$/i.test(file.name);
  const sizeOk = file.size <= MAX_SIZE_B;
  return extOk && sizeOk;
}


/* ─── Show / Hide file preview ─── */
function showFilePreview(file) {
  fileName.textContent = file.name;
  fileSize.textContent = formatBytes(file.size);
  filePreview.classList.remove('d-none');
  analyzeBtn.disabled  = false;
}

function clearFilePreview() {
  selectedFile         = null;
  videoInput.value     = '';
  fileName.textContent = '';
  fileSize.textContent = '';
  filePreview.classList.add('d-none');
  analyzeBtn.disabled  = true;
  dropZone.classList.remove('drag-over');
}


/* ─── File selected handler ─── */
function handleFileSelected(file) {
  if (!file) return;

  if (!isValidFile(file)) {
    alert(`Invalid file.\n• Allowed: MP4, AVI, MOV, MKV, WEBM\n• Max size: ${MAX_SIZE_MB} MB`);
    clearFilePreview();
    return;
  }

  selectedFile = file;
  showFilePreview(file);
}


/* ─── Input[type=file] change ─── */
videoInput.addEventListener('change', () => {
  handleFileSelected(videoInput.files[0] || null);
});


/* ─── Remove file button ─── */
fileRemoveBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  clearFilePreview();
  showPanel('idle');
});


/* ─── Drag-and-Drop ─── */
dropZone.addEventListener('dragenter', (e) => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', (e) => {
  if (!dropZone.contains(e.relatedTarget)) {
    dropZone.classList.remove('drag-over');
  }
});

dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0] || null;
  if (file) {
    // sync to hidden input
    const dt = new DataTransfer();
    dt.items.add(file);
    videoInput.files = dt.files;
    handleFileSelected(file);
  }
});

/* Keyboard accessibility for drop zone */
dropZone.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault();
    videoInput.click();
  }
});


/* ─── Panel switcher (used by Parts 2.5b & 2.5c) ─── */
function showPanel(panel) {
  idlePanel.classList.add('d-none');
  loadingPanel.classList.add('d-none');
  resultsPanel.classList.add('d-none');

  if (panel === 'idle')    idlePanel.classList.remove('d-none');
  if (panel === 'loading') loadingPanel.classList.remove('d-none');
  if (panel === 'results') resultsPanel.classList.remove('d-none');
}

/* ===================================================================
   Part 2.5b: API Call & Loading State
   =================================================================== */

/* ─── Loading step elements ─── */
const steps = [
  document.getElementById('step-extract'),
  document.getElementById('step-sequence'),
  document.getElementById('step-infer'),
  document.getElementById('step-agg'),
];
const loadingPulseEl = document.getElementById('loading-pulse');

const PULSE_MESSAGES = [
  'Extracting frames…',
  'Building 16-frame sequences…',
  'Running CNN-LSTM inference…',
  'Aggregating predictions…',
];

let stepTimer = null;


/* ─── Animate loading steps ─── */
function startLoadingAnimation() {
  let current = 0;

  // Reset all steps
  steps.forEach(s => {
    s.classList.remove('active', 'done');
  });
  steps[0].classList.add('active');
  loadingPulseEl.textContent = PULSE_MESSAGES[0];

  stepTimer = setInterval(() => {
    // Mark current as done
    steps[current].classList.remove('active');
    steps[current].classList.add('done');

    current++;
    if (current < steps.length) {
      steps[current].classList.add('active');
      loadingPulseEl.textContent = PULSE_MESSAGES[current];
    }

    // Stop cycling at the last step
    if (current >= steps.length - 1) {
      clearInterval(stepTimer);
      stepTimer = null;
    }
  }, 2200);
}

function stopLoadingAnimation() {
  if (stepTimer) {
    clearInterval(stepTimer);
    stepTimer = null;
  }
  // Mark all remaining steps as done
  steps.forEach(s => {
    s.classList.remove('active');
    s.classList.add('done');
  });
}


/* ─── Lock / unlock UI during processing ─── */
function setProcessingState(isProcessing) {
  analyzeBtn.disabled = isProcessing;
  videoInput.disabled = isProcessing;
  fileRemoveBtn.disabled = isProcessing;

  if (isProcessing) {
    analyzeBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Analyzing…';
  } else {
    analyzeBtn.innerHTML = '<i class="bi bi-play-circle-fill me-2"></i>Analyze Video';
  }
}


/* ─── Form submit → Fetch POST /predict ─── */
uploadForm.addEventListener('submit', async (e) => {
  e.preventDefault();

  if (!selectedFile) return;

  // Switch to loading panel & start animation
  showPanel('loading');
  setProcessingState(true);
  startLoadingAnimation();

  const formData = new FormData();
  formData.append('video', selectedFile);

  let responseData = null;

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      body: formData,
    });

    stopLoadingAnimation();

    // Parse JSON regardless of HTTP status
    try {
      responseData = await response.json();
    } catch {
      responseData = {
        success: false,
        error: `Server returned an invalid response (HTTP ${response.status}).`,
      };
    }

  } catch (networkErr) {
    stopLoadingAnimation();
    responseData = {
      success: false,
      error: 'Network error — could not reach the server. Is Flask running?',
    };
  } finally {
    setProcessingState(false);
  }

  // Hand off to Part 2.5c
  displayResults(responseData);
});

/* ===================================================================
   Part 2.5c: Results Display & Error Handling
   =================================================================== */

/* ─── Result DOM refs ─── */
const resultHeader  = document.getElementById('result-header');
const resultIcon    = document.getElementById('result-icon');
const resultClass   = document.getElementById('result-class');
const confValue     = document.getElementById('conf-value');
const confBar       = document.getElementById('conf-bar');
const statWindows   = document.getElementById('stat-windows');
const statFrames    = document.getElementById('stat-frames');
const statDominant  = document.getElementById('stat-dominant-count');
const breakdownBars = document.getElementById('breakdown-bars');
const resultError   = document.getElementById('result-error');
const errorMessage  = document.getElementById('error-message');
const resetBtn      = document.getElementById('reset-btn');

/* ─── Color config per class ─── */
const CLASS_CONFIG = {
  normal:     { icon: 'bi-check-circle-fill',   colorClass: 'result-normal' },
  panic:      { icon: 'bi-exclamation-triangle-fill', colorClass: 'result-panic' },
  aggressive: { icon: 'bi-shield-exclamation',  colorClass: 'result-aggressive' },
  idle:       { icon: 'bi-pause-circle-fill',   colorClass: 'result-idle' },
};

const ALL_CLASSES = ['normal', 'panic', 'aggressive', 'idle'];


/* ─── Show error inside results panel ─── */
function showError(message) {
  // Show results panel with error sub-panel visible
  showPanel('results');

  // Hide all the data sections
  resultHeader.style.display  = 'none';
  document.querySelector('.result-confidence').style.display = 'none';
  document.getElementById('result-stats').style.display      = 'none';
  document.getElementById('result-breakdown').style.display  = 'none';
  resetBtn.style.display = '';

  // Show error
  errorMessage.textContent = message;
  resultError.classList.remove('d-none');
}


/* ─── Restore hidden sections ─── */
function restoreResultSections() {
  resultHeader.style.display  = '';
  document.querySelector('.result-confidence').style.display = '';
  document.getElementById('result-stats').style.display      = '';
  document.getElementById('result-breakdown').style.display  = '';
  resultError.classList.add('d-none');
}


/* ─── Build breakdown bars ─── */
function buildBreakdown(summary, totalWindows) {
  breakdownBars.innerHTML = '';

  ALL_CLASSES.forEach(cls => {
    const count = summary[cls] || 0;
    const pct   = totalWindows > 0 ? Math.round((count / totalWindows) * 100) : 0;

    const row = document.createElement('div');
    row.className = 'breakdown-row';
    row.innerHTML = `
      <span class="breakdown-label">${cls}</span>
      <div class="breakdown-track">
        <div class="breakdown-fill fill-${cls}" style="width: 0%"></div>
      </div>
      <span class="breakdown-count">${count}</span>
    `;
    breakdownBars.appendChild(row);

    // Animate bar width after a small delay
    setTimeout(() => {
      row.querySelector('.breakdown-fill').style.width = pct + '%';
    }, 80);
  });
}


/* ─── Main results renderer ─── */
function displayResults(data) {
  showPanel('results');
  restoreResultSections();

  // Handle backend error response
  if (!data || !data.success) {
    const msg = data?.error || 'An unknown error occurred. Please try again.';
    showError(msg);
    return;
  }

  const dominant   = (data.dominant_prediction || 'unknown').toLowerCase();
  const config     = CLASS_CONFIG[dominant] || { icon: 'bi-question-circle', colorClass: '' };
  const summary    = data.prediction_summary  || {};
  const totalWin   = data.total_windows_analysed || 0;
  // Average confidence across all windows that predicted the dominant class
  const dominantWindows = (data.window_details || []).filter(w => w.predicted_class === dominant);
  const confidence = dominantWindows.length
    ? dominantWindows.reduce((sum, w) => sum + w.confidence, 0) / dominantWindows.length
    : 0;

  // ── Result header color class ──
  resultHeader.className = 'result-header ' + config.colorClass;
  resultIcon.innerHTML   = `<i class="bi ${config.icon}"></i>`;
  resultClass.textContent = dominant.charAt(0).toUpperCase() + dominant.slice(1);

  // ── Confidence bar ──
  const confPct = Math.round(confidence * 100);
  confValue.textContent = confPct + '%';
  confBar.style.width   = '0%';
  confBar.setAttribute('aria-valuenow', confPct);
  // Animate after paint
  setTimeout(() => { confBar.style.width = confPct + '%'; }, 80);

  // ── Stats ──
  statWindows.textContent  = totalWin;
  statFrames.textContent   = data.total_frames_extracted ?? '—';
  statDominant.textContent = data.dominant_count ?? '—';

  // ── Breakdown bars ──
  buildBreakdown(summary, totalWin);
}


/* ─── Reset button ─── */
resetBtn.addEventListener('click', () => {
  clearFilePreview();
  restoreResultSections();
  showPanel('idle');
  // Re-enable analyze button state is handled by clearFilePreview()
});

