/* ─── Demo page JavaScript ───────────────────────────────────────── */

const CLASS_COLORS = [
  '#7c6ff7','#00d4ff','#00e676','#ffb300','#ff4081',
  '#ff6d00','#40c4ff','#69f0ae','#ea80fc','#ff6e40',
];

let state = window.__INIT_STATE__ || { classes:[], counters:[], face_detect:false, emotion:false, auto_predict:false, trained:false, training:false };

/* ─── Toast helper ───────────────────────────────────────────────── */
function showToast(msg, type = 'info', duration = 2800) {
  let container = document.getElementById('toastContainer');
  if (!container) {
    container = document.createElement('div');
    container.id = 'toastContainer';
    container.className = 'toast-container';
    document.body.appendChild(container);
  }
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.textContent = msg;
  container.appendChild(toast);
  setTimeout(() => toast.remove(), duration);
}

/* ─── API helpers ────────────────────────────────────────────────── */
async function api(path, method = 'GET', body = null) {
  const opts = { method, headers: { 'Content-Type': 'application/json' } };
  if (body) opts.body = JSON.stringify(body);
  const res = await fetch(path, opts);
  return res.json();
}

/* ─── Toggle features ────────────────────────────────────────────── */
async function toggle(feature) {
  const data = await api(`/api/toggle/${feature}`, 'POST');
  if (data.error) { showToast(data.error, 'error'); return; }
  state[feature] = data.enabled;
  updateTogglePill(feature, data.enabled);
  if (feature === 'emotion' && data.enabled) {
    showToast('Loading emotion model… this may take a few seconds', 'info', 5000);
  }
}

function updateTogglePill(feature, enabled) {
  const pillMap = { face_detect: 'pillFace', emotion: 'pillEmotion', auto_predict: 'pillAuto' };
  const btnMap  = { face_detect: 'togFace',  emotion: 'togEmotion',  auto_predict: 'togAuto' };
  const pill = document.getElementById(pillMap[feature]);
  const btn  = document.getElementById(btnMap[feature]);
  if (pill) { pill.textContent = enabled ? 'ON' : 'OFF'; pill.className = `tog-pill ${enabled ? 'on' : 'off'}`; }
  if (btn)  { btn.classList.toggle('active', enabled); }
}

/* ─── Add a new class ────────────────────────────────────────────── */
async function addNewClass() {
  const input = document.getElementById('newClassInput');
  const name  = input.value.trim();
  if (!name) { input.focus(); return; }

  const data = await api('/api/add_class', 'POST', { name });
  if (data.error) { showToast(data.error, 'error'); return; }

  input.value = '';
  state.classes.push(data.name);
  state.counters.push(0);
  renderClassItem(data.class_idx, data.name, 0);
  updateClassCount();
  showToast(`Class "${data.name}" added`, 'success');
}

/* ─── Render a class card ────────────────────────────────────────── */
function renderClassItem(idx, name, count) {
  const list = document.getElementById('classList');
  const empty = document.getElementById('emptyMsg');
  if (empty) empty.remove();

  const color = CLASS_COLORS[(idx - 1) % CLASS_COLORS.length];
  const item  = document.createElement('div');
  item.className = 'class-item';
  item.id = `classItem${idx}`;
  item.innerHTML = `
    <span class="class-color-dot" style="background:${color}"></span>
    <span class="class-name" title="${name}">${name}</span>
    <span class="class-count" id="classCount${idx}">${count} samples</span>
    <button class="capture-btn" id="captureBtn${idx}" onclick="captureFrame(${idx})">📸 Capture</button>
  `;
  list.appendChild(item);
}

function updateClassCount() {
  const n = state.classes.length;
  const el = document.getElementById('classCount');
  if (el) el.textContent = `${n} class${n !== 1 ? 'es' : ''}`;
}

/* ─── Capture a frame ────────────────────────────────────────────── */
async function captureFrame(classNum) {
  const btn = document.getElementById(`captureBtn${classNum}`);
  if (btn) { btn.disabled = true; btn.textContent = '…'; }

  const data = await api(`/api/capture/${classNum}`, 'POST');

  if (btn) { btn.disabled = false; btn.innerHTML = '📸 Capture'; }

  if (data.error) { showToast(data.error, 'error'); return; }

  state.counters[classNum - 1] = data.count;
  const countEl = document.getElementById(`classCount${classNum}`);
  if (countEl) countEl.textContent = `${data.count} sample${data.count !== 1 ? 's' : ''}`;
  showToast(`Captured sample ${data.count} for class ${classNum}`, 'success', 1400);
}

/* ─── Train model ────────────────────────────────────────────────── */
async function trainModel() {
  if (state.classes.length < 2) { showToast('Add at least 2 classes first', 'error'); return; }

  const btn = document.getElementById('btnTrain');
  if (btn) { btn.disabled = true; }

  setModelStatus('training', 'Training…');
  const data = await api('/api/train', 'POST');

  if (data.error) {
    showToast(data.error, 'error');
    setModelStatus('untrained', 'Not trained');
    if (btn) btn.disabled = false;
    return;
  }

  showToast('Training started — check status bar', 'info');
  if (btn) btn.disabled = false;
}

/* ─── Predict once ───────────────────────────────────────────────── */
async function predictOnce() {
  const data = await api('/api/predict', 'POST');
  if (data.error) { showToast(data.error, 'error'); return; }
  if (data.class_name) {
    updateResultDisplay(data.class_name, data.confidence, null);
    showToast(`Predicted: ${data.class_name} (${Math.round(data.confidence * 100)}%)`, 'success');
  }
}

/* ─── Reset ──────────────────────────────────────────────────────── */
async function resetAll() {
  if (!confirm('Reset all classes, samples, and the trained model?')) return;

  const data = await api('/api/reset', 'POST');
  if (data.error) { showToast(data.error, 'error'); return; }

  state.classes  = [];
  state.counters = [];
  state.trained  = false;

  const list = document.getElementById('classList');
  list.innerHTML = '<p class="empty-msg" id="emptyMsg">No classes yet. Add one below.</p>';
  updateClassCount();
  setModelStatus('untrained', 'Not trained');
  updateResultDisplay(null, 0, null);
  showToast('Reset complete', 'info');
}

/* ─── UI helpers ─────────────────────────────────────────────────── */
function setModelStatus(type, text) {
  const dot  = document.getElementById('mStatusDot');
  const txt  = document.getElementById('mStatusText');
  if (dot) dot.className = `mstatus-dot ${type}`;
  if (txt) txt.textContent = text;
}

function updateResultDisplay(className, confidence, emotion) {
  const resClass   = document.getElementById('resClass');
  const confFill   = document.getElementById('confFill');
  const confPct    = document.getElementById('confPct');
  const resEmotion = document.getElementById('resEmotion');
  const overlay    = document.getElementById('feedOverlay');
  const oClass     = document.getElementById('overlayClass');
  const oConf      = document.getElementById('overlayConf');

  if (resClass)   resClass.textContent   = className || '—';
  if (confFill)   confFill.style.width   = className ? `${Math.round(confidence * 100)}%` : '0%';
  if (confPct)    confPct.textContent    = className ? `${Math.round(confidence * 100)}%` : '0%';
  if (resEmotion) resEmotion.textContent = emotion   || '—';

  if (overlay) {
    overlay.style.display = className ? 'flex' : 'none';
    if (oClass) oClass.textContent = className || '';
    if (oConf)  oConf.textContent  = className ? `${Math.round(confidence * 100)}%` : '';
  }
}

/* ─── Status polling ─────────────────────────────────────────────── */
async function pollStatus() {
  try {
    const s = await api('/api/status');

    /* Sync toggle pills */
    ['face_detect', 'emotion', 'auto_predict'].forEach(f => {
      if (state[f] !== s[f]) updateTogglePill(f, s[f]);
    });
    state = s;

    /* Model status */
    if (s.training) {
      setModelStatus('training', 'Training…');
    } else if (s.trained) {
      setModelStatus('trained', `Trained on ${s.classes.length} classes`);
    } else {
      setModelStatus('untrained', 'Not trained');
    }

    /* Update class sample counts */
    s.counters.forEach((count, i) => {
      const el = document.getElementById(`classCount${i + 1}`);
      if (el) el.textContent = `${count} sample${count !== 1 ? 's' : ''}`;
    });

    /* Prediction results */
    const pred = s.prediction;
    const emo  = s.emotion_result;
    const emotionStr = (emo && emo.emotion && emo.emotion !== 'No face')
      ? `${emo.emotion.charAt(0).toUpperCase() + emo.emotion.slice(1)} (${Math.round(emo.confidence * 100)}%)`
      : (emo && emo.emotion ? emo.emotion : null);

    updateResultDisplay(
      pred.class_name,
      pred.confidence,
      s.emotion ? emotionStr : null
    );

  } catch (_) { /* ignore transient network errors */ }
}

/* ─── Boot ───────────────────────────────────────────────────────── */
(function boot() {
  /* Re-render classes from server-seeded state */
  if (state.classes && state.classes.length > 0) {
    state.classes.forEach((name, i) => {
      renderClassItem(i + 1, name, state.counters[i] || 0);
    });
    updateClassCount();
  }

  /* Sync toggle pills */
  ['face_detect', 'emotion', 'auto_predict'].forEach(f => updateTogglePill(f, !!state[f]));

  /* Sync model status */
  if (state.trained)        setModelStatus('trained',   `Trained on ${state.classes.length} classes`);
  else if (state.training)  setModelStatus('training',  'Training…');
  else                      setModelStatus('untrained', 'Not trained');

  /* Start polling every 1.2 seconds */
  setInterval(pollStatus, 1200);
})();
