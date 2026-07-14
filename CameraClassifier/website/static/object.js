/* ─── Object Detection page JavaScript ───────────────────────────── */

const _init = window.__OBJ_INIT__ || { enabled: false, available: false };

let detEnabled  = _init.enabled;
let pollCount   = 0;
let frameCount  = 0;
let _lastPoll   = Date.now();

/* ─── Toast (shared helper) ─────────────────────────────────────── */
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

/* ─── API helper ─────────────────────────────────────────────────── */
async function api(path, method = 'GET', body = null) {
  const opts = { method, headers: { 'Content-Type': 'application/json' } };
  if (body) opts.body = JSON.stringify(body);
  const res = await fetch(path, opts);
  return res.json();
}

/* ─── Toggle detection on/off ────────────────────────────────────── */
async function toggleDetect() {
  const data = await api('/api/toggle/object_detect', 'POST');
  if (data.error) { showToast(data.error, 'error'); return; }
  detEnabled = data.enabled;
  updateToggleUI(data.enabled);
  if (data.enabled) {
    showToast('Loading YOLOv8 model… first run downloads ~6 MB', 'info', 6000);
  }
}

function updateToggleUI(enabled) {
  const btn  = document.getElementById('togDetect');
  const pill = document.getElementById('pillDetect');
  const overlay = document.getElementById('offOverlay');
  if (btn)  { btn.classList.toggle('active', enabled); }
  if (pill) { pill.textContent = enabled ? 'ON' : 'OFF'; pill.className = `tog-pill ${enabled ? 'on' : 'off'}`; }
  if (overlay) { overlay.style.display = enabled ? 'none' : 'flex'; }
  if (!enabled) {
    clearDetections();
    document.getElementById('objBadge').style.display = 'none';
  }
}

/* ─── Confidence threshold ───────────────────────────────────────── */
function onThresholdChange(val) {
  const lbl = document.getElementById('confLabel');
  if (lbl) lbl.textContent = val + '%';
}

async function applyThreshold(val) {
  await api('/api/object/threshold', 'POST', { threshold: parseFloat(val) / 100 });
}

/* ─── Render detection list ──────────────────────────────────────── */
function renderDetections(detections) {
  const list  = document.getElementById('detList');
  const empty = document.getElementById('detEmpty');
  const badge = document.getElementById('objBadge');
  const count = document.getElementById('detCount');

  const total = detections.length;
  document.getElementById('statTotal').textContent  = total;

  if (!total || !detEnabled) {
    if (empty)  empty.style.display  = 'block';
    if (badge)  badge.style.display  = 'none';
    if (count)  count.textContent    = '';
    clearDetections();
    return;
  }

  if (empty) empty.style.display = 'none';
  if (badge) { badge.style.display = 'flex'; document.getElementById('badgeCount').textContent = total; }
  if (count) count.textContent = `${total} found`;

  /* Group by label, sort by count desc */
  const groups = {};
  for (const d of detections) {
    if (!groups[d.label]) groups[d.label] = { count: 0, maxConf: 0 };
    groups[d.label].count++;
    if (d.confidence > groups[d.label].maxConf) groups[d.label].maxConf = d.confidence;
  }
  const sorted = Object.entries(groups).sort((a, b) => b[1].count - a[1].count);

  const unique = sorted.length;
  document.getElementById('statUnique').textContent = unique;

  /* Rebuild list only if content changed */
  const newHtml = sorted.map(([label, info]) => {
    const pct = Math.round(info.maxConf * 100);
    return `
      <div class="obj-det-item">
        <span class="obj-det-icon">${labelEmoji(label)}</span>
        <div class="obj-det-body">
          <span class="obj-det-label">${label}</span>
          <div class="obj-conf-bar-wrap">
            <div class="obj-conf-bar">
              <div class="obj-conf-fill" style="width:${pct}%"></div>
            </div>
            <span class="obj-conf-pct">${pct}%</span>
          </div>
        </div>
        <span class="obj-det-badge">${info.count > 1 ? '×' + info.count : ''}</span>
      </div>`;
  }).join('');

  if (list.innerHTML !== newHtml) list.innerHTML = newHtml;
}

function clearDetections() {
  const list = document.getElementById('detList');
  const html = '<p class="empty-msg" id="detEmpty">Enable detection to see objects</p>';
  if (list && list.innerHTML !== html) list.innerHTML = html;
  document.getElementById('statTotal').textContent  = '0';
  document.getElementById('statUnique').textContent = '0';
}

/* ─── Emoji mapping for common COCO labels ───────────────────────── */
const _emojiMap = {
  person:'🧑', bicycle:'🚲', car:'🚗', motorcycle:'🏍️', airplane:'✈️',
  bus:'🚌', train:'🚆', truck:'🚛', boat:'⛵', 'traffic light':'🚦',
  'fire hydrant':'🚒', 'stop sign':'🛑', bench:'🪑', bird:'🐦',
  cat:'🐱', dog:'🐶', horse:'🐴', sheep:'🐑', cow:'🐄', elephant:'🐘',
  bear:'🐻', zebra:'🦓', giraffe:'🦒', backpack:'🎒', umbrella:'☂️',
  handbag:'👜', tie:'👔', suitcase:'🧳', frisbee:'🥏', skis:'⛷️',
  'sports ball':'⚽', kite:'🪁', 'baseball bat':'🏏', skateboard:'🛹',
  'tennis racket':'🎾', bottle:'🍶', 'wine glass':'🍷', cup:'☕',
  fork:'🍴', knife:'🔪', spoon:'🥄', bowl:'🥣', banana:'🍌',
  apple:'🍎', sandwich:'🥪', orange:'🍊', broccoli:'🥦', carrot:'🥕',
  'hot dog':'🌭', pizza:'🍕', donut:'🍩', cake:'🎂', chair:'🪑',
  couch:'🛋️', bed:'🛏️', toilet:'🚽', tv:'📺', laptop:'💻',
  mouse:'🖱️', remote:'📱', keyboard:'⌨️', 'cell phone':'📱',
  microwave:'📦', oven:'🔲', sink:'🚰', refrigerator:'🧊',
  book:'📚', clock:'🕐', vase:'🏺', scissors:'✂️', 'teddy bear':'🧸',
  toothbrush:'🪥',
};
function labelEmoji(label) { return _emojiMap[label] || '📦'; }

/* ─── Poll backend for latest detections ─────────────────────────── */
async function pollDetections() {
  try {
    const now  = Date.now();
    const fps  = Math.round(1000 / (now - _lastPoll));
    _lastPoll  = now;
    pollCount++;
    frameCount += detEnabled ? 1 : 0;

    document.getElementById('statFrames').textContent = frameCount;
    document.getElementById('statFps').textContent    = fps + '/s';

    const data = await api('/api/object/status');
    detEnabled = data.enabled;
    updateToggleUI(data.enabled);

    if (data.enabled && data.detections) {
      renderDetections(data.detections);
    } else {
      clearDetections();
    }
  } catch (_) { /* ignore transient errors */ }
}

/* ─── Boot ───────────────────────────────────────────────────────── */
(function boot() {
  updateToggleUI(detEnabled);
  setInterval(pollDetections, 800);
})();
