'use strict';

// ── Config ────────────────────────────────────────────────────────────────────
const WS_URL      = `ws://${location.host}/ws`;
const CAPTURE_FPS = 8;       // frames sent to server for analysis
const CHART_MAX   = 150;     // data points in timeline chart

const PALETTE = ['#58a6ff','#3fb950','#d29922','#f85149','#bc8cff','#79c0ff','#56d364','#e3b341'];
const OBJ_COLORS = { 'using phone':'#f85149', 'cell phone':'#d29922', 'cup':'#bc8cff', 'bottle':'#79c0ff' };

function personColor(id)  { return PALETTE[id % PALETTE.length]; }
function engColor(score)  { return score >= 0.6 ? '#3fb950' : score >= 0.35 ? '#d29922' : '#f85149'; }
function engLevel(score)  { return score >= 0.6 ? 'high' : score >= 0.35 ? 'medium' : 'low'; }

// ── State ─────────────────────────────────────────────────────────────────────
let ws              = null;
let stream          = null;
let captureInterval = null;
let waitingForReply = false;
let sessionStart    = null;
let clockInterval   = null;
let lastResult      = null;   // most recent server response
let camW = 640, camH = 480;   // camera native resolution (for box scaling)

const chartData = { labels: [], datasets: {} };

// ── DOM refs ──────────────────────────────────────────────────────────────────
const webcamEl     = document.getElementById('webcam');
const overlayCanvas= document.getElementById('overlay-canvas');
const noSignal     = document.getElementById('no-signal');
const statusDot    = document.getElementById('status-dot');
const statusLabel  = document.getElementById('status-label');
const btnStart     = document.getElementById('btn-start');
const btnStop      = document.getElementById('btn-stop');
const btnReset     = document.getElementById('btn-reset');
const statPeople   = document.getElementById('stat-people');
const statAvgEng   = document.getElementById('stat-avg-eng');
const statFps      = document.getElementById('stat-fps');
const statTime     = document.getElementById('stat-time');
const personsBox   = document.getElementById('persons-container');
const chartLegend  = document.getElementById('chart-legend');

const octx = overlayCanvas.getContext('2d');

// ── Chart ─────────────────────────────────────────────────────────────────────
const engChart = new Chart(document.getElementById('engagement-chart'), {
  type: 'line',
  data: { labels: [], datasets: [] },
  options: {
    responsive: true, maintainAspectRatio: false, animation: false,
    plugins: {
      legend: { display: false },
      tooltip: { callbacks: { label: c => ` P${c.dataset.label}: ${(c.raw*100).toFixed(0)}%` } },
    },
    scales: {
      x: { ticks:{ color:'#8b949e', maxTicksLimit:6, font:{size:10} }, grid:{ color:'#21262d' } },
      y: { min:0, max:1, ticks:{ color:'#8b949e', font:{size:10}, callback:v=>`${(v*100).toFixed(0)}%` }, grid:{ color:'#21262d' } },
    },
  },
});

// ── Annotation rendering loop (runs at ~30fps independent of server) ──────────
function renderOverlay() {
  requestAnimationFrame(renderOverlay);

  // Use getBoundingClientRect for reliable sizing (handles all layout scenarios)
  const rect = overlayCanvas.getBoundingClientRect();
  const dpr  = window.devicePixelRatio || 1;
  const dw   = Math.round(rect.width);
  const dh   = Math.round(rect.height);
  if (dw === 0 || dh === 0) return;

  // Sync intrinsic resolution to display resolution (×dpr for sharp Retina rendering)
  const pw = Math.round(rect.width  * dpr);
  const ph = Math.round(rect.height * dpr);
  if (overlayCanvas.width !== pw || overlayCanvas.height !== ph) {
    overlayCanvas.width  = pw;
    overlayCanvas.height = ph;
    octx.scale(dpr, dpr);
  }

  octx.clearRect(0, 0, dw, dh);

  // ── DEBUG: semi-transparent overlay so we know canvas is alive ──
  if (lastResult) {
    octx.fillStyle = 'rgba(88,166,255,0.08)';
    octx.fillRect(0, 0, dw, dh);
  }

  // Status dot — top-right corner
  octx.beginPath();
  octx.arc(dw - 14, 14, 8, 0, 2 * Math.PI);
  octx.fillStyle = lastResult ? '#3fb950' : '#8b949e';
  octx.fill();

  if (!lastResult) return;

  // Scale factors: map from camera resolution to display resolution
  const [srcW, srcH] = lastResult.frame_size || [camW, camH];
  // Account for object-fit: contain letterboxing
  const videoAR   = srcW / srcH;
  const canvasAR  = dw / dh;
  let drawW, drawH, offX, offY;
  if (videoAR > canvasAR) {
    drawW = dw; drawH = dw / videoAR;
    offX  = 0;  offY  = (dh - drawH) / 2;
  } else {
    drawH = dh; drawW = dh * videoAR;
    offY  = 0;  offX  = (dw - drawW) / 2;
  }
  const sx = drawW / srcW;
  const sy = drawH / srcH;

  // ── Draw object boxes ────────────────────────────────────────────
  (lastResult.objects || []).forEach(o => {
    const [x1,y1,x2,y2] = o.box;
    const rx = offX + x1*sx, ry = offY + y1*sy;
    const rw = (x2-x1)*sx,   rh = (y2-y1)*sy;
    const color = OBJ_COLORS[o.name] || '#bc8cff';

    octx.strokeStyle = color;
    octx.lineWidth   = 2;
    octx.strokeRect(rx, ry, rw, rh);

    const label = `${o.name} ${Math.round(o.conf*100)}%`;
    octx.font = 'bold 11px system-ui';
    const tw   = octx.measureText(label).width;
    octx.fillStyle = color;
    octx.fillRect(rx, ry - 18, tw + 8, 18);
    octx.fillStyle = '#000';
    octx.fillText(label, rx + 4, ry - 5);
  });

  // ── Draw face boxes ──────────────────────────────────────────────
  (lastResult.persons || []).forEach(p => {
    const [x1,y1,x2,y2] = p.box;
    const rx = offX + x1*sx, ry = offY + y1*sy;
    const rw = (x2-x1)*sx,   rh = (y2-y1)*sy;
    const color = engColor(p.engagement);

    // Box
    octx.strokeStyle = color;
    octx.lineWidth   = 2.5;
    octx.strokeRect(rx, ry, rw, rh);

    // Engagement fill bar at bottom of box
    octx.fillStyle = 'rgba(0,0,0,0.45)';
    octx.fillRect(rx, ry + rh - 6, rw, 6);
    octx.fillStyle = color;
    octx.fillRect(rx, ry + rh - 6, rw * p.engagement, 6);

    // Label above box
    const label = `P${p.id}  ${Math.round(p.engagement*100)}%`;
    octx.font  = 'bold 12px system-ui';
    const tw   = octx.measureText(label).width;
    const ly   = ry - 2;
    octx.fillStyle = color + 'cc';
    octx.fillRect(rx, ly - 16, tw + 8, 17);
    octx.fillStyle = '#000';
    octx.fillText(label, rx + 4, ly - 3);

    // Status icons inline
    let iconX = rx + 4;
    const iy  = ry + 14;
    if (p.eyes_closed) {
      drawIcon(octx, '😴', iconX, iy); iconX += 20;
    }
    if (p.hand_raise > 0.5) {
      drawIcon(octx, '✋', iconX, iy); iconX += 20;
    }
    if (p.distraction) {
      drawIcon(octx, '📱', iconX, iy);
    }
  });
}

function drawIcon(ctx, emoji, x, y) {
  ctx.font = '14px system-ui';
  ctx.fillText(emoji, x, y);
}

// Start the render loop immediately
renderOverlay();

// ── Helpers ──────────────────────────────────────────────────────────────────
function setStatus(state) {
  statusDot.className = 'dot dot-' + state;
  statusLabel.textContent = { off:'Disconnected', connecting:'Connecting…', on:'Live', error:'Error' }[state] ?? state;
}

function formatTime(s) {
  return `${String(Math.floor(s/60)).padStart(2,'0')}:${String(Math.floor(s%60)).padStart(2,'0')}`;
}

// ── Session control ───────────────────────────────────────────────────────────
async function startSession() {
  btnStart.disabled = true;

  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width:1280, height:720 }, audio: false });
  } catch (err) {
    alert('Camera access denied: ' + err.message);
    btnStart.disabled = false;
    return;
  }

  webcamEl.srcObject = stream;
  await new Promise(res => { webcamEl.onloadedmetadata = res; });
  await webcamEl.play();

  camW = webcamEl.videoWidth  || 640;
  camH = webcamEl.videoHeight || 480;

  setStatus('connecting');
  ws = new WebSocket(WS_URL);
  ws.binaryType = 'arraybuffer';

  ws.onopen = () => {
    setStatus('on');
    btnStop.disabled  = false;
    btnReset.disabled = false;
    noSignal.classList.add('hidden');

    sessionStart  = Date.now();
    clockInterval = setInterval(() => { statTime.textContent = formatTime((Date.now()-sessionStart)/1000); }, 1000);

    // Hidden canvas for JPEG capture only
    captureInterval = setInterval(captureAndSend, 1000 / CAPTURE_FPS);
  };

  ws.onmessage = evt => onServerMessage(JSON.parse(evt.data));
  ws.onerror   = err  => { console.error('WS error', err); setStatus('error'); };
  ws.onclose   = ()   => { setStatus('off'); stopCapture(); };
}

function stopSession() {
  stopCapture();
  if (stream)       { stream.getTracks().forEach(t => t.stop()); stream = null; }
  if (ws)           { ws.close(); ws = null; }
  if (clockInterval){ clearInterval(clockInterval); clockInterval = null; }
  btnStart.disabled = false;
  btnStop.disabled  = true;
  btnReset.disabled = true;
  noSignal.classList.remove('hidden');
  lastResult = null;
  setStatus('off');
}

function stopCapture() {
  if (captureInterval) { clearInterval(captureInterval); captureInterval = null; }
  waitingForReply = false;
}

async function resetSession() {
  await fetch('/api/reset', { method:'POST' });
  lastResult            = null;
  chartData.labels      = [];
  chartData.datasets    = {};
  engChart.data.labels  = [];
  engChart.data.datasets= [];
  engChart.update('none');
  chartLegend.innerHTML = '';
  personsBox.innerHTML  = '<div class="empty-msg">No persons detected yet.</div>';
  statPeople.textContent= '0';
  statAvgEng.textContent= '—';
  sessionStart = Date.now();
}

// ── Frame capture ─────────────────────────────────────────────────────────────
const _capCanvas = document.createElement('canvas');
const _capCtx    = _capCanvas.getContext('2d', { willReadFrequently: true });

function captureAndSend() {
  if (waitingForReply || !ws || ws.readyState !== WebSocket.OPEN) return;
  if (webcamEl.readyState < 2) return;

  _capCanvas.width  = camW;
  _capCanvas.height = camH;
  _capCtx.drawImage(webcamEl, 0, 0, camW, camH);

  _capCanvas.toBlob(blob => {
    if (!blob) return;
    blob.arrayBuffer().then(buf => {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(buf);
        waitingForReply = true;
      }
    });
  }, 'image/jpeg', 0.8);
}

// ── Server response handler ───────────────────────────────────────────────────
let _dbgLast = 0;
function onServerMessage(data) {
  waitingForReply = false;
  lastResult = data;
  const now = Date.now();
  if (now - _dbgLast > 1000) {
    console.log('[MSG] persons:', data.num_persons, 'objects:', data.objects?.length,
                'frame_size:', data.frame_size, 'canvas:', overlayCanvas.offsetWidth, overlayCanvas.offsetHeight);
    _dbgLast = now;
  }

  // Stats bar
  statPeople.textContent = data.num_persons;
  statFps.textContent    = data.fps + ' fps';
  if (data.num_persons > 0) {
    const pct = Math.round(data.avg_engagement * 100);
    statAvgEng.textContent = pct + '%';
    statAvgEng.style.color = pct >= 60 ? 'var(--green)' : pct >= 35 ? 'var(--yellow)' : 'var(--red)';
  }

  updateChart(data.elapsed, data.persons);
  updatePersonCards(data.persons);
}

// ── Chart ─────────────────────────────────────────────────────────────────────
function updateChart(elapsed, persons) {
  const label = elapsed.toFixed(1) + 's';
  if (chartData.labels.length >= CHART_MAX) {
    chartData.labels.shift();
    Object.values(chartData.datasets).forEach(ds => ds.data.shift());
  }
  chartData.labels.push(label);

  const seenIds = new Set();
  persons.forEach(p => {
    seenIds.add(p.id);
    if (!chartData.datasets[p.id]) {
      const color   = personColor(p.id);
      const dataset = {
        label: String(p.id), data: new Array(chartData.labels.length-1).fill(null),
        borderColor: color, backgroundColor: color+'22',
        borderWidth: 2, pointRadius: 0, tension: 0.3, fill: false,
      };
      chartData.datasets[p.id] = dataset;
      engChart.data.datasets.push(dataset);
      const item = document.createElement('span');
      item.className = 'legend-item';
      item.innerHTML = `<span class="legend-dot" style="background:${color}"></span>P${p.id}`;
      chartLegend.appendChild(item);
    }
    chartData.datasets[p.id].data.push(p.engagement);
  });
  Object.entries(chartData.datasets).forEach(([id, ds]) => {
    if (!seenIds.has(Number(id))) ds.data.push(null);
  });
  engChart.data.labels = [...chartData.labels];
  engChart.update('none');
}

// ── Person cards ──────────────────────────────────────────────────────────────
function updatePersonCards(persons) {
  if (!persons.length) {
    personsBox.innerHTML = '<div class="empty-msg">No persons detected.</div>';
    return;
  }
  const existing = {};
  personsBox.querySelectorAll('.person-card').forEach(el => { existing[el.dataset.id] = el; });
  const currentIds = new Set(persons.map(p => String(p.id)));
  Object.keys(existing).forEach(id => { if (!currentIds.has(id)) existing[id].remove(); });

  persons.forEach(p => {
    const level = engLevel(p.engagement);
    const color = personColor(p.id);
    const tags  = [];
    if (p.eyes_closed)      tags.push('<span class="tag tag-sleep">😴 Drowsy</span>');
    if (p.distraction)      tags.push(`<span class="tag tag-dist">📱 ${p.distraction}</span>`);
    if (p.hand_raise > 0.5) tags.push('<span class="tag tag-raise">✋ Hand Up</span>');

    const html = `
      <div class="person-header">
        <span class="person-id" style="color:${color}">Person ${p.id}</span>
        <span class="eng-badge ${level}">${Math.round(p.engagement*100)}%</span>
      </div>
      <div class="metric-bars">
        <div class="metric-row">
          <span class="metric-name">Focus</span>
          <div class="bar-track"><div class="bar-fill bar-focus" style="width:${p.focus*100}%"></div></div>
          <span class="bar-val">${Math.round(p.focus*100)}%</span>
        </div>
        <div class="metric-row">
          <span class="metric-name">Talk</span>
          <div class="bar-track"><div class="bar-fill bar-talk" style="width:${p.talk*100}%"></div></div>
          <span class="bar-val">${Math.round(p.talk*100)}%</span>
        </div>
      </div>
      ${tags.length ? `<div class="person-tags">${tags.join('')}</div>` : ''}`;

    const sid = String(p.id);
    if (existing[sid]) {
      existing[sid].className = `person-card ${level}`;
      existing[sid].innerHTML = html;
    } else {
      const card = document.createElement('div');
      card.className = `person-card ${level}`; card.dataset.id = sid;
      card.innerHTML = html;
      personsBox.appendChild(card);
    }
  });
}

// ── Poll until models ready ───────────────────────────────────────────────────
(async function checkReady() {
  try {
    const data = await (await fetch('/api/status')).json();
    if (!data.ready) {
      setStatus('connecting'); statusLabel.textContent = 'Loading models…';
      setTimeout(checkReady, 2000);
    } else {
      setStatus('off');
    }
  } catch { setTimeout(checkReady, 3000); }
})();
