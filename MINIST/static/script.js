/* ============================================================
   script.js  —  Premium MNIST UI
   ============================================================ */

/* ── Particle System ────────────────────────────────────────── */
const pCanvas = document.getElementById('particleCanvas');
const pCtx = pCanvas.getContext('2d');
let particles = [];

function resizeParticleCanvas() {
  pCanvas.width = window.innerWidth;
  pCanvas.height = window.innerHeight;
}
resizeParticleCanvas();
window.addEventListener('resize', resizeParticleCanvas);

function createParticle() {
  return {
    x: Math.random() * pCanvas.width,
    y: Math.random() * pCanvas.height,
    r: Math.random() * 1.5 + 0.4,
    vx: (Math.random() - 0.5) * 0.3,
    vy: (Math.random() - 0.5) * 0.3,
    alpha: Math.random() * 0.4 + 0.1,
    color: ['#7c6ef5', '#5e9fff', '#f472b6', '#34d399'][Math.floor(Math.random() * 4)],
  };
}

for (let i = 0; i < 90; i++) particles.push(createParticle());

(function animateParticles() {
  pCtx.clearRect(0, 0, pCanvas.width, pCanvas.height);
  particles.forEach(p => {
    pCtx.beginPath();
    pCtx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
    pCtx.fillStyle = p.color;
    pCtx.globalAlpha = p.alpha;
    pCtx.fill();
    p.x += p.vx; p.y += p.vy;
    if (p.x < 0 || p.x > pCanvas.width) p.vx *= -1;
    if (p.y < 0 || p.y > pCanvas.height) p.vy *= -1;
  });
  pCtx.globalAlpha = 1;
  requestAnimationFrame(animateParticles);
})();

/* ── Cursor Glow ────────────────────────────────────────────── */
const cursorGlow = document.getElementById('cursorGlow');
document.addEventListener('mousemove', e => {
  cursorGlow.style.left = e.clientX + 'px';
  cursorGlow.style.top = e.clientY + 'px';
});

/* ── Canvas Drawing ─────────────────────────────────────────── */
const canvas = document.getElementById('drawCanvas');
const ctx = canvas.getContext('2d');
const canvasOverlay = document.getElementById('canvasOverlay');
const brushInput = document.getElementById('brushSize');
const brushValEl = document.getElementById('brushVal');
const clearBtn = document.getElementById('clearBtn');
const predictBtn = document.getElementById('predictBtn');

let isDrawing = false, hasDrawn = false;
let lastX = 0, lastY = 0;

function initCanvas() {
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}
initCanvas();

function getPos(e) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  if (e.touches) {
    return {
      x: (e.touches[0].clientX - rect.left) * scaleX,
      y: (e.touches[0].clientY - rect.top) * scaleY
    };
  }
  return {
    x: (e.clientX - rect.left) * scaleX,
    y: (e.clientY - rect.top) * scaleY
  };
}

function startDraw(e) {
  e.preventDefault();
  isDrawing = true;
  const { x, y } = getPos(e);
  lastX = x; lastY = y;
  if (!hasDrawn) {
    hasDrawn = true;
    canvasOverlay.classList.add('hidden');
  }
}

function draw(e) {
  e.preventDefault();
  if (!isDrawing) return;
  const { x, y } = getPos(e);
  ctx.lineWidth = parseInt(brushInput.value);
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.strokeStyle = '#ffffff';
  ctx.shadowColor = 'rgba(255,255,255,0.4)';
  ctx.shadowBlur = 8;
  ctx.beginPath();
  ctx.moveTo(lastX, lastY);
  ctx.lineTo(x, y);
  ctx.stroke();
  ctx.shadowBlur = 0;
  lastX = x; lastY = y;
}

function stopDraw() { isDrawing = false; }

canvas.addEventListener('mousedown', startDraw);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDraw);
canvas.addEventListener('mouseleave', stopDraw);
canvas.addEventListener('touchstart', startDraw, { passive: false });
canvas.addEventListener('touchmove', draw, { passive: false });
canvas.addEventListener('touchend', stopDraw);

brushInput.addEventListener('input', () => {
  brushValEl.textContent = brushInput.value;
});

clearBtn.addEventListener('click', () => {
  initCanvas();
  hasDrawn = false;
  canvasOverlay.classList.remove('hidden');
  showState('idle');
});

/* ── Predict ────────────────────────────────────────────────── */
predictBtn.addEventListener('click', async () => {
  if (!hasDrawn) { shakeCanvas(); return; }
  showState('loading');

  try {
    const resp = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: canvas.toDataURL('image/png') }),
    });
    const data = await resp.json();
    if (!resp.ok || data.error) throw new Error(data.error || 'Server error');

    renderResult(data);
    addToHistory(data);
    showState('prediction');
  } catch (err) {
    document.getElementById('errorMsg').textContent = err.message;
    showState('error');
  }
});

/* ── Render result ──────────────────────────────────────────── */
function renderResult({ digit, confidence, probs }) {
  /* Big digit */
  document.getElementById('bigDigit').textContent = digit;

  /* Circular arc */
  const arc = document.getElementById('confArc');
  const pct = document.getElementById('confPct');
  const circ = 2 * Math.PI * 34;       // 2πr, r=34
  const offset = circ * (1 - confidence / 100);
  setTimeout(() => { arc.style.strokeDashoffset = offset; }, 50);
  animateCounter(pct, 0, Math.round(confidence), 900, v => v + '%');

  /* Confidence label */
  const statusEl = document.getElementById('resultStatus');
  if (confidence >= 95) { statusEl.textContent = 'Very High'; statusEl.style.color = 'var(--green)'; }
  else if (confidence >= 80) { statusEl.textContent = 'High'; statusEl.style.color = 'var(--green)'; }
  else if (confidence >= 60) { statusEl.textContent = 'Moderate'; statusEl.style.color = 'var(--yellow)'; }
  else { statusEl.textContent = 'Low'; statusEl.style.color = 'var(--red)'; }

  /* Prob bars */
  const list = document.getElementById('probBars');
  list.innerHTML = '';
  probs.forEach((p, d) => {
    const isTop = d === digit;
    const row = document.createElement('div');
    row.className = 'prob-row';
    row.innerHTML = `
      <span class="prob-lbl ${isTop ? 'top' : ''}">${d}</span>
      <div class="prob-track">
        <div class="prob-fill ${isTop ? 'top' : ''}" data-pct="${p}"></div>
      </div>
      <span class="prob-pct ${isTop ? 'top' : ''}">${p.toFixed(1)}%</span>
    `;
    list.appendChild(row);
  });
  requestAnimationFrame(() => {
    document.querySelectorAll('.prob-fill').forEach(el => {
      el.style.width = el.dataset.pct + '%';
    });
  });
}

/* ── History ────────────────────────────────────────────────── */
const historyList = document.getElementById('historyList');
const clearHistoryBtn = document.getElementById('clearHistoryBtn');
let historyData = [];

function addToHistory({ digit, confidence }) {
  historyData.unshift({ digit, confidence });
  if (historyData.length > 12) historyData.pop();
  renderHistory();
}

function renderHistory() {
  if (historyData.length === 0) {
    historyList.innerHTML = '<span class="history-empty">No predictions yet</span>';
    return;
  }
  historyList.innerHTML = historyData.map(h => `
    <div class="history-chip">
      <span class="history-chip-digit">${h.digit}</span>
      <span class="history-chip-conf">${h.confidence.toFixed(0)}%</span>
    </div>
  `).join('');
}

clearHistoryBtn.addEventListener('click', () => {
  historyData = [];
  renderHistory();
});

/* ── State machine ──────────────────────────────────────────── */
const states = { idle: 'idleState', loading: 'loadingState', error: 'errorState', prediction: 'predictionState' };

function showState(name) {
  Object.values(states).forEach(id => document.getElementById(id).classList.add('hidden'));
  document.getElementById(states[name]).classList.remove('hidden');
}

/* ── Shake canvas ───────────────────────────────────────────── */
function shakeCanvas() {
  const moves = [10, -10, 7, -7, 4, -4, 0];
  let i = 0;
  const tick = () => {
    canvas.style.transform = `translateX(${moves[i]}px)`;
    i++;
    if (i < moves.length) setTimeout(tick, 55);
    else canvas.style.transform = '';
  };
  tick();
}

/* ── Counter animation ──────────────────────────────────────── */
function animateCounter(el, from, to, duration, format) {
  const start = performance.now();
  (function tick(now) {
    const t = Math.min((now - start) / duration, 1);
    const ease = 1 - Math.pow(1 - t, 3);
    el.textContent = format(Math.round(from + (to - from) * ease));
    if (t < 1) requestAnimationFrame(tick);
  })(start);
}
