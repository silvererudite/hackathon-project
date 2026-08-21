const $ = (selector) => document.querySelector(selector);
const promptEl = $('#prompt');
const runButton = $('#runButton');
const terminal = $('#terminal');
let currentRun = null;
let cursor = 0;
let eventCount = 0;

const escapeHtml = (value = '') => String(value)
  .replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;')
  .replaceAll('"', '&quot;').replaceAll("'", '&#039;');

function pretty(value) {
  if (value === null || value === undefined || value === '') return '—';
  try { return JSON.stringify(typeof value === 'string' ? JSON.parse(value) : value, null, 2); }
  catch { return String(value); }
}

function setStatus(kind, text) {
  const status = $('#status');
  status.className = `status ${kind}`;
  status.querySelector('span').textContent = text;
}

function setRunning(running) {
  runButton.disabled = running;
  runButton.querySelector('span').textContent = running ? 'Investigation running…' : 'Run investigation';
}

async function startRun() {
  const prompt = promptEl.value.trim();
  if (!prompt) { promptEl.focus(); return; }
  setRunning(true);
  setStatus('running', 'RUNNING');
  $('#runTitle').textContent = 'Agent is exploring';
  $('#emptyState').classList.add('hidden');
  $('#execution').classList.remove('hidden');
  $('#summary').classList.add('hidden');
  $('#traceHeading').classList.add('hidden');
  $('#conclusion').classList.add('hidden');
  $('#timeline').innerHTML = '';
  terminal.textContent = '';
  cursor = 0; eventCount = 0;
  $('#streamCount').textContent = '0 events';

  try {
    const response = await fetch('/api/runs', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({prompt, backend: $('#backend').value, max_steps: Number($('#maxSteps').value)})
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Unable to start the run.');
    currentRun = data.run_id;
    pollRun();
  } catch (error) {
    failRun(error.message);
  }
}

async function pollRun() {
  if (!currentRun) return;
  try {
    const response = await fetch(`/api/runs/${currentRun}?after=${cursor}`);
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Unable to read run status.');
    if (data.logs?.length) {
      terminal.textContent += `${data.logs.join('\n')}\n`;
      terminal.scrollTop = terminal.scrollHeight;
      eventCount += data.logs.filter(Boolean).length;
      $('#streamCount').textContent = `${eventCount} events`;
    }
    cursor = data.next;
    if (data.status === 'completed') {
      renderTrajectory(data.trajectory, data.download_url);
      setRunning(false); setStatus('completed', 'COMPLETED');
      $('#runTitle').textContent = 'Investigation complete';
      return;
    }
    if (data.status === 'failed') { failRun(data.error || 'The agent run failed.'); return; }
    setTimeout(pollRun, 650);
  } catch (error) { failRun(error.message); }
}

function failRun(message) {
  terminal.textContent += `\nERROR: ${message}\n`;
  setRunning(false); setStatus('failed', 'FAILED');
  $('#runTitle').textContent = 'Investigation failed';
}

function renderTrajectory(t, downloadUrl) {
  const m = t.metadata || {};
  $('#summary').innerHTML = [
    [m.total_steps ?? 0, 'Exploration steps'], [m.total_code_executions ?? 0, 'Code executions'],
    [m.total_failures ?? 0, 'Recoverable errors'], [`${m.wall_time_seconds ?? 0}s`, 'Wall time']
  ].map(([v,l]) => `<div class="metric"><b>${escapeHtml(v)}</b><span>${escapeHtml(l)}</span></div>`).join('');
  $('#summary').classList.remove('hidden');
  $('#traceHeading').classList.remove('hidden');
  $('#stepCount').textContent = `${t.trace.length} recorded steps`;
  $('#timeline').innerHTML = t.trace.map((step, index) => renderStep(step, index)).join('');
  renderConclusion(t.outcome || {}, downloadUrl);
}

function renderStep(step, index) {
  const action = step.action || {};
  const error = step.error || {};
  const code = action.input || '# No code returned';
  return `<article class="trace-card ${error.occurred ? 'error' : ''}">
    <div class="trace-top">
      <span class="trace-index">${String(index + 1).padStart(2, '0')}</span>
      <div><span class="trace-name">Python execution</span><span class="phase-pill">${escapeHtml(step.phase)}</span></div>
      <span class="trace-meta">conf ${step.confidence ?? '—'} · ${step.wall_time ?? 0}s</span>
    </div>
    <div class="trace-body">
      <div class="thought"><strong>Agent rationale</strong><br>${escapeHtml(step.thought)}</div>
      <div class="detail-box generated-code"><h4>MODEL-GENERATED PYTHON</h4><pre>${escapeHtml(code)}</pre></div>
      <div class="detail-box generated-code"><h4>RAW EXECUTION OBSERVATION</h4><pre>${escapeHtml(pretty(step.observation || action.output))}</pre></div>
      ${renderArtifacts(step.observation || action.output)}
      ${step.revision_trigger ? `<div class="error-box"><strong>Revision trigger:</strong> ${escapeHtml(step.revision_trigger)}</div>` : ''}
      ${error.occurred ? `<div class="error-box"><strong>${escapeHtml(error.type || 'Execution error')}:</strong> ${escapeHtml(error.message)}</div>` : ''}
    </div>
  </article>`;
}

function renderArtifacts(rawObservation) {
  let observation;
  try { observation = typeof rawObservation === 'string' ? JSON.parse(rawObservation) : rawObservation; }
  catch { return ''; }
  const artifacts = observation?.artifacts || [];
  return artifacts.filter(item => item.type === 'image' && item.path).map(item => {
    const normalized = String(item.path).replaceAll('\\', '/');
    const marker = '/figures/';
    const relative = normalized.includes(marker) ? normalized.split(marker).pop() : normalized.split('/').pop();
    const url = `/figures/${relative.split('/').map(encodeURIComponent).join('/')}`;
    return `<figure class="artifact-card"><div class="artifact-head"><span>GENERATED FIGURE</span><a href="${url}" download>Download PNG ↓</a></div><img src="${url}" alt="Correlation matrix generated by the agent"></figure>`;
  }).join('');
}

function renderConclusion(outcome, downloadUrl) {
  const sensitivity = outcome.selection_sensitivity || {};
  const limitations = outcome.limitations || [];
  $('#conclusion').innerHTML = `<p class="label">Final conclusion</p>
    <h2>${escapeHtml(outcome.final_claim || 'No structured conclusion was returned.')}</h2>
    <div class="verdict-row"><span>${escapeHtml(sensitivity.verdict || 'NO VERDICT')}</span><span>CONFIDENCE ${outcome.confidence ?? '—'}</span><span>R SPREAD ${sensitivity.r_spread ?? '—'}</span></div>
    <div class="conclusion-grid">
      <div><h3>Limitations</h3>${limitations.length ? `<ul>${limitations.map(x => `<li>${escapeHtml(x)}</li>`).join('')}</ul>` : '<p>None reported.</p>'}</div>
      <div><h3>What would resolve the uncertainty?</h3><p>${escapeHtml(outcome.resolving_measurement || 'No additional measurement was proposed.')}</p></div>
    </div><a class="download" href="${escapeHtml(downloadUrl)}">Download complete JSON trace ↓</a>`;
  $('#conclusion').classList.remove('hidden');
}

async function openTools() {
  $('#drawer').classList.add('open'); $('#drawer').setAttribute('aria-hidden', 'false');
  $('#drawerBackdrop').classList.remove('hidden');
  try {
    const data = await fetch('/api/runtime').then(r => r.json());
    $('#toolList').innerHTML = Object.entries(data.sources).map(([name, source]) =>
      `<details><summary>${escapeHtml(name)}</summary><pre>${escapeHtml(source)}</pre></details>`).join('');
  } catch { $('#toolList').innerHTML = '<p class="muted">Unable to load the runtime policy.</p>'; }
}
function closeTools() { $('#drawer').classList.remove('open'); $('#drawer').setAttribute('aria-hidden', 'true'); $('#drawerBackdrop').classList.add('hidden'); }

runButton.addEventListener('click', startRun);
$('#toolLibraryButton').addEventListener('click', openTools);
$('#closeDrawer').addEventListener('click', closeTools);
$('#drawerBackdrop').addEventListener('click', closeTools);
document.addEventListener('keydown', (event) => { if (event.key === 'Escape') closeTools(); if ((event.metaKey || event.ctrlKey) && event.key === 'Enter' && !runButton.disabled) startRun(); });
