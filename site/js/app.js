/* Application wiring: data loading, explorer, live tuner, dashboard. */

import { PointCloudViewer, webglAvailable } from './viewer.js';
import { Grid, sor, ror, idsor, removalScores, meanKnnDistance } from './filters.js';
import { barChart, scatterChart, groupedBarChart, lineChart, C } from './charts.js';

const DATA = 'data/';
const SEVERITIES = [
  ['light', 'Light', '10 mm/h'],
  ['moderate', 'Moderate', '25 mm/h'],
  ['heavy', 'Heavy', '50 mm/h'],
];
const METHODS = ['noisy', 'clean', 'SOR+ROR', 'IDSOR', 'DGCNN', 'PointNet-AE', 'RangeRestoreNet'];
const LEARNED = new Set(['DGCNN', 'PointNet-AE', 'RangeRestoreNet']);
const fileSafe = (s) => s.replace(/[+\-]/g, '');

// ── loading ──────────────────────────────────────────────────────────────
async function fetchBuffer(path, Type) {
  const r = await fetch(DATA + path);
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return new Type(await r.arrayBuffer());
}
const fetchJSON = (p) => fetch(DATA + p).then((r) => r.json());

/** Int16 centimetres -> Float32 metres. */
function dequantise(i16, scale) {
  const out = new Float32Array(i16.length);
  for (let i = 0; i < i16.length; i++) out[i] = i16[i] / scale;
  return out;
}

const store = { severities: {} };

async function loadSeverity(sev, entry) {
  const clouds = {};
  for (const [name, meta] of Object.entries(entry.clouds)) {
    if (meta.keep_indices) {
      clouds[name] = {
        keepIdx: await fetchBuffer(`${sev}_${fileSafe(name)}.keep.i32`, Int32Array),
        derived: true,
      };
    } else {
      const base = `${sev}_${fileSafe(name)}`;
      const c = { xyz: dequantise(await fetchBuffer(`${base}.xyz.i16`, Int16Array), meta.scale), n: meta.n };
      if (meta.intensity) c.intensity = await fetchBuffer(`${base}.intensity.u8`, Uint8Array);
      if (meta.label) c.label = await fetchBuffer(`${base}.label.u8`, Uint8Array);
      clouds[name] = c;
    }
  }
  // removal-only methods are stored as an index set into the noisy cloud
  for (const c of Object.values(clouds)) {
    if (!c.derived) continue;
    c.keep = new Uint8Array(clouds.noisy.n);
    for (let i = 0; i < c.keepIdx.length; i++) c.keep[c.keepIdx[i]] = 1;
  }
  return { clouds, metrics: entry.metrics, rain: entry.rain_rate, nFull: entry.n_full };
}

// ── small DOM helpers ────────────────────────────────────────────────────
function segmented(host, items, initial, onChange) {
  host.innerHTML = '';
  const buttons = items.map(([value, label, tag]) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.setAttribute('role', 'radio');
    b.innerHTML = tag ? `<span>${label}</span><span class="tag">${tag}</span>` : label;
    b.addEventListener('click', () => { select(value); onChange(value); });
    host.appendChild(b);
    return [value, b];
  });
  const select = (v) => buttons.forEach(([val, b]) =>
    b.setAttribute('aria-checked', String(val === v)));
  select(initial);
  return select;
}

const rows = (pairs) => pairs.map(([k, v, cls]) =>
  k === null ? '<div class="sep"></div>'
    : `<div class="k">${k}</div><div class="v ${cls || ''}">${v}</div>`).join('');

const pct = (v) => `${v.toFixed(1)}%`;
const f3 = (v) => v.toFixed(3);

// ── explorer ─────────────────────────────────────────────────────────────
function initExplorer(manifest) {
  const viewer = new PointCloudViewer(document.getElementById('main-canvas'));
  const overlay = document.getElementById('canvas-overlay');
  const readout = document.getElementById('readout');
  const hint = document.getElementById('hint-severity');

  const state = { sev: 'moderate', method: 'noisy', colour: 'height', showRemoved: true };

  segmented(document.getElementById('seg-severity'),
    SEVERITIES.map(([v, l, t]) => [v, l, t]), state.sev,
    (v) => { state.sev = v; draw(); });

  const methodItems = METHODS.map((m) => {
    const met = manifest.severities.moderate.metrics[m === 'clean' ? 'oracle' : m];
    const label = m === 'noisy' ? 'Corrupted input' : m === 'clean' ? 'Clean (ground truth)' : m;
    const tag = LEARNED.has(m) ? 'learned' : (m === 'clean' || m === 'noisy') ? '' : 'filter';
    return [m, label, tag];
  });
  segmented(document.getElementById('seg-method'), methodItems, state.method,
    (v) => { state.method = v; draw(); });

  segmented(document.getElementById('seg-colour'), [
    ['height', 'Height'], ['intensity', 'Intensity'], ['label', 'True noise'],
  ], state.colour, (v) => { state.colour = v; draw(); });

  document.getElementById('chk-removed').addEventListener('change', (e) => {
    state.showRemoved = e.target.checked; draw();
  });
  const size = document.getElementById('rng-size');
  size.addEventListener('input', () => {
    viewer.setPointSize(+size.value);
    document.getElementById('out-size').textContent = (+size.value).toFixed(2);
  });
  document.getElementById('btn-reset').addEventListener('click', () => viewer.resetView());
  const rot = document.getElementById('btn-rotate');
  rot.addEventListener('click', () => {
    viewer.autoRotate = !viewer.autoRotate;
    rot.setAttribute('aria-pressed', String(viewer.autoRotate));
  });

  function draw() {
    const S = store.severities[state.sev];
    const noisy = S.clouds.noisy;
    const entry = S.clouds[state.method];
    let shown;

    if (state.method === 'clean') {
      shown = viewer.render(S.clouds.clean.xyz, S.clouds.clean.n, {
        mode: state.colour === 'label' ? 'height' : state.colour,
        intensity: S.clouds.clean.intensity,
      });
    } else if (entry.derived) {
      shown = viewer.render(noisy.xyz, noisy.n, {
        mode: state.colour, intensity: noisy.intensity, labels: noisy.label,
        keep: entry.keep, showRemoved: state.showRemoved,
      });
    } else {
      shown = viewer.render(entry.xyz, entry.n, {
        mode: state.colour === 'label' ? 'height' : state.colour,
        intensity: entry.intensity, labels: entry.label,
      });
    }

    const rain = SEVERITIES.find((s) => s[0] === state.sev)[2];
    hint.textContent = `${rain}. The model destroys ${
      (100 - S.metrics.oracle.retention_pct).toFixed(0)}% of returns outright at this severity.`;

    overlay.innerHTML = `<b>${methodItems.find((m) => m[0] === state.method)[1]}</b><br>` +
      `${shown.toLocaleString()} points shown · ${rain}`;

    const key = state.method === 'clean' ? 'oracle' : state.method;
    const m = S.metrics[key];
    if (!m) { readout.innerHTML = ''; return; }

    const list = [];
    if (state.method === 'clean') {
      list.push(['Reference', 'ground truth']);
      list.push(['Oracle CD', `${f3(S.metrics.oracle.cd)} m`, 'good']);
    } else {
      list.push(['Chamfer Distance', `${f3(m.cd)} m`,
        m.cd < S.metrics.noisy.cd * 0.6 ? 'good' : m.cd > S.metrics.noisy.cd * 0.95 ? 'bad' : '']);
      list.push(['Retention', pct(m.retention_pct)]);
      if (m.removal_f1 !== undefined) {
        list.push([null]);
        list.push(['Removal precision', f3(m.removal_precision)]);
        list.push(['Removal recall', f3(m.removal_recall)]);
        list.push(['Removal F1', f3(m.removal_f1),
          m.removal_f1 > 0.75 ? 'good' : m.removal_f1 === 0 ? 'bad' : '']);
      }
      if (m.time_ms !== undefined) {
        list.push([null]);
        list.push(['Latency', `${m.time_ms.toFixed(0)} ms`,
          m.time_ms < 100 ? 'good' : m.time_ms > 500 ? 'bad' : '']);
      }
    }
    readout.innerHTML = rows(list);
  }

  draw();
  return viewer;
}

// ── live tuner ───────────────────────────────────────────────────────────
const FILTER_DEFS = {
  'SOR': {
    label: 'SOR', tag: 'statistical',
    params: {
      k: { label: 'Neighbours k', min: 4, max: 48, step: 1, value: 20 },
      stdRatio: { label: 'Std-dev multiplier', min: 0.5, max: 3.5, step: 0.1, value: 2.0 },
    },
  },
  'SOR+ROR': {
    label: 'SOR + ROR', tag: 'baseline',
    params: {
      k: { label: 'Neighbours k', min: 4, max: 48, step: 1, value: 20 },
      stdRatio: { label: 'Std-dev multiplier', min: 0.5, max: 3.5, step: 0.1, value: 2.0 },
      minPoints: { label: 'ROR min neighbours', min: 1, max: 12, step: 1, value: 4 },
      radius: { label: 'ROR radius (m)', min: 0.2, max: 1.5, step: 0.05, value: 0.6 },
    },
  },
  'IDSOR': {
    label: 'IDSOR', tag: 'uses intensity',
    params: {
      k: { label: 'Neighbours k', min: 4, max: 48, step: 1, value: 20 },
      stdRatio: { label: 'Std-dev multiplier', min: 0.5, max: 3.5, step: 0.1, value: 1.5 },
      rangeBeta: { label: 'Range normalisation', min: 0, max: 6, step: 0.25, value: 3.0 },
      intensityGamma: { label: 'Intensity exponent', min: 0, max: 4, step: 0.25, value: 2.0 },
      intensityFloor: { label: 'Weak-return floor', min: 0.1, max: 1, step: 0.05, value: 0.7 },
    },
  },
};

function initTuner() {
  const viewer = new PointCloudViewer(document.getElementById('tune-canvas'));
  const overlay = document.getElementById('tune-overlay');
  const readout = document.getElementById('tune-readout');
  const timing = document.getElementById('tune-timing');
  const controls = document.getElementById('filter-controls');

  const S = store.severities.moderate;
  const noisy = S.clouds.noisy;
  const grid = new Grid(noisy.xyz, noisy.n);
  const dCache = new Map(); // mean k-NN distance is the expensive part; k is the only key

  let current = 'SOR+ROR';
  let params = {};

  const resetParams = () => {
    params = {};
    for (const [k, def] of Object.entries(FILTER_DEFS[current].params)) params[k] = def.value;
  };

  segmented(document.getElementById('seg-filter'),
    Object.entries(FILTER_DEFS).map(([k, v]) => [k, v.label, v.tag]), current,
    (v) => { current = v; resetParams(); buildControls(); run(); });

  function buildControls() {
    controls.innerHTML = '';
    for (const [key, def] of Object.entries(FILTER_DEFS[current].params)) {
      const wrap = document.createElement('div');
      wrap.className = 'field';
      wrap.innerHTML =
        `<label for="p-${key}">${def.label} <output id="o-${key}">${params[key]}</output></label>
         <input type="range" id="p-${key}" min="${def.min}" max="${def.max}"
                step="${def.step}" value="${params[key]}">`;
      controls.appendChild(wrap);
      const input = wrap.querySelector('input');
      const out = wrap.querySelector('output');
      input.addEventListener('input', () => {
        params[key] = +input.value;
        out.textContent = def.step < 1 ? (+input.value).toFixed(2) : input.value;
        schedule();
      });
    }
  }

  let pending = null;
  const schedule = () => {
    if (pending) cancelAnimationFrame(pending);
    pending = requestAnimationFrame(() => { pending = null; run(); });
  };

  let lastKnnMs = 0;
  function knn(k) {
    if (!dCache.has(k)) {
      const t = performance.now();
      dCache.set(k, meanKnnDistance(grid, k));
      lastKnnMs = performance.now() - t;
    } else {
      lastKnnMs = 0; // reused: k did not change
    }
    return dCache.get(k);
  }

  function run() {
    const t0 = performance.now();
    const d = knn(params.k);
    let keep;
    if (current === 'IDSOR') {
      keep = idsor(grid, noisy.intensity, params, d).keep;
    } else {
      keep = sor(grid, params, d).keep;
      if (current === 'SOR+ROR') {
        const second = ror(grid, params, keep).keep;
        for (let i = 0; i < keep.length; i++) keep[i] = keep[i] && second[i] ? 1 : 0;
      }
    }
    const ms = performance.now() - t0;

    const s = removalScores(keep, noisy.label);
    viewer.render(noisy.xyz, noisy.n, {
      mode: 'label', labels: noisy.label, keep, showRemoved: true,
    });

    overlay.innerHTML =
      `<b>${FILTER_DEFS[current].label}</b>, computed live<br>` +
      `<span style="color:#3987e5">■</span> real kept &nbsp;` +
      `<span style="color:#e66767">■</span> noise kept &nbsp;` +
      `<span style="color:#8a3a3a">■</span> removed`;

    readout.innerHTML = rows([
      ['Points kept', `${s.kept.toLocaleString()} / ${noisy.n.toLocaleString()}`],
      ['Retention', pct(s.retention)],
      [null],
      ['Removal precision', f3(s.precision), s.precision > 0.85 ? 'good' : ''],
      ['Removal recall', f3(s.recall), s.recall > 0.7 ? 'good' : ''],
      ['Removal F1', f3(s.f1), s.f1 > 0.7 ? 'good' : s.f1 < 0.4 ? 'bad' : ''],
    ]);
    timing.textContent = lastKnnMs
      ? `Recomputed in ${ms.toFixed(0)} ms on ${noisy.n.toLocaleString()} points ` +
        `(${lastKnnMs.toFixed(0)} ms of it rebuilding the k-NN graph), in your browser.`
      : `Recomputed in ${ms.toFixed(0)} ms on ${noisy.n.toLocaleString()} points, ` +
        `reusing the cached k-NN graph. Change k to force a rebuild.`;
  }

  document.getElementById('btn-tune-reset').addEventListener('click', () => viewer.resetView());
  document.getElementById('btn-defaults').addEventListener('click',
    () => { resetParams(); buildControls(); run(); });

  resetParams();
  buildControls();
  run();
}

// ── dashboard ────────────────────────────────────────────────────────────
const PRETTY = {
  'oracle (perfect removal)': 'Oracle (perfect removal)',
  dgcnn: 'DGCNN', IDSOR: 'IDSOR', RangeRestoreNet: 'RangeRestoreNet',
  'SOR+ROR': 'SOR+ROR', 'Temporal+SOR': 'Temporal+SOR',
  pointnet_ae: 'PointNet-AE', 'noisy (no denoising)': 'Corrupted input', noisy: 'Corrupted input',
};
const ORDER = ['oracle (perfect removal)', 'dgcnn', 'IDSOR', 'RangeRestoreNet',
  'SOR+ROR', 'Temporal+SOR', 'pointnet_ae'];

function initDashboard(metrics, detection, ablations) {
  const M = metrics.summary, D = detection.summary;
  const has = (n) => n in M;

  barChart(document.getElementById('chart-cd'),
    [{ label: 'Corrupted input', value: M['noisy (no denoising)'].cd, colour: C.bad },
      ...ORDER.filter(has).map((n) => ({
        label: PRETTY[n], value: M[n].cd,
        emphasis: n === 'dgcnn',
        colour: n.startsWith('oracle') ? C.ref : undefined,
      }))],
    { unit: ' m', reference: M['oracle (perfect removal)'].cd, referenceLabel: 'achievable floor' });

  barChart(document.getElementById('chart-gap'),
    ORDER.filter((n) => has(n) && isFinite(M[n].oracle_gap_pct)).map((n) => ({
      label: PRETTY[n], value: M[n].oracle_gap_pct,
      emphasis: n === 'dgcnn', colour: n.startsWith('oracle') ? C.ref : undefined,
    })), { unit: '%', fmt: (v) => v.toFixed(1), max: 100 });

  const detRows = ['noisy', ...ORDER.filter((n) => n in D)];
  barChart(document.getElementById('chart-det'),
    detRows.map((n) => {
      const sig = detection.significance_vs_oracle[`${n} vs oracle (perfect removal)`];
      const ns = sig && !sig['significant_at_0.05'];
      return {
        label: PRETTY[n] + (ns ? '  (n.s.)' : ''), value: D[n].recall,
        emphasis: ns, colour: n === 'pointnet_ae' ? C.bad
          : n.startsWith('oracle') ? C.ref : n === 'noisy' ? C.muted : undefined,
      };
    }), { reference: D['oracle (perfect removal)'].recall, referenceLabel: 'oracle', max: 0.62 });

  const prfNames = ORDER.filter((n) => has(n) && M[n].removal_f1 !== undefined
    && !n.startsWith('oracle'));
  groupedBarChart(document.getElementById('chart-prf'),
    prfNames.map((n) => PRETTY[n]), [
      { label: 'precision', colour: C.accent, values: prfNames.map((n) => M[n].removal_precision) },
      { label: 'recall', colour: C.hi, values: prfNames.map((n) => M[n].removal_recall) },
      { label: 'F1', colour: C.good, values: prfNames.map((n) => M[n].removal_f1) },
    ]);

  // hand-placed: seven labels in a small plot collide under any automatic
  // rule, and identity here is carried by the label, so it has to be legible
  const place = {
    'oracle (perfect removal)': ['left', 4], dgcnn: ['right', 4],
    IDSOR: ['center', 22], RangeRestoreNet: ['center', -14],
    'SOR+ROR': ['center', 22], 'Temporal+SOR': ['left', 4], pointnet_ae: ['right', 4],
  };
  scatterChart(document.getElementById('chart-scatter'),
    [...ORDER.filter((n) => has(n) && n in D).map((n) => ({
      x: M[n].cd, y: D[n].recall, label: PRETTY[n],
      align: (place[n] || ['left', 4])[0], dy: (place[n] || ['left', 4])[1],
      colour: n === 'pointnet_ae' ? C.bad : n.startsWith('oracle') ? C.ref : C.accent,
    })), {
      x: M['noisy (no denoising)'].cd, y: D.noisy.recall,
      label: 'Corrupted input', align: 'right', colour: C.muted,
    }],
    { xLabel: 'Chamfer Distance (m), lower is better',
      yLabel: 'detection recall, higher is better' });

  const lat = ORDER.filter((n) => has(n) && M[n].time_ms > 0);
  barChart(document.getElementById('chart-latency'),
    lat.map((n) => ({
      label: PRETTY[n], value: M[n].time_ms,
      emphasis: M[n].time_ms < 100,
      colour: M[n].time_ms < 100 ? C.good : undefined,
    })), { unit: ' ms', fmt: (v) => v.toFixed(0), reference: 100, referenceLabel: '10 Hz budget' });

  const sev = ablations.severity;
  const keys = ['noisy', 'dgcnn', 'IDSOR', 'RangeRestoreNet', 'SOR+ROR', 'pointnet_ae']
    .filter((k) => k in sev.series);
  const hues = [C.muted, C.accent, C.good, C.hi, '#c98500', C.bad];
  lineChart(document.getElementById('chart-severity'),
    sev.levels.map((l, i) => `${l}\n${[10, 25, 50][i]} mm/h`),
    keys.map((k, i) => ({ label: PRETTY[k] || k, colour: hues[i % hues.length], values: sev.series[k] })),
    { xLabel: 'corruption severity' });
}

// ── boot ─────────────────────────────────────────────────────────────────
/**
 * Replace a dead canvas with an explanation.
 *
 * The 3D views are the best part of this site but they are not the whole of
 * it: the charts, the findings and the methodology are plain DOM. A machine
 * without WebGL — headless, locked-down, or an old integrated GPU — should
 * still get everything else rather than a blank page, so viewer failures are
 * contained here instead of aborting the boot.
 */
function canvasFallback(id, message) {
  const canvas = document.getElementById(id);
  if (!canvas) return;
  const shell = canvas.parentElement;
  canvas.remove();
  const box = document.createElement('div');
  box.className = 'canvas-fallback';
  box.innerHTML = `<p><b>3D view unavailable</b></p><p>${message}</p>`;
  shell.appendChild(box);
}

(async function main() {
  const loading = document.getElementById('loading');
  let manifest, metrics, detection, ablations;

  // 1. data — genuinely fatal if it fails, since everything reads from it
  try {
    manifest = await fetchJSON('manifest.json');
    [metrics, detection, ablations] = await Promise.all([
      fetchJSON('metrics_summary.json'),
      fetchJSON('detection_consistency.json'),
      fetchJSON('ablations.json'),
    ]);
    for (const [sev, entry] of Object.entries(manifest.severities)) {
      store.severities[sev] = await loadSeverity(sev, entry);
    }
  } catch (err) {
    console.error('data load failed', err);
    loading.innerHTML = `<p style="color:#e66767;max-width:40ch;text-align:center">
      Could not load the point-cloud data.<br><code>${err.message}</code><br><br>
      <span style="color:#868b94">If you opened this page from the file system, serve it over
      HTTP instead, because browsers block <code>fetch</code> on <code>file://</code>.</span></p>`;
    return;
  }

  // 2. charts and narrative — no WebGL needed, so do these first
  try {
    initDashboard(metrics, detection, ablations);
  } catch (err) {
    console.error('dashboard failed', err);
  }

  // 3. 3D views — degrade individually
  const NO_GL = 'This browser could not create a WebGL context. Every chart and result ' +
    'on this page is still available below.';
  if (!webglAvailable()) {
    ['hero-canvas', 'main-canvas', 'tune-canvas'].forEach((id) => canvasFallback(id, NO_GL));
  } else {
    try {
      const hero = new PointCloudViewer(document.getElementById('hero-canvas'));
      const n = store.severities.moderate.clouds.noisy;
      hero.render(n.xyz, n.n, { mode: 'height' });
      hero.autoRotate = true;
    } catch (err) { console.error(err); canvasFallback('hero-canvas', NO_GL); }

    try { initExplorer(manifest); }
    catch (err) { console.error(err); canvasFallback('main-canvas', NO_GL); }

    try { initTuner(); }
    catch (err) { console.error(err); canvasFallback('tune-canvas', NO_GL); }
  }

  loading.classList.add('hidden');
})();
