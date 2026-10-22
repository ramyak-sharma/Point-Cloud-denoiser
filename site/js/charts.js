/* Minimal SVG charts.
 *
 * No chart library: every chart here is a bar chart or a labelled scatter, and
 * a dependency would cost more bytes than the ~200 lines it replaces.
 *
 * Colour follows the same rule as the report figures. A chart of one measure
 * across seven methods is MAGNITUDE, not seven series, so it gets one hue plus
 * an accent for the row being emphasised — giving each bar its own colour
 * would imply a grouping that does not exist. The scatter carries identity in
 * direct labels because no palette separates seven categories reliably for
 * colour-vision-deficient readers across all pairs.
 */

const NS = 'http://www.w3.org/2000/svg';
export const C = {
  accent: '#3987e5', hi: '#d95926', ref: '#9aa0a6',
  good: '#199e70', bad: '#e66767', ink: '#e8e6df', muted: '#8b8f96',
  grid: '#26262c',
};

const el = (tag, attrs = {}, text = null) => {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (text !== null) n.textContent = text;
  return n;
};

function frame(host, w, h) {
  host.innerHTML = '';
  const svg = el('svg', {
    viewBox: `0 0 ${w} ${h}`, width: '100%', role: 'img',
    preserveAspectRatio: 'xMidYMid meet',
  });
  host.appendChild(svg);
  return svg;
}

/**
 * Horizontal bar chart — used wherever categories have long names, because
 * rotated x-labels are harder to read than left-aligned ones.
 */
export function barChart(host, rows, {
  valueKey = 'value', max = null, fmt = (v) => v.toFixed(3),
  reference = null, referenceLabel = '', unit = '',
} = {}) {
  const W = 640, rowH = 30, padL = 168, padR = 62, padT = 8;
  const H = padT + rows.length * rowH + 26;
  const svg = frame(host, W, H);
  const hi = max ?? Math.max(...rows.map((r) => r[valueKey])) * 1.08;
  const x = (v) => padL + (v / hi) * (W - padL - padR);

  if (reference !== null) {
    svg.appendChild(el('line', {
      x1: x(reference), x2: x(reference), y1: padT - 2, y2: padT + rows.length * rowH,
      stroke: C.ref, 'stroke-width': 1.2, 'stroke-dasharray': '3 3',
    }));
    svg.appendChild(el('text', {
      x: x(reference), y: H - 8, fill: C.ref, 'font-size': 11, 'text-anchor': 'middle',
    }, referenceLabel));
  }

  rows.forEach((r, i) => {
    const y = padT + i * rowH;
    svg.appendChild(el('text', {
      x: padL - 10, y: y + 15, fill: r.emphasis ? C.ink : C.muted,
      'font-size': 12, 'text-anchor': 'end',
      'font-weight': r.emphasis ? 600 : 400,
    }, r.label));
    const bw = Math.max(2, x(r[valueKey]) - padL);
    svg.appendChild(el('rect', {
      x: padL, y: y + 5, width: bw, height: 15, rx: 3,
      fill: r.colour || (r.emphasis ? C.hi : C.accent),
    }));
    svg.appendChild(el('text', {
      x: padL + bw + 7, y: y + 16, fill: C.ink, 'font-size': 11.5,
    }, fmt(r[valueKey]) + unit));
  });
  return svg;
}

/** Labelled scatter — identity in text, never colour alone. */
export function scatterChart(host, points, {
  xLabel = '', yLabel = '', xMax = null, yMax = null,
} = {}) {
  const W = 640, H = 400, padL = 62, padB = 46, padT = 16, padR = 16;
  const svg = frame(host, W, H);
  const xs = points.map((p) => p.x), ys = points.map((p) => p.y);
  const xhi = xMax ?? Math.max(...xs) * 1.12, yhi = yMax ?? Math.max(...ys) * 1.18;
  const X = (v) => padL + (v / xhi) * (W - padL - padR);
  const Y = (v) => H - padB - (v / yhi) * (H - padB - padT);

  for (let i = 0; i <= 4; i++) {
    const gy = padT + (i / 4) * (H - padB - padT);
    svg.appendChild(el('line', { x1: padL, x2: W - padR, y1: gy, y2: gy, stroke: C.grid }));
    svg.appendChild(el('text', {
      x: padL - 8, y: gy + 4, fill: C.muted, 'font-size': 10.5, 'text-anchor': 'end',
    }, (yhi * (1 - i / 4)).toFixed(2)));
  }
  for (let i = 0; i <= 4; i++) {
    const gx = padL + (i / 4) * (W - padL - padR);
    svg.appendChild(el('text', {
      x: gx, y: H - padB + 16, fill: C.muted, 'font-size': 10.5, 'text-anchor': 'middle',
    }, (xhi * (i / 4)).toFixed(2)));
  }

  points.forEach((p) => {
    svg.appendChild(el('circle', {
      cx: X(p.x), cy: Y(p.y), r: 6, fill: p.colour || C.accent,
      stroke: '#101012', 'stroke-width': 2,
    }));
    const anchor = p.align === 'right' ? 'end' : p.align === 'center' ? 'middle' : 'start';
    const dx = p.align === 'right' ? -11 : p.align === 'center' ? 0 : 11;
    svg.appendChild(el('text', {
      x: X(p.x) + dx, y: Y(p.y) + (p.dy ?? 4), fill: p.colour || C.ink,
      'font-size': 11.5, 'text-anchor': anchor,
    }, p.label));
  });

  svg.appendChild(el('text', {
    x: (W + padL) / 2, y: H - 6, fill: C.muted, 'font-size': 11.5, 'text-anchor': 'middle',
  }, xLabel));
  const yt = el('text', {
    x: 14, y: (H - padB + padT) / 2, fill: C.muted, 'font-size': 11.5,
    'text-anchor': 'middle', transform: `rotate(-90 14 ${(H - padB + padT) / 2})`,
  }, yLabel);
  svg.appendChild(yt);
  return svg;
}

/** Grouped bars — for precision / recall / F1 side by side. */
export function groupedBarChart(host, categories, series) {
  const W = 640, H = 300, padL = 46, padB = 62, padT = 26, padR = 12;
  const svg = frame(host, W, H);
  const bandW = (W - padL - padR) / categories.length;
  const barW = Math.min(17, (bandW - 12) / series.length);
  const Y = (v) => H - padB - v * (H - padB - padT);

  for (let i = 0; i <= 4; i++) {
    const gy = padT + (i / 4) * (H - padB - padT);
    svg.appendChild(el('line', { x1: padL, x2: W - padR, y1: gy, y2: gy, stroke: C.grid }));
    svg.appendChild(el('text', {
      x: padL - 8, y: gy + 4, fill: C.muted, 'font-size': 10.5, 'text-anchor': 'end',
    }, (1 - i / 4).toFixed(2)));
  }

  categories.forEach((cat, ci) => {
    const cx = padL + ci * bandW + bandW / 2;
    series.forEach((s, si) => {
      const v = s.values[ci];
      const x = cx - (series.length * barW) / 2 + si * barW;
      svg.appendChild(el('rect', {
        x: x + 1, y: Y(v), width: barW - 2, height: Math.max(1, H - padB - Y(v)),
        rx: 2, fill: s.colour,
      }));
    });
    svg.appendChild(el('text', {
      x: cx, y: H - padB + 15, fill: C.muted, 'font-size': 10.5, 'text-anchor': 'end',
      transform: `rotate(-22 ${cx} ${H - padB + 15})`,
    }, cat));
  });

  series.forEach((s, i) => {
    const lx = padL + i * 96;
    svg.appendChild(el('rect', { x: lx, y: 4, width: 10, height: 10, rx: 2, fill: s.colour }));
    svg.appendChild(el('text', { x: lx + 15, y: 13, fill: C.muted, 'font-size': 11 }, s.label));
  });
  return svg;
}

/** Multi-series line chart with direct labels (no legend box needed). */
export function lineChart(host, xTicks, series, { yLabel = '', xLabel = '' } = {}) {
  const W = 640, H = 300, padL = 54, padB = 46, padT = 14, padR = 128;
  const svg = frame(host, W, H);
  const all = series.flatMap((s) => s.values);
  const yhi = Math.max(...all) * 1.1, ylo = 0;
  const X = (i) => padL + (i / (xTicks.length - 1)) * (W - padL - padR);
  const Y = (v) => H - padB - ((v - ylo) / (yhi - ylo)) * (H - padB - padT);

  for (let i = 0; i <= 4; i++) {
    const gy = padT + (i / 4) * (H - padB - padT);
    svg.appendChild(el('line', { x1: padL, x2: W - padR, y1: gy, y2: gy, stroke: C.grid }));
    svg.appendChild(el('text', {
      x: padL - 8, y: gy + 4, fill: C.muted, 'font-size': 10.5, 'text-anchor': 'end',
    }, (yhi * (1 - i / 4)).toFixed(2)));
  }
  xTicks.forEach((t, i) => svg.appendChild(el('text', {
    x: X(i), y: H - padB + 17, fill: C.muted, 'font-size': 11, 'text-anchor': 'middle',
  }, t)));

  series.forEach((s) => {
    const d = s.values.map((v, i) => `${i ? 'L' : 'M'}${X(i)},${Y(v)}`).join(' ');
    svg.appendChild(el('path', { d, fill: 'none', stroke: s.colour, 'stroke-width': 2 }));
    s.values.forEach((v, i) => svg.appendChild(el('circle', {
      cx: X(i), cy: Y(v), r: 3.5, fill: s.colour,
    })));
    svg.appendChild(el('text', {
      x: X(s.values.length - 1) + 9, y: Y(s.values[s.values.length - 1]) + 4,
      fill: s.colour, 'font-size': 11,
    }, s.label));
  });

  svg.appendChild(el('text', {
    x: (W - padR + padL) / 2, y: H - 6, fill: C.muted, 'font-size': 11.5,
    'text-anchor': 'middle',
  }, xLabel));
  return svg;
}
