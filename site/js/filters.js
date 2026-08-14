/* Statistical denoising filters, reimplemented for the browser.
 *
 * These are the real algorithms from src/lidar_denoise/filters.py, not a
 * mock-up: SOR, ROR and IDSOR are geometric procedures with no learned
 * parameters, so they can run live while a visitor drags a slider. The
 * learned models (DGCNN, PointNet-AE, RangeRestoreNet) cannot — PyTorch does
 * not run in a browser — and are served precomputed. The UI says which is
 * which rather than blurring the distinction.
 *
 * Neighbour search uses a uniform voxel grid rather than a k-d tree: point
 * density here is roughly uniform after voxel downsampling, which is the case
 * a grid handles best, and it builds in a single linear pass.
 */

const CELL = 1.0; // metres — ~1.5x the largest radius we query

export class Grid {
  constructor(xyz, n) {
    this.xyz = xyz;
    this.n = n;
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    for (let i = 0; i < n; i++) {
      const x = xyz[i * 3], y = xyz[i * 3 + 1], z = xyz[i * 3 + 2];
      if (x < minX) minX = x; if (x > maxX) maxX = x;
      if (y < minY) minY = y; if (y > maxY) maxY = y;
      if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
    }
    this.min = [minX, minY, minZ];
    this.dim = [
      Math.max(1, Math.ceil((maxX - minX) / CELL) + 1),
      Math.max(1, Math.ceil((maxY - minY) / CELL) + 1),
      Math.max(1, Math.ceil((maxZ - minZ) / CELL) + 1),
    ];
    const nCells = this.dim[0] * this.dim[1] * this.dim[2];

    // counting sort into cells: two passes, no per-cell arrays
    const counts = new Int32Array(nCells + 1);
    const cellOf = new Int32Array(n);
    for (let i = 0; i < n; i++) {
      const c = this.cellIndex(xyz[i * 3], xyz[i * 3 + 1], xyz[i * 3 + 2]);
      cellOf[i] = c;
      counts[c + 1]++;
    }
    for (let c = 0; c < nCells; c++) counts[c + 1] += counts[c];
    this.start = counts;
    this.items = new Int32Array(n);
    const cursor = counts.slice(0, nCells);
    for (let i = 0; i < n; i++) this.items[cursor[cellOf[i]]++] = i;
  }

  cellIndex(x, y, z) {
    const [mx, my, mz] = this.min, [dx, dy] = this.dim;
    const i = Math.min(this.dim[0] - 1, Math.max(0, Math.floor((x - mx) / CELL)));
    const j = Math.min(this.dim[1] - 1, Math.max(0, Math.floor((y - my) / CELL)));
    const k = Math.min(this.dim[2] - 1, Math.max(0, Math.floor((z - mz) / CELL)));
    return (k * dy + j) * dx + i;
  }

  /** Visit every point in the 3x3x3 cell block around point `p`. */
  forEachNeighbour(p, fn) {
    const { xyz, dim, min, start, items } = this;
    const x = xyz[p * 3], y = xyz[p * 3 + 1], z = xyz[p * 3 + 2];
    const ci = Math.floor((x - min[0]) / CELL);
    const cj = Math.floor((y - min[1]) / CELL);
    const ck = Math.floor((z - min[2]) / CELL);
    for (let k = Math.max(0, ck - 1); k <= Math.min(dim[2] - 1, ck + 1); k++)
      for (let j = Math.max(0, cj - 1); j <= Math.min(dim[1] - 1, cj + 1); j++)
        for (let i = Math.max(0, ci - 1); i <= Math.min(dim[0] - 1, ci + 1); i++) {
          const c = (k * dim[1] + j) * dim[0] + i;
          for (let s = start[c]; s < start[c + 1]; s++) {
            const q = items[s];
            if (q !== p) fn(q);
          }
        }
  }
}

/** Mean distance from each point to its k nearest neighbours. */
export function meanKnnDistance(grid, k) {
  const { xyz, n } = grid;
  const out = new Float32Array(n);
  const heap = new Float64Array(k); // max-heap of the k smallest squared dists

  for (let p = 0; p < n; p++) {
    let size = 0;
    const px = xyz[p * 3], py = xyz[p * 3 + 1], pz = xyz[p * 3 + 2];
    grid.forEachNeighbour(p, (q) => {
      const dx = xyz[q * 3] - px, dy = xyz[q * 3 + 1] - py, dz = xyz[q * 3 + 2] - pz;
      const d2 = dx * dx + dy * dy + dz * dz;
      if (size < k) {
        // sift up
        let i = size++;
        heap[i] = d2;
        while (i > 0) {
          const parent = (i - 1) >> 1;
          if (heap[parent] >= heap[i]) break;
          const t = heap[parent]; heap[parent] = heap[i]; heap[i] = t;
          i = parent;
        }
      } else if (d2 < heap[0]) {
        heap[0] = d2;
        let i = 0;
        for (;;) { // sift down
          const l = 2 * i + 1, r = l + 1;
          let big = i;
          if (l < k && heap[l] > heap[big]) big = l;
          if (r < k && heap[r] > heap[big]) big = r;
          if (big === i) break;
          const t = heap[big]; heap[big] = heap[i]; heap[i] = t;
          i = big;
        }
      }
    });
    if (size === 0) { out[p] = Infinity; continue; } // isolated: definitely noise
    let sum = 0;
    for (let i = 0; i < size; i++) sum += Math.sqrt(heap[i]);
    out[p] = sum / size;
  }
  return out;
}

function meanStd(arr) {
  let n = 0, sum = 0;
  for (let i = 0; i < arr.length; i++) if (isFinite(arr[i])) { sum += arr[i]; n++; }
  const mean = n ? sum / n : 0;
  let acc = 0;
  for (let i = 0; i < arr.length; i++) if (isFinite(arr[i])) acc += (arr[i] - mean) ** 2;
  return [mean, n ? Math.sqrt(acc / n) : 0];
}

/** Statistical Outlier Removal — global threshold on mean k-NN distance. */
export function sor(grid, { k = 20, stdRatio = 2.0 }, cachedD = null) {
  const d = cachedD || meanKnnDistance(grid, k);
  const [mean, std] = meanStd(d);
  const thresh = mean + stdRatio * std;
  const keep = new Uint8Array(grid.n);
  for (let i = 0; i < grid.n; i++) keep[i] = d[i] <= thresh ? 1 : 0;
  return { keep, d, thresh };
}

/** Radius Outlier Removal — minimum neighbour count inside a radius. */
export function ror(grid, { minPoints = 4, radius = 0.6 }, mask = null) {
  const r2 = radius * radius;
  const { xyz, n } = grid;
  const keep = new Uint8Array(n);
  for (let p = 0; p < n; p++) {
    if (mask && !mask[p]) continue;
    const px = xyz[p * 3], py = xyz[p * 3 + 1], pz = xyz[p * 3 + 2];
    let count = 0;
    grid.forEachNeighbour(p, (q) => {
      if (mask && !mask[q]) return;
      const dx = xyz[q * 3] - px, dy = xyz[q * 3 + 1] - py, dz = xyz[q * 3 + 2] - pz;
      if (dx * dx + dy * dy + dz * dz <= r2) count++;
    });
    keep[p] = count >= minPoints ? 1 : 0;
  }
  return { keep };
}

/**
 * IDSOR — intensity- and range-adaptive SOR (arXiv:2602.05876).
 *
 * Two corrections to plain SOR, in opposite directions. Angular resolution is
 * fixed, so genuine returns spread out with range; dividing the statistic by
 * (1 + beta*r/ref) makes it comparable across the scan instead of eroding
 * far-field structure. Separately, weather backscatter returns less energy,
 * so the threshold is tightened for weak returns.
 */
export function idsor(grid, intensity, {
  k = 20, stdRatio = 1.5, rangeBeta = 3.0, rangeRef = 30.0,
  intensityGamma = 2.0, intensityFloor = 0.7,
}, cachedD = null) {
  const { xyz, n } = grid;
  const d = cachedD || meanKnnDistance(grid, k);

  const dNorm = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const r = Math.hypot(xyz[i * 3], xyz[i * 3 + 1], xyz[i * 3 + 2]);
    dNorm[i] = d[i] / (1 + rangeBeta * r / Math.max(rangeRef, 1e-6));
  }
  const [mean, std] = meanStd(dNorm);
  const base = mean + stdRatio * std;

  // normalise intensity against the 95th percentile, not the max: one
  // specular return would otherwise squash everything else toward zero
  const sorted = Float32Array.from(intensity).sort();
  const scale = sorted[Math.min(n - 1, Math.floor(0.95 * n))] || 1;

  const keep = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const iNorm = Math.min(1, Math.max(0, intensity[i] / scale));
    const f = Math.pow(intensityFloor + (1 - intensityFloor) * iNorm, intensityGamma);
    keep[i] = dNorm[i] <= base * f ? 1 : 0;
  }
  return { keep, d };
}

/** Precision / recall / F1 of removal against ground-truth noise labels. */
export function removalScores(keep, labels) {
  let tp = 0, fp = 0, fn = 0, kept = 0;
  for (let i = 0; i < keep.length; i++) {
    const isNoise = labels[i] > 0, removed = !keep[i];
    if (keep[i]) kept++;
    if (removed && isNoise) tp++;
    else if (removed && !isNoise) fp++;
    else if (!removed && isNoise) fn++;
  }
  const precision = tp + fp ? tp / (tp + fp) : 0;
  const recall = tp + fn ? tp / (tp + fn) : 0;
  const f1 = precision + recall ? (2 * precision * recall) / (precision + recall) : 0;
  return { precision, recall, f1, retention: (100 * kept) / keep.length, kept };
}
