/* three.js point-cloud viewer. */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// Dark-surface palette steps (validated as a set for the dark background).
export const PALETTE = {
  surface: 0x101012,
  clean: 0x199e70,
  noise: 0xe66767,
  real: 0x3987e5,
  removed: 0x8a3a3a,
  neutral: 0x9aa0a6,
};

const HEIGHT_RAMP = [
  [0.0, 0x1c3f6e], [0.35, 0x2a78d6], [0.6, 0x199e70],
  [0.8, 0xc98500], [1.0, 0xe8e6df],
];

function rampColour(t, out, o) {
  let a = HEIGHT_RAMP[0], b = HEIGHT_RAMP[HEIGHT_RAMP.length - 1];
  for (let i = 0; i < HEIGHT_RAMP.length - 1; i++) {
    if (t >= HEIGHT_RAMP[i][0] && t <= HEIGHT_RAMP[i + 1][0]) {
      a = HEIGHT_RAMP[i]; b = HEIGHT_RAMP[i + 1]; break;
    }
  }
  const f = b[0] === a[0] ? 0 : (t - a[0]) / (b[0] - a[0]);
  const ca = a[1], cb = b[1];
  out[o] = (((ca >> 16 & 255) * (1 - f) + (cb >> 16 & 255) * f) / 255) ** 2.2;
  out[o + 1] = (((ca >> 8 & 255) * (1 - f) + (cb >> 8 & 255) * f) / 255) ** 2.2;
  out[o + 2] = (((ca & 255) * (1 - f) + (cb & 255) * f) / 255) ** 2.2;
}

function hexToLinear(hex, out, o, dim = 1) {
  out[o] = ((hex >> 16 & 255) / 255) ** 2.2 * dim;
  out[o + 1] = ((hex >> 8 & 255) / 255) ** 2.2 * dim;
  out[o + 2] = ((hex & 255) / 255) ** 2.2 * dim;
}

/** True if this browser can actually give us a WebGL context. */
export function webglAvailable() {
  try {
    const c = document.createElement('canvas');
    return !!(c.getContext('webgl2') || c.getContext('webgl'));
  } catch { return false; }
}

export class PointCloudViewer {
  constructor(canvas) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({
      canvas, antialias: true, alpha: false, powerPreference: 'high-performance',
    });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(PALETTE.surface);
    this.scene.fog = new THREE.Fog(PALETTE.surface, 55, 130);

    this.camera = new THREE.PerspectiveCamera(52, 1, 0.1, 500);
    this.camera.position.set(-24, -24, 16);
    this.camera.up.set(0, 0, 1);

    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.target.set(2, 0, -1);
    this.controls.maxDistance = 160;
    this.controls.minDistance = 3;

    this.#addGround();

    this.geometry = new THREE.BufferGeometry();
    this.material = new THREE.PointsMaterial({
      size: 0.13, vertexColors: true, sizeAttenuation: true,
    });
    this.points = new THREE.Points(this.geometry, this.material);
    this.points.frustumCulled = false;
    this.scene.add(this.points);

    this.autoRotate = false;
    new ResizeObserver(() => this.resize()).observe(canvas.parentElement);
    this.resize();
    this.#loop();
  }

  #addGround() {
    const grid = new THREE.GridHelper(120, 24, 0x2a2a30, 0x1c1c22);
    grid.rotation.x = Math.PI / 2;
    grid.position.z = -1.75;
    this.scene.add(grid);
    // sensor origin marker — orients the viewer to where the LiDAR sits
    const s = new THREE.Mesh(
      new THREE.SphereGeometry(0.42, 16, 12),
      new THREE.MeshBasicMaterial({ color: 0xeda100 })
    );
    this.scene.add(s);
  }

  resize() {
    const el = this.canvas.parentElement;
    const w = el.clientWidth, h = el.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  #loop = () => {
    requestAnimationFrame(this.#loop);
    if (this.autoRotate) {
      const a = 0.0016;
      const { x, y } = this.camera.position;
      this.camera.position.x = x * Math.cos(a) - y * Math.sin(a);
      this.camera.position.y = x * Math.sin(a) + y * Math.cos(a);
    }
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  };

  setPointSize(v) { this.material.size = v; }

  resetView() {
    this.camera.position.set(-24, -24, 16);
    this.controls.target.set(2, 0, -1);
  }

  /**
   * Render a cloud.
   * `keep` (optional Uint8Array) dims removed points instead of hiding them,
   * so the visitor can see WHAT a filter deleted — the most informative view
   * when tuning a threshold, and impossible to judge from the survivors alone.
   */
  render(xyz, n, { mode = 'height', intensity = null, labels = null,
                   keep = null, showRemoved = true } = {}) {
    const positions = new Float32Array(n * 3);
    const colours = new Float32Array(n * 3);
    let w = 0;

    let zMin = Infinity, zMax = -Infinity;
    for (let i = 0; i < n; i++) {
      const z = xyz[i * 3 + 2];
      if (z < zMin) zMin = z; if (z > zMax) zMax = z;
    }
    const span = Math.max(1e-6, zMax - zMin);

    for (let i = 0; i < n; i++) {
      const removed = keep ? !keep[i] : false;
      if (removed && !showRemoved) continue;
      const o = w * 3;
      positions[o] = xyz[i * 3];
      positions[o + 1] = xyz[i * 3 + 1];
      positions[o + 2] = xyz[i * 3 + 2];

      if (removed) {
        hexToLinear(PALETTE.removed, colours, o, 0.55);
      } else if (mode === 'intensity' && intensity) {
        const t = Math.min(1, intensity[i] / 200);
        colours[o] = t ** 2.2; colours[o + 1] = (t * 0.72) ** 2.2; colours[o + 2] = (t * 0.28) ** 2.2;
      } else if (mode === 'label' && labels) {
        hexToLinear(labels[i] > 0 ? PALETTE.noise : PALETTE.real, colours, o);
      } else {
        rampColour((xyz[i * 3 + 2] - zMin) / span, colours, o);
      }
      w++;
    }

    this.geometry.setAttribute('position',
      new THREE.BufferAttribute(positions.subarray(0, w * 3), 3));
    this.geometry.setAttribute('color',
      new THREE.BufferAttribute(colours.subarray(0, w * 3), 3));
    this.geometry.attributes.position.needsUpdate = true;
    this.geometry.attributes.color.needsUpdate = true;
    this.geometry.setDrawRange(0, w);
    this.geometry.computeBoundingSphere();
    return w;
  }
}
