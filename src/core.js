export const CLASSES = [
  { id: 0, label: 'person', color: '#00a0ff' },
  { id: 24, label: 'backpack', color: '#ff8c00' },
  { id: 26, label: 'handbag', color: '#ffc800' },
  { id: 28, label: 'suitcase', color: '#ff50b4' },
];
const targetIds = new Set(CLASSES.map((item) => item.id));
export const emptyCounts = () => Object.fromEntries(CLASSES.map(({ label }) => [label, 0]));

export class PausableClock {
  constructor(read = () => performance.now() / 1000) {
    this.read = read;
    this.start = read();
    this.pausedAt = null;
    this.pausedTotal = 0;
  }
  now() { return (this.pausedAt ?? this.read()) - this.start - this.pausedTotal; }
  pause() { if (this.pausedAt === null) this.pausedAt = this.read(); }
  resume() {
    if (this.pausedAt !== null) {
      this.pausedTotal += this.read() - this.pausedAt;
      this.pausedAt = null;
    }
  }
}

export class SmileTrigger {
  constructor(hold = 0.25, cooldown = 3) {
    this.hold = hold;
    this.cooldown = cooldown;
    this.started = null;
    this.fired = false;
    this.lastFire = -Infinity;
  }
  update(smiling, now) {
    if (!smiling) {
      this.started = null;
      this.fired = false;
      return false;
    }
    if (this.started === null) this.started = now;
    if (this.fired || now - this.started < this.hold || now - this.lastFire < this.cooldown) return false;
    this.fired = true;
    this.lastFire = now;
    return true;
  }
}

export function insetProgress(elapsed, slide = 0.4, hold = 3) {
  if (elapsed >= hold + 2 * slide) return null;
  const t = elapsed < slide ? elapsed / slide : elapsed < slide + hold ? 1 : (hold + 2 * slide - elapsed) / slide;
  return t * t * (3 - 2 * t);
}

export function intersectionOverUnion(a, b) {
  const intersection = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0])) *
    Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const area = (box) => Math.max(0, box[2] - box[0]) * Math.max(0, box[3] - box[1]);
  return intersection / (area(a) + area(b) - intersection || 1);
}

export function letterbox(width, height, size = 320) {
  const scale = Math.min(size / width, size / height);
  const drawWidth = Math.round(width * scale), drawHeight = Math.round(height * scale);
  return { size, drawWidth, drawHeight, left: Math.floor((size - drawWidth) / 2),
    top: Math.floor((size - drawHeight) / 2), scaleX: drawWidth / width, scaleY: drawHeight / height };
}

export function decodeYolo(data, dims, transform, width, height, confidence = 0.1, nms = 0.45) {
  if (dims.length !== 3 || dims[0] !== 1 || dims[1] !== 84) {
    throw new Error(`Unexpected YOLO output shape ${dims.join('×')}; use scripts/export-model.py.`);
  }
  const count = dims[2], candidates = [];
  const clamp = (value, max) => Math.max(0, Math.min(value, max));
  for (let i = 0; i < count; i++) {
    let cls = -1, score = confidence;
    for (let c = 0; c < 80; c++) {
      const value = data[(c + 4) * count + i];
      if (value > score) { score = value; cls = c; }
    }
    if (!targetIds.has(cls)) continue;
    const cx = data[i], cy = data[count + i], w = data[2 * count + i], h = data[3 * count + i];
    const box = [clamp((cx - w / 2 - transform.left) / transform.scaleX, width),
      clamp((cy - h / 2 - transform.top) / transform.scaleY, height),
      clamp((cx + w / 2 - transform.left) / transform.scaleX, width),
      clamp((cy + h / 2 - transform.top) / transform.scaleY, height)];
    if (box[2] > box[0] && box[3] > box[1]) candidates.push({ cls, conf: score, box });
  }
  candidates.sort((a, b) => b.conf - a.conf);
  const kept = [];
  for (const detection of candidates) {
    if (!kept.some((other) => other.cls === detection.cls && intersectionOverUnion(other.box, detection.box) > nms)) {
      kept.push(detection);
      if (kept.length === 100) break;
    }
  }
  return kept;
}

// Global minimum-cost assignment, padded so either side may remain unmatched.
export function assignment(costs) {
  if (!costs.length || !costs[0].length) return [];
  const rows = costs.length, columns = costs[0].length, n = rows + columns;
  const u = new Float64Array(n + 1), v = new Float64Array(n + 1);
  const p = new Int32Array(n + 1), way = new Int32Array(n + 1);
  const cost = (i, j) => i < rows && j < columns ? costs[i][j] : i >= rows && j >= columns ? 0 : 1;
  for (let i = 1; i <= n; i++) {
    p[0] = i;
    let j0 = 0;
    const min = new Float64Array(n + 1).fill(Infinity), used = new Uint8Array(n + 1);
    do {
      used[j0] = 1;
      const i0 = p[j0];
      let delta = Infinity, j1 = 0;
      for (let j = 1; j <= n; j++) if (!used[j]) {
        const current = cost(i0 - 1, j - 1) - u[i0] - v[j];
        if (current < min[j]) { min[j] = current; way[j] = j0; }
        if (min[j] < delta) { delta = min[j]; j1 = j; }
      }
      for (let j = 0; j <= n; j++) {
        if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
        else min[j] -= delta;
      }
      j0 = j1;
    } while (p[j0] !== 0);
    do { const j1 = way[j0]; p[j0] = p[j1]; j0 = j1; } while (j0);
  }
  const matches = [];
  for (let j = 1; j <= columns; j++) {
    if (p[j] > 0 && p[j] <= rows && costs[p[j] - 1][j - 1] < 1) matches.push([p[j] - 1, j - 1]);
  }
  return matches;
}

// Browser tracker: two confidence passes with global IoU assignment and motion prediction.
// Preserves short-lived IDs, but is deliberately not advertised as Python ByteTrack parity.
export class ObjectTracker {
  constructor({ maxAge = 2.5, countAfter = 0.5 } = {}) {
    this.maxAge = maxAge;
    this.countAfter = countAfter;
    this.tracks = [];
    this.nextId = 1;
    this.totals = emptyCounts();
  }
  update(detections, now, displayConfidence = 0.4) {
    this.tracks = this.tracks.filter((track) => now - track.lastSeen <= this.maxAge);
    const matched = new Set(), used = new Set(), observed = [];
    const associate = (tracks, indices, minimumIou) => {
      const costs = tracks.map((track) => indices.map((index) => {
        const detection = detections[index];
        const dt = Math.min(now - track.lastSeen, 0.5);
        const prediction = track.box.map((value, axis) => value + track.velocity[axis] * dt);
        const iou = Math.max(intersectionOverUnion(prediction, detection.box), intersectionOverUnion(track.box, detection.box));
        return track.cls === detection.cls && iou >= minimumIou ? 1 - iou : 100;
      }));
      for (const [row, column] of assignment(costs)) {
        const track = tracks[row], index = indices[column], detection = detections[index];
        const dt = now - track.lastSeen;
        if (dt > 0) track.velocity = detection.box.map((value, axis) => (value - track.box[axis]) / dt);
        track.box = detection.box;
        track.conf = detection.conf;
        track.lastSeen = now;
        track.hits++;
        matched.add(track.id);
        used.add(index);
        observed.push(track);
      }
    };
    const high = detections.flatMap((d, i) => d.conf >= 0.25 ? [i] : []);
    const low = detections.flatMap((d, i) => d.conf >= 0.1 && d.conf < 0.25 ? [i] : []);
    associate(this.tracks, high, 0.2);
    associate(this.tracks.filter((t) => !matched.has(t.id) && t.hits >= 2), low, 0.5);
    for (const index of high) if (!used.has(index)) {
      const detection = detections[index];
      const track = { ...detection, id: this.nextId++, firstSeen: now, lastSeen: now,
        hits: 1, counted: false, velocity: [0, 0, 0, 0] };
      this.tracks.push(track);
      observed.push(track);
    }
    const visible = observed.filter((t) => t.conf >= displayConfidence);
    for (const track of visible) {
      const label = CLASSES.find((c) => c.id === track.cls).label;
      if (!track.counted && track.hits >= 2 && now - track.firstSeen >= this.countAfter) {
        this.totals[label]++;
        track.counted = true;
      }
    }
    return visible.map((track) => ({ cls: track.cls, conf: track.conf, box: track.box,
      trackId: track.hits >= 2 ? track.id : null }));
  }
}

export function faceResults(result, width, height) {
  return result.faceLandmarks.map((landmarks, index) => {
    const xs = landmarks.map((p) => p.x), ys = landmarks.map((p) => p.y);
    const scores = Object.fromEntries((result.faceBlendshapes[index]?.categories ?? []).map((c) => [c.categoryName, c.score]));
    // MediaPipe JS Matrix.data is column-major; index 8..10 is the rotation's third column.
    const matrix = result.facialTransformationMatrixes[index]?.data;
    return { box: [Math.min(...xs) * width, Math.min(...ys) * height, Math.max(...xs) * width, Math.max(...ys) * height],
      smile: ((scores.mouthSmileLeft ?? 0) + (scores.mouthSmileRight ?? 0)) / 2,
      yaw: matrix ? Math.atan2(matrix[8], matrix[10]) * 180 / Math.PI : Infinity,
      pitch: matrix ? Math.atan2(matrix[9], matrix[10]) * 180 / Math.PI : Infinity };
  });
}
