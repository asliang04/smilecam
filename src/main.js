import './style.css';
import { CLASSES, emptyCounts, PausableClock, SmileTrigger, insetProgress } from './core.js';

const element = (id) => document.getElementById(id);
const video = element('video'), canvas = element('output'), context = canvas.getContext('2d');
const snapshot = document.createElement('canvas'), snapshotContext = snapshot.getContext('2d');
const live = document.createElement('canvas'), liveContext = live.getContext('2d');
const photo = document.createElement('canvas');
const baseUrl = new URL(import.meta.env.BASE_URL, location.href).href;
let stream, worker, pending = new Map(), requestId = 0, generation = 0, analysisEpoch = 0;
let state = 'off', busy = false, clock, trigger, photoTaken = null;
let detections = [], faces = [], totals = emptyCounts(), captures = 0;
let lastVideoTime = -1, fps = 0, lastInferenceTime = 0, backend = 'CPU';

element('counts').innerHTML = CLASSES.map(({ label, color }) => `<tr><td style="color:${color}">${label}</td><td id="now-${label}">0</td><td id="seen-${label}">0</td></tr>`).join('');

function refreshCounts() {
  const now = emptyCounts();
  for (const detection of detections) now[CLASSES.find((c) => c.id === detection.cls).label]++;
  for (const { label } of CLASSES) {
    element(`now-${label}`).textContent = now[label];
    element(`seen-${label}`).textContent = totals[label];
  }
  element('faces').textContent = faces.length;
  element('captures').textContent = captures;
}

function setState(next) {
  state = next;
  element('welcome').hidden = next !== 'off';
  element('loading').hidden = next !== 'loading';
  canvas.hidden = next === 'off' || next === 'loading';
  element('paused-overlay').hidden = next !== 'paused';
  element('pause').disabled = next !== 'running' && next !== 'paused';
  element('stop').disabled = next === 'off';
  element('pause').textContent = next === 'paused' ? 'Resume [space]' : 'Pause [space]';
  element('pause').classList.toggle('is-paused', next === 'paused');
  element('camera').disabled = next === 'loading';
  element('cpu').disabled = next !== 'off';
  if (next === 'off') element('performance').textContent = 'camera off';
  if (next === 'loading') element('performance').textContent = 'loading';
  if (next === 'paused') element('performance').textContent = 'paused';
}

function showError(message) {
  element('error').textContent = message;
  element('error').hidden = !message;
}

function makeWorker() {
  worker = new Worker(new URL('./inference-worker.js', import.meta.url), { type: 'module' });
  worker.onmessage = ({ data }) => {
    if (data.type === 'status') { element('loading-message').textContent = data.message; return; }
    const request = pending.get(data.id);
    if (!request) return;
    clearTimeout(request.timeout);
    pending.delete(data.id);
    if (data.error) request.reject(new Error(data.error));
    else request.resolve(data.result);
  };
  worker.onerror = (event) => {
    event.preventDefault();
    const message = event.message || 'The processing worker stopped. Try CPU only or restart the camera.';
    stopCamera();
    showError(message);
  };
}

function callWorker(type, data = {}, transfer = []) {
  return new Promise((resolve, reject) => {
    const id = ++requestId;
    const timeout = setTimeout(() => {
      pending.delete(id);
      reject(new Error('Model processing timed out. Check the connection, or restart with CPU only enabled.'));
    }, 120_000);
    pending.set(id, { resolve, reject, timeout });
    try { worker.postMessage({ type, id, ...data }, transfer); }
    catch (error) { clearTimeout(timeout); pending.delete(id); reject(error); }
  });
}

function stopCamera() {
  generation++;
  analysisEpoch++;
  worker?.terminate();
  worker = null;
  for (const request of pending.values()) {
    clearTimeout(request.timeout);
    request.reject(new Error('Camera stopped.'));
  }
  pending.clear();
  stream?.getTracks().forEach((track) => track.stop());
  stream = null;
  video.srcObject = null;
  busy = false;
  photoTaken = null;
  for (const buffer of [canvas, snapshot, live, photo]) { buffer.width = 0; buffer.height = 0; }
  detections = [];
  faces = [];
  refreshCounts();
  setState('off');
}

async function listCameras() {
  const current = element('camera').value;
  const devices = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === 'videoinput');
  element('camera').replaceChildren(new Option('Default camera', ''));
  devices.forEach((device, index) => element('camera').add(new Option(device.label || `Camera ${index + 1}`, device.deviceId)));
  if (devices.some((d) => d.deviceId === current)) element('camera').value = current;
}

async function startCamera() {
  stopCamera();
  showError('');
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    showError('Camera access requires HTTPS or localhost. Open the hosted site, or run npm run dev.');
    return;
  }
  if (!window.Worker || !window.OffscreenCanvas || !window.createImageBitmap || !window.DecompressionStream) {
    showError('This browser is missing camera processing features. Use a current version of Chrome or Edge.');
    return;
  }
  const ownGeneration = generation;
  totals = emptyCounts(); captures = 0; fps = 0; lastInferenceTime = 0; lastVideoTime = -1;
  clock = new PausableClock(); trigger = new SmileTrigger();
  refreshCounts();
  setState('loading');
  element('loading-message').textContent = 'Loading models…';
  makeWorker();
  const deviceId = element('camera').value;
  const cameraPromise = navigator.mediaDevices.getUserMedia({ audio: false, video: {
    width: { ideal: 1280 }, height: { ideal: 720 }, ...(deviceId ? { deviceId: { exact: deviceId } } : {}),
  } }).then((cameraStream) => {
    if (ownGeneration !== generation) { cameraStream.getTracks().forEach((t) => t.stop()); return null; }
    stream = cameraStream;
    return cameraStream;
  });
  try {
    const [initialized, cameraStream] = await Promise.all([
      callWorker('init', { baseUrl, forceCpu: element('cpu').checked }), cameraPromise,
    ]);
    if (ownGeneration !== generation || !cameraStream) return;
    backend = initialized.backend;
    video.srcObject = cameraStream;
    await video.play();
    if (ownGeneration !== generation) return;
    for (const buffer of [canvas, snapshot, live, photo]) { buffer.width = video.videoWidth; buffer.height = video.videoHeight; }
    cameraStream.getVideoTracks()[0].addEventListener('ended', () => {
      if (ownGeneration === generation) { stopCamera(); showError('The camera disconnected or camera access was revoked.'); }
    });
    await listCameras();
    if (ownGeneration === generation) {
      setState('running');
      if (document.hidden) togglePause();
    }
  } catch (error) {
    if (ownGeneration !== generation) return;
    stopCamera();
    const hints = {
      NotAllowedError: 'Camera permission was denied. Allow camera access in the browser and Windows camera settings, then try again.',
      NotFoundError: 'No camera was found. Connect a webcam and try again.',
      NotReadableError: 'The camera could not be opened. Close other apps using it and try again.',
      OverconstrainedError: 'The selected camera is unavailable. Select Default camera and try again.',
    };
    showError(hints[error.name] || `${error.message} Restart with CPU only enabled if GPU initialization failed.`);
  }
}

function togglePause() {
  if (state === 'running') { clock.pause(); analysisEpoch++; setState('paused'); }
  else if (state === 'paused') { clock.resume(); analysisEpoch++; lastVideoTime = -1; lastInferenceTime = 0; setState('running'); }
}

function drawVideo(target) {
  target.save();
  if (element('mirror').checked) { target.translate(canvas.width, 0); target.scale(-1, 1); }
  target.drawImage(video, 0, 0, canvas.width, canvas.height);
  target.restore();
}

function label(text, x, y, color, below = false) {
  const size = Math.max(12, Math.round(canvas.height / 45));
  liveContext.font = `${size}px Arial`;
  const textWidth = liveContext.measureText(text).width;
  x = Math.max(0, Math.min(x, canvas.width - textWidth - 8));
  y = Math.max(size + 6, Math.min(below ? y + size + 8 : y, canvas.height));
  liveContext.fillStyle = color;
  liveContext.fillRect(x, y - size - 6, textWidth + 8, size + 6);
  liveContext.fillStyle = '#000';
  liveContext.fillText(text, x + 4, y - 4);
}

function drawBox(box, color) {
  liveContext.strokeStyle = color;
  liveContext.lineWidth = 2;
  liveContext.strokeRect(box[0], box[1], box[2] - box[0], box[3] - box[1]);
}

const isHeadOn = (face) => Math.abs(face.yaw) <= 15 && Math.abs(face.pitch) <= 20;

async function analyzeFrame() {
  if (busy || state !== 'running' || video.readyState < 2 || video.currentTime === lastVideoTime) return;
  busy = true;
  lastVideoTime = video.currentTime;
  const ownGeneration = generation, ownEpoch = analysisEpoch, now = clock.now();
  try {
    drawVideo(snapshotContext);
    const bitmap = await createImageBitmap(snapshot);
    if (ownGeneration !== generation) { bitmap.close(); return; }
    const result = await callWorker('infer', { bitmap, timestamp: performance.now(), now }, [bitmap]);
    if (ownGeneration !== generation || ownEpoch !== analysisEpoch || state !== 'running') return;
    detections = result.detections; faces = result.faces; totals = result.totals;
    const completed = performance.now();
    if (lastInferenceTime) {
      const currentFps = 1000 / (completed - lastInferenceTime);
      fps = fps ? fps * 0.9 + currentFps * 0.1 : currentFps;
    }
    lastInferenceTime = completed;
    if (photoTaken === null && trigger.update(faces.some((f) => isHeadOn(f) && f.smile >= Number(element('threshold').value)), now)) {
      photo.getContext('2d').drawImage(snapshot, 0, 0);
      photoTaken = clock.now();
      captures++;
    }
    refreshCounts();
    element('performance').textContent = `${fps.toFixed(1)} fps  ${backend}`;
  } catch (error) {
    if (ownGeneration === generation) { stopCamera(); showError(error.message); }
  } finally { if (ownGeneration === generation) busy = false; }
}

function render() {
  if (state === 'running' && video.readyState >= 2) {
    drawVideo(liveContext);
    for (const detection of detections) {
      const { color, label: name } = CLASSES.find((c) => c.id === detection.cls);
      drawBox(detection.box, color);
      label(`${name}${detection.trackId === null ? '' : ` #${detection.trackId}`} ${detection.conf.toFixed(2)}`, ...detection.box.slice(0, 2), color);
    }
    for (const face of faces) {
      const headOn = isHeadOn(face), smiling = headOn && face.smile >= Number(element('threshold').value);
      const color = smiling ? '#50dc50' : headOn ? '#fff' : '#808080';
      drawBox(face.box, color);
      label(`smile ${face.smile.toFixed(2)}  yaw ${face.yaw.toFixed(0)}  pitch ${face.pitch.toFixed(0)}`, face.box[0], face.box[3], color, true);
    }
    const progress = photoTaken === null ? null : insetProgress(clock.now() - photoTaken);
    if (photoTaken !== null && progress === null) {
      photoTaken = null;
      photo.getContext('2d').clearRect(0, 0, photo.width, photo.height);
    }
    if (progress === null) context.drawImage(live, 0, 0);
    else {
      context.drawImage(photo, 0, 0);
      const margin = 16, w = canvas.width, h = canvas.height;
      const x1 = (w - margin - w * 0.3) * progress, y1 = (h - margin - h * 0.3) * progress;
      const x2 = w - margin * progress, y2 = h - margin * progress;
      context.fillStyle = '#fff';
      context.fillRect(x1 - 3, y1 - 3, x2 - x1 + 6, y2 - y1 + 6);
      context.drawImage(live, x1, y1, x2 - x1, y2 - y1);
    }
    void analyzeFrame();
  }
  requestAnimationFrame(render);
}

element('start').addEventListener('click', startCamera);
element('cancel').addEventListener('click', stopCamera);
element('stop').addEventListener('click', stopCamera);
element('pause').addEventListener('click', togglePause);
element('resume').addEventListener('click', togglePause);
element('threshold').addEventListener('input', () => { element('threshold-value').textContent = Number(element('threshold').value).toFixed(2); });
element('camera').addEventListener('change', () => { if (state === 'running' || state === 'paused') void startCamera(); });
element('mirror').addEventListener('change', () => {
  if (state === 'running' || state === 'paused') void startCamera();
});
document.addEventListener('keydown', (event) => {
  if (event.code === 'Escape') { stopCamera(); return; }
  if (['INPUT', 'SELECT', 'TEXTAREA', 'BUTTON'].includes(event.target.tagName)) return;
  if (event.code === 'Space') { event.preventDefault(); togglePause(); }
  if (event.key.toLowerCase() === 'q') stopCamera();
});
document.addEventListener('visibilitychange', () => { if (document.hidden && state === 'running') togglePause(); });
window.addEventListener('pagehide', stopCamera);
if (navigator.mediaDevices?.addEventListener) navigator.mediaDevices.addEventListener('devicechange', () => { void listCameras().catch(() => {}); });
setState('off');
requestAnimationFrame(render);
