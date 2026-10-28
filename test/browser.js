import { SmileTrigger } from '../src/core.js';

const output = document.getElementById('result');
const button = document.getElementById('run');
button.onclick = async () => {
  button.disabled = true;
  const messages = [];
  const log = (value) => { messages.push(value); output.textContent = messages.join('\n'); };
  const worker = new Worker(new URL('../src/inference-worker.js', import.meta.url), { type: 'module' });
  const requests = new Map();
  let id = 0;
  worker.onmessage = ({ data }) => {
    if (data.type === 'status') return log(data.message);
    const pending = requests.get(data.id);
    if (!pending) return;
    requests.delete(data.id);
    if (data.error) pending.reject(new Error(data.error)); else pending.resolve(data.result);
  };
  worker.onerror = (event) => { log(`FAIL: ${event.message}`); for (const p of requests.values()) p.reject(new Error(event.message)); };
  const call = (type, data = {}, transfer = []) => new Promise((resolve, reject) => {
    const request = ++id; requests.set(request, { resolve, reject }); worker.postMessage({ id: request, type, ...data }, transfer);
  });
  try {
    const initialized = await call('init', { baseUrl: new URL('../', import.meta.url).href,
      forceCpu: new URLSearchParams(location.search).get('gpu') !== '1' });
    log(`Backend: ${initialized.backend}`);
    let timestamp = 0;
    const inferImage = async (name, now) => {
      const response = await fetch(new URL(`fixtures/${name}.jpg`, import.meta.url));
      if (!response.ok) throw new Error(`Fixture missing: ${name}`);
      const bitmap = await createImageBitmap(await response.blob());
      timestamp += 500;
      return call('infer', { bitmap, timestamp, now }, [bitmap]);
    };
    const bus = await inferImage('bus', 0);
    log(`People in bus image: ${bus.detections.filter((d) => d.cls === 0).length}`);
    if (!bus.detections.some((d) => d.cls === 0)) throw new Error('YOLO detected no people in the bus fixture.');
    await call('reset-tracks');
    const portrait = await inferImage('portrait', 0);
    log(`Faces in portrait: ${portrait.faces.length}`);
    log(`Face results: ${JSON.stringify(portrait.faces)}`);
    if (!portrait.faces.length) throw new Error('MediaPipe detected no face in portrait.');
    const second = await inferImage('portrait', 0.6);
    if (!second.detections.some((d) => d.trackId !== null)) throw new Error('No confirmed object track.');
    if (second.totals.person !== 1) throw new Error(`Expected one seen person, got ${second.totals.person}.`);
    const trigger = new SmileTrigger();
    const smiling = second.faces.some((f) => f.smile >= 0.7 && Math.abs(f.yaw) <= 15 && Math.abs(f.pitch) <= 20);
    if (!smiling) throw new Error('Portrait does not meet default smile/head-pose thresholds.');
    trigger.update(smiling, 0);
    if (!trigger.update(smiling, 0.3) || trigger.update(smiling, 5)) throw new Error('Smile capture should fire once.');
    log('PASS: local models, detection, face pose, smile capture, tracking, and seen counts.');
  } catch (error) { log(`FAIL: ${error.stack || error.message}`); }
  finally { worker.terminate(); button.disabled = false; }
};
