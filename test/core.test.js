import test from 'node:test';
import assert from 'node:assert/strict';
import { SmileTrigger, PausableClock, insetProgress, ObjectTracker, letterbox, decodeYolo, assignment, faceResults } from '../src/core.js';

test('smile requires a hold, fires once, and requires release plus cooldown', () => {
  const trigger = new SmileTrigger();
  assert.equal(trigger.update(true, 0), false);
  assert.equal(trigger.update(true, 0.24), false);
  assert.equal(trigger.update(true, 0.25), true);
  assert.equal(trigger.update(true, 8), false);
  trigger.update(false, 8.1);
  assert.equal(trigger.update(true, 8.2), false);
  assert.equal(trigger.update(true, 8.5), true);
  trigger.update(false, 8.6);
  trigger.update(true, 8.7);
  assert.equal(trigger.update(true, 9), false);
  assert.equal(trigger.update(true, 11.5), true);
});

test('pause freezes timers without counting time spent paused', () => {
  let time = 10;
  const clock = new PausableClock(() => time);
  time = 11; clock.pause(); time = 60;
  assert.equal(clock.now(), 1);
  clock.resume(); time = 61;
  assert.equal(clock.now(), 2);
  assert.equal(insetProgress(0), 0);
  assert.equal(insetProgress(0.4), 1);
  assert.equal(insetProgress(3.4), 1);
  assert.equal(insetProgress(3.8), null);
});

const person = (box = [0, 0, 100, 200], conf = 0.9, cls = 0) => ({ box, conf, cls });
test('tracks preserve IDs through brief occlusion and count each tracked visit once', () => {
  const tracker = new ObjectTracker();
  tracker.update([person()], 0);
  const first = tracker.update([person([4, 0, 104, 200])], 0.3)[0].trackId;
  tracker.update([person([6, 0, 106, 200])], 0.6);
  assert.equal(tracker.totals.person, 1);
  tracker.update([], 1);
  assert.equal(tracker.update([person([8, 0, 108, 200])], 1.6)[0].trackId, first);
  assert.equal(tracker.totals.person, 1);
  tracker.update([], 5);
  tracker.update([person()], 5.1);
  const returned = tracker.update([person()], 5.7)[0].trackId;
  assert.notEqual(returned, first);
  assert.equal(tracker.totals.person, 2);
});

test('low-confidence observations keep a track alive without inflating visible counts', () => {
  const tracker = new ObjectTracker();
  tracker.update([person()], 0);
  const id = tracker.update([person()], 0.3)[0].trackId;
  assert.deepEqual(tracker.update([person(undefined, 0.2)], 2.5), []);
  assert.equal(tracker.update([person()], 3)[0].trackId, id);
  assert.equal(tracker.totals.person, 1);
  tracker.update([person(undefined, 0.9, 24)], 3.1);
  assert.equal(tracker.totals.backpack, 0);
});

test('assignment makes a global match rather than greedily taking the best pair', () => {
  const matches = assignment([[0.1, 0.2], [0.2, 100]]);
  assert.deepEqual(matches.sort((a, b) => a[0] - b[0]), [[0, 1], [1, 0]]);
  assert.deepEqual(assignment([[100]]), []);
  assert.deepEqual(assignment([[]]), []);
});

test('YOLO decoding removes letterbox padding, filters classes and suppresses duplicates', () => {
  const n = 4, data = new Float32Array(84 * n);
  for (let i = 0; i < n; i++) {
    data[i] = 160; data[n + i] = 160;
    data[2 * n + i] = 160; data[3 * n + i] = 90;
  }
  data[4 * n] = 0.9; // person
  data[4 * n + 1] = 0.8; // duplicate person
  data[6 * n + 2] = 0.95; // car: ignored
  data[28 * n + 3] = 0.85; // backpack: retained despite overlap
  const result = decodeYolo(data, [1, 84, n], letterbox(1280, 720), 1280, 720);
  assert.equal(result.length, 2);
  assert.deepEqual(result[0].box, [320, 180, 960, 540]);
  assert.deepEqual(result.map((d) => d.cls), [0, 24]);
  assert.throws(() => decodeYolo(data, [1, 6, n], {}, 100, 100), /Unexpected YOLO/);
});

test('face conversion averages the two smile blendshapes and reads column-major pose', () => {
  const a = 10 * Math.PI / 180;
  const face = faceResults({ faceLandmarks: [[{ x: 0.2, y: 0.1 }, { x: 0.8, y: 0.9 }]],
    faceBlendshapes: [{ categories: [{ categoryName: 'mouthSmileLeft', score: 0.6 }, { categoryName: 'mouthSmileRight', score: 0.8 }] }],
    facialTransformationMatrixes: [{ data: [Math.cos(a), 0, -Math.sin(a), 0, 0, 1, 0, 0, Math.sin(a), 0, Math.cos(a), 0, 0, 0, 0, 1] }],
  }, 100, 100)[0];
  assert.equal(face.smile, 0.7);
  assert.ok(Math.abs(face.yaw - 10) < 1e-6);
  assert.deepEqual(face.box, [20, 10, 80, 90]);
});
