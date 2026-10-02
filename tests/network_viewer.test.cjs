/* Run with node --test tests/network_viewer.test.cjs; no npm dependencies. */
const test = require('node:test');
const assert = require('node:assert/strict');
const {prepare, lookAt, mm, mv, tube, visibleSegment} = require('../src/plume_advanced/web_assets/network_viewer.js');

test('3D lengths and layer separation use real elevations without mutating input', () => {
  const input = {flowAngle: 0, nodes: [{id: 3}, {id: 9}], segments: [{
    id: 42, source: 3, target: 9, xyz: [[100, 0, 150], [106, 0, 142]], widths: [2, 3]
  }]};
  const before = JSON.stringify(input), result = prepare(input), s = result.segments[0];
  assert.equal(s.length3d, 10);
  assert.ok(s.points.flat().every((v, i) => Math.abs(v - [-3, 4, 0, 3, -4, 0][i]) < 1e-10));
  assert.deepEqual(result.nodes.map(n => n.point), s.points);
  assert.equal(JSON.stringify(input), before);
});

test('host flow basis rotates plan coordinates, preserving lengths', () => {
  const result = prepare({flowAngle: 90, nodes: [], segments: [
    {source: 0, target: 1, xyz: [[0, 0, 0], [0, 10, 0]], widths: [1, 1]}
  ]});
  assert.ok(Math.abs(result.segments[0].points[0][0] + 5) < 1e-10);
  assert.ok(Math.abs(result.segments[0].points[0][2]) < 1e-10);
  assert.equal(result.segments[0].length3d, 10);
});

test('paired views share a world origin instead of recentering changed geometry', () => {
  const base = {flowAngle: 0, nodes: [], segments: [{source: 0, target: 1,
    xyz: [[0, 0, 0], [10, 0, 0]], widths: [1, 1]}]};
  const coarse = prepare(base);
  const changed = {...base, segments: [{...base.segments[0], xyz: [[0, 0, 0], [10, 4, 0]]}]};
  const detailed = prepare(changed, coarse.origin);
  assert.deepEqual(detailed.segments[0].points[0], coarse.segments[0].points[0]);
  assert.deepEqual(detailed.origin, coarse.origin);
});

test('longitudinal sampling is never simplified', () => {
  const points = Array.from({length: 3200}, (_, i) => [i, Math.sin(i / 50), 150 - i / 100]);
  const result = prepare({flowAngle: 17, nodes: [], segments: [{source: 0, target: 1, xyz: points}]});
  assert.equal(result.segments[0].points.length, points.length);
  assert.deepEqual(result.segments[0].xyz, points);
});

test('width glyphs retain finite coordinates and unique picking identifiers', () => {
  const data = tube([[0, 0, 0], [1, 0, 0], [1, 2, 0]], [1, 2, 1], [.1, .2, .3], 513);
  assert.equal(data.length, (2 * 8 * 6 + 2 * 8 * 3) * 12);
  assert.ok(data.every(Number.isFinite));
  for (let i = 0; i < data.length; i += 12) {
    assert.deepEqual(data.slice(i + 9, i + 12), [1 / 255, 2 / 255, 0]);
    assert.ok(Math.abs(Math.hypot(...data.slice(i + 3, i + 6)) - 1) < 1e-10);
  }
});

test('inter-layer connectors require both endpoint layers and the connector switch', () => {
  const ramp = {layer: 0, endLayer: 2};
  assert.equal(visibleSegment(ramp, new Set([0, 2]), true), true);
  assert.equal(visibleSegment(ramp, new Set([0]), true), false);
  assert.equal(visibleSegment(ramp, new Set([0, 2]), false), false);
  assert.equal(visibleSegment({layer: 0, endLayer: 0}, new Set([0]), false), true);
});

test('camera centres its target and matrix multiplication preserves identity', () => {
  const view = lookAt([10, 5, 10], [2, 1, 2]);
  const p = mv(view, [2, 1, 2]);
  assert.ok(Math.abs(p[0]) < 1e-10 && Math.abs(p[1]) < 1e-10);
  const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  assert.ok(mm(view, identity).every((v, i) => Math.abs(v - view[i]) < 1e-10));
});
