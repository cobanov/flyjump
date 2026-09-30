import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRunner } from "../src/lib/runner.ts";
import {
  CONTROLLERS,
  NETWORK,
  decideDirect,
  directInputs,
  forward,
  observation,
  validModel,
} from "../src/lib/policy.ts";
import { Connectome, CIRCUIT } from "../src/lib/connectome.ts";
import { Trainer, episode, rng, randomWeights } from "../src/lib/training.ts";
import { benchmark } from "../src/lib/benchmark.ts";
const read = (path) =>
  JSON.parse(readFileSync(new URL(`../${path}`, import.meta.url), "utf8"));
const model = read("public/benchmarks/model.json");
const direct = ["8-12-3", "8-20-3"].map((shape) =>
  read(`public/benchmarks/direct/${shape}/20260912/model.json`),
);
test("actual learned weights and hidden activations determine the selected score", () => {
  const weights = Array(NETWORK.parameters).fill(0);
  weights[229] = 2;
  const result = forward(weights, Array(16).fill(0));
  assert.deepEqual(result.scores, [0, 2, 0]);
  assert.equal(result.action, 1);
  assert.equal(episode(weights, 42, 20).jumps, 1);
});
test("measured circuit activity depends on input and resets per episode", () => {
  const a = new Connectome(),
    b = new Connectome(),
    input = observation(createRunner());
  const first = a.step(input);
  assert.deepEqual(first, b.step(input));
  assert.ok(a.activity.some((v) => v !== 0));
  const altered = input.slice();
  altered[0] = 1;
  for (let i = 0; i < 10; i++) {
    a.step(input);
    b.step(altered);
  }
  assert.notDeepEqual(Array.from(a.activity), Array.from(b.activity));
  assert.notDeepEqual(
    CIRCUIT.outputs.map((i) => a.activity[i]),
    CIRCUIT.outputs.map((i) => b.activity[i]),
  );
  assert.deepEqual(a.step(input, true), Array(16).fill(0));
  assert.ok(a.activity.every((v) => v === 0));
});
test("every circuit edge points to real cells and output has a path from driven input", () => {
  assert.equal(new Set(CIRCUIT.nodes.map((n) => n.id)).size, 80);
  assert.equal(CIRCUIT.edges.length, 1296);
  assert.equal(
    CIRCUIT.edges.reduce((n, e) => n + e[2], 0),
    26029,
  );
  const reached = new Set(CIRCUIT.inputs.map(([i]) => i));
  for (let t = 0; t < 80; t++)
    for (const [a, b, w] of CIRCUIT.edges) {
      assert.ok(
        Number.isInteger(w) && w > 0 && a >= 0 && a < 80 && b >= 0 && b < 80,
      );
      if (reached.has(a) && CIRCUIT.nodes[a].sign) reached.add(b);
    }
  for (const i of CIRCUIT.outputs)
    assert.ok(reached.has(i), `unreachable ${i}`);
});
test("direct controllers see the circuit's input encoding and no circuit state", () => {
  for (const [kind, hidden] of [
    ["direct-8-12-3", 12],
    ["direct-8-20-3", 20],
  ]) {
    const shape = CONTROLLERS[kind];
    assert.equal(shape.parameters, (8 + 1) * hidden + (hidden + 1) * 3);
    assert.equal(randomWeights(rng(1), kind).length, shape.parameters);
  }
  assert.equal(CONTROLLERS["direct-8-20-3"].parameters, NETWORK.parameters);
  // Input cells receive exactly this drive in connectome.ts.
  const state = createRunner(),
    input = observation(state),
    brain = new Connectome();
  brain.step(input);
  for (const [cell, channel] of CIRCUIT.inputs)
    assert.equal(directInputs(input)[channel], brain.drive[cell]);
  const shape = CONTROLLERS["direct-8-12-3"],
    weights = Array(shape.parameters).fill(0);
  weights[0] = 1; // hidden 0 reads observation 0 (proximity)
  weights[9 * 12 + 13] = 5; // Jump score reads hidden 0
  const far = decideDirect(weights, state, shape);
  assert.deepEqual(far.inputs, directInputs(far.observations));
  assert.equal(far.action, 0);
  state.obstacles.push({ x: 60, width: 20, height: 40, bottom: 0 });
  assert.equal(decideDirect(weights, state, shape).action, 1);
});
test("malformed and incompatible checkpoints are rejected", () => {
  assert.ok(validModel(model));
  assert.ok(validModel(direct[0], "direct-8-12-3"));
  assert.ok(validModel(direct[1], "direct-8-20-3"));
  assert.equal(validModel(direct[1]), false);
  assert.equal(validModel(model, "direct-8-20-3"), false);
  for (const bad of [
    null,
    {},
    { ...model, version: "flyjump-v1" },
    { ...model, weights: [] },
    { ...model, weights: model.weights.map(() => NaN) },
  ])
    assert.equal(validModel(bad), false);
});
test("seeded learning changes the search distribution through actual rollouts", () => {
  const a = new Trainer(42),
    b = new Trainer(42),
    original = a.champion.weights.slice();
  assert.deepEqual(original, randomWeights(rng(42)));
  const first = a.step();
  assert.deepEqual(first, b.step());
  assert.equal(first.episodes, 196);
  assert.ok(a.mean.some((n) => n !== 0));
  assert.notDeepEqual(first.model.weights, original);
  const d = new Trainer(42, "direct-8-12-3");
  assert.deepEqual(d.champion.weights, randomWeights(rng(42), "direct-8-12-3"));
  const directFirst = d.step();
  assert.equal(directFirst.episodes, 196);
  assert.equal(directFirst.model.version, CONTROLLERS["direct-8-12-3"].version);
  assert.equal(directFirst.model.weights.length, 147);
});
test("published benchmark reproduces exactly against the shipped graph, checkpoint and Chromium", () => {
  const saved = JSON.parse(
    readFileSync(
      new URL("../public/benchmarks/benchmark.json", import.meta.url),
      "utf8",
    ),
  );
  const actual = benchmark(model, direct);
  assert.deepEqual(actual.results, saved.results);
  const row = (name) => actual.results.find((r) => r.name === name);
  assert.ok(
    row("connectome + trained readout").meanScore >
      row("circuit output zeroed").meanScore * 10,
  );
  // Zeroed output gives the readout one constant input, hence one constant action.
  const zeroed = row("circuit output zeroed").runs.map((r) => r.actions);
  assert.equal(new Set(zeroed.map((a) => a.findIndex((n) => n > 0))).size, 1);
  assert.ok(zeroed.every((a) => a.filter((n) => n > 0).length === 1));
});
