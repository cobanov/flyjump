import { RUNNER, type State, type Input } from "./runner.ts";
import { CIRCUIT, Connectome } from "./connectome.ts";
export const OBSERVATION_LABELS = [
  "Proximity",
  "Width",
  "Height",
  "Altitude",
  "Speed",
  "Player Y",
  "Velocity",
  "Grounded",
];
export const INPUT_LABELS = CIRCUIT.outputs.map(
  (i) => CIRCUIT.nodes[i].type ?? String(CIRCUIT.nodes[i].id),
);
export const ACTIONS = ["Run", "Jump", "Duck"] as const;
export const CONTROLLERS = {
  connectome: {
    inputs: 16,
    hidden: 12,
    outputs: 3,
    parameters: 243,
    version: "flydino-chromium98-connectome-v2",
  },
  /** Control: the same 8 observations, encoded as the circuit's input cells receive them, with no circuit. */
  "direct-8-12-3": {
    inputs: 8,
    hidden: 12,
    outputs: 3,
    parameters: 147,
    version: "flydino-chromium98-direct-8-12-3-v1",
  },
  /** Control with the connectome readout's 243-parameter search budget. */
  "direct-8-20-3": {
    inputs: 8,
    hidden: 20,
    outputs: 3,
    parameters: 243,
    version: "flydino-chromium98-direct-8-20-3-v1",
  },
} as const;
export type ControllerKind = keyof typeof CONTROLLERS;
export type Shape = { inputs: number; hidden: number };
export const NETWORK = { ...CONTROLLERS.connectome, decisionSteps: 2 } as const;
export type Model = {
  version: (typeof CONTROLLERS)[ControllerKind]["version"];
  weights: number[];
  generation: number;
  trainingSeed: number;
  validation: number;
};
export const controllerKind = (model: Model) =>
  (Object.keys(CONTROLLERS) as ControllerKind[]).find(
    (kind) => CONTROLLERS[kind].version === model.version,
  ) ?? "connectome";
export type Decision = {
  inputs: number[];
  hidden: number[];
  scores: number[];
  action: number;
  observations?: number[];
  activity?: number[];
};
/** CodeBullet-inspired structured observations; engineered state access, not pixels. */
export function observation(s: State): number[] {
  const o = s.obstacles.find((o) => o.x + o.width > RUNNER.x);
  return [
    o ? 1 - Math.max(0, Math.min(1, (o.x - RUNNER.x) / 600)) : 0,
    o ? o.width / 75 : 0,
    o ? o.height / 60 : 0,
    o ? o.bottom / 60 : 0,
    s.speed / 780,
    s.y / 100,
    (s.vy / 900 + 1) / 2,
    s.y === 0 ? 1 : 0,
  ];
}
export function forward(
  weights: number[],
  inputs: number[],
  shape: Shape = NETWORK,
): Decision {
  const hidden = new Array<number>(shape.hidden),
    scores = new Array<number>(3);
  let k = 0;
  for (let h = 0; h < shape.hidden; h++) {
    let z = 0;
    for (let i = 0; i < shape.inputs; i++) z += weights[k++] * inputs[i];
    hidden[h] = Math.tanh(z + weights[k++]);
  }
  for (let a = 0; a < 3; a++) {
    let z = 0;
    for (let h = 0; h < shape.hidden; h++) z += weights[k++] * hidden[h];
    scores[a] = z + weights[k++];
  }
  let action = 0;
  for (let a = 1; a < 3; a++) if (scores[a] > scores[action]) action = a;
  return { inputs, hidden, scores, action };
}
export function decide(
  weights: number[],
  s: State,
  brain: Connectome,
  ablated = false,
): Decision {
  const observations = observation(s),
    inputs = brain.step(observations, ablated);
  return { ...forward(weights, inputs), observations };
}
/** Same drive the circuit's input cells receive (connectome.ts), fed straight to the readout. */
export const directInputs = (observations: number[]) =>
  observations.map((x) => 2 * (x - 0.5));
export function decideDirect(
  weights: number[],
  s: State,
  shape: Shape,
): Decision {
  const observations = observation(s);
  return {
    ...forward(weights, directInputs(observations), shape),
    observations,
  };
}
export const actionInput = (action: number): Input => ({
  jump: action === 1,
  duck: action === 2,
});
export function validModel(
  value: unknown,
  kind: ControllerKind = "connectome",
): value is Model {
  if (!value || typeof value !== "object") return false;
  const m = value as Model;
  return (
    m.version === CONTROLLERS[kind].version &&
    Array.isArray(m.weights) &&
    m.weights.length === CONTROLLERS[kind].parameters &&
    m.weights.every((n) => Number.isFinite(n) && Math.abs(n) < 1e4) &&
    Number.isInteger(m.generation) &&
    m.generation >= 0 &&
    Number.isInteger(m.trainingSeed) &&
    Number.isFinite(m.validation)
  );
}
