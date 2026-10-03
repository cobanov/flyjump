import { episode, rng, randomWeights, type Episode } from "./training.ts";
import { type Model, type ControllerKind, controllerKind } from "./policy.ts";
export const BENCHMARK = {
  seconds: 180,
  seeds: Array.from({ length: 100 }, (_, i) => 2100001 + i),
};
/** Result-row name of each trained controller. */
export const CONTROLLER_NAMES: Record<ControllerKind, string> = {
  connectome: "connectome + trained readout",
  "direct-8-12-3": "direct readout 8-12-3 (no connectome)",
  "direct-8-20-3": "direct readout 8-20-3 (no connectome)",
};
export function summarize(runs: Episode[]) {
  const sorted = runs.map((r) => r.seconds).sort((a, b) => a - b);
  return {
    courses: runs.length,
    survived: runs.filter((r) => !r.dead).length,
    meanSeconds: runs.reduce((n, r) => n + r.seconds, 0) / runs.length,
    medianSeconds: sorted[Math.floor(sorted.length / 2)],
    meanScore: runs.reduce((n, r) => n + r.score, 0) / runs.length,
  };
}
type Policy = readonly [
  string,
  number[] | null,
  "rule" | "idle" | "random" | "ablated",
  ControllerKind,
];
/**
 * Evaluates a checkpoint and the control conditions on the held-out courses.
 * Optional direct-input checkpoints (no connectome) are evaluated alongside.
 */
export function benchmark(model: Model, direct: Model[] = []) {
  const kind = controllerKind(model),
    untrained = randomWeights(rng(model.trainingSeed), kind);
  const trained: Policy[] =
    kind === "connectome"
      ? [
          [CONTROLLER_NAMES.connectome, model.weights, "idle", kind],
          ...direct.map((d): Policy => {
            const k = controllerKind(d);
            return [CONTROLLER_NAMES[k], d.weights, "idle", k];
          }),
          // Zeroes every circuit output: the readout receives no game information.
          ["circuit output zeroed", model.weights, "ablated", kind],
        ]
      : [[CONTROLLER_NAMES[kind], model.weights, "idle", kind]];
  const policies: Policy[] = [
    ...trained,
    ["untrained readout", untrained, "idle", kind],
    ["rule", null, "rule", kind],
    ["random", null, "random", kind],
    ["idle", null, "idle", kind],
  ];
  return {
    model,
    environment: model.version,
    modelGeneration: model.generation,
    trainingSeed: model.trainingSeed,
    direct,
    config: BENCHMARK,
    results: policies.map(([name, weights, baseline, k]) => {
      const runs = BENCHMARK.seeds.map((seed) =>
        episode(weights, seed, BENCHMARK.seconds, baseline, k),
      );
      return { name, ...summarize(runs), runs };
    }),
  };
}
export type Benchmark = ReturnType<typeof benchmark>;
