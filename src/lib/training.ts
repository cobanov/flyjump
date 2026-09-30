import { Connectome } from "./connectome.ts";
import { createRunner, tickRunner, RUNNER, score } from "./runner.ts";
import {
  CONTROLLERS,
  NETWORK,
  decide,
  decideDirect,
  actionInput,
  type ControllerKind,
  type Model,
} from "./policy.ts";
export const TRAINING = {
  population: 64,
  elites: 8,
  generations: 80,
  courseSeconds: 180,
  trainingCourses: 3,
  validationSeeds: [1100001, 1100002, 1100003, 1100004],
  validationSeconds: 180,
};
export function rng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return (s + 0.5) / 4294967296;
  };
}
export function randomWeights(
  random: () => number,
  kind: ControllerKind = "connectome",
) {
  return Array.from(
    { length: CONTROLLERS[kind].parameters },
    () => gaussian(random) * 0.7,
  );
}
const gaussian = (r: () => number) =>
  Math.sqrt(-2 * Math.log(r())) * Math.cos(2 * Math.PI * r());
export type Episode = {
  seed: number;
  seconds: number;
  score: number;
  dead: boolean;
  jumps: number;
  ducks: number;
  actions: number[];
};
export function episode(
  weights: number[] | null,
  seed: number,
  seconds: number,
  baseline: "rule" | "idle" | "random" | "ablated" = "idle",
  kind: ControllerKind = "connectome",
): Episode {
  const s = createRunner(seed),
    brain = kind === "connectome" ? new Connectome() : null,
    r = rng(seed),
    actions = [0, 0, 0];
  const steps = Math.round(seconds / RUNNER.step);
  let action = 0;
  for (let i = 0; i < steps && !s.dead; i++) {
    if (i % NETWORK.decisionSteps === 0) {
      action = weights
        ? brain
          ? decide(weights, s, brain, baseline === "ablated").action
          : decideDirect(weights, s, CONTROLLERS[kind]).action
        : baseline === "random"
          ? Math.floor(r() * 3)
          : 0;
      actions[action]++;
    }
    tickRunner(s, baseline === "rule" && !weights, actionInput(action));
  }
  return {
    seed,
    seconds: s.time,
    score: score(s),
    dead: s.dead,
    jumps: s.jumps,
    ducks: s.ducks,
    actions,
  };
}
const average = (values: number[]) =>
  values.reduce((a, b) => a + b, 0) / values.length;
export type Progress = {
  generation: number;
  episodes: number;
  bestFitness: number;
  meanFitness: number;
  validation: number;
  model: Model;
};
/** Cross-entropy neuroevolution. No scripted actions, labels, or teacher policy. */
export class Trainer {
  seed: number;
  kind: ControllerKind;
  random: () => number;
  mean: number[];
  sigma: number[];
  generation = 0;
  episodes = 0;
  champion: Model;
  history: Omit<Progress, "model">[] = [];
  constructor(seed = 20260912, kind: ControllerKind = "connectome") {
    this.seed = seed;
    this.kind = kind;
    this.random = rng(seed);
    this.mean = Array(CONTROLLERS[kind].parameters).fill(0);
    this.sigma = Array(CONTROLLERS[kind].parameters).fill(0.8);
    this.champion = {
      version: CONTROLLERS[kind].version,
      weights: randomWeights(this.random, kind),
      generation: 0,
      trainingSeed: seed,
      validation: 0,
    };
  }
  step(): Progress {
    const generation = ++this.generation;
    // All candidates face identical courses this generation; new training seeds each generation.
    const seeds = Array.from(
      { length: TRAINING.trainingCourses },
      () => 1 + Math.floor(this.random() * 900000),
    );
    const population = Array.from({ length: TRAINING.population }, (_, i) => {
      const weights =
        i === 0
          ? this.champion.weights.slice()
          : this.mean.map((m, k) => m + this.sigma[k] * gaussian(this.random));
      const fitness = average(
        seeds.map(
          (seed) =>
            episode(weights, seed, TRAINING.courseSeconds, "idle", this.kind)
              .score,
        ),
      );
      this.episodes += seeds.length;
      return { weights, fitness };
    }).sort((a, b) => b.fitness - a.fitness);
    const elite = population.slice(0, TRAINING.elites);
    for (let k = 0; k < this.mean.length; k++) {
      const mean = average(elite.map((e) => e.weights[k]));
      const deviation = Math.sqrt(
        average(elite.map((e) => (e.weights[k] - mean) ** 2)),
      );
      this.mean[k] = 0.3 * this.mean[k] + 0.7 * mean;
      this.sigma[k] = Math.max(0.07, 0.3 * this.sigma[k] + 0.7 * deviation);
    }
    const candidate = population[0];
    const validation = average(
      TRAINING.validationSeeds.map(
        (seed) =>
          episode(
            candidate.weights,
            seed,
            TRAINING.validationSeconds,
            "idle",
            this.kind,
          ).score,
      ),
    );
    this.episodes += TRAINING.validationSeeds.length;
    if (validation > this.champion.validation)
      this.champion = {
        version: CONTROLLERS[this.kind].version,
        weights: candidate.weights.slice(),
        generation,
        trainingSeed: this.seed,
        validation,
      };
    const progress = {
      generation,
      episodes: this.episodes,
      bestFitness: candidate.fitness,
      meanFitness: average(population.map((p) => p.fitness)),
      validation: this.champion.validation,
      model: this.champion,
    };
    const { model: _, ...row } = progress;
    this.history.push(row);
    return progress;
  }
}
