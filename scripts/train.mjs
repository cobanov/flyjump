import { writeFileSync, mkdirSync } from "node:fs";
import { Trainer, TRAINING } from "../src/lib/training.ts";
import { CONTROLLERS } from "../src/lib/policy.ts";
// Usage: train.mjs [seed] [generations] [output] [--controller=connectome|direct-8-12-3|direct-8-20-3]
const args = process.argv.slice(2),
  flag = args.find((a) => a.startsWith("--controller=")),
  controller = flag ? flag.slice("--controller=".length) : "connectome",
  [seedArg, generationsArg, outputArg] = args.filter((a) => a !== flag);
if (!(controller in CONTROLLERS))
  throw new Error(
    `Unknown controller ${controller}; expected ${Object.keys(CONTROLLERS).join(", ")}`,
  );
const seed = Number(seedArg ?? 20260912),
  generations = Number(generationsArg ?? TRAINING.generations);
const output =
  outputArg ??
  (controller === "connectome"
    ? "public/benchmarks"
    : `public/benchmarks/direct/${controller.slice("direct-".length)}/${seed}`);
mkdirSync(output, { recursive: true });
const trainer = new Trainer(seed, controller),
  start = performance.now();
for (let i = 0; i < generations; i++) {
  const p = trainer.step();
  console.log(
    `generation=${p.generation} train=${p.bestFitness.toFixed(1)} validation=${p.validation.toFixed(1)} episodes=${p.episodes}`,
  );
}
writeFileSync(`${output}/model.json`, JSON.stringify(trainer.champion));
writeFileSync(
  `${output}/training.json`,
  JSON.stringify(
    {
      algorithm: "CEM neuroevolution",
      controller,
      seed,
      config: { ...TRAINING, generations },
      wallSeconds: (performance.now() - start) / 1000,
      history: trainer.history,
    },
    null,
    2,
  ),
);
