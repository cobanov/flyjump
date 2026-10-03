import { readFileSync, writeFileSync } from "node:fs";
import { benchmark, CONTROLLER_NAMES } from "../src/lib/benchmark.ts";
import { CONTROLLERS, validModel } from "../src/lib/policy.ts";
// Ten predeclared training seeds per controller, identical CEM budget.
const seeds = Array.from({ length: 10 }, (_, i) => 20260912 + i);
const path = (controller, seed) =>
  controller === "connectome"
    ? seed === seeds[0]
      ? "public/benchmarks"
      : `public/benchmarks/replicates/${seed}`
    : `public/benchmarks/direct/${controller.slice("direct-".length)}/${seed}`;
const arms = Object.keys(CONTROLLERS).map((controller) => {
  const results = seeds.map((seed) => {
    const dir = path(controller, seed),
      model = JSON.parse(readFileSync(`${dir}/model.json`, "utf8"));
    if (!validModel(model, controller) || model.trainingSeed !== seed)
      throw new Error(`Invalid ${controller} model in ${dir}`);
    const result = benchmark(model);
    // The published checkpoint's evaluation is written by scripts/benchmark.mjs.
    if (dir !== "public/benchmarks")
      writeFileSync(`${dir}/benchmark.json`, JSON.stringify(result, null, 2));
    const trained = result.results[0];
    return {
      seed,
      generation: model.generation,
      validation: model.validation,
      survived: trained.survived,
      meanSeconds: trained.meanSeconds,
      meanScore: trained.meanScore,
    };
  });
  const survived = results.map((r) => r.survived).sort((a, b) => a - b);
  return {
    controller,
    name: CONTROLLER_NAMES[controller],
    parameters: CONTROLLERS[controller].parameters,
    summary: {
      runs: results.length,
      meanSurvived: survived.reduce((a, b) => a + b, 0) / survived.length,
      medianSurvived:
        (survived[(survived.length - 1) >> 1] +
          survived[survived.length >> 1]) /
        2,
      minSurvived: survived[0],
      maxSurvived: survived.at(-1),
      runsAtLeast95: survived.filter((n) => n >= 95).length,
      meanSeconds:
        results.reduce((n, r) => n + r.meanSeconds, 0) / results.length,
    },
    results,
  };
});
writeFileSync(
  "public/benchmarks/replicates.json",
  JSON.stringify(
    {
      note: "Same protocol, CEM budget and predeclared training seeds for every controller. The published connectome checkpoint is the first seed, not selected using test performance. Test courses are shared across runs; each arm is 10 training runs on the same 100 courses, not 1,000 independent test courses.",
      seeds,
      arms,
    },
    null,
    2,
  ),
);
for (const arm of arms) {
  console.log(arm.name, arm.summary);
  console.table(arm.results);
}
