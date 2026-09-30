import { useEffect, useRef, useState } from "react";
import { TRAINING, type Progress } from "../lib/training";
import { type Model } from "../lib/policy";
import { type Benchmark } from "../lib/benchmark";
export type History = Omit<Progress, "model">[];
export function downloadJson(name: string, data: unknown) {
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }),
  );
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
const controllerLabels: Record<string, { label: string; detail: string }> = {
  "connectome + trained readout": {
    label: "Trained agent",
    detail: "Circuit + trained decision network",
  },
  "direct readout 8-12-3 (no connectome)": {
    label: "No connectome, 147 weights",
    detail:
      "8 game inputs straight into an 8-12-3 network, trained the same way",
  },
  "direct readout 8-20-3 (no connectome)": {
    label: "No connectome, 243 weights",
    detail:
      "8 game inputs straight into an 8-20-3 network, trained the same way",
  },
  "circuit output zeroed": {
    label: "Circuit output zeroed",
    detail: "Same network, no game information reaches it",
  },
  "untrained readout": {
    label: "Untrained network",
    detail: "Same circuit, random initial readout weights",
  },
  rule: { label: "Simple rules", detail: "Handwritten distance thresholds" },
  random: {
    label: "Random actions",
    detail: "Run, jump or duck chosen at random",
  },
  idle: { label: "No input", detail: "Run without pressing any keys" },
};
export function TrainingBench({
  model,
  history,
  busy,
  status,
  benchmark,
  onTrain,
  onStop,
  onBenchmark,
  onImport,
  onRestore,
}: {
  model: Model | null;
  history: History;
  busy: "train" | "benchmark" | null;
  status: string;
  benchmark: Benchmark | null;
  onTrain: (seed: number) => void;
  onStop: () => void;
  onBenchmark: () => void;
  onImport: (file: File) => void;
  onRestore: () => void;
}) {
  const [seed, setSeed] = useState("20260912");
  const validSeed =
    Number.isInteger(Number(seed)) &&
    Number(seed) >= 1 &&
    Number(seed) <= 4294967295;
  const last = history.at(-1);
  const chartRef = useRef<SVGSVGElement>(null);
  const [chartWidth, setChartWidth] = useState(800);
  const hasHistory = history.length > 0;
  useEffect(() => {
    const el = chartRef.current;
    if (!el) return;
    const observer = new ResizeObserver(([entry]) =>
      setChartWidth(Math.max(260, entry.contentRect.width)),
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, [hasHistory]);
  // One scale for the two visible series; the unseen candidate maximum is irrelevant.
  const chartMax = Math.max(
    500,
    Math.ceil(
      Math.max(
        0,
        ...history.map((p) => Math.max(p.meanFitness, p.validation)),
      ) / 500,
    ) * 500,
  );
  const chartEnd = chartWidth - 28;
  const chartX = (generation: number) =>
    56 +
    ((generation - 1) / Math.max(1, TRAINING.generations - 1)) *
      (chartEnd - 56);
  const chartY = (value: number) => 172 - (value / chartMax) * 140;
  const path = (key: "meanFitness" | "validation") =>
    history
      .map(
        (p, i) => `${i ? "L" : "M"}${chartX(p.generation)},${chartY(p[key])}`,
      )
      .join(" ");
  const highlights = [
    "connectome + trained readout",
    "direct readout 8-20-3 (no connectome)",
    "circuit output zeroed",
    "untrained readout",
  ]
    .flatMap((name) => benchmark?.results.find((r) => r.name === name) ?? [])
    .slice(0, 3);
  return (
    <section className="training-bench" aria-labelledby="training-title">
      <div className="bench-heading">
        <div>
          <p className="eyebrow">TRAINING & EVALUATION</p>
          <h2 id="training-title">Teach the decision network</h2>
          <p>
            Train the small action readout while the 80-cell circuit stays
            fixed.
          </p>
        </div>
        <span role="status">{status}</span>
      </div>
      <div className="bench-controls">
        <div className="training-actions">
          <label>
            Training seed
            <input
              aria-label="Training seed"
              aria-describedby="seed-help"
              aria-invalid={!validSeed}
              type="number"
              min="1"
              max="4294967295"
              step="1"
              value={seed}
              disabled={!!busy}
              onChange={(e) => setSeed(e.target.value)}
            />
          </label>
          <button
            className="primary-action"
            disabled={!!busy || !validSeed}
            onClick={() => onTrain(Number(seed))}
          >
            Train from scratch
          </button>
          {busy && (
            <button className="stop-action" onClick={onStop}>
              Stop {busy === "train" ? "training" : "evaluation"}
            </button>
          )}
          <button disabled={!!busy || !model} onClick={onBenchmark}>
            Evaluate current model
          </button>
        </div>
        <div className="model-actions" role="group" aria-label="Model files">
          <button disabled={!!busy} onClick={onRestore}>
            Restore published model
          </button>
          <button
            disabled={!model}
            onClick={() => downloadJson("flydino-model.json", model)}
          >
            Export model
          </button>
          <label className={`import-model${busy ? " is-disabled" : ""}`}>
            Import model
            <input
              aria-label="Import model"
              type="file"
              accept=".json,application/json"
              disabled={!!busy}
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) onImport(file);
                e.target.value = "";
              }}
            />
          </label>
        </div>
      </div>
      <p
        id="seed-help"
        className={`field-help${validSeed ? "" : " field-error"}`}
      >
        {validSeed
          ? "The seed makes a training run repeatable. Training starts from random weights and replaces the model playing above."
          : "Enter a whole number from 1 to 4,294,967,295."}
      </p>
      <div className="training-readout">
        <div>
          <span>Generations logged</span>
          <strong>
            {last?.generation ?? 0}
            <small> / {TRAINING.generations}</small>
          </strong>
        </div>
        <div>
          <span>Episodes evaluated</span>
          <strong>{(last?.episodes ?? 0).toLocaleString("en-US")}</strong>
        </div>
        <div>
          <span>Best validation score</span>
          <strong>
            {Math.round(
              last?.validation ?? model?.validation ?? 0,
            ).toLocaleString("en-US")}
          </strong>
        </div>
        <div>
          <span>Candidates → selected</span>
          <strong>
            {TRAINING.population}
            <small> → {TRAINING.elites}</small>
          </strong>
        </div>
      </div>
      {history.length > 0 ? (
        <figure className="learning-figure">
          <div className="figure-heading">
            <h3>Learning over generations</h3>
            <button
              className="quiet-action"
              onClick={() =>
                downloadJson("flydino-training.json", {
                  algorithm: "CEM neuroevolution",
                  seed: model?.trainingSeed,
                  config: TRAINING,
                  history,
                })
              }
            >
              Export training log
            </button>
          </div>
          <svg
            ref={chartRef}
            className="training-chart"
            viewBox={`0 0 ${chartWidth} 212`}
            role="img"
            aria-label={`Training through generation ${last?.generation}. Latest population average ${Math.round(last?.meanFitness ?? 0)}, best validation score ${Math.round(last?.validation ?? 0)}. Higher scores are better.`}
          >
            {[0, chartMax / 2, chartMax].map((value) => (
              <g key={value}>
                <line
                  x1="56"
                  y1={chartY(value)}
                  x2={chartEnd}
                  y2={chartY(value)}
                  className="chart-grid"
                />
                <text x="44" y={chartY(value) + 4} textAnchor="end">
                  {value.toLocaleString("en-US")}
                </text>
              </g>
            ))}
            {[1, 20, 40, 60, TRAINING.generations].map((generation) => (
              <text
                key={generation}
                x={chartX(generation)}
                y="191"
                textAnchor="middle"
              >
                {generation}
              </text>
            ))}
            <text x="56" y="17">
              Game score ↑
            </text>
            <text x={chartEnd} y="17" textAnchor="end">
              Generation →
            </text>
            <path
              d={path("meanFitness")}
              fill="none"
              className="chart-population"
              strokeWidth="2"
              strokeDasharray="6 4"
            />
            <path
              d={path("validation")}
              fill="none"
              className="chart-validation"
              strokeWidth="2.5"
            />
          </svg>
          <figcaption>
            <span>
              <i className="legend-line population-line" />
              Population average · training courses
            </span>
            <span>
              <i className="legend-line validation-line" />
              Best saved model · validation courses
            </span>
          </figcaption>
          <p className="chart-note">
            Higher is better. The two lines use separate course sets; validation
            selects the saved model.
          </p>
        </figure>
      ) : (
        <p className="training-empty">
          {busy === "train"
            ? "Evaluating the first generation. The graph will appear when its results are ready."
            : "No training history loaded. Start a new run, or restore the published model to inspect its log."}
        </p>
      )}
      <p className="training-explainer">
        Each generation tests {TRAINING.population} candidates and selects the
        best {TRAINING.elites}. Only the decision network’s 243 parameters are
        updated.
      </p>
      <section
        id="results"
        className="benchmark-result"
        aria-labelledby="results-title"
      >
        <div className="figure-heading">
          <div>
            <p className="eyebrow">HELD-OUT EVALUATION</p>
            <h3 id="results-title">How well does it play?</h3>
          </div>
          {benchmark && (
            <button
              className="quiet-action"
              onClick={() => downloadJson("flydino-benchmark.json", benchmark)}
            >
              Export results
            </button>
          )}
        </div>
        {benchmark ? (
          <>
            <p className="result-context">
              {benchmark.config.seeds.length} unseen courses ·{" "}
              {benchmark.config.seconds} seconds each · model generation{" "}
              {benchmark.modelGeneration} · training seed{" "}
              {benchmark.trainingSeed}
            </p>
            <div className="result-summary">
              {highlights?.map((r) => (
                <div key={r.name}>
                  <span>{controllerLabels[r.name]?.label ?? r.name}</span>
                  <strong>
                    {r.survived}
                    <small> / {r.courses}</small>
                  </strong>
                  <span>courses completed</span>
                </div>
              ))}
            </div>
            <p className="table-scroll-hint">
              Scroll sideways for survival time and score →
            </p>
            <div
              className="benchmark-table"
              role="region"
              aria-label="Controller comparison, scroll horizontally on narrow screens"
              tabIndex={0}
            >
              <table>
                <caption>
                  Same courses and time limit for every controller. A completed
                  course means surviving the full {benchmark.config.seconds}{" "}
                  seconds.
                </caption>
                <thead>
                  <tr>
                    <th scope="col">Controller</th>
                    <th scope="col">Finished</th>
                    <th scope="col">Avg. survival</th>
                    <th scope="col">Avg. score</th>
                  </tr>
                </thead>
                <tbody>
                  {benchmark.results.map((r) => (
                    <tr key={r.name}>
                      <th scope="row">
                        <span>{controllerLabels[r.name]?.label ?? r.name}</span>
                        <small>{controllerLabels[r.name]?.detail}</small>
                      </th>
                      <td>
                        {r.survived}
                        <span className="muted"> / {r.courses}</span>
                      </td>
                      <td className="survival-cell">
                        <span>
                          {r.meanSeconds.toFixed(1)}
                          <small> s</small>
                        </span>
                        <div className="survival-track" aria-hidden="true">
                          <i
                            style={{
                              width: `${Math.max(0, Math.min(100, (r.meanSeconds / benchmark.config.seconds) * 100))}%`,
                            }}
                          />
                        </div>
                      </td>
                      <td>{Math.round(r.meanScore).toLocaleString("en-US")}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="result-note">
              Zeroing the circuit output leaves the trained network with a
              constant input, so it receives no information about the game. That
              control shows the information pathway is needed, not that the
              circuit’s computation helps. The no-connectome networks see the
              same eight game inputs directly and are trained with the same
              budget; they use the published training seed. Ten training runs
              per controller are compared in the replicate file.
            </p>
          </>
        ) : (
          <p className="training-empty" role="status">
            {busy === "benchmark"
              ? "Evaluating the current model on 100 unseen courses alongside the control conditions. Results will appear when all runs finish."
              : busy === "train"
                ? "Training is running. Stop or finish the run, then evaluate the saved model on unseen courses."
                : !model
                  ? "Loading the published model and its evaluation…"
                  : "No evaluation loaded for this model. Choose Evaluate current model to compare it with the control conditions."}
          </p>
        )}
      </section>
      <div className="bench-downloads" aria-label="Published experiment files">
        <span>Published files</span>
        <a href="/benchmarks/model.json" download>
          Model
        </a>
        <a href="/benchmarks/training.json" download>
          Training log
        </a>
        <a href="/benchmarks/replicates.json" download>
          Ten training runs per controller
        </a>
        <a href="/benchmarks/benchmark.json" download>
          Evaluation
        </a>
        <a href="https://github.com/cobanov/flyjump/blob/main/docs/experiment.md">
          Full protocol ↗
        </a>
      </div>
    </section>
  );
}
