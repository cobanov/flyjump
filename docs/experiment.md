# Fly Dino v2: protocol and evidence

Recorded 2026-09-12. Environment ID: `flydino-chromium98-connectome-v2`.

## Question and scope

Can a trainable readout learn the original Dino task when its only inputs are the computed activities of a small, measured MaleCNS circuit? How does it compare with the same kind of readout trained on the game observations directly, with no circuit at all?

This experiment demonstrates numerical learning and a real causal computation path. The direct-input control (added after publication, see [below](#direct-input-control-no-connectome)) shows the task can be solved as well without the connectome. The experiment does **not** test whether biological topology is better than an equally sized artificial or rewired recurrent network. It does not model the entire brain, measured electrophysiology, muscles, learning in a living animal, or plasticity of biological synapses.

## Original environment

The Chromium Authors' `offline.js`, sprite definitions and 2× sprite sheet are pinned to Chromium **98.0.4758.55**, commit `0b0619d8287f51c4fca09d0385c163d30bd35c4b`. Source files are vendored unchanged under `vendor/chromium`; hashes cover those files and the generated wrapper. BSD-3-Clause license is bundled and linked on the page.

`build-chromium.mjs` surrounds the original source with a lexical platform adapter: each episode has its own seeded RNG and fixed clock. `runner.ts` instantiates original Trex, Horizon and DistanceMeter classes and calls the original `Runner.update()`, including original jump physics, cactus/bird generation, detailed collision boxes, speed curve, score and night-mode logic.

Integration choices: 600×150 logical game area on all devices; desktop obstacle rules; start at the post-intro running position; omit browser offline-page DOM, audio and restart-icon animation; externally schedule a fixed **60 Hz** tick; automatic controllers restart after collision. Held actions use key transitions, without OS key-repeat synthesis. Original keyboard behavior, including minimum jump height and fast fall, is retained through Trex methods. No rewritten collision model or modified obstacle physics is used.

The rendered game uses the same update path and consumes the same random sequence as headless rollouts. A regression test compares every state step with/without a drawing surface. Headless rendering is a no-op, not a separate simulator. Score is Chromium's `getActualDistance(ceil(distanceRan))` (coefficient 0.025), not v1's custom distance scale.

## Measured circuit and selection

Data: [MaleCNS v1.0 downloads](https://male-cns.janelia.org/download/), minimum confidence 0.5. [Manifest](../public/data/connectome/manifest.json) pins SHA-256 of all raw tables and the exact exported graph. Data creators: FlyEM / HHMI Janelia and collaborators; CC BY 4.0.

Selection is deterministic and uses anatomy alone, before training:

1. For each of `LC4, LC11, LC9, LC15, LC16, LC17, LC21, LPLC2`, retain four visual cells with largest total direct contact count onto descending neurons with soma coordinates.
2. Take the union of the top two descending targets per visual type; fill to 16 by total contact rank. Break ties by body ID.
3. Add the 32 strongest two-hop bridge cells, ranked by the minimum of summed selected-visual input and selected-descending output contact counts.
4. Retain **every** measured directed edge among the selected cells, including one-contact and recurrent edges. Do not synthesize edges.

Result: **80 cells, 1,296 directed edges, 26,029 contacts**, 32 driven visual cells, 32 intermediates, 16 readout cells. All output cells are reachable from input through nonzero-sign measured edges. The one unclear transmitter cell retains its anatomical connections but contributes zero outgoing modeled drive. There are 61 acetylcholine, 13 GABA, 5 glutamate and 1 unclear annotations.

Selection favors a compact, directly connected visual-to-descending circuit. This selection and its boundary truncation are strong inductive biases, not a representative sample of the complete CNS. No physiological receptive fields are inferred.

## Observation encoder and recurrence

Eight engineered state features, using the nearest obstacle whose trailing edge has not passed the dinosaur's x=50:

| Channel | Feature | Encoding | Driven cell type |
| --- | --- | --- | --- |
| 0 | Proximity | `1 - clamp((obstacle.x - 50)/600, 0, 1)`; 0 when absent | LC4 |
| 1 | Obstacle width | `width / 75`; 0 absent | LC11 |
| 2 | Obstacle height | `height / 60`; 0 absent | LC9 |
| 3 | Obstacle altitude | `bottom / 60`; 0 absent | LC15 |
| 4 | Speed | `pixelsPerSecond / 780` | LC16 |
| 5 | Player jump height | `heightAboveGround / 100` | LC17 |
| 6 | Player vertical velocity | `(velocity / 900 + 1) / 2` | LC21 |
| 7 | Ground contact | 1 grounded, 0 airborne | LPLC2 |

These are structured game observations inspired by CodeBullet's input design, **not pixels**. Assignment to cell types is an arbitrary, fixed engineering encoder and has no claimed biological interpretation. Inputs can briefly exceed nominal normalization bounds under original physics; they are not all clipped. No rule policy supplies targets or actions during neural training.

A driven cell receives `u[i] = 2 * (feature[channel] - 0.5)`; all others receive zero external drive. Let `c[j,i]` be the measured contact count and `s[j]` the assumed presynaptic sign: acetylcholine +1, GABA/glutamate −1, unclear/modulatory 0. Incoming signed weights are normalized by total absolute signed contact count at each postsynaptic cell:

```text
W[j,i] = c[j,i] * s[j] / sum_k(c[k,i] * abs(s[k]))
h_new[i] = 0.3 * h[i] + 0.7 * tanh(u[i] + 1.4 * sum_j W[j,i] * h[j])
```

Zero denominators produce zero drive. Three synchronous iterations run at each 30 Hz action decision. State is reset to zero for each episode and persists between decisions. JavaScript Float64 arithmetic is used in all environments. The normalized signed activity is dimensionless; it is neither firing rate nor membrane voltage. The transmitter sign mapping, gain, leak and three iterations are simplified assumptions, not calibrated physiology.

The readout receives `4 * h[outputCell]` from 16 selected descending cells. Its architecture is **16 inputs → 12 tanh hidden units → 3 linear scores**, with biases, totaling **243 parameters**. Argmax chooses Run, Jump or Duck. There is no direct observation connection to the trainable network. Brain-view colors use the same `h` state; the decision-network labels show the gain-scaled inputs. Network edges depict weighted signal contribution; scores are not probabilities.

## Actual learning

Diagonal Gaussian cross-entropy method, independently implemented from the standard algorithm. This is score-based policy search / neuroevolution in an RL environment, not DQN, PPO, NEAT or backpropagation.

- Training seed: **20260912**; additional predeclared replicas: **20260913**, **20260914**.
- 80 generations, 64 candidates, 8 elites; 3 fresh shared training courses per generation.
- Training course seeds sampled from `1..900000` by seeded LCG. Box–Muller Gaussian weight sampling.
- Each rollout ends on collision or 180 seconds. Fitness: mean original Dino score over the three courses.
- Candidate 0 preserves the current validation champion; others are sampled from per-parameter mean/sigma.
- Initial mean 0, sigma 0.8; initial champion random Gaussian sigma 0.7.
- Mean and sigma use 0.3 old + 0.7 elite statistics; sigma floor 0.07.
- Every generation's best training candidate is evaluated on validation seeds **1100001–1100004**, 180 seconds each. Champion replaced only on strictly improved mean validation score.
- **15,680 episodes per run**: 80 × (64 × 3 + 4). Published training time: 456.33 seconds on the development Mac mini (not a portable performance guarantee).

Only readout weights change. The biological-edge graph, recurrence and encoder remain fixed. Additional connectome seeds **20260915–20260921** were declared and trained with the same settings when the direct-input control was added, giving ten runs per controller. Normal gameplay runs inference only. Browser training performs genuine candidate rollouts in a Web Worker; the visible game uses the current validation champion. Stop terminates the worker and retains its latest checkpoint.

## Direct-input control (no connectome)

Proposed by Ben Caunt in [issue #1](https://github.com/cobanov/flyjump/issues/1), who first ran it in [his fork](https://github.com/BenCaunt/flyjump/blob/d2e69a46f13b694db5550921eb4b433fc46e0fe3/analysis/experiments/direct-input/FINDINGS.md). The controller receives the eight observations directly, each encoded as `2 * (feature - 0.5)`: exactly the drive the circuit's input cells receive. There is no circuit, recurrence or hidden state between decisions. Two widths are trained:

- **8 → 12 tanh → 3**, 147 parameters: the connectome readout's hidden width. This matches the network in Ben Caunt's fork, which reached it by folding the same encoding into the weights after a validation-only input-scaling sweep. Here the encoding is fixed in advance, so no scaling search was run.
- **8 → 20 tanh → 3**, 243 parameters: the connectome readout's parameter count, so CEM searches a space of the same dimension. With the same training seed this arm also consumes the random stream identically, so it faces the same training courses in every generation as the connectome run.

Everything else is unchanged: CEM settings, 80 generations, 15,680 episodes per run, validation seeds and checkpoint rule, 30 Hz decisions and the held-out test seeds. Ten predeclared training seeds, **20260912–20260921**, were run for each width before any direct controller was evaluated on test courses. `npm run train -- 20260912 80 --controller=direct-8-12-3` writes to `public/benchmarks/direct/8-12-3/20260912`.

## Held-out evaluation

Exactly **100 test seeds: 2100001–2100100**, each capped at 180 seconds. These seeds are absent from training and validation, and changed from the already-inspected v1 split. The first declared training seed remains the published checkpoint regardless of other replicas' test scores. Architecture and weights were frozen before this test.

| Controller | Completed / 100 | Mean survival | Mean original score |
| --- | ---: | ---: | ---: |
| Connectome + trained readout | **99** | **179.372 s** | **2885.75** |
| Direct readout 8-12-3, no connectome (147 parameters) | 92 | 176.888 s | 2837.34 |
| Direct readout 8-20-3, no connectome (243 parameters) | 100 | 180.000 s | 2898.00 |
| Same connectome readout, circuit output zeroed | 0 | 4.5085 s | 41.00 |
| Initial untrained readout | 0 | 4.4918 s | 41.00 |
| Handwritten rule baseline | 0 | 46.4927 s | 535.27 |
| Uniform random actions at 30 Hz | 0 | 4.6622 s | 42.77 |
| Idle | 0 | 4.5085 s | 41.00 |

Published champion: **generation 32**, validation score **2591.75**. The run continues to generation 80 without replacing that champion. The direct rows use the direct checkpoints from the same first declared seed, 20260912, not chosen by test score.

**What the zeroed-output control shows.** It was previously labeled "silenced connectome". Zeroing clears the circuit state and returns 16 zeros, and those zeros are the readout's only input. The feed-forward readout therefore sees one constant input and chooses one constant action: it pressed Jump on all 13,551 of its decisions and survived exactly as long as Idle. The 99 versus 0 comparison establishes that the readout needs information from the game to arrive through the circuit. It does **not** show that the circuit's computation, or fly-specific connectivity, adds anything over the raw observations; the direct-input control addresses that question.

The handwritten rule baseline is a simple, untuned distance threshold; it is not a strong optimized controller. Do not interpret this table as outperforming all rule-based methods.

[Per-course benchmark](../public/benchmarks/benchmark.json) includes actions, jumps, ducks, deaths and exact checkpoint. [Training history](../public/benchmarks/training.json) records every generation. [Replicas](../public/benchmarks/replicates.json) report all ten training runs per controller on the **same** held-out courses; those are not 1,000 independent courses. All per-replica models, logs and results are published. A topology-benefit claim would require matched artificial/rewired recurrent controls trained with equal budgets; that study is not included.

### Independent training replicas

Ten training runs per controller, identical budget, each evaluated on the same 100 held-out courses. Cells show completed courses out of 100 and the selected generation; validation score is 2898.00 (the four-course maximum) unless noted.

| Training seed | Connectome + readout | Direct 8-12-3 | Direct 8-20-3 |
| --- | ---: | ---: | ---: |
| 20260912 (published) | 99 (gen. 32, val. 2591.75) | 92 (gen. 55) | 100 (gen. 24) |
| 20260913 | 85 (gen. 46) | 100 (gen. 18) | 100 (gen. 20) |
| 20260914 | 100 (gen. 33) | 5 (gen. 79, val. 1176.00) | 95 (gen. 32) |
| 20260915 | 87 (gen. 42) | 98 (gen. 23) | 98 (gen. 32) |
| 20260916 | 92 (gen. 52) | 88 (gen. 44) | 95 (gen. 26) |
| 20260917 | 99 (gen. 32) | 95 (gen. 10) | 79 (gen. 31) |
| 20260918 | 97 (gen. 51) | 89 (gen. 30) | 100 (gen. 18) |
| 20260919 | 96 (gen. 23) | 84 (gen. 19) | 100 (gen. 33) |
| 20260920 | 78 (gen. 29) | 86 (gen. 28) | 56 (gen. 52) |
| 20260921 | 89 (gen. 30) | 47 (gen. 73, val. 2517.50) | 99 (gen. 18) |
| **Mean completed** | **92.2** | **78.4** | **92.2** |
| Median / range | 94 / 78–100 | 88.5 / 5–100 | 98.5 / 56–100 |
| Runs with at least 95 | 5 of 10 | 3 of 10 | 8 of 10 |
| Mean survival | 174.90 s | 163.64 s | 171.37 s |

The parameter-matched direct controller (8-20-3) has the same mean completion as the connectome controller across ten seeds, with a higher median and a wider spread. The 147-parameter direct controller is less reliable: two of its runs never reached a perfect validation score within 80 generations, and one completed only 5 courses. No run of any arm was excluded. Several runs have a perfect four-course validation score but lower held-out completion, which is why validation and test results are reported separately. There was no test-based checkpoint selection.

These are ten training runs per arm on shared test courses, not 1,000 independent courses, and no significance test is claimed. The results do not support a claim that the connectome improves performance or training reliability on this engineered-state task, and they do not show that it harms it. Matched random or degree-preserving rewired circuits, which would isolate the contribution of fly-specific topology from that of any fixed recurrent layer, have not been trained.

## Rebuilding data and code

Raw source directory (approximately 1.1 GB; do not commit raw files):

```sh
mkdir -p /tmp/pinfly-data
base=https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome
curl -fL "$base/body-annotations-male-cns-v1.0-minconf-0.5.feather" -o /tmp/pinfly-data/annotations.feather
curl -fL "$base/connectome-weights-male-cns-v1.0-minconf-0.5.feather" -o /tmp/pinfly-data/edges.feather
curl -fL "$base/body-neurotransmitters-male-cns-v1.0.feather" -o /tmp/pinfly-data/neurotransmitters.feather
uv run --with pyarrow --with numpy python scripts/build-connectome.py /tmp/pinfly-data
npm run build:chromium
npm run check:assets
```

The data builder rejects unexpected source hashes. The Chromium generator rebuilds a wrapper around unchanged vendored files; asset checks verify pinned source, sprites, graph and anatomical asset hashes. `npm test` reproduces the main benchmark (including both direct-input rows), tests deterministic learning, input sensitivity, zeroed circuit output, the direct controllers' input encoding, graph reachability, key transitions, original collision geometry and rendered/headless parity.

## Prior version

[v1 protocol](archive/v1/experiment.md) and [v1 artifacts](../public/benchmarks/archive-v1) are retained as historical records. They used a different, independently drawn runner and a conventional 8–12–3 controller. Their 54/100 result and illustrative anatomy overlay do not describe v2. v1 checkpoints are rejected by the v2 loader.
