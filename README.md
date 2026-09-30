# Fly Dino

**[Open the experiment](https://flydino.cobanov.dev/)** · [Method and evidence](docs/experiment.md) · [Credits](THIRD_PARTY_NOTICES.md)

An 80-cell measured fly connectome circuit drives the **original Chromium Dino** through a learned neural readout. Watch the decisions and computed circuit activity side by side, train from random weights in your browser, and reproduce the benchmark locally.

- Original Chromium engine and sprites, pinned source, jump / short hop / duck / fast fall, genuine collision boxes.
- **8 engineered observations → 80 fixed MaleCNS cells → 16 descending-cell activities → 16–12–3 trained readout → 3 keys.**
- Real cross-entropy neuroevolution (CEM), 64 candidates, 8 elites, 80 generations, 15,680 episodes. Only the 243 readout parameters learn.
- Separate live decision network and anatomical activity view. Colored cells use the exact states that feed the action decoder. Gray atlas cells are unmodeled context.
- Seeded training, import/export checkpoints, learning curves, held-out benchmark and control conditions, including trained direct-input networks with no connectome.

The published checkpoint completed **99/100** held-out 180-second courses (mean survival **179.37 seconds**). Its initial untrained weights completed 0/100.

**Control without a connectome.** The same eight observations fed straight into an ordinary network, trained with the identical CEM budget, completed **100/100** (8-20-3, 243 parameters, the readout's parameter count) and **92/100** (8-12-3, 147 parameters) for the same training seed. Across ten training seeds per controller, mean completion was **92.2** for the connectome controller (range 78–100), **92.2** for the 243-parameter direct network (range 56–100) and **78.4** for the 147-parameter one (range 5–100). The task does not need the connectome, and these runs show no performance or reliability advantage from it. This control was proposed by [Ben Caunt](https://github.com/BenCaunt), who first ran it in [his fork](https://github.com/BenCaunt/flyjump) ([issue #1](https://github.com/cobanov/flyjump/issues/1)).

**What the zeroed-output control shows.** With the circuit output zeroed (previously called "silenced"), the trained readout receives 16 constant zeros and no game information, so it chooses the same key on every decision and completed 0/100, exactly like Idle. That shows the game information must reach the readout through the circuit; it does not show that the circuit's computation helps.

These results demonstrate learned control, **not superiority of biological topology**. Rewired or random circuits of the same size have not been tested. See per-course results and all training replicas in [public/benchmarks](public/benchmarks) and [the protocol](docs/experiment.md#held-out-evaluation).

## Run and reproduce

Node 22.18 or newer:

```sh
npm ci
npm run dev
# A complete reproducible training run, then independent evaluation:
npm run train -- 20260912 80
npm run benchmark
npm test
npm run check:assets
npm run build
```

The training command overwrites the published checkpoint and log. To preserve them, train replicas (seeds 20260913 to 20260921) and the direct-input controls (seeds 20260912 to 20260921) into their own directories:

```sh
npm run train -- 20260913 80 public/benchmarks/replicates/20260913
npm run train -- 20260912 80 --controller=direct-8-12-3   # public/benchmarks/direct/8-12-3/20260912
npm run train -- 20260912 80 --controller=direct-8-20-3   # public/benchmarks/direct/8-20-3/20260912
npm run benchmark:replicates
```

The original run took about 7.6 minutes on the development Mac mini; browser/device speeds vary. Use Node 22 as pinned in `.nvmrc`: with Node 26.8.1 the per-course `seconds` field of benchmark output was observed to differ from Node 22 for identical rollouts (actions, scores and trained weights matched). Training and evaluation run in a worker and can be stopped. Ordinary gameplay uses frozen weights. Browser-local checkpoints persist on the current origin; Export/Import transfers them across domains or devices.

## Controls

Space / Up / Jump: jump. Release after the minimum jump height for a shorter jump. Hold Down / S / Duck to crouch or drop faster in midair. Manual input takes over. The controller selector returns to automatic play. Manual collisions wait for Restart (or keyboard/canvas jump); automatic modes restart after 1.4 seconds. Pause freezes the game; Stop terminates a background training/evaluation job.

The adapter treats actions as held keys with explicit transitions. Holding Jump does not synthesize browser key-repeat events. This convention is identical in live play and training.

## What is real, and what is modeled?

**Real source/data:** Chromium game code and pixel sprites; MaleCNS v1.0 cell identities, soma coordinates, neurotransmitter annotations and all 1,296 selected directed connections (26,029 synaptic contacts); Flybody anatomical meshes.

**Engineered model:** an explicitly bounded 80-cell subgraph, game-state-to-visual-cell mapping, signed normalized leaky tanh dynamics, artificial trainable action readout and keyboard rig animation. Computed activity is dimensionless, not measured spikes or membrane voltage. The remaining 124,289-cell atlas is anatomical context, not a whole-brain simulation. No biological fly, muscle simulation or internal synaptic plasticity is claimed.

The circuit is selected using anatomy alone. Source hashes, extraction procedure, channel mapping, equations, benchmark split and limitations are in [the protocol](docs/experiment.md). Rebuild scripts are included; the approximately 1 GB raw connectivity table is downloaded separately, not bundled into the website.

## Shared work and attribution

- **The Chromium Authors:** original Dino code and artwork, BSD 3-Clause.
- **FlyEM / HHMI Janelia and MaleCNS collaborators:** measured connectome and anatomy, CC BY 4.0.
- **Turaga Lab / Flybody:** fly anatomy, Apache 2.0.
- **CodeBullet:** Dino structured observations, score-driven neuroevolution and decision-network visualization inspired this integration. CodeBullet uses NEAT; Fly Dino uses CEM. No unlicensed Processing code was copied.
- **Ben Caunt:** proposed and first ran the direct-input control in [BenCaunt/flyjump](https://github.com/BenCaunt/flyjump); the controls here reimplement it within the benchmark.
- **aome510/chrome-dino-game-rl:** reviewed DQN alternative; not our implemented algorithm.
- **nftechie/doomfly, liuzihe02/fly-craftax, eganeganegan/flydoom:** connectome learning architecture and experimental-control references. Their methods are distinguished in the [source review](docs/connectome-review.md).
- **de Boer, Kroese, Mannor and Rubinstein:** cross-entropy method tutorial.
- Built with [fly-connectome-template](https://github.com/cobanov/fly-connectome-template) by [Mert Cobanov](https://github.com/cobanov).

Full source pins, transformations and licenses: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Original application/template work retains the [Cobanov Template Attribution License 1.0](LICENSE). Third-party licenses remain separate. Independent project, not affiliated with Google Chrome.

## Deployment

Static Vite application. Existing Cloudflare Pages project and GitHub repository remain named `flyjump`; public branding and canonical URL are **Fly Dino / flydino.cobanov.dev**. Deploy with `wrangler pages deploy dist --project-name flyjump --branch main`. The old `flyjump.cobanov.dev` alias remains compatible.
