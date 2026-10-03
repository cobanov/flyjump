# Changelog

## Unreleased

- Add trained direct-input controls with no connectome (8-12-3, 147 parameters; 8-20-3, 243 parameters), proposed by Ben Caunt in issue #1. They receive the same eight observations with the drive the circuit's input cells get, and use the same CEM budget.
- Train ten predeclared seeds per controller (seven new connectome seeds) and report every run on the same 100 held-out courses.
- Rename the silenced condition to "circuit output zeroed" and state that it only removes game information from the readout.

## 0.2.0 · 2026-09-12

- Rename the public experiment Fly Dino and use flydino.cobanov.dev.
- Replace approximate runner physics/art with pinned original Chromium Dino source and sprites, preserving BSD attribution.
- Add an extracted 80-cell MaleCNS recurrent circuit with measured connectivity, signed model activity and a genuinely trained 243-parameter action readout.
- Show decision network and real computed circuit state in separate, simultaneous panels; remove the illustrative image-driven brain overlay.
- Retrain and evaluate on fresh validation/test seed ranges; archive incompatible v1 results; publish controls, checkpoints, logs and replication runs.
- Refine responsive navigation, readable typography, controls, training console, methodology and upstream credits.

## 0.1.0 · 2026-09-12

- Independent runner and 8–12–3 artificial CEM controller with training, checkpoint export and benchmark.
- Anatomy was an illustrative overlay, not a connectome simulation. Its results apply only to the archived v1 environment.
