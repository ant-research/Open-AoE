# Script layout

The `scripts/` root is reserved for production orchestration, input adapters,
route selection, provenance, and backend entrypoints.

- `diagnostics/`: dataset screening, read-only audits, trajectory inspection,
  explicit repair tools, and numerical validators.
- `review/`: RGB overlays, mesh visualizations, contact sheets, triptychs, and
  review-pool summaries.
- `maintenance/`: third-party setup, experiment pruning, and explicit runtime
  patch installers. These tools change or maintain an execution environment;
  they are not ordinary retarget entrypoints.

Run tools from the repository root, for example:

```bash
python scripts/diagnostics/validate_dai_raw_to_processed_hoi.py --help
python scripts/diagnostics/screen_aoe_retarget_candidates.py --help
python scripts/review/compose_triptych.py --help
python scripts/maintenance/prune_experiments.py --help
```

On a newly configured third-party machine, run this before any GPU job:

```bash
cp local_env.example.sh local_env.sh
source local_env.sh
python3 scripts/check_third_party_config.py --json third_party_config_report.json
```

Production runners use the configured executables, dataset root, and local
`third_party/` paths.

Production entrypoints remain directly under `scripts/`, including
`run_fresh_aoe_auto_window.py`,
`run_fresh_aoe_scene_12_demos.sh`, `run_full_12_demos.sh`,
`run_v4_two_full_pipelines.sh`, `run_do_as_i_do_official_retarget.sh`, and
`run_spider_retarget.sh`.

## AoE input modes

Both modes enter the same reconstruction and DAI/SPIDER retarget pipeline.

- Automatic window selection: `run_fresh_aoe_auto_window.py` tests the
  configured durations longest-first with the unchanged SAM3 prompt/mask
  gates, then launches one formal full run with the first passing window.
- Explicit input window: use `--window-selection-mode direct` together with
  `--direct-duration-sec` and pass an audited `--ref-source-frame` after `--`.
  This bypasses automatic window selection. It does not bypass reconstruction
  or silently weaken SAM3/DAI/SPIDER validation.

Example automatic invocation:

```bash
python scripts/run_fresh_aoe_auto_window.py \
  --runner scripts/run_fresh_aoe_scene_12_demos.sh \
  --experiments-root experiments \
  --source-run cooking_wine_auto_source \
  --matrix-run cooking_wine_auto_matrix \
  --window-selection-mode auto \
  --candidate-durations-sec 3.0,2.6,2.4,2.2,2.0 \
  --manifest experiments/cooking_wine_auto_window.json -- \
  --scene cooking_wine_bottle --segment SEGMENT --annotation-id 6 \
  --object-name "cooking wine bottle" --task cooking_wine_bottle_left \
  --hand-type left --anchor-hand left --ref-source-frame 411
```

Example explicit-window invocation:

```bash
python scripts/run_fresh_aoe_auto_window.py \
  --runner scripts/run_fresh_aoe_scene_12_demos.sh \
  --experiments-root experiments \
  --source-run cooking_wine_direct_source \
  --matrix-run cooking_wine_direct_matrix \
  --window-selection-mode direct --direct-duration-sec 2.4 \
  --manifest experiments/cooking_wine_direct_window.json -- \
  --scene cooking_wine_bottle --segment SEGMENT --annotation-id 6 \
  --object-name "cooking wine bottle" --task cooking_wine_bottle_left \
  --hand-type left --anchor-hand left --ref-source-frame 411
```

Candidate screening is an optional diagnostic step. It is kept under
`scripts/diagnostics/` because the automatic and explicit production entries
do not depend on a precomputed candidate list.
