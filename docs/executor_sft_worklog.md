# Executor LoRA SFT execution log

## Objective

Complete the executor model configuration, exact model-attempt recording, validated
decision datasets, isolated training environment, LoRA training, comparative
evaluation, integration and rollback documentation. Preserve default planner
behavior and existing user edits. No paid APIs, cloud purchases, commits or pushes.

## Initial evidence (2026-09-10)

- GPU: NVIDIA RTX 5060 Laptop, 8151 MiB VRAM; 160 MiB occupied at inspection.
- PyTorch 2.11.0+cu130 reports CUDA available; transformers 4.57.6; PEFT 0.20.0.
- Workspace disk: approximately 505 GB free.
- HF cache contains BGE embedding/reranker models, no generative executor model.
- Candidate: Qwen/Qwen3-0.6B, non-thinking tool calling, ordinary BF16 LoRA first.
  This is a feasibility baseline, not a claim of production capability.
- Existing worktree edits are user documentation and evidence artifacts; preserve.
- Initial targeted tests: 39 passed, 2 setup errors caused by inaccessible system
  pytest temporary directory. Retrying with a fresh workspace-local basetemp.
- Initial model download encountered inherited HF_HUB_OFFLINE. Retrying with
  online access explicitly enabled for this download only and no auth token.

## Completion checklist

- [x] Independent executor endpoint/model/context and role-specific fallback
- [x] Shared total request/token/concurrency limits preserved
- [x] Exact attempt recording including governed/emergency/retried requests
- [x] Dataset export, privacy handling, pairing and leakage validation
- [x] Authored training/dev/test scenarios, provenance and frozen split manifest
- [x] Isolated reproducible training environment and target-only loss checks
- [x] Real adapter training, save/load and generation evidence if resources allow
- [x] Base/adapter held-out comparison and actual Agent integration (quality failures retained)
- [x] Regression checks, reproducible commands, rollback and limitations

## Progress evidence

- Model downloaded to `data/models/Qwen3-0.6B` at revision
  `c1899de289a04d12100db370d81485cdf75e47ca`. Actual weights SHA-256 matches
  download ETag: `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`.
- Isolated `training/.venv` created; actual dependency freeze in
  `training/environment-lock.txt`. No paid model calls or cloud resources used.
- Routing/recorder regression: 44 tests passed. Subsequent full suite: 183 passed;
  later additions still require final verification.
- Smoke training `training/outputs/smoke-01`: 8 training examples, 2 epochs,
  4,587,520 trainable parameters, peak allocated CUDA memory 3,769,779,200 bytes.
  Development loss 0.86228 -> 0.61084; adapter saved and reloaded. Four preliminary
  development generations: 2 passed, 2 omitted requested top_k. This is a load/
  generation smoke test, not evidence of overall quality gain.
- Dataset review corrected JPY precision using production Money, aligned compressed
  TaskState constraints to its schema, and added preference conflict, category-to-
  search and transient retry cases. Final authored split: 224 train / 56 dev / 84
  test. No held-out test generation has been run at this point.
- First full run `executor-v1` failed with CUDA OOM. Retained config and recorded
  failure; no adapter or successful full-training claim from that run.
- Replaced full prompt vocabulary logits with target-position logits. Tiny Qwen
  equivalence check: both losses 4.149051666259766, maximum gradient error 0.0.
- Full run `executor-v2` launched with the validated memory reduction. Running;
  evaluate only after the process finishes and adapter is verified.
- Actual production item_search on a synthetic lexical fixture returns CUP-OK and
  filters CUP-OVER / CUP-NOSHIP correctly. Real model Agent integration pending.
- Updated uv.lock for the JSON Schema dev dependency; four validator packages added.

## Full training complete; held-out evaluation in progress

- `executor-v2` completed both epochs. Script-measured duration 1447.916 seconds;
  peak allocated CUDA memory 1,933,725,696 bytes. Development loss by epoch:
  0: 0.8335391; 1: 0.1284545; 2: 0.1495943. The saved adapter is epoch 1,
  selected solely by development loss before held-out generation.
- Adapter: `training/outputs/executor-v2/adapter/adapter_model.safetensors`,
  18,380,008 bytes. Current base/adapter generation comparison is pending.
- Exact emergency request accounting was added after finding that a governance-
  internal retry could otherwise bypass the outer gateway request count. Tests
  prove 1-request caps stop the second physical call and 2-request caps permit it,
  with distinct captured requests under the same logical step.
- Final full regression after these changes: 190 passed, 1 existing Starlette
  deprecation warning. Machine-readable report: `eval/reports/executor/regression.xml`.
- Base held-out evaluation launched with all 84 frozen cases, no fallback/cache,
  greedy decoding and max_new_tokens=256. Active tool session: 72708.
  Output directory: `eval/reports/executor/base-test-v1`.
- Next: wait for this actual process to finish, then run the selected adapter with
  identical settings into `adapter-test-v1`; compare, perform live local Agent
  integration, review sampled failures, and finish the requirement audit.

## Baseline finished; selected adapter evaluation started

- Baseline summary contains all 84 cases: 40 passed (47.619%). This is the
  authored single-decision rule score, not end-to-end business success.
- Selected epoch-1 adapter evaluation launched with the same frozen evaluator,
  model, dataset and decoding settings into `adapter-test-v1` (session 39780).
- Latest complete application regression: 193 passed, with one existing
  Starlette warning; `eval/reports/executor/regression-final.xml`.
- Remaining gates: completed paired comparison, real local Agent integration,
  reviewed runtime export/tokenization and final evidence audit.
- Verified running evaluator launcher/worker IDs 37072/35744, both matching
  `training.evaluate_executor ... adapter-test-v1`. A serial continuation is
  waiting on those exact processes (session 17821; script retained at
  `eval/reports/executor/finish-v1.ps1`). It requires a complete 84-case summary
  and successful comparator before starting the local adapter server, running
  `integration_smoke`, and stopping only that owned server process tree.
  Do not start duplicate integration/GPU jobs while this continuation is live.
- At the last observation, evaluator session 39780 had completed 39/84 cases.
  Startup provenance comparison confirms identical data, base files, evaluator,
  template, decoding, context and generation limits; both runs disable fallback
  and response cache. Complete score comparison is still pending.

## Final evidence and limits

- Both held-out runs completed, with matching experimental provenance. Base:
  40/84; adapter: 72/84. Paired improvements/regressions: 33/1. Tool parameter
  rate 57.41% -> 98.15%; p95 latency 28.75 -> 44.11 seconds. Clarify 0/6 and
  preference conflict 1/6 remain weak. Full comparison in `comparison.json`.
- Real local integration-v1 used the correct call path but simplified the catalog
  category, producing an empty filtered result. integration-v2 supplied an
  explicit catalog category: arguments and production filtering were correct,
  but the final answer duplicated the one valid product. The initial path-only
  gate missed this; the added answer-quality gate and audit mark quality failed.
  Original outputs were not overwritten and the frozen model test was not changed.
- Reviewed two actual executor captures from integration-v2. The correct tool
  response is unchanged; the duplicated final answer has an explicit reviewed
  correction. Export records target_origin and response/review hashes. Both
  samples are synthetic-session train data, not newly claimed test successes.
  Export validation includes local tokenizer prefix/mask/length checks (1213 and
  1642 tokens). No retraining was performed on these post-evaluation samples.
- Final full regression: 194 passed, one existing Starlette warning. Ruff,
  whitespace and offline dependency-lock checks passed. Current model/data hashes,
  actual capture records, export reviews and test XML cross-checked in
  `completion_evidence.json`. The initial audit command incorrectly compared
  canonical data hashes with raw file hashes; this audit error was corrected to
  use the same canonical hashing used by training and evaluation. No data changed.
- Owned train/evaluate/server processes exited, port 8010 has no listener and
  GPU memory returned to 30 MiB. No .env edits, paid APIs, purchases, commits or pushes.
- The requested engineering/experiment loop is delivered. Production model
  readiness is not claimed: category mapping, clarification, conflicting
  preferences and duplicate answers need further development and independent
  evaluation before replacing the default executor.
