# scripts/

Every number in [the documentation](../docs/) came out of one of these. They are
grouped by what they are for, because "fifteen Python files" is not a map.

## Measure the system

| | Measures | Costs |
| --- | --- | --- |
| `ablation.py` | What each retrieval stage buys, one configuration per row, plus a control row that measures the harness's own noise floor. `--resume` and `--only` because a 45-minute run should not be all-or-nothing. | ~$1.12 |
| `ablation_variance.py` | Reads an ablation's raw output and asks how much of a gap is real — which cases flipped between repeats, and between two configurations that were the same pipeline. | free |
| `eval_gate.py` | The golden set against thresholds. `--smoke` needs no API key and runs in CI on every push; `--full` runs on pushes to main. | ~$0.06 |
| `loadtest.py` | What one replica serves: the cache-hit ceiling, head-of-line blocking under a slow request, and the cold path. Run from inside the cluster — `port-forward` serialises connections and would measure itself. | ~$0.02 |
| `corpus_stats.py` | Whether a corpus behaves like natural language, before any retrieval result measured on it is believed. Heaps β, Zipf slope, and the concentration checks that were added after they were needed. | free |

## Prove a specific claim by breaking it

These exist because "it should work" and "it does work" are different
statements. Each one causes the failure it claims to handle.

| | Causes | Checks |
| --- | --- | --- |
| `reconcile_probe.py` | An orphan vector and an unrecoverable record | Both repaired, dry run changes nothing, a second pass reports clean |
| `requeue_probe.py` | Deletes one document's vectors, leaves the record saying `ready` | It comes back and gets cited in an answer, not merely counted |
| `replica_check.py` | Two API replicas, addressed individually | Byte-identical rankings from each, for all three retrievers |
| `injection_probe.py` | 17 attacks across 4 defence configurations | What leaks, adjudicated by a judge rather than by substring matching |
| `distractor_probe.py` | — | Whether added documents actually compete, or are scenery |

## Build the corpus

| | |
| --- | --- |
| `generate_corpus.py` | Ten thousand adversarial distractors around the real handbook. Deterministic from `--seed`, so the generator is committed and 42MB of derived files is not. |
| `bulk_index.py` | Loads them: same chunking and same upsert as the worker, batched embeddings, resumable, and it reports what it spent. |
| `calibrate_cache.py` | Sets the paraphrase cache threshold from measured similarity pairs rather than from feel. |

## Operate and publish

| | |
| --- | --- |
| `k8s-deploy.ps1` | Build, load images, apply manifests. Generates the API key secret once; `-RotateApiKey` to replace it. |
| `dev.ps1` | Local stack without Kubernetes. |
| `capture_demo.py` | Records the README GIF against the real UI. Refuses to write one whose trace shows a cache hit, because that records the wrong pipeline and the difference is two words nobody checks. |
| `check_links.py` | Every link in every markdown file. Runs in CI, and was verified by planting three broken links of each kind. |

## Running them

Most import from `backend/app`, so they run from the repo root or from `/app`
inside a pod — both paths are tried. The ones that talk to the deployment want
to run inside the cluster:

```bash
kubectl exec -n atlas deploy/worker -- python /tmp/<script>.py
```

The ones that spend money report what they spent, via the ledger in
`backend/app/usage.py`, rather than leaving it to be estimated. Two estimates
were made before that existed and [both were wrong](../docs/cost.md), in
opposite directions.
