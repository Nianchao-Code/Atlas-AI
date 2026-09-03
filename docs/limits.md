# Limits

Measured and known, not hidden. Each of these is a deliberate stopping point
for a handbook-scale corpus, with the replacement named.

**The SLI counters are still per-process.** Retrieval state is not: dense and
sparse vectors both live in Qdrant, so replicas share one view of the corpus
and a write is visible to all of them at once. See
[the sparse index moved into Qdrant](sparse-in-qdrant.md) for
what that replaced and what it measured.

`/api/v1/metrics` drives the UI and reports the replica that served the
request, so with more than one it shows a slice rather than the system. The
Prometheus endpoint is the answer for anything that has to be true across
replicas; the JSON one stays because a demo should show numbers without asking
you to stand up a scraper first. It is the last piece of per-process state in
the service, and unlike the sparse index it never affected an answer.

**Auth is service-level, not user identity.** Every `/api/v1` route requires
an API key mapped to a named principal, and the semantic cache and rate limit
budget are both keyed by that principal. What it does not have is per-user
login: the nginx container presents one key on behalf of every browser that
reaches it, so anyone who can load the page can query. Real user identity means
a session layer in front, and principals would come from it rather than from a
secret.

**The user-input guard is a regex, and regexes lose this game.** It catches
the phrasings it was written against and nothing else, at a measured 7/11
bypass rate. It is kept as a cheap deterministic first pass, not as the defense
— that is the model's instruction-following, and the probe is what says so.

**The paraphrase cache catches roughly two thirds of rewordings.** At the
calibrated threshold it hits 6 of 9 measured paraphrases. Looser rephrasings
fall through to the full pipeline, which is the safe direction to fail, but it
means the hit rate is bounded by how people happen to phrase things. Raising
recall here needs a threshold that also fuses questions one decisive word
apart, so it is not a knob to turn without a better signal than cosine
similarity.

**The cross-encoder is off, and the measurement is the reason.** It changed
answer correctness by zero on the golden set while adding latency, so it stays
disabled in `infra/k8s/atlas.yaml`. Turning it on also costs a HuggingFace
fetch on the first request — about 12s, and egress the cluster may not have —
so the weights would need baking into the image first. That work is not worth
doing until a corpus exists where the reranker earns it.

**A document whose source is gone cannot be recovered.** Redis holds the
catalogue and runs with AOF on a PVC, so a restart no longer empties it, and
[reconciliation](reconciliation.md) repairs the two stores
when they disagree. What it cannot do is re-index an upload whose bytes died
with the pod's `emptyDir`; those are marked `failed` rather than silently left
listed. A persistent volume for uploads is the fix, and it is not there because
the demo corpus is seeded from a ConfigMap.


---

Back to the [README](../README.md).
