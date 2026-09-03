# The eval endpoint became a job

Running the golden set takes minutes, and it used to happen inside the request
that asked for it. That shape had four problems, and only one of them was ever
visible: the nginx in front of the API needed `proxy_read_timeout 600s` to stop
cutting the connection. A timeout being configured around is a symptom, and
raising it treated the symptom.

The other three were quieter:

- **A closed tab threw the work away.** Four minutes of model calls, already
  paid for, discarded because nobody was still listening.
- **Two clicks were two runs.** Nothing stopped a second request starting a
  second full pass over the same corpus, and billing for it.
- **A spinner is indistinguishable from a hang.** With no progress, the only
  way to learn the run was still alive was to wait longer than the timeout —
  which is how the timeout got found in the first place.

So `POST /api/v1/eval` now returns `202` with a job id and keeps going;
`GET /api/v1/eval/{job_id}` reports progress and, when it is finished, the
report. `GET /api/v1/eval` returns the most recent run, so reloading the tab
does not mean re-running the set.

**The state lives in Redis, not in the process** — for the same reason
[the sparse index does](sparse-in-qdrant.md). A poll can land
on a different replica than the one doing the work, and a job that only one
process can see is a job that breaks the moment there are two.

**The failure Redis cannot cover is the process dying mid-run.** The job
heartbeats as it goes, and a record that stops beating is reported as
`abandoned` rather than left claiming to be running forever. The heartbeat also
refreshes the single-flight claim, so the two cannot disagree about whether a
run is still alive — and a crashed process cannot block every future run.

Verified against the deployed stack, twice. First the machinery, while the
account behind the model calls happened to be out of credits — which turned out
to be a useful accident, because it exercised the failure path:

```
POST         202 in 14ms  job=20f0dd8ac6d5 0/53
second click job=20f0dd8ac6d5 joined=True (one run, not two)
final        status=failed 0/53 error=RateLimitError: 429 ... no credits remaining
```

The start returns in **14 milliseconds** where it used to hold the connection
for minutes; the second click **joined** rather than starting a second paid run;
the case count is known at `0/53` before the first question finishes rather than
`0/?`, because "0 of ?" is the same non-answer the old spinner gave; and a
provider error is reported *through* the job instead of vanishing.

Then the whole golden set, once there were credits to run it with:

```
POST 202 in 18ms  job=554ea24e8810  0/53
   1/53  (9s)  ...  53/53  (281s)
final: done  53/53  281s
  recall 0.961  faithful 0.974  correct 0.943  halluc 0.000
  abstention 0.857  tokens 288 (naive 542, -46.8%)
  TOTAL $0.0574  ($0.00108/case)
```

All six gate thresholds pass, and the bill matches what
[the ledger](cost.md) predicted from three questions to within a
tenth of a cent.

With the long request gone, the proxy timeouts it forced came down with it:
600s to 120s in the frontend nginx and 3600 to 120 on the Ingress. The longest
response left is a streamed answer, which is seconds. It is not tighter than
that because SSE is idle between tokens, and a timeout firing mid-answer
produces a truncated answer rather than an error anyone can see.


---

Back to the [README](../README.md).
