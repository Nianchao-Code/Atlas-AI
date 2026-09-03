| Configuration | Recall | Ctx precision | Faithful | Correct | Halluc. | p95 ms | Tokens |
|---|---|---|---|---|---|---|---|
| **Dense only** | 1.000 ±0.000 | 0.505 ±0.002 | 0.981 ±0.000 | 0.925 ±0.000 | 0.000 ±0.000 | 462 ±135 | 1010 ±2 |
| **Control: dense only again** | 1.000 ±0.000 | 0.505 ±0.002 | 0.981 ±0.000 | 0.925 ±0.000 | 0.000 ±0.000 | 1094 ±1060 | 1019 ±6 |
| **Sparse only** | 1.000 ±0.000 | 0.412 ±0.002 | 0.981 ±0.000 | 0.896 ±0.013 | 0.000 ±0.000 | 805 ±614 | 1047 ±18 |
| **Hybrid + RRF** | 1.000 ±0.000 | 0.458 ±0.002 | 0.980 ±0.001 | 0.925 ±0.000 | 0.000 ±0.000 | 426 ±62 | 1067 ±1 |
| **+ LLM grading (full)** | 0.961 ±0.000 | 0.850 ±0.000 | 0.975 ±0.003 | 0.934 ±0.013 | 0.000 ±0.000 | 377 ±29 | 288 ±0 |
| **Full - query rewrite** | 0.961 ±0.000 | 0.850 ±0.000 | 0.976 ±0.001 | 0.925 ±0.000 | 0.000 ±0.000 | 1189 ±1191 | 277 ±0 |

53 golden questions, 2 run(s) per configuration, mean ±sd. Judge: `gpt-4o-mini`, generation: `gpt-4o`. Query embeddings are content-hash cached in Redis, so p95 covers vector search and fusion, not embedding API time.
