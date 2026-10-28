# sm90-sanitize

7/7 representative cells pass, and 27/27 sanitizer invocations report zero memcheck errors, racecheck hazards or synccheck errors.

Each invocation runs the `paired-reload-v2-check` path under compute-sanitizer: the same shared buffers and module reload that definitive timing uses, with two queued complete calls per image in each half instead of the full timed sequence. Numerical checks run before the work, after the reload and after the work. A control-reference run executes the linked and control images; a stock-reference run executes stock as well.

Long cells are too slow to race-check, so the sanitizers run on shorter representative cells. A timed cell without its own record counts as covered when a passing record exists for another cell of the same pipeline with the same executable inputs in the same build; a recorded failure is never covered this way. These are correctness checks and make no timing claim.

| Representative | Pipeline | Paired reference / tool | Hazards or errors | Verdict |
|---|---|---|---:|---|
| W1-8k | selected | paired-control/memcheck | 0 | PASS |
| W1-8k | selected | paired-control/racecheck | 0 | PASS |
| W1-8k | selected | paired-control/synccheck | 0 | PASS |
| W1-8k | selected | paired-stock/memcheck | 0 | PASS |
| W1-8k | selected | paired-stock/racecheck | 0 | PASS |
| W1-8k | selected | paired-stock/synccheck | 0 | PASS |
| W2a-8k | window | paired-control/memcheck | 0 | PASS |
| W2a-8k | window | paired-control/racecheck | 0 | PASS |
| W2a-8k | window | paired-control/synccheck | 0 | PASS |
| W2a-8k | window | paired-stock/memcheck | 0 | PASS |
| W2a-8k | window | paired-stock/racecheck | 0 | PASS |
| W2a-8k | window | paired-stock/synccheck | 0 | PASS |
| W3a-8k | window-softcap | paired-control/memcheck | 0 | PASS |
| W3a-8k | window-softcap | paired-control/racecheck | 0 | PASS |
| W3a-8k | window-softcap | paired-control/synccheck | 0 | PASS |
| W3b-8k | softcap | paired-control/memcheck | 0 | PASS |
| W3b-8k | softcap | paired-control/racecheck | 0 | PASS |
| W3b-8k | softcap | paired-control/synccheck | 0 | PASS |
| W4-32k | selected | paired-control/memcheck | 0 | PASS |
| W4-32k | selected | paired-control/racecheck | 0 | PASS |
| W4-32k | selected | paired-control/synccheck | 0 | PASS |
| W5-32k | phi | paired-control/memcheck | 0 | PASS |
| W5-32k | phi | paired-control/racecheck | 0 | PASS |
| W5-32k | phi | paired-control/synccheck | 0 | PASS |
| W6-prefix-32k | msa | paired-control/memcheck | 0 | PASS |
| W6-prefix-32k | msa | paired-control/racecheck | 0 | PASS |
| W6-prefix-32k | msa | paired-control/synccheck | 0 | PASS |
