# sm90-definitive

Each row compares the linked image (A) with a reference image (B) on one cell. R1 uses the same-source control as the reference and must reach a ratio of at least 0.97. R3 uses stock FA3 and is reported for context, with no threshold.

The `paired-reload-v2` protocol times eight blocks in balanced ABBA/BAAB order. Each block has four batches of exactly ten complete calls, so a comparison scores 32 batches (320 calls) after 32 unscored warmup calls. After the fourth block both modules are unloaded and reloaded, reference first, in the same context and with the same fixture. Each block yields a B/A time ratio, and the reported ratio is sqrt(median of the first four × median of the last four), so values above 1 mean the linked image is faster. The eight blocks balance order effects; they are not independent samples.

Each comparison runs once; no result triggers a retry. It is admitted when the native command completes inside an uncontended GPU window under a healthy, qualified clock lease and returns a valid result. There is no minimum duration and no per-image telemetry requirement.

| Cell | Pipeline | Comparison | Ratio | Blocks | Calls/batch | Clock MHz | Admitted | Verdict |
|---|---|---|---:|---:|---:|---:|---|---|
| W1-32k | selected | R1 | 1.000189682 | 8 | 10 | 1230 | True | PASS |
| W1-32k | selected | R3 | 0.981644705 | 8 | 10 | 1230 | True | DIAGNOSTIC |
| W1-8k | selected | R1 | 0.999696417 | 8 | 10 | 1230 | True | PASS |
| W1-8k | selected | R3 | 0.956977257 | 8 | 10 | 1230 | True | DIAGNOSTIC |
| W2a-32k | window | R1 | 0.999960612 | 8 | 10 | 1230 | True | PASS |
| W2a-32k | window | R3 | 0.924808993 | 8 | 10 | 1230 | True | DIAGNOSTIC |
| W2a-8k | window | R1 | 0.999784835 | 8 | 10 | 1230 | True | PASS |
| W2a-8k | window | R3 | 0.937498710 | 8 | 10 | 1230 | True | DIAGNOSTIC |
| W2b-32k | window | R1 | 1.000018110 | 8 | 10 | 1230 | True | PASS |
| W2b-32k | window | R3 | 0.839808875 | 8 | 10 | 1230 | True | DIAGNOSTIC |
| W2b-8k | window | R1 | 1.000272940 | 8 | 10 | 1230 | True | PASS |
| W2b-8k | window | R3 | 0.845840818 | 8 | 10 | 1230 | True | DIAGNOSTIC |
| W3a-32k | window-softcap | R1 | 0.999994186 | 8 | 10 | 1230 | True | PASS |
| W3a-8k | window-softcap | R1 | 0.999838455 | 8 | 10 | 1230 | True | PASS |
| W3b-8k | softcap | R1 | 0.999980007 | 8 | 10 | 1230 | True | PASS |
| W4-128k | selected | R1 | 0.999804989 | 8 | 10 | 1230 | True | PASS |
| W4-32k | selected | R1 | 0.999134025 | 8 | 10 | 1230 | True | PASS |
| W5-128k | phi | R1 | 0.999969955 | 8 | 10 | 1230 | True | PASS |
| W5-32k | phi | R1 | 1.000495400 | 8 | 10 | 1230 | True | PASS |
| W6-prefix-128k | msa | R1 | 0.999436155 | 8 | 10 | 1230 | True | PASS |
| W6-prefix-32k | msa | R1 | 1.000240464 | 8 | 10 | 1230 | True | PASS |
| W6-uniform-128k | msa | R1 | 0.999395016 | 8 | 10 | 1230 | True | PASS |
| W6-uniform-32k | msa | R1 | 0.999972635 | 8 | 10 | 1230 | True | PASS |

## Admission and prerequisites

All 23 comparisons were admitted: the 17 R1 comparisons pass, and the six R3 comparisons completed as context. Every timing window was uncontended, every native command returned zero, and every 1230 MHz clock lease stayed healthy and was released.
R1 ranges from **0.999134025** to **1.000495400** against the 0.97 floor; R3 has no threshold.

Timing required passing [numerical gates on all 17 cells](sm90-gate.md) and [sanitizer runs on seven representative cells, 27 tool invocations](sm90-sanitize.md) from the same build.

The JSON record keeps every raw block, event duration, wall interval, numerical check and admission record; it and the raw native, window and clock logs stay local and are not published.
