# sm90-sanitize

R1 = linked/control (scored, >= 0.97); R2 = control/stock and R3 = linked/stock (diagnostic).
Ratios are medians of paired iterations/s ratios; these are not absolute-placement timings.
Attempts/discards count timing pairs; pairs that violate the clock lease are discarded.

| Cell | Pipeline | Comparison | Reference iterations/s | Candidate iterations/s | Ratio | Clock mode | Observed MHz | Valid pairs | Attempts | Discards | Verdict |
|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| W1-32k | selected | sm90-sanitize | | | | not timed | | | | | PASS |
| W1-8k | selected | sm90-sanitize | | | | not timed | | | | | PASS |
| W2a-32k | window | sm90-sanitize | | | | not timed | | | | | PASS |
| W2a-8k | window | sm90-sanitize | | | | not timed | | | | | PASS |
| W2b-32k | window | sm90-sanitize | | | | not timed | | | | | PASS |
| W2b-8k | window | sm90-sanitize | | | | not timed | | | | | PASS |
| W3a-32k | window-softcap | sm90-sanitize | | | | not timed | | | | | PASS |
| W3a-8k | window-softcap | sm90-sanitize | | | | not timed | | | | | PASS |
| W3b-8k | softcap | sm90-sanitize | | | | not timed | | | | | PASS |
| W4-32k | selected | sm90-sanitize | | | | not timed | | | | | PASS |
| W5-32k | phi | sm90-sanitize | | | | not timed | | | | | PASS |
| W6-prefix-32k | msa | sm90-sanitize | | | | not timed | | | | | PASS |
| W6-uniform-32k | msa | sm90-sanitize | | | | not timed | | | | | PASS |
