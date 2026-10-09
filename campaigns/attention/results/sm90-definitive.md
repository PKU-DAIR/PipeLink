# sm90-definitive

R1 = linked/control (scored, >= 0.97); R2 = control/stock and R3 = linked/stock (diagnostic).
Ratios are medians of paired iterations/s ratios; these are not absolute-placement timings.
Attempts/discards count timing pairs; pairs that violate the clock lease are discarded.

| Cell | Pipeline | Comparison | Reference iterations/s | Candidate iterations/s | Ratio | Clock mode | Observed MHz | Valid pairs | Attempts | Discards | Verdict |
|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| W1-32k | selected | R1 | 69.88319667435262 | 69.88837362869683 | 1.0001347026631469 | locked | 1305.0 | 8 | 8 | 0 | PASS |
| W1-32k | selected | R3 | 71.19575608956335 | 69.86731181243957 | 0.9813902637779545 | locked | 1305.0 | 8 | 8 | 0 | PASS |
| W1-8k | selected | R1 | 1030.1854806128322 | 1030.0576104096285 | 0.9997492552760295 | locked | 1305.0 | 8 | 8 | 0 | PASS |
| W1-8k | selected | R3 | 1077.1201484566764 | 1029.9381138575832 | 0.9562206262998196 | locked | 1305.0 | 8 | 8 | 0 | BELOW_RATIO_FLOOR |
| W2a-32k | window | R1 | 250.9012719122049 | 250.9484945904748 | 1.0000127698282726 | locked | 1305.0 | 8 | 8 | 0 | PASS |
| W2a-32k | window | R3 | 271.49496720896707 | 250.84078441517906 | 0.9241110729866904 | locked | 1305.0 | 8 | 8 | 0 | BELOW_RATIO_FLOOR |
| W2a-8k | window | R1 | 1237.817878416046 | 1237.7967240262624 | 1.000216318681772 | locked | 1305.0 | 8 | 8 | 0 | PASS |
| W2a-8k | window | R3 | 1320.5795306647603 | 1238.0516781991323 | 0.9376019920563332 | locked | 1305.0 | 8 | 8 | 0 | BELOW_RATIO_FLOOR |
| W2b-32k | window | R1 | 677.3139978125089 | 677.3180948537063 | 0.9999835927739098 | locked | 1305.0 | 8 | 8 | 0 | PASS |
| W2b-32k | window | R3 | 806.8970218330269 | 677.3787002522852 | 0.8394590694552028 | locked | 1305.0 | 8 | 8 | 0 | BELOW_RATIO_FLOOR |
| W2b-8k | window | R1 | 2743.6836659605797 | 2749.585145549253 | 1.0004202360646985 | locked | 1305.0 | 8 | 8 | 0 | PASS |
| W2b-8k | window | R3 | 3245.927218229185 | 2746.836797081659 | 0.8461621117596307 | locked | 1305.0 | 8 | 8 | 0 | BELOW_RATIO_FLOOR |
| W3a-32k | window-softcap | R1 | 183.20758581956542 | 183.21573066327193 | 1.0001529676896226 | locked | 1305.0 | 8 | 8 | 0 | PASS |
| W3a-8k | window-softcap | R1 | 913.8986944520897 | 914.2035922916643 | 1.00055408070854 | locked | 1305.0 | 8 | 8 | 0 | PASS |
| W3b-8k | softcap | R1 | 477.2356745550146 | 477.2576077524675 | 1.0001041033954774 | locked | 1305.0 | 8 | 8 | 0 | PASS |
| W4-128k | selected | R1 | 52.28916704500909 | 52.2869113843219 | 1.0000623363569277 | locked | 1305.0 | 8 | 8 | 0 | PASS |
| W4-32k | selected | R1 | 208.6600288295162 | 208.641680887125 | 1.0000510060842998 | locked | 1305.0 | 8 | 8 | 0 | PASS |
| W5-128k | phi | R1 | 28.653397289330577 | 28.653799090044973 | 1.0000511832447927 | locked | 1305.0 | 8 | 8 | 0 | PASS |
| W5-32k | phi | R1 | 305.606457729192 | 305.61089248541424 | 1.0000177636500838 | locked | 1305.0 | 8 | 8 | 0 | PASS |
| W6-prefix-128k | msa | R1 | 48.83552884956431 | 48.846201847981455 | 1.0002185500513874 | locked | 1305.0 | 8 | 8 | 0 | PASS |
| W6-prefix-32k | msa | R1 | 197.20949665213453 | 197.29172914578058 | 0.9997836318391965 | locked | 1305.0 | 8 | 8 | 0 | PASS |
| W6-uniform-128k | msa | R1 | 17.214512136701217 | 17.214507472268433 | 0.9999109053239581 | locked | 1305.0 | 8 | 8 | 0 | PASS |
| W6-uniform-32k | msa | R1 | 71.81117202345948 | 71.80874522856118 | 1.0000232269015146 | locked | 1305.0 | 8 | 8 | 0 | PASS |
