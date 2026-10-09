# sm90-smoke

R1 = linked/control (scored, >= 0.97); R2 = control/stock and R3 = linked/stock (diagnostic).
Ratios are medians of paired iterations/s ratios; these are not absolute-placement timings.
Attempts/discards count timing pairs; pairs that violate the clock lease are discarded.

| Cell | Pipeline | Comparison | Reference iterations/s | Candidate iterations/s | Ratio | Clock mode | Observed MHz | Valid pairs | Attempts | Discards | Verdict |
|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| W1-32k | selected | R1 | 69.77478372980838 | 69.79004942856841 | 1.0002187853253568 | locked | 1305.0 | 3 | 3 | 0 | PASS |
| W1-32k | selected | R3 | 71.15766505949139 | 69.83490734749837 | 0.9808208743829945 | locked | 1305.0 | 3 | 3 | 0 | PASS |
| W1-8k | selected | R1 | 1029.0909838972036 | 1028.5234898304257 | 0.9994485482084112 | locked | 1305.0 | 3 | 3 | 0 | PASS |
| W1-8k | selected | R3 | 1076.0514907451625 | 1028.8713756841428 | 0.9561544076033502 | locked | 1305.0 | 3 | 3 | 0 | BELOW_RATIO_FLOOR |
| W2a-32k | window | R1 | 250.68083445607832 | 250.59079435151386 | 0.9997865194133794 | locked | 1305.0 | 3 | 3 | 0 | PASS |
| W2a-32k | window | R3 | 271.39783605126127 | 250.59524536394431 | 0.9233501969286602 | locked | 1305.0 | 3 | 3 | 0 | BELOW_RATIO_FLOOR |
| W2a-8k | window | R1 | 1236.4761308538548 | 1237.4751247591812 | 1.00073156531345 | locked | 1305.0 | 3 | 3 | 0 | PASS |
| W2a-8k | window | R3 | 1320.2602002066938 | 1236.8201180274812 | 0.9372636771932668 | locked | 1305.0 | 3 | 3 | 0 | BELOW_RATIO_FLOOR |
| W2b-32k | window | R1 | 677.335019083546 | 677.0347673100715 | 0.9995567160046137 | locked | 1305.0 | 3 | 3 | 0 | PASS |
| W2b-32k | window | R3 | 806.274806159696 | 676.9395511103669 | 0.839589115198383 | locked | 1305.0 | 3 | 3 | 0 | BELOW_RATIO_FLOOR |
| W2b-8k | window | R1 | 2757.08758893355 | 2758.2525231207283 | 1.0002493958056147 | locked | 1305.0 | 3 | 3 | 0 | PASS |
| W2b-8k | window | R3 | 3244.6398578723624 | 2756.466533306693 | 0.8492586603956598 | locked | 1305.0 | 3 | 3 | 0 | BELOW_RATIO_FLOOR |
| W3a-32k | window-softcap | R1 | 183.15845932002605 | 183.17013331142738 | 0.9996974158033651 | locked | 1305.0 | 3 | 3 | 0 | PASS |
| W3a-8k | window-softcap | R1 | 913.6604708361583 | 914.0725076608022 | 1.0004509736798253 | locked | 1305.0 | 3 | 3 | 0 | PASS |
| W3b-8k | softcap | R1 | 477.24855375730584 | 476.96315466137526 | 0.9994019906531226 | locked | 1305.0 | 3 | 3 | 0 | PASS |
| W4-128k | selected | R1 | 52.21847362855776 | 52.232918519717444 | 1.0005310056980548 | locked | 1305.0 | 3 | 3 | 0 | PASS |
| W4-32k | selected | R1 | 208.50012542291995 | 208.373161433693 | 0.9998984727212908 | locked | 1305.0 | 3 | 3 | 0 | PASS |
| W5-128k | phi | R1 | 28.64570249127873 | 28.64407002161874 | 1.0000836080909077 | locked | 1305.0 | 3 | 3 | 0 | PASS |
| W5-32k | phi | R1 | 305.52379620402905 | 305.55549079990584 | 1.0002309598026886 | locked | 1305.0 | 3 | 3 | 0 | PASS |
| W6-prefix-128k | msa | R1 | 48.81209547888496 | 48.82724204138811 | 1.0005584049341607 | locked | 1305.0 | 3 | 3 | 0 | PASS |
| W6-prefix-32k | msa | R1 | 197.1683040339492 | 197.13169039326903 | 0.9997702231413841 | locked | 1305.0 | 3 | 3 | 0 | PASS |
| W6-uniform-128k | msa | R1 | 17.220131593094017 | 17.215567613139257 | 0.9998465283608852 | locked | 1305.0 | 3 | 3 | 0 | PASS |
| W6-uniform-32k | msa | R1 | 71.82058442992357 | 71.82715276172263 | 1.0001601481010665 | locked | 1305.0 | 3 | 3 | 0 | PASS |
