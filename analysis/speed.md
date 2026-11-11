# Quality-run speed (from the sweep log)

quality-run throughput: batched vLLM (16 in flight), bs1 native ref, bs8 ref-r2, single-sequence llama.cpp; these numbers are NOT the isolated latency benchmark, which is in the `latency` block (speed.json) and the section below (speed.md). unit_secs excludes model/server load.

Source: analysis/inputs/sweep.log: 372 set END lines used (exit=0, latest per set) of 422 END lines; non-zero exits: 0. END unit_secs vs the per-set sidecar's unit_wall_seconds_sum: 372 sets compared, max abs difference 0.05 s.

## clef-flash

| config | run regime | scope | sets | records | sum unit_secs | s/record | records/s | public: records / s per record | private: records / s per record | file GB | peak GB | server start s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ref | native transformers, batch 1, PyTorch GDN fallback | full sets | 12 | 40318 | 19040.8 | 0.4723 | 2.117 | 26903 / 0.5731 | 13415 / 0.2700 | 19.06 | 20.04 | - |
| q8_0 | llama.cpp, single sequence | full sets | 16 | 44690 | 13815.1 | 0.3091 | 3.235 | 31275 / 0.3458 | 13415 / 0.2237 | 9.66 | 12.95 | 15.0 |
| q4_k_m | llama.cpp, single sequence | full sets | 16 | 44690 | 13991.4 | 0.3131 | 3.194 | 31275 / 0.3561 | 13415 / 0.2129 | 5.72 | 9.53 | 10.0 |
| fp8 | vLLM, max-num-seqs 16, 16 in flight | full sets | 16 | 44690 | 10841.5 | 0.2426 | 4.122 | 31275 / 0.2754 | 13415 / 0.1661 | 12.15 | 13.56 | 160.0 |
| nvfp4 | vLLM, max-num-seqs 16, 16 in flight | full sets | 16 | 44690 | 5297.0 | 0.1185 | 8.437 | 31275 / 0.1378 | 13415 / 0.0736 | 9.13 | 10.50 | 160.0 |
| bf16-srv | vLLM, max-num-seqs 16, 16 in flight | full sets | 16 | 44690 | 8138.3 | 0.1821 | 5.491 | 31275 / 0.2074 | 13415 / 0.1231 | 19.06 | 20.43 | 205.1 |
| gguf-bf16 | llama.cpp, single sequence | full sets | 16 | 44690 | 12431.5 | 0.2782 | 3.595 | 31275 / 0.3179 | 13415 / 0.1857 | 18.16 | 20.55 | 25.0 |
| ref-r2 | native transformers, batch 8, PyTorch GDN fallback | full sets | 16 | 44690 | 27488.9 | 0.6151 | 1.626 | 31275 / 0.6576 | 13415 / 0.5160 | 19.06 | 27.02 | - |
| bf16-srv-r2 | vLLM, max-num-seqs 16, 16 in flight | subset (first 200 per public task) | 12 | 2400 | 373.8 | 0.1557 | 6.421 | 2400 / 0.1557 | 0 / - | 19.06 | 20.43 | 210.1 |
| fp8-r2 | vLLM, max-num-seqs 16, 16 in flight | subset (first 200 per public task) | 12 | 2400 | 470.4 | 0.1960 | 5.102 | 2400 / 0.1960 | 0 / - | 12.15 | 13.56 | 160.1 |
| nvfp4-r2 | vLLM, max-num-seqs 16, 16 in flight | subset (first 200 per public task) | 12 | 2400 | 241.5 | 0.1006 | 9.938 | 2400 / 0.1006 | 0 / - | 9.13 | 10.50 | 150.0 |
| gguf-bf16-r2 | llama.cpp, single sequence | subset (first 200 per public task) | 12 | 2400 | 593.2 | 0.2472 | 4.046 | 2400 / 0.2472 | 0 / - | 18.16 | 20.52 | 10.0 |
| ref-fla | native transformers, batch 1, flash-linear-attention GDN | subset (first 200 per public task) | 12 | 2400 | 602.3 | 0.2510 | 3.985 | 2400 / 0.2510 | 0 / - | 19.06 | 19.55 | - |

- clef-flash/ref: no END line for anli_r1, anli_r2, anli_r3, arc_challenge; those sets are excluded from the totals above.

## clef

| config | run regime | scope | sets | records | sum unit_secs | s/record | records/s | public: records / s per record | private: records / s per record | file GB | peak GB | server start s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ref | native transformers, batch 1, PyTorch GDN fallback | full sets | 16 | 44690 | 61834.9 | 1.3836 | 0.723 | 31275 / 1.5977 | 13415 / 0.8847 | 54.97 | 56.38 | - |
| q8_0 | llama.cpp, single sequence | full sets | 16 | 44690 | 43448.6 | 0.9722 | 1.029 | 31275 / 1.0994 | 13415 / 0.6757 | 28.73 | 33.16 | 40.1 |
| q4_k_m | llama.cpp, single sequence | full sets | 16 | 44690 | 45753.0 | 1.0238 | 0.977 | 31275 / 1.1699 | 13415 / 0.6832 | 16.59 | 21.66 | 20.0 |
| fp8 | vLLM, max-num-seqs 16, 16 in flight | full sets | 16 | 44690 | 66547.1 | 1.4891 | 0.672 | 31275 / 1.8087 | 13415 / 0.7440 | 30.65 | 32.96 | 340.2 |
| nvfp4 | vLLM, max-num-seqs 16, 16 in flight | full sets | 16 | 44690 | 23526.9 | 0.5264 | 1.900 | 31275 / 0.6089 | 13415 / 0.3341 | 20.00 | 22.45 | 220.1 |
| bf16-srv | vLLM, max-num-seqs 16, 16 in flight | full sets | 16 | 44690 | 28321.0 | 0.6337 | 1.578 | 31275 / 0.7129 | 13415 / 0.4492 | 54.97 | 57.17 | 455.3 |
| gguf-bf16 | llama.cpp, single sequence | full sets | 16 | 44690 | 39179.5 | 0.8767 | 1.141 | 31275 / 0.9991 | 13415 / 0.5914 | 54.06 | 57.37 | 55.0 |
| ref-r2 | native transformers, batch 8, PyTorch GDN fallback | full sets | 16 | 44690 | 84334.2 | 1.8871 | 0.530 | 31275 / 2.0129 | 13415 / 1.5939 | 54.97 | 66.41 | - |
| bf16-srv-r2 | vLLM, max-num-seqs 16, 16 in flight | subset (first 200 per public task) | 12 | 2400 | 1207.5 | 0.5031 | 1.988 | 2400 / 0.5031 | 0 / - | 54.97 | 57.17 | 455.3 |
| fp8-r2 | vLLM, max-num-seqs 16, 16 in flight | subset (first 200 per public task) | 12 | 2400 | 1668.2 | 0.6951 | 1.439 | 2400 / 0.6951 | 0 / - | 30.65 | 32.96 | 335.1 |
| nvfp4-r2 | vLLM, max-num-seqs 16, 16 in flight | subset (first 200 per public task) | 12 | 2400 | 768.8 | 0.3203 | 3.122 | 2400 / 0.3203 | 0 / - | 20.00 | 22.45 | 230.1 |
| gguf-bf16-r2 | llama.cpp, single sequence | subset (first 200 per public task) | 12 | 2400 | 1868.9 | 0.7787 | 1.284 | 2400 / 0.7787 | 0 / - | 54.06 | 57.33 | 55.1 |
| ref-fla | native transformers, batch 1, flash-linear-attention GDN | subset (first 200 per public task) | 12 | 2400 | 1903.0 | 0.7929 | 1.261 | 2400 / 0.7929 | 0 / - | 54.97 | 55.65 | - |

## Like-for-like: full-set clef-flash configs on the same sets

full-set configs only (ref, q8_0, q4_k_m, fp8, nvfp4, bf16-srv, gguf-bf16, ref-r2); restricted to the sets where every one has an END line; speed_vs_ref = ref s/record divided by the config's.

Sets used (12): banking77, boolq, clinc150, contractnli, mmlu, r1, r2, r4a, r4b, ragtruth, sst5, when2call.

| config | records | sum unit_secs | s/record | records/s | speed vs ref (x) |
|---|---|---|---|---|---|
| ref | 40318 | 19040.8 | 0.4723 | 2.117 | 1.00 |
| q8_0 | 40318 | 13333.1 | 0.3307 | 3.024 | 1.43 |
| q4_k_m | 40318 | 13564.7 | 0.3364 | 2.972 | 1.40 |
| fp8 | 40318 | 10624.9 | 0.2635 | 3.795 | 1.79 |
| nvfp4 | 40318 | 5161.7 | 0.1280 | 7.811 | 3.69 |
| bf16-srv | 40318 | 7869.9 | 0.1952 | 5.123 | 2.42 |
| gguf-bf16 | 40318 | 11937.2 | 0.2961 | 3.378 | 1.60 |
| ref-r2 | 40318 | 26764.6 | 0.6638 | 1.506 | 0.71 |

clef-flash/ref over all 16 sets with the per-set sidecar standing in for the sets without an END line: 44690 records, 19708.1 s, 0.4410 s/record.

## Seconds per set (sum of unit_secs; records in brackets)

| set | clef-flash/ref | clef-flash/q8_0 | clef-flash/q4_k_m | clef-flash/fp8 | clef-flash/nvfp4 | clef-flash/bf16-srv | clef-flash/gguf-bf16 | clef-flash/ref-r2 | clef-flash/bf16-srv-r2 | clef-flash/fp8-r2 | clef-flash/nvfp4-r2 | clef-flash/gguf-bf16-r2 | clef-flash/ref-fla | clef/ref | clef/q8_0 | clef/q4_k_m | clef/fp8 | clef/nvfp4 | clef/bf16-srv | clef/gguf-bf16 | clef/ref-r2 | clef/bf16-srv-r2 | clef/fp8-r2 | clef/nvfp4-r2 | clef/gguf-bf16-r2 | clef/ref-fla |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| anli_r1 | no END | 111.2 (1000) | 103.8 (1000) | 56.5 (1000) | 34.5 (1000) | 66.0 (1000) | 117.0 (1000) | 174.0 (1000) | 13.5 (200) | 11.7 (200) | 7.0 (200) | 23.6 (200) | 22.3 (200) | 502.4 (1000) | 415.3 (1000) | 367.3 (1000) | 286.5 (1000) | 161.1 (1000) | 206.6 (1000) | 377.3 (1000) | 542.4 (1000) | 42.0 (200) | 43.2 (200) | 21.7 (200) | 76.4 (200) | 72.6 (200) |
| anli_r2 | no END | 111.1 (1000) | 104.0 (1000) | 55.7 (1000) | 33.7 (1000) | 65.8 (1000) | 117.4 (1000) | 171.8 (1000) | 13.3 (200) | 11.3 (200) | 6.9 (200) | 23.3 (200) | 22.2 (200) | 501.9 (1000) | 415.1 (1000) | 366.3 (1000) | 286.3 (1000) | 155.3 (1000) | 207.1 (1000) | 375.8 (1000) | 535.0 (1000) | 41.5 (200) | 42.1 (200) | 21.4 (200) | 74.8 (200) | 72.2 (200) |
| anli_r3 | no END | 135.3 (1200) | 120.3 (1200) | 61.6 (1200) | 37.8 (1200) | 76.2 (1200) | 136.9 (1200) | 215.1 (1200) | 12.8 (200) | 10.6 (200) | 6.5 (200) | 22.8 (200) | 21.9 (200) | 588.5 (1200) | 478.1 (1200) | 414.8 (1200) | 234.7 (1200) | 180.1 (1200) | 248.6 (1200) | 438.3 (1200) | 671.1 (1200) | 39.8 (200) | 37.6 (200) | 20.1 (200) | 72.4 (200) | 70.3 (200) |
| arc_challenge | no END | 124.4 (1172) | 98.6 (1172) | 42.8 (1172) | 29.3 (1172) | 60.4 (1172) | 123.0 (1172) | 163.4 (1172) | 10.5 (200) | 7.3 (200) | 5.0 (200) | 21.0 (200) | 19.9 (200) | 520.0 (1172) | 394.4 (1172) | 327.8 (1172) | 144.8 (1172) | 132.4 (1172) | 195.0 (1172) | 386.7 (1172) | 512.8 (1172) | 34.1 (200) | 23.4 (200) | 15.1 (200) | 65.5 (200) | 63.2 (200) |
| banking77 | 2846.1 (3080) | 1796.6 (3080) | 1945.0 (3080) | 1658.0 (3080) | 802.4 (3080) | 1148.3 (3080) | 1692.2 (3080) | 3114.2 (3080) | 74.4 (200) | 107.0 (200) | 52.1 (200) | 112.0 (200) | 121.6 (200) | 9006.5 (3080) | 5615.6 (3080) | 6322.4 (3080) | 11286.5 (3080) | 3483.8 (3080) | 5009.3 (3080) | 5419.3 (3080) | 9552.1 (3080) | 241.9 (200) | 379.5 (200) | 170.9 (200) | 349.4 (200) | 377.5 (200) |
| boolq | 534.3 (3270) | 402.9 (3270) | 357.4 (3270) | 196.1 (3270) | 111.0 (3270) | 222.3 (3270) | 389.3 (3270) | 748.9 (3270) | 13.6 (200) | 11.9 (200) | 7.0 (200) | 23.8 (200) | 22.5 (200) | 1696.0 (3270) | 1373.2 (3270) | 1217.7 (3270) | 818.4 (3270) | 530.8 (3270) | 698.4 (3270) | 1243.1 (3270) | 2338.3 (3270) | 42.5 (200) | 44.3 (200) | 21.9 (200) | 75.9 (200) | 72.7 (200) |
| clinc150 | 8087.3 (5000) | 5153.1 (5000) | 5551.8 (5000) | 4551.7 (5000) | 2259.6 (5000) | 3155.7 (5000) | 4651.7 (5000) | 8692.4 (5000) | 125.8 (200) | 181.3 (200) | 90.7 (200) | 191.2 (200) | 200.1 (200) | 24328.4 (5000) | 16387.6 (5000) | 18322.0 (5000) | 31921.7 (5000) | 9769.5 (5000) | 10254.1 (5000) | 14370.8 (5000) | 26335.3 (5000) | 408.2 (200) | 640.1 (200) | 287.0 (200) | 568.8 (200) | 614.8 (200) |
| contractnli | 345.0 (1991) | 265.7 (1991) | 240.5 (1991) | 142.3 (1991) | 77.8 (1991) | 148.3 (1991) | 249.7 (1991) | 549.7 (1991) | 15.1 (200) | 14.6 (200) | 7.6 (200) | 25.1 (200) | 24.0 (200) | 1096.2 (1991) | 900.1 (1991) | 814.2 (1991) | 537.7 (1991) | 357.9 (1991) | 465.5 (1991) | 796.5 (1991) | 1701.8 (1991) | 47.6 (200) | 56.3 (200) | 25.4 (200) | 79.8 (200) | 77.7 (200) |
| mmlu | 768.4 (5000) | 597.3 (5000) | 500.2 (5000) | 238.7 (5000) | 149.0 (5000) | 306.0 (5000) | 571.8 (5000) | 1188.8 (5000) | 12.5 (200) | 9.9 (200) | 6.0 (200) | 22.8 (200) | 21.7 (200) | 2428.0 (5000) | 1917.4 (5000) | 1656.1 (5000) | 1199.8 (5000) | 723.1 (5000) | 962.0 (5000) | 1800.2 (5000) | 3694.2 (5000) | 38.5 (200) | 36.0 (200) | 20.9 (200) | 91.4 (200) | 69.4 (200) |
| r1 | 467.3 (1477) | 340.6 (1477) | 364.9 (1477) | 301.9 (1477) | 130.8 (1477) | 218.2 (1477) | 308.1 (1477) | 718.6 (1477) | - | - | - | - | - | 1506.9 (1477) | 1064.4 (1477) | 1169.1 (1477) | 2094.9 (1477) | 610.3 (1477) | 710.3 (1477) | 994.7 (1477) | 2195.3 (1477) | - | - | - | - | - |
| r2 | 1057.5 (3908) | 847.9 (3908) | 828.2 (3908) | 637.9 (3908) | 292.0 (3908) | 486.9 (3908) | 723.4 (3908) | 1822.1 (3908) | - | - | - | - | - | 3513.9 (3908) | 2630.3 (3908) | 2671.2 (3908) | 2879.8 (3908) | 1339.7 (3908) | 2248.3 (3908) | 2319.7 (3908) | 5616.5 (3908) | - | - | - | - | - |
| r4a | 1958.6 (7461) | 1707.4 (7461) | 1563.6 (7461) | 1216.8 (7461) | 532.1 (7461) | 888.4 (7461) | 1362.1 (7461) | 4033.0 (7461) | - | - | - | - | - | 6396.0 (7461) | 5048.7 (7461) | 5007.3 (7461) | 4617.9 (7461) | 2381.6 (7461) | 2881.6 (7461) | 4314.9 (7461) | 12496.8 (7461) | - | - | - | - | - |
| r4b | 138.5 (569) | 104.7 (569) | 99.1 (569) | 72.2 (569) | 31.9 (569) | 57.3 (569) | 96.9 (569) | 349.0 (569) | - | - | - | - | - | 451.5 (569) | 321.4 (569) | 317.5 (569) | 388.6 (569) | 151.0 (569) | 186.2 (569) | 304.2 (569) | 1073.4 (569) | - | - | - | - | - |
| ragtruth | 1186.4 (2700) | 871.6 (2700) | 905.2 (2700) | 710.5 (2700) | 353.9 (2700) | 529.8 (2700) | 772.2 (2700) | 2295.9 (2700) | 39.4 (200) | 53.6 (200) | 26.2 (200) | 58.1 (200) | 58.5 (200) | 3908.8 (2700) | 2649.7 (2700) | 2882.9 (2700) | 4870.5 (2700) | 1567.6 (2700) | 1736.5 (2700) | 2473.1 (2700) | 7033.0 (2700) | 131.0 (200) | 187.3 (200) | 82.0 (200) | 182.4 (200) | 190.1 (200) |
| sst5 | 308.1 (2210) | 235.0 (2210) | 186.9 (2210) | 81.1 (2210) | 52.6 (2210) | 114.2 (2210) | 229.7 (2210) | 284.3 (2210) | 10.6 (200) | 7.3 (200) | 5.1 (200) | 20.8 (200) | 19.8 (200) | 982.2 (2210) | 737.6 (2210) | 606.8 (2210) | 270.0 (2210) | 262.4 (2210) | 368.8 (2210) | 735.1 (2210) | 889.0 (2210) | 34.8 (200) | 23.6 (200) | 15.2 (200) | 79.4 (200) | 63.3 (200) |
| when2call | 1343.3 (3652) | 1010.3 (3652) | 1021.9 (3652) | 817.7 (3652) | 368.6 (3652) | 594.5 (3652) | 890.1 (3652) | 2967.7 (3652) | 32.3 (200) | 43.9 (200) | 21.4 (200) | 48.7 (200) | 47.8 (200) | 4407.7 (3652) | 3099.7 (3652) | 3289.6 (3652) | 4709.0 (3652) | 1720.3 (3652) | 1942.7 (3652) | 2829.8 (3652) | 9147.2 (3652) | 105.6 (200) | 154.8 (200) | 67.2 (200) | 152.7 (200) | 159.2 (200) |

## Seconds per record by set

| set | clef-flash/ref | clef-flash/q8_0 | clef-flash/q4_k_m | clef-flash/fp8 | clef-flash/nvfp4 | clef-flash/bf16-srv | clef-flash/gguf-bf16 | clef-flash/ref-r2 | clef-flash/bf16-srv-r2 | clef-flash/fp8-r2 | clef-flash/nvfp4-r2 | clef-flash/gguf-bf16-r2 | clef-flash/ref-fla | clef/ref | clef/q8_0 | clef/q4_k_m | clef/fp8 | clef/nvfp4 | clef/bf16-srv | clef/gguf-bf16 | clef/ref-r2 | clef/bf16-srv-r2 | clef/fp8-r2 | clef/nvfp4-r2 | clef/gguf-bf16-r2 | clef/ref-fla |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| anli_r1 | no END | 0.1112 | 0.1038 | 0.0565 | 0.0345 | 0.0660 | 0.1170 | 0.1740 | 0.0675 | 0.0585 | 0.0350 | 0.1180 | 0.1115 | 0.5024 | 0.4153 | 0.3673 | 0.2865 | 0.1611 | 0.2066 | 0.3773 | 0.5424 | 0.2100 | 0.2160 | 0.1085 | 0.3820 | 0.3630 |
| anli_r2 | no END | 0.1111 | 0.1040 | 0.0557 | 0.0337 | 0.0658 | 0.1174 | 0.1718 | 0.0665 | 0.0565 | 0.0345 | 0.1165 | 0.1110 | 0.5019 | 0.4151 | 0.3663 | 0.2863 | 0.1553 | 0.2071 | 0.3758 | 0.5350 | 0.2075 | 0.2105 | 0.1070 | 0.3740 | 0.3610 |
| anli_r3 | no END | 0.1128 | 0.1002 | 0.0513 | 0.0315 | 0.0635 | 0.1141 | 0.1792 | 0.0640 | 0.0530 | 0.0325 | 0.1140 | 0.1095 | 0.4904 | 0.3984 | 0.3457 | 0.1956 | 0.1501 | 0.2072 | 0.3653 | 0.5593 | 0.1990 | 0.1880 | 0.1005 | 0.3620 | 0.3515 |
| arc_challenge | no END | 0.1061 | 0.0841 | 0.0365 | 0.0250 | 0.0515 | 0.1049 | 0.1394 | 0.0525 | 0.0365 | 0.0250 | 0.1050 | 0.0995 | 0.4437 | 0.3365 | 0.2797 | 0.1235 | 0.1130 | 0.1664 | 0.3299 | 0.4375 | 0.1705 | 0.1170 | 0.0755 | 0.3275 | 0.3160 |
| banking77 | 0.9241 | 0.5833 | 0.6315 | 0.5383 | 0.2605 | 0.3728 | 0.5494 | 1.0111 | 0.3720 | 0.5350 | 0.2605 | 0.5600 | 0.6080 | 2.9242 | 1.8232 | 2.0527 | 3.6644 | 1.1311 | 1.6264 | 1.7595 | 3.1013 | 1.2095 | 1.8975 | 0.8545 | 1.7470 | 1.8875 |
| boolq | 0.1634 | 0.1232 | 0.1093 | 0.0600 | 0.0339 | 0.0680 | 0.1191 | 0.2290 | 0.0680 | 0.0595 | 0.0350 | 0.1190 | 0.1125 | 0.5187 | 0.4199 | 0.3724 | 0.2503 | 0.1623 | 0.2136 | 0.3802 | 0.7151 | 0.2125 | 0.2215 | 0.1095 | 0.3795 | 0.3635 |
| clinc150 | 1.6175 | 1.0306 | 1.1104 | 0.9103 | 0.4519 | 0.6311 | 0.9303 | 1.7385 | 0.6290 | 0.9065 | 0.4535 | 0.9560 | 1.0005 | 4.8657 | 3.2775 | 3.6644 | 6.3843 | 1.9539 | 2.0508 | 2.8742 | 5.2671 | 2.0410 | 3.2005 | 1.4350 | 2.8440 | 3.0740 |
| contractnli | 0.1733 | 0.1335 | 0.1208 | 0.0715 | 0.0391 | 0.0745 | 0.1254 | 0.2761 | 0.0755 | 0.0730 | 0.0380 | 0.1255 | 0.1200 | 0.5506 | 0.4521 | 0.4089 | 0.2701 | 0.1798 | 0.2338 | 0.4001 | 0.8547 | 0.2380 | 0.2815 | 0.1270 | 0.3990 | 0.3885 |
| mmlu | 0.1537 | 0.1195 | 0.1000 | 0.0477 | 0.0298 | 0.0612 | 0.1144 | 0.2378 | 0.0625 | 0.0495 | 0.0300 | 0.1140 | 0.1085 | 0.4856 | 0.3835 | 0.3312 | 0.2400 | 0.1446 | 0.1924 | 0.3600 | 0.7388 | 0.1925 | 0.1800 | 0.1045 | 0.4570 | 0.3470 |
| r1 | 0.3164 | 0.2306 | 0.2471 | 0.2044 | 0.0886 | 0.1477 | 0.2086 | 0.4865 | - | - | - | - | - | 1.0202 | 0.7206 | 0.7915 | 1.4183 | 0.4132 | 0.4809 | 0.6735 | 1.4863 | - | - | - | - | - |
| r2 | 0.2706 | 0.2170 | 0.2119 | 0.1632 | 0.0747 | 0.1246 | 0.1851 | 0.4662 | - | - | - | - | - | 0.8992 | 0.6731 | 0.6835 | 0.7369 | 0.3428 | 0.5753 | 0.5936 | 1.4372 | - | - | - | - | - |
| r4a | 0.2625 | 0.2288 | 0.2096 | 0.1631 | 0.0713 | 0.1191 | 0.1826 | 0.5405 | - | - | - | - | - | 0.8573 | 0.6767 | 0.6711 | 0.6189 | 0.3192 | 0.3862 | 0.5783 | 1.6749 | - | - | - | - | - |
| r4b | 0.2434 | 0.1840 | 0.1742 | 0.1269 | 0.0561 | 0.1007 | 0.1703 | 0.6134 | - | - | - | - | - | 0.7935 | 0.5649 | 0.5580 | 0.6830 | 0.2654 | 0.3272 | 0.5346 | 1.8865 | - | - | - | - | - |
| ragtruth | 0.4394 | 0.3228 | 0.3353 | 0.2631 | 0.1311 | 0.1962 | 0.2860 | 0.8503 | 0.1970 | 0.2680 | 0.1310 | 0.2905 | 0.2925 | 1.4477 | 0.9814 | 1.0677 | 1.8039 | 0.5806 | 0.6431 | 0.9160 | 2.6048 | 0.6550 | 0.9365 | 0.4100 | 0.9120 | 0.9505 |
| sst5 | 0.1394 | 0.1063 | 0.0846 | 0.0367 | 0.0238 | 0.0517 | 0.1039 | 0.1286 | 0.0530 | 0.0365 | 0.0255 | 0.1040 | 0.0990 | 0.4444 | 0.3338 | 0.2746 | 0.1222 | 0.1187 | 0.1669 | 0.3326 | 0.4023 | 0.1740 | 0.1180 | 0.0760 | 0.3970 | 0.3165 |
| when2call | 0.3678 | 0.2766 | 0.2798 | 0.2239 | 0.1009 | 0.1628 | 0.2437 | 0.8126 | 0.1615 | 0.2195 | 0.1070 | 0.2435 | 0.2390 | 1.2069 | 0.8488 | 0.9008 | 1.2894 | 0.4711 | 0.5320 | 0.7749 | 2.5047 | 0.5280 | 0.7740 | 0.3360 | 0.7635 | 0.7960 |

## 27B / 9B per-record time (reference configs, native batch 1)

ratio = 27B seconds per record / 9B seconds per record, both reference (`ref`, native, batch 1), same records; 'pooled' weights by records. The 27B has no clinc150 yet, so the pooled ratio excludes it.

| set | kind | records | 27B s/record | 9B s/record | ratio | 9B timing source |
|---|---|---|---|---|---|---|
| anli_r1 | public | 1000 | 0.5024 | 0.1585 | 3.17 | per-set sidecar wall_seconds (no END line in the log) |
| anli_r2 | public | 1000 | 0.5019 | 0.1596 | 3.14 | per-set sidecar wall_seconds (no END line in the log) |
| anli_r3 | public | 1200 | 0.4904 | 0.1556 | 3.15 | per-set sidecar wall_seconds (no END line in the log) |
| arc_challenge | public | 1172 | 0.4437 | 0.1386 | 3.20 | per-set sidecar wall_seconds (no END line in the log) |
| banking77 | public | 3080 | 2.9242 | 0.9241 | 3.16 | END line |
| boolq | public | 3270 | 0.5187 | 0.1634 | 3.17 | END line |
| clinc150 | public | 5000 | 4.8657 | 1.6175 | 3.01 | END line |
| contractnli | public | 1991 | 0.5506 | 0.1733 | 3.18 | END line |
| mmlu | public | 5000 | 0.4856 | 0.1537 | 3.16 | END line |
| r1 | private | 1477 | 1.0202 | 0.3164 | 3.22 | END line |
| r2 | private | 3908 | 0.8992 | 0.2706 | 3.32 | END line |
| r4a | private | 7461 | 0.8573 | 0.2625 | 3.27 | END line |
| r4b | private | 569 | 0.7935 | 0.2434 | 3.26 | END line |
| ragtruth | public | 2700 | 1.4477 | 0.4394 | 3.29 | END line |
| sst5 | public | 2210 | 0.4444 | 0.1394 | 3.19 | END line |
| when2call | public | 3652 | 1.2069 | 0.3678 | 3.28 | END line |

9B timing for anli_r1, anli_r2, anli_r3, arc_challenge comes from the per-set sidecar (no END line in the log).

- Pooled (END lines only): 12 sets, 40318 records, ratio 3.14 (per-set range 3.01 to 3.32).
- Pooled (with sidecar fallback for sets lacking a 9B END line): 16 sets, 44690 records, ratio 3.14 (per-set range 3.01 to 3.32).
- Pooled (public sets only, with sidecar fallback): 12 sets, 31275 records, ratio 3.11 (per-set range 3.01 to 3.29).

Build lines in the sweep log (quantized 27B builds, not in any results meta.json): build/clef/fp8 size_gb=30.39 secs=508.8; build/clef/nvfp4 size_gb=19.74 secs=2288.1.

## Isolated latency benchmark (results-latency)

isolated latency: batch size 1, one request in flight, 300 sampled records per set, per-request milliseconds. Timing differs by engine: vLLM and llama.cpp configs are an HTTP round trip to localhost (server-side tokenisation included); ref and ref-r2 are in-process (encode, collate, forward, cuda sync, no HTTP). ref-r2 is an alias of ref, not a second measurement. Peak memory is measured differently per engine (see peak_memory_source).

Public pooled mean = the n-weighted mean of the per-set mean_ms over the public sets. A pooled p50 cannot be derived from per-set p50s, so `median_of_set_p50_ms` is the median of the per-set p50s (an unweighted median over sets, not the p50 of the pooled requests). Private sets are pooled the same way and appear only as that aggregate.

`ratio_27b_over_9b[config]` = clef mean_ms / clef-flash mean_ms of the same config (overall mean_ms, all 16 sets). `mean_ms_over_ref` = the config's overall mean_ms / the same model's `ref` overall mean_ms (above 1 is slower than ref).

### clef-flash

| config | engine | n | errors | mean ms | p50 ms | p95 ms | p99 ms | public pooled mean ms | public median of set p50s ms | private pooled mean ms | mean vs ref (x) | peak GB | file GB | other GPU procs at start (count / MiB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ref | transformers-native | 4800 | 0 | 354.0 | 172.4 | 1577.1 | 1582.1 | 380.7 | 158.6 | 273.7 | 1.00 | 19.97 | 19.06 | not recorded |
| q8_0 | llama.cpp | 4800 | 0 | 254.9 | 127.3 | 1052.8 | 1067.9 | 267.1 | 115.7 | 218.6 | 0.72 | 12.95 | 9.66 | 1 / 4742 MiB |
| q4_k_m | llama.cpp | 4800 | 0 | 253.6 | 121.0 | 1120.5 | 1132.4 | 267.4 | 103.0 | 212.1 | 0.72 | 9.52 | 5.72 | 2 / 10448 MiB |
| fp8 | vllm+clef_bench_vllm | 4800 | 0 | 122.5 | 68.1 | 507.3 | 511.8 | 130.9 | 64.9 | 97.4 | 0.35 | 13.06 | 12.15 | 2 / 10446 MiB |
| nvfp4 | vllm+clef_bench_vllm | 4800 | 0 | 87.6 | 49.5 | 351.5 | 355.1 | 92.7 | 45.6 | 72.5 | 0.25 | 10.00 | 9.13 | 1 / 4742 MiB |
| bf16-srv | vllm+clef_bench_vllm | 4800 | 0 | 183.8 | 104.8 | 704.4 | 710.6 | 195.7 | 98.9 | 148.0 | 0.52 | 19.93 | 19.06 | 1 / 4742 MiB |
| gguf-bf16 | llama.cpp | 4800 | 0 | 235.6 | 124.7 | 975.4 | 983.0 | 251.0 | 117.2 | 189.5 | 0.67 | 20.54 | 18.16 | 2 / 10448 MiB |
| ref-r2 (alias of ref) | transformers-native | 4800 | 0 | 354.0 | 172.4 | 1577.1 | 1582.1 | 380.7 | 158.6 | 273.7 | 1.00 | 19.97 | 19.06 | not recorded |

Peak memory source: ref, ref-r2: torch.cuda.max_memory_allocated (weights + activations, no cache pool); q8_0, q4_k_m, gguf-bf16: nvidia-smi compute-apps of the server process; fp8, nvfp4, bf16-srv: vllm plugin (weights + peak activation).

### clef

| config | engine | n | errors | mean ms | p50 ms | p95 ms | p99 ms | public pooled mean ms | public median of set p50s ms | private pooled mean ms | mean vs ref (x) | peak GB | file GB | other GPU procs at start (count / MiB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ref | transformers-native | 4800 | 0 | 1161.7 | 593.5 | 5004.7 | 5017.0 | 1242.8 | 535.6 | 918.6 | 1.00 | 56.27 | 54.97 | not recorded |
| q8_0 | llama.cpp | 4800 | 0 | 799.1 | 457.1 | 3259.3 | 3309.1 | 845.6 | 413.8 | 659.6 | 0.69 | 33.16 | 28.73 | 1 / 4742 MiB |
| q4_k_m | llama.cpp | 4800 | 0 | 802.1 | 413.7 | 3512.1 | 3557.5 | 848.3 | 353.8 | 663.5 | 0.69 | 21.65 | 16.59 | 1 / 4742 MiB |
| fp8 | vllm+clef_bench_vllm | 4800 | 0 | 381.3 | 198.1 | 1775.6 | 1788.4 | 410.5 | 187.4 | 293.5 | 0.33 | 32.26 | 30.65 | 1 / 4742 MiB |
| nvfp4 | vllm+clef_bench_vllm | 4800 | 0 | 270.0 | 147.8 | 1047.6 | 1055.4 | 283.5 | 136.8 | 229.5 | 0.23 | 21.75 | 20.00 | 1 / 4742 MiB |
| bf16-srv | vllm+clef_bench_vllm | 4800 | 0 | 596.3 | 347.8 | 2169.8 | 2186.3 | 634.6 | 333.2 | 481.3 | 0.51 | 55.90 | 54.97 | 1 / 4742 MiB |
| gguf-bf16 | llama.cpp | 4800 | 0 | 727.0 | 405.8 | 2862.9 | 2872.6 | 772.0 | 376.0 | 592.2 | 0.63 | 57.35 | 54.06 | 1 / 4742 MiB |
| ref-r2 (alias of ref) | transformers-native | 4800 | 0 | 1161.7 | 593.5 | 5004.7 | 5017.0 | 1242.8 | 535.6 | 918.6 | 1.00 | 56.27 | 54.97 | not recorded |

Peak memory source: ref, ref-r2: torch.cuda.max_memory_allocated (weights + activations, no cache pool); q8_0, q4_k_m, gguf-bf16: nvidia-smi compute-apps of the server process; fp8, nvfp4, bf16-srv: vllm plugin (weights + peak activation).

### 27B / 9B overall mean_ms by config

| config | 27B mean ms | 9B mean ms | ratio |
|---|---|---|---|
| ref | 1161.7 | 354.0 | 3.28 |
| q8_0 | 799.1 | 254.9 | 3.13 |
| q4_k_m | 802.1 | 253.6 | 3.16 |
| fp8 | 381.3 | 122.5 | 3.11 |
| nvfp4 | 270.0 | 87.6 | 3.08 |
| bf16-srv | 596.3 | 183.8 | 3.24 |
| gguf-bf16 | 727.0 | 235.6 | 3.09 |
| ref-r2 | 1161.7 | 354.0 | 3.28 |

### clef-flash: mean ms by public set

| set | ref | q8_0 | q4_k_m | fp8 | nvfp4 | bf16-srv | gguf-bf16 | ref-r2 |
|---|---|---|---|---|---|---|---|---|
| anli_r1 | 158.4 | 112.5 | 105.9 | 65.0 | 46.4 | 98.4 | 116.8 | 158.4 |
| anli_r2 | 158.4 | 111.6 | 104.1 | 64.9 | 44.7 | 98.5 | 116.3 | 158.4 |
| anli_r3 | 155.2 | 114.3 | 101.6 | 64.0 | 43.6 | 98.6 | 114.3 | 155.2 |
| arc_challenge | 138.1 | 107.1 | 85.6 | 59.2 | 39.2 | 93.0 | 104.5 | 138.1 |
| banking77 | 932.0 | 600.0 | 637.7 | 278.6 | 198.1 | 436.3 | 571.0 | 932.0 |
| boolq | 163.6 | 124.4 | 111.1 | 65.7 | 46.0 | 101.5 | 120.3 | 163.6 |
| clinc150 | 1582.2 | 1060.3 | 1126.5 | 509.8 | 353.0 | 707.2 | 978.9 | 1582.2 |
| contractnli | 175.8 | 139.9 | 125.1 | 69.9 | 50.8 | 106.0 | 128.0 | 175.8 |
| mmlu | 153.7 | 120.0 | 101.3 | 63.5 | 45.9 | 98.7 | 114.2 | 153.7 |
| ragtruth | 438.4 | 325.7 | 335.7 | 145.0 | 108.0 | 225.1 | 291.4 | 438.4 |
| sst5 | 138.6 | 107.2 | 85.4 | 59.1 | 41.5 | 92.8 | 103.4 | 138.6 |
| when2call | 374.4 | 281.7 | 288.4 | 126.1 | 94.6 | 193.0 | 252.6 | 374.4 |

### clef: mean ms by public set

| set | ref | q8_0 | q4_k_m | fp8 | nvfp4 | bf16-srv | gguf-bf16 | ref-r2 |
|---|---|---|---|---|---|---|---|---|
| anli_r1 | 535.3 | 416.9 | 362.8 | 188.0 | 138.6 | 345.5 | 376.4 | 535.3 |
| anli_r2 | 542.0 | 415.2 | 360.1 | 186.8 | 136.7 | 340.7 | 375.3 | 542.0 |
| anli_r3 | 520.3 | 399.5 | 341.3 | 181.0 | 133.3 | 321.2 | 364.3 | 520.3 |
| arc_challenge | 460.2 | 337.5 | 270.9 | 160.1 | 121.3 | 293.5 | 330.3 | 460.2 |
| banking77 | 2993.5 | 1834.3 | 1985.2 | 877.7 | 633.4 | 1485.7 | 1773.5 | 2993.5 |
| boolq | 549.1 | 423.5 | 368.5 | 189.1 | 139.7 | 328.5 | 385.2 | 549.1 |
| clinc150 | 5010.9 | 3293.0 | 3540.2 | 1781.5 | 1050.8 | 2181.0 | 2867.7 | 5010.9 |
| contractnli | 592.3 | 462.4 | 415.5 | 200.8 | 148.9 | 344.2 | 409.2 | 592.3 |
| mmlu | 513.2 | 386.3 | 326.0 | 177.1 | 131.6 | 312.9 | 359.5 | 513.2 |
| ragtruth | 1470.2 | 980.5 | 1041.4 | 438.9 | 346.4 | 731.3 | 906.0 | 1470.2 |
| sst5 | 462.2 | 335.6 | 269.0 | 160.0 | 119.3 | 292.3 | 330.5 | 462.2 |
| when2call | 1264.5 | 862.6 | 898.4 | 385.0 | 302.3 | 638.5 | 785.9 | 1264.5 |

