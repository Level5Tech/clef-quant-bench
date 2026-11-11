# Restart probes (clef)

Exploratory (not pre-registered). Third vLLM instance per format, 27B, same 2,400 records as the -r2 repeats.

Unit: 2400 records, 2400 questions. Statistics are bench.decompose's (same bootstrap, same Clopper-Pearson, same bit-identical test).

## fp8

Instances present: fp8, fp8-r2, fp8-r3

### Instance pairs

| pair | records | flips | flip rate | 95% CI | exact CP | bit-identical | mean TVD [95% CI] | max abs dprob | max abs dlogit |
|---|---|---|---|---|---|---|---|---|---|
| fp8 vs fp8-r2 | 2400 | 0 | 0.00% | [0.00, 0.00] | [0.00, 0.15] | 2288 (95.3%) | 0.00055 [0.00036, 0.00078] | 0.1170 | 0.4688 |
| fp8 vs fp8-r3 | 2400 | 0 | 0.00% | [0.00, 0.00] | [0.00, 0.15] | 2291 (95.5%) | 0.00045 [0.00029, 0.00065] | 0.1170 | 0.4688 |
| fp8-r2 vs fp8-r3 | 2400 | 0 | 0.00% | [0.00, 0.00] | [0.00, 0.15] | 2317 (96.5%) | 0.00029 [0.00017, 0.00045] | 0.1110 | 0.4531 |

### Each instance against ref

| instance | flips | flip rate | 95% CI |
|---|---|---|---|
| fp8 | 33 | 1.38% | [0.92, 1.83] |
| fp8-r2 | 33 | 1.38% | [0.92, 1.83] |
| fp8-r3 | 33 | 1.38% | [0.92, 1.83] |

Flip count across instances: min 33, max 33.

### Discreteness

Context, base vs r2 (per record, all questions bit-identical):

| split | records | share |
|---|---|---|
| identical | 2288 | 95.3% |
| different | 112 | 4.7% |

r3 against base and r2, per record:

| r3 equals | records | share |
|---|---|---|
| equals base only | 33 | 1.4% |
| equals r2 only | 59 | 2.5% |
| equals both | 2258 | 94.1% |
| equals neither | 50 | 2.1% |

Records where r2 and r3 agree and base differs (59), by position inside their task: 32 in the first half, 27 in the second.

## nvfp4

Instances present: nvfp4, nvfp4-r2, nvfp4-r3

### Instance pairs

| pair | records | flips | flip rate | 95% CI | exact CP | bit-identical | mean TVD [95% CI] | max abs dprob | max abs dlogit |
|---|---|---|---|---|---|---|---|---|---|
| nvfp4 vs nvfp4-r2 | 2400 | 22 | 0.92% | [0.54, 1.33] | [0.58, 1.38] | 2087 (87.0%) | 0.00710 [0.00581, 0.00856] | 0.5891 | 1.8555 |
| nvfp4 vs nvfp4-r3 | 2400 | 22 | 0.92% | [0.58, 1.33] | [0.58, 1.38] | 1887 (78.6%) | 0.00754 [0.00627, 0.00899] | 0.5891 | 1.8555 |
| nvfp4-r2 vs nvfp4-r3 | 2400 | 4 | 0.17% | [0.04, 0.33] | [0.05, 0.43] | 2082 (86.8%) | 0.00141 [0.00097, 0.00199] | 0.4358 | 1.3984 |

### Each instance against ref

| instance | flips | flip rate | 95% CI |
|---|---|---|---|
| nvfp4 | 137 | 5.71% | [4.83, 6.62] |
| nvfp4-r2 | 139 | 5.79% | [4.92, 6.71] |
| nvfp4-r3 | 137 | 5.71% | [4.83, 6.62] |

Flip count across instances: min 137, max 139.

### Discreteness

Context, base vs r2 (per record, all questions bit-identical):

| split | records | share |
|---|---|---|
| identical | 2087 | 87.0% |
| different | 313 | 13.0% |

r3 against base and r2, per record:

| r3 equals | records | share |
|---|---|---|
| equals base only | 29 | 1.2% |
| equals r2 only | 224 | 9.3% |
| equals both | 1858 | 77.4% |
| equals neither | 289 | 12.0% |

Records where r2 and r3 agree and base differs (224), by position inside their task: 120 in the first half, 104 in the second.

