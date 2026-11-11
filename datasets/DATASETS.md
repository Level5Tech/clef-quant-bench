# Clef quantization benchmark: datasets

All data comes from public Hugging Face datasets, loaded at pinned commit hashes (`SOURCES` in
`datasets/build.py`), so `build.py` reproduces the records byte for byte. Nothing private is used.
Everything below was checked on 2026-10-15 unless marked **(not verified)**.

Licenses are recorded as declared on the dataset card (metadata tag and card body) and, where reachable,
against the upstream repository's LICENSE file. The "derived records" column is my reading of the license
text, not legal advice: have counsel confirm before the public release. `release` is the policy encoded
in `build.py` and in `data/record_ids.jsonl`: `records` = raw derived records may be published (with the
attribution / ShareAlike terms noted), `ids_only` = publish `datasets/build.py` + `data/record_ids.jsonl`
and let users regenerate the text from the source.

## Counts

<!-- BEGIN GENERATED COUNTS -->

Seed 20261015; cap 5000 scored records per task; calibration target 768.

| task | eval split | split size | scored records | capped | calibration records | release |
|---|---|---:|---:|---|---:|---|
| banking77 | test | 3080 | 3080 | no (full split) | 64 | records |
| clinc150 | test | 5500 | 5000 | yes (stratified) | 64 | records |
| anli_r1 | test_r1 | 1000 | 1000 | no (full split) | 64 | ids_only |
| anli_r2 | test_r2 | 1000 | 1000 | no (full split) | 64 | ids_only |
| anli_r3 | test_r3 | 1200 | 1200 | no (full split) | 64 | ids_only |
| contractnli | test | 1991 | 1991 | no (full split) | 64 | ids_only |
| mmlu | test | 14042 | 5000 | yes (stratified) | 64 | records |
| arc_challenge | test | 1172 | 1172 | no (full split) | 64 | records |
| boolq | validation | 3270 | 3270 | no (full split) | 64 | records |
| ragtruth | test | 2700 | 2700 | no (full split) | 64 | ids_only |
| when2call | mcq | 3652 | 3652 | no (full split) | 64 | records |
| sst5 | test | 2210 | 2210 | no (full split) | 64 | ids_only |
| **total** | | | **31275** | | **768** | |

Skipped tasks: none.

Calibration records dropped for overlapping a scored record's text: 1.

Label distribution of the scored records (counts):

- banking77: activate_my_card: 40, age_limit: 40, apple_pay_or_google_pay: 40, atm_support: 40, automatic_top_up: 40, balance_not_updated_after_bank_transfer: 40, balance_not_updated_after_cheque_or_cash_deposit: 40, beneficiary_not_allowed: 40, ... (77 labels)
- clinc150: accept_reservations: 28, account_blocked: 28, alarm: 28, application_status: 28, apr: 28, are_you_a_bot: 28, balance: 28, bill_balance: 28, ... (151 labels)
- anli_r1: contradiction: 333, entailment: 334, neutral: 333
- anli_r2: contradiction: 333, entailment: 334, neutral: 333
- anli_r3: contradiction: 396, entailment: 402, neutral: 402
- contractnli: contradiction: 210, entailment: 878, neutral: 903
- mmlu: A: 1147, B: 1233, C: 1275, D: 1345
- arc_challenge: A: 266, B: 311, C: 310, D: 285
- boolq: false: 1237, true: 2033
- ragtruth: false: 1757, true: 943
- when2call: cannot_answer: 1295, request_for_info: 1062, tool_call: 1295
- sst5: 0: 279, 1: 633, 2: 389, 3: 510, 4: 399

<!-- END GENERATED COUNTS -->

## Sources, licenses and mapping

| task | HF id @ pinned revision | config / eval split / calibration split | license (card; upstream check) | derived records may be redistributed? | release |
|---|---|---|---|---|---|
| `banking77` | [legacy-datasets/banking77](https://huggingface.co/datasets/legacy-datasets/banking77) @ `f54121560de48f2852f90be299010d1d6dc612ec` | - / `test` (3,080) / `train` (10,003) | CC BY 4.0. Card metadata `cc-by-4.0`; card body "Creative Commons Attribution 4.0 International"; upstream [PolyAI-LDN/task-specific-datasets LICENSE](https://github.com/PolyAI-LDN/task-specific-datasets) is CC BY 4.0 (fetched) | Yes, with attribution and a note that the data was modified | records |
| `clinc150` | [clinc/clinc_oos](https://huggingface.co/datasets/clinc/clinc_oos) @ `155b9c710419136e17307b80d0a13e68cd46b4ec` | `plus` / `test` (5,500, capped to 5,000) / `train` (15,250) | CC BY 3.0. Card metadata `cc-by-3.0`; card body says "[More Information Needed]"; upstream [clinc/oos-eval LICENSE](https://github.com/clinc/oos-eval) is "Attribution 3.0 Unported" (fetched) | Yes, with attribution | records |
| `anli_r1`, `anli_r2`, `anli_r3` | [facebook/anli](https://huggingface.co/datasets/facebook/anli) @ `8e4813d81f46d313dac7892e1c28076917cfcdf9` | - / `test_r1` (1,000), `test_r2` (1,000), `test_r3` (1,200) / `train_r1`, `train_r2`, `train_r3` | CC BY-NC 4.0. Card metadata `cc-by-nc-4.0`; card body links the [facebookresearch/anli LICENSE](https://github.com/facebookresearch/anli/blob/main/LICENSE) (fetched: "Attribution-NonCommercial 4.0 International") | Only non-commercial, with attribution. Level5 is a commercial publisher, so treat as **not** publishable as text | ids_only |
| `contractnli` | [kiddothe2b/contract-nli](https://huggingface.co/datasets/kiddothe2b/contract-nli) @ `059927b91122a6827e7dbb4f296f6da8f5dcee1c` | `contract_nli.zip` / `test` subset (1,991) / `train` subset (6,819) | CC BY-NC-SA 4.0 per the card metadata (the card has no body text). **The sources disagree**: the original ContractNLI release at [stanfordnlp.github.io/contract-nli](https://stanfordnlp.github.io/contract-nli/) states "released under CC BY 4.0" (fetched). I apply the stricter HF-card terms | Under the HF card: non-commercial and ShareAlike only. Under the upstream terms: yes with attribution. Unresolved, so treat as **not** publishable as text until counsel decides | ids_only |
| `mmlu` | [cais/mmlu](https://huggingface.co/datasets/cais/mmlu) @ `c30699e8356da336a370243923dbaf21066bb9fe` | `all` / `test` (14,042, capped to 5,000) / `auxiliary_train` (99,842) | MIT. Card metadata `mit`; card body links the [hendrycks/test LICENSE](https://github.com/hendrycks/test/blob/master/LICENSE) (fetched: MIT, Copyright 2020 Dan Hendrycks). The card does not address the copyright of the third-party sources the questions were drawn from **(not verified)** | Yes under the stated MIT terms (keep the copyright notice) | records |
| `arc_challenge` | [allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc) @ `210d026faf9955653af8916fad021475a3f00453` | `ARC-Challenge` / `test` (1,172) / `train` (1,119) | CC BY-SA 4.0 per the card metadata `cc-by-sa-4.0`; the card body says "[More Information Needed]", and I could not confirm it against an upstream license file **(not verified upstream)** | Yes, with attribution; ShareAlike means the derived ARC records must also be offered under CC BY-SA 4.0 | records |
| `boolq` | [google/boolq](https://huggingface.co/datasets/google/boolq) @ `35b264d03638db9f4ce671b711558bf7ff0f80d5` | - / `validation` (3,270; the test split has no public labels) / `train` (9,427) | CC BY-SA 3.0. Card metadata `cc-by-sa-3.0`; card body: "BoolQ is released under the Creative Commons Share-Alike 3.0 license" | Yes, with attribution; ShareAlike: derived records must be released under CC BY-SA 3.0 (or a later version / compatible license as its section 4(b) permits) | records |
| `ragtruth` | [wandb/RAGTruth-processed](https://huggingface.co/datasets/wandb/RAGTruth-processed) @ `eb4f4b9d1b68eb7092d3e1a61c0cd82d9808737b` | - / `test` (2,700) / `train` (15,090) | MIT per the card body ("released under the MIT License"; no metadata tag); upstream [ParticleMedia/RAGTruth LICENSE](https://github.com/ParticleMedia/RAGTruth) is MIT (fetched). The card does not say what terms apply to the source passages inside the records (news, MS MARCO-style QA, business data) **(not verified)** | The MIT terms permit it; the unverified third-party passage terms are why I chose the conservative setting | ids_only |
| `when2call` | [nvidia/When2Call](https://huggingface.co/datasets/nvidia/When2Call) @ `0582f7749df63a96fdc3070932e83e72396ace53` | `test` split `mcq` (3,652) / `train_pref` split `train` | CC BY 4.0. Card metadata `cc-by-4.0`; card body: "licensed under a Creative Commons Attribution 4.0 International License" and "ready for commercial use" | Yes, with attribution | records |
| `sst5` (ordinal score task) | [SetFit/sst5](https://huggingface.co/datasets/SetFit/sst5) @ `e51bdcd8cd3a30da231967c1a249ba59361279a3` | - / `test` (2,210) / `train` (8,544) | **No license is declared** (no metadata tag; the word "license" does not appear in the card). The card says it is the Stanford Sentiment Treebank with 5 labels, with splits taken from AcademiaSinicaNLPLab/sentiment_dataset. The terms of the original SST data were not checked **(not verified)** | Unknown. Without a declared license there is no permission to redistribute | ids_only |

Skipped tasks: **none**. ContractNLI, RAGTruth and When2Call all have usable public HF copies (above).

Why SST-5 for the ordinal task: it is a 5-level ordered scale with a public, pinned HF copy. Yelp
`Yelp/yelp_review_full` (5 stars) was passed over because its card license is `other`;
`SetFit/amazon_reviews_multi_en` declares `apache-2.0` on its card, but I did not verify the terms of the
underlying review data, so I did not use it. If SST-5's missing license is a blocker, swapping the task
only means replacing `_sst5()` in `build.py`.

## How each source becomes a Clef question

Every record has one question with id `answer`. `request.model` is `"clef"`; the runner overrides it
for Clef-flash. `state` is a small JSON object whose keys name the fields. Instruction wording is
neutral, fixed per task, and identical across every config because the same JSONL is fed to every engine.
Option ids are the criteria keys; Cloudflare's reference encoder sorts them (`sorted(str(key))`), and
gold labels use those ids, so a runner never has to re-map anything.

| task | question type | `state` | instructions | option ids -> criteria | gold |
|---|---|---|---|---|---|
| banking77 | choice (77) | `{"message"}` | "Select the intent that best describes the message in the state." | label names, normalised (lower case, `?` removed; the source has `Refund_not_showing_up` and `reverted_card_payment?`) -> readable name (underscores to spaces) | source label |
| clinc150 | choice (151) | `{"utterance"}` | "Select the intent that best describes the utterance in the state. If none of the other listed intents applies, select out_of_scope." | 150 intent names -> readable names, plus `out_of_scope` -> "None of the other intents applies; the request is outside the supported intents." (source label `oos`) | source label |
| anli_r1..r3 | choice (3) | `{"premise", "hypothesis"}` | "Decide how the hypothesis relates to the premise in the state." | `entailment` / `neutral` / `contradiction` -> one-sentence definitions | source label |
| contractnli | choice (3) | `{"premise", "hypothesis"}` (a contract excerpt and one of 17 fixed hypotheses) | same as ANLI | same as ANLI. `neutral` is the source's neutral label; that it corresponds to the original "NotMentioned" label is my assumption **(not verified)** | source label |
| mmlu | choice (4) | `{"question", "subject"}` (subject omitted when blank, as in `auxiliary_train`) | "Select the correct answer to the question in the state." | `A`..`D` -> answer text | answer index as a letter |
| arc_challenge | choice (3-5) | `{"question"}` | same as MMLU | letters by position -> answer text (the source's numeric labels `1..4` are re-lettered) | `answerKey` position as a letter |
| boolq | noul | `{"passage", "question"}` | "The answer to the question in the state is yes, based on the passage in the state." | `true` / `false` (reference defaults) | `answer` |
| ragtruth | noul | `{"task_type", "request", "source_passages", "response"}` | "The response in the state contains at least one claim that is not supported by, or that contradicts, the source passages in the state." | `true` / `false` | `true` when `evident_conflict` or `baseless_info` is non-zero in `hallucination_labels_processed`. All quality labels are kept (no filtering) |
| when2call | choice (4) | `{"user_message", "available_tools"}` (tool definitions parsed from JSON) | "Select the best next action for the assistant." | `tool_call`, `request_for_info`, `cannot_answer`, `direct` -> one-sentence behaviour descriptions | the `correct_answer` category. `direct` is a distractor and is never gold in the test MCQ (counts below) |
| sst5 | score (5) | `{"text"}` | "Rate the sentiment of the text in the state on the ordered scale provided." | `0`..`4` -> "Very negative", "Negative", "Neutral", "Positive", "Very positive" | label `0`..`4` (mapping asserted against `label_text` at build time) |

When2Call calibration rows come from `train_pref` (single user-turn rows only), and that split has no
category column. Their gold is a heuristic on the `chosen_response`: starts with `<TOOLCALL>` gives
`tool_call`; otherwise a `?` gives `request_for_info`; otherwise `cannot_answer`. I spot-checked 14
random rows and all 14 looked right. Calibration gold is never scored.

## Selection, ordering and caps

- Eval split: the full held-out split (`test`, or `validation` where the test labels are hidden), capped at
  5,000 records per task. CLINC150 (5,500) and MMLU (14,042) exceed the cap and are stratified-sampled.
- Seed `20261015`; each task and split gets its own `random.Random("<seed>:<task>:<split>")`.
- Order: records are written in a stratified-prefix order. Labels are shuffled within each class, then
  interleaved by the largest-deficit rule, so every prefix of a file holds each label's share of the split
  to within one record. The cap and `--limit` both take prefixes, so a stratified sample is the first
  N records and the ids (`<task>-<n>`) are the same in the full and the dev build.
- `--limit 200` builds a dev subset (default output `data_dev/`, 200 per task, same deterministic order).
- Calibration: 768 records, 64 per task (12 tasks), drawn only from the train splits listed above, using the
  same ordering. Ids are `<task>-cal-<n>`; the file is shuffled (seeded) so tasks are mixed.
  A calibration record is dropped and replaced if its normalised state text equals any scored record's
  text in any task, or another calibration record's. `build.py` asserts the final data has no overlap
  (text or id); the number dropped is in the counts above. MMLU's train split is `auxiliary_train`, which the
  card describes as "auxiliary multiple-choice training questions from ARC, MC_TEST, OBQA, RACE, etc.", so the
  MMLU slice of the calibration set is not MMLU-style subject questions.
- `data/record_ids.jsonl`: one row per record (scored and calibration) with `id`, `task`, `kind`,
  `dataset`, `config`, `revision`, `source_split`, `source_index` (row index in that split as loaded at the
  pinned revision; for ContractNLI the index within the subset, with the file line number in `source_id`;
  for When2Call train, the row index in `train_pref`), `source_id` (the source's own id where it has one) and
  `release`. A task released as `ids_only` publishes this file plus `build.py`.
- `data/manifest.json`: counts, label distributions, revisions, licenses, and SHA-256 of every output file.

## Reproduce

```bash
cd clef-quant-bench
uv venv --python 3.12 && uv pip install -r requirements.txt
.venv/bin/python datasets/build.py                 # full build -> data/
.venv/bin/python datasets/build.py --limit 200     # dev subset -> data_dev/
.venv/bin/python -m pytest tests -q
```

Verified: three independent full builds (one with `PYTHONHASHSEED=123`) produced identical SHA-256 hashes
for all 14 output files. Network access to huggingface.co is required on the first run; later runs use the
local Hugging Face cache.
