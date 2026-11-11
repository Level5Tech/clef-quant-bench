# Notes on the result tree

## `concurrent_workers_max` in the `clef-flash` `meta.json` files

Twelve of the thirteen `clef-flash/*/meta.json` files read `concurrent_workers_max: 0`, and so do the matching private
configs. That value is wrong: the real value is 1 wherever it was recorded.

**What happened.** The files' timestamps and the code point to a driver relaunch on 2026-10-18 at 07:08 (Spark clock),
after the 9B configs had finished. `sweep/sched.py` rebuilt `meta.json` for every finished config with that process's
worker count, and a process with nothing to run has a count of 0. `ref-fla`, stamped 07:21, reads 1 because it ran after.

**How the real value is known.** Each result set has a sidecar, `<set>.jsonl.meta.json`, written when the set was merged.
Its `concurrent_workers_hint` field reads 1 in all 12 sets of 11 of these configs, and in 8 of the 12 sets of `ref`.
The other 4 `ref` sets (`anli_r1`, `anli_r2`, `anli_r3`, `arc_challenge`) have no hint recorded, and
`sweep/README.md` describes `anli_r1` as written by a serial single-worker run. The same check on
the `clef` (27B) configs, whose `meta.json` files were restored, gives 1 from both the sidecars and `meta.json`.

**Impact.** No analysis reads this field and the paper doesn't cite it, so no reported number depends on it. The files
are left as they are so their hashes keep matching the reviewed package. Read the sidecars, or this note, for the value.

## The same rewrite on the `clef` (27B) configs

A relaunch on 2026-11-03 rewrote `meta.json` for 13 finished `clef` configs in the same way. They were restored from
copies taken before the relaunch. Compared with the restored files, the rewrite changed `concurrent_workers_max` in all
13 and the wording of `repeat_note` in 4, and nothing else.

## The guard

Since 2026-11-04 `sweep/sched.py` leaves an existing `meta.json` alone when a process neither ran a worker nor merged a
set for that config. See `sweep/README.md`.
