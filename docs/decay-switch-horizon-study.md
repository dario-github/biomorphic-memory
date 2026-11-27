# How switch day affects inherited edge decay

On the frozen [ranker-cadence fixture](ranker-cadence-study.md), later switches from legacy to corrected daily maintenance preserved larger day-30 weight deficits. The day-23 switch was the first tested switch to inherit a legacy pass on the two lift edges, which became eligible on day 22. Both lift weights finished at 0.4503 versus 0.4594 in the clean corrected and one-pass references. The lift query kept the same full rank order in every arm, with the superseded order second.

| Day-30 arm | Each long-aged edge | Each lift edge | Kiln Top-1 | Beacon Top-1 |
| --- | ---: | ---: | --- | --- |
| One legacy pass on a fresh graph | 0.4939 | 0.4594 | units | units |
| Clean corrected daily | 0.4939 | 0.4594 | units | units |
| Full legacy daily | 0.0001 | 0.2236 | vendor | registry |
| Switch on day 8 | 0.2821 | 0.4594 | units | units |
| Switch on day 16 | 0.0448 | 0.4594 | units | units |
| Switch on day 23 | 0.0031 | 0.4503 | vendor | registry |

The long-aged edges are kiln registry → units and beacon field slip → units. All three switches had full-legacy Top-2 membership on both queries. The voided kiln memo was in Top-2 and the obsolete beacon field slip was out, the reverse of the clean references. Day 23 also matched full-legacy Top-1 on both queries. The day-23 lift deficit arose when the first corrected call used the inherited 0.5391 as its origin and applied age from the existing activation date. The two short-aged edges became eligible on day 26 and matched the clean weights under all switches. The undated greenhouse control stayed unchanged.

## Reproduce

From the repository root, use Python 3.10 or newer, an existing NumPy installation and a local copy of Git object `7f11e26b81177d0679d91c926945e12ac2e3f30b`.

```sh
GIT_NO_LAZY_FETCH=1 GIT_ALLOW_PROTOCOL= python3 -I -B tests/test_decay_switch_horizon.py -v
GIT_NO_LAZY_FETCH=1 GIT_ALLOW_PROTOCOL= python3 -I -B tests/test_decay_switch_horizon.py --json
```

Pass the archived [round-3 study](decay-migration-study.md) measurements JSON path after `--json` to check its four day-30 arms exactly before running the two new arms. The harness loads public-baseline maintenance from local Git, runs each arm in a temporary directory and blocks Python socket events. Output includes the frozen inputs, all day-30 ranks and components, and edge origin snapshots on the three first corrected days. The comparison covers the Python ranker after fixed anchor selection. Rebuilt live graphs, changed anchors, downstream retrieval and answer quality are outside its scope.

### Follow-up on switch days 17–22

On the same fixture, day 17 was the first measured switch to retain full-legacy Top-1 on both conflict queries at day 30. Kiln vendor and beacon registry each led their units node by 0.0001. Day 16 still led with units. Top-2 membership matched full legacy throughout days 16–23. Each long-aged edge fell from 0.0448 on day 16 to 0.0325 on day 17 and 0.0049 on day 22. The lift edges finished at the clean 0.4594 for every switch through day 22 and at 0.4503 for day 23, with identical lift Top-2 order.

Pass the archived round-4 measurements JSON path after `--json` to reproduce the six reference endpoints before running days 17–22. The export adds their pre-switch ranks and edge weights, first corrected edge states, and complete day-30 rows.

### Follow-up on switch days 1–7 and 9–15

Both conflict queries kept units first at day 30 on all 14 new switch days. With days 8 and 16 reproduced from the archive, day 17 remains the first measured switch with full-legacy Top-1 on both queries across days 1–23. The day-1 arm made zero legacy calls and matched the clean corrected endpoint exactly.

Top-2 membership changed earlier, between switch days 4 and 5. Kiln vendor displaced kiln registry, and beacon registry displaced beacon field slip. The incoming source was superseded in the kiln case and authoritative in the beacon case. Their opposing exposure changes left the aggregate superseded-conflict measure at 2/3. Each long-aged edge dropped from 0.4381 to 0.4044 at that boundary. For switches 7, 8, 9, 15 and 16, the corresponding weights were 0.3245, 0.2821, 0.2404, 0.0605 and 0.0448. The lift edges held at 0.4594 throughout days 1–16.

Pass the archived round-5 measurements JSON path after `--json` to check all 12 reference endpoints before scoring the 14 early switches. The export includes full per-case rows and first-corrected edge snapshots. All 22 positive pre-switch graphs matched the uninterrupted legacy graph byte-for-byte at the same day.

### Follow-up on beacon registry anchor similarity

With only the beacon registry query-anchor similarity lowered from 1.00 to 0.97, a day-5 switch kept beacon units and field slip in the day-30 Top-2. Field slip scored 0.5564 and registry 0.5514. Kiln still switched to units and vendor, with vendor ahead of registry by 0.0083. Clean and day-4 Top-2 membership stayed the same for both queries, and full legacy at 0.97 still ranked beacon registry first and units second. The day-4 and day-5 long-aged weights stayed at 0.4381 and 0.4044 respectively, with lift weights at 0.4594. This measures sensitivity to one fixed anchor on the frozen fixture.

Use `--anchor-json /path/to/round-6/measurements.json` with the reproduction command above. It checks five original-anchor endpoints against the archive and exports all nine endpoints plus the pre-switch and first-corrected graph snapshots for both 0.97 switches.
