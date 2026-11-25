# Switching decay operators after legacy daily maintenance

On the frozen [ranker-cadence fixture](ranker-cadence-study.md), switching from legacy to corrected daily decay on day 16 partly recovered day-30 exposure. Both mirrored conflict cases returned to the clean Top-1 node, `units`, while keeping the full-legacy Top-2 membership. The voided kiln memo stayed in Top-2 and the obsolete beacon field slip stayed out. The lift and greenhouse cases kept the same exposure in all four arms.

| Day-30 arm | Authoritative current at Top-1 | Superseded conflict in Top-2 | Uncontested recall at Top-2 |
| --- | ---: | ---: | ---: |
| One legacy pass on a fresh graph | 0/3 | 2/3 | 4/4 |
| Clean corrected daily | 0/3 | 2/3 | 4/4 |
| Full legacy daily | 1/3 | 2/3 | 4/4 |
| Switch on day 16 | 0/3 | 2/3 | 4/4 |

The identical aggregate conflict counts hide opposite changes on the two mirrored cases. The two edges already over-decayed at day 15 were at 0.0816 versus 0.6667 under clean corrected daily. The first corrected call saved the inherited 0.0816 as `decay_origin.weight` and applied 16 days of age from the existing `last_activation`, yielding 0.0593. Each edge finished at 0.0448, compared with 0.4939 in both clean references and 0.0001 under uninterrupted legacy daily. Four edges that first became eligible after the switch matched the clean weights, and the undated control edge stayed at 0.6000.

## Reproduce

From the repository root, with Python 3.10 or newer, an existing NumPy installation, and the local Git object for `7f11e26b81177d0679d91c926945e12ac2e3f30b`:

```sh
python3 -I -B tests/test_decay_migration.py -v
python3 -I -B tests/test_decay_migration.py --json
```

The commands block Python socket events and use temporary graphs. The JSON includes the frozen inputs, maintenance schedule, full ranks and components at days 15 and 30 where planned, and edge state at the first corrected migration call. The comparison covers the Python ranking path after fixed anchor selection. Rebuilt graphs, changed anchors, downstream reranking and answer accuracy are outside its scope.
