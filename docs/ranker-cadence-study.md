# Edge maintenance cadence and ranking

In a fixed-anchor synthetic study, public-baseline daily edge decay changed Top-1 and Top-2 membership in two of four queries. Corrected daily decay matched a single baseline pass at the same endpoint in all four queries, including both propagation components and the final scores. This connects the [edge-decay correction](edge-decay-study.md) to a change in the Python ranker's output.

The changes had opposite effects on source-resolved evidence. Daily baseline decay moved a voided kiln memo into Top-2 when the standing register was older. With the same graph shape and fixed similarities, it moved an obsolete field slip out of Top-2 when the valid beacon registry was newer. The same-authority lift case kept the superseded order in Top-2 under all schedules. The negative control was unchanged. Source authority, document timestamp and explicit supersession are labels on these fixtures. The Python ranker uses fixed similarities, edge-weighted spreading activation, personalized PageRank, emotional weight and path-keyed Q-values.

## Reproduce

Use a local clone containing `7f11e26b81177d0679d91c926945e12ac2e3f30b`, Python 3.10 or newer, and an existing NumPy installation for the production ranker. From the repo root, run the unittest or emit the full experiment as JSON.

```sh
python3 -I -B tests/test_ranker_cadence.py -v
python3 -I -B tests/test_ranker_cadence.py --json
```

The harness uses the standard library, blocks and records Python socket audit events when run directly, and loads the baseline source from local Git. Each arm makes its own temporary graph and Q files. It runs decay once on day 30 or daily on days 1–30 at the default rate of `0.02`, then uses the production loaders and scoring functions. The eleven original fixture nodes and all anchors are in the test. JSON includes initial metadata, final edge weights, Top-1, Top-2, gold labels, and every ranked row and component score.

The experiment covers the Python ranker after anchor selection with a frozen graph lifecycle. Embedding selection, the TypeScript hook's file-mtime rerank, file access for snippets and answer quality are outside its scope. The cases support a cadence-sensitivity finding on these inputs.
