# Edge decay and maintenance frequency

`src/q_value.py decay` previously multiplied the persisted edge weight by the full age factor on every call. At age 30 days, the same synthetic edge with starting weight `0.8` and decay rate `0.02` per day gave these persisted weights.

| Version | Once on day 30 | Twice on day 30 | Daily from day 1 to 30 |
| --- | ---: | ---: | ---: |
| Baseline `7f11e26` | 0.4390 | 0.2409 | 0.0001 |
| Updated command | 0.4390 | 0.4390 | 0.4390 |

## Reproduce the comparison

Run from the repository root with Python 3.10 or newer and the baseline commit available locally. The fixtures use the standard library, a fixed clock, and temporary graph JSON. Both source versions run the same tests.

```sh
study=$(mktemp -d "${TMPDIR:-$PWD}/edge-decay.XXXXXX")
mkdir -p "$study/tmp" "$study/baseline/src" "$study/baseline/tests"
GIT_NO_LAZY_FETCH=1 git show 7f11e26b81177d0679d91c926945e12ac2e3f30b:src/q_value.py \
    > "$study/baseline/src/q_value.py"
cp tests/test_edge_decay.py "$study/baseline/tests/"

env -i PATH="$PATH" TMPDIR="$study/tmp" python3 -I -S -B tests/test_edge_decay.py -v
baseline_status=0
env -i PATH="$PATH" TMPDIR="$study/tmp" python3 -I -S -B \
    "$study/baseline/tests/test_edge_decay.py" -v || baseline_status=$?
test "$baseline_status" -eq 1

env -i PATH="$PATH" TMPDIR="$study/tmp" python3 -I -S -B \
    "$study/baseline/tests/test_edge_decay.py" --json > "$study/baseline.json"
env -i PATH="$PATH" TMPDIR="$study/tmp" python3 -I -S -B \
    tests/test_edge_decay.py --json > "$study/patched.json"
printf '%s\n' "$study"
```

The baseline run is expected to report assertion failures and the updated run is expected to pass. Each JSON report contains 24 settings from ages 0, 1, 7, and 30 days, three schedules and two edge orders. The table uses the `medium` edge in the forward-order day-30 settings. The age-zero, zero-weight, future-date, missing-date and malformed-date controls keep their original weights in both versions. Both versions agree with the exponential on single-run cases.

## Weight lifecycle

On the first eligible run for an activation, the command records the persisted weight as its decay origin. Later runs under a fixed rate and a forward-moving clock compute the same four-decimal weight at equal ages. The stored magnitude also caps the result, so an earlier clock, a lower rate or a zeroed weight cannot revive weight within an activation. A changed activation timestamp takes the current stored weight as the new origin. The builder writes fresh edge objects and timestamps when either `build` or `incremental` runs, starting a new lifecycle.

For a legacy edge with no origin, the first eligible upgraded run applies the full age factor to the weight still on disk. The JSON contains no record of whether that weight has already decayed, so the command cannot reconstruct the historical origin. Rebuilding computes fresh weights from the current inputs and resets their timestamps.

The comparison measures maintenance cadence on persisted synthetic weights. Source authority, semantic correctness, retrieval and answer accuracy, and deletion are outside its scope.
