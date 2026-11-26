"""Compare decay switch days on the unchanged ranker-cadence fixture."""

import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_decay_migration import snapshot
from test_ranker_cadence import (BASELINE, DAY_ZERO, END_DAY, RATE, ROOT, TOP_K,
                                 fixture, load_baseline)

REFERENCE_SWITCH_DAYS = (16, 8, 23)
FOLLOWUP_SWITCH_DAYS = (17, 18, 19, 20, 21, 22)
R5_SWITCH_DAYS = (*REFERENCE_SWITCH_DAYS, *FOLLOWUP_SWITCH_DAYS)
EARLY_SWITCH_DAYS = (1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15)
R3_ARMS = ("baseline_once", "corrected_daily_clean", "baseline_daily_full", "migrate_at_16")
R4_ARMS = (*R3_ARMS, "migrate_at_8", "migrate_at_23")
R5_ARMS = (*R4_ARMS, *(f"migrate_at_{day}" for day in FOLLOWUP_SWITCH_DAYS))


def run_arm(schedule, switch_day, pre_switch_days, legacy_pre_graphs,
            operators, ranker, graph, q_values, queries):
    snapshots = {}
    pre_switch_graphs = {}
    with TemporaryDirectory() as directory:
        root = Path(directory)
        graph_path, q_path = root / "graph.json", root / "q-values.json"
        graph_path.write_text(json.dumps(graph), encoding="utf-8")
        q_payload = json.dumps(q_values)
        q_path.write_text(q_payload, encoding="utf-8")
        for day, operator in schedule.items():
            decay = operators[operator]
            with (
                patch.multiple(decay, GRAPH_FILE=graph_path, EDGE_DECAY_RHO=RATE),
                patch.object(decay, "datetime", wraps=datetime) as clock,
                redirect_stdout(io.StringIO()),
            ):
                clock.now.return_value = DAY_ZERO + timedelta(days=day)
                decay.cmd_decay()
            is_pre_switch = day in pre_switch_days and (
                switch_day is None or day == switch_day - 1
            )
            if is_pre_switch:
                graph_bytes = graph_path.read_bytes()
                if switch_day is not None and graph_bytes != legacy_pre_graphs[day]:
                    raise ValueError(f"migrate_at_{switch_day} has different pre-switch bytes "
                                     "from full legacy")
                pre_switch_graphs[day] = graph_bytes
            if day == END_DAY or is_pre_switch:
                snapshots[day] = snapshot(graph_path, q_path, graph, q_payload, ranker, queries)
            if day == switch_day:
                snapshots[day] = {"edges": json.loads(graph_path.read_bytes())["edges"]}
    arm = {"schedule": schedule, "snapshots": snapshots}
    if switch_day == 1:
        arm["pre_switch_day"] = None
    return arm, pre_switch_graphs


def run_experiment(prior=None):
    sys.path.insert(0, str(ROOT))
    from src import q_value, spreading_activation as ranker

    operators = {"legacy": load_baseline(), "corrected": q_value}
    graph, q_values, queries = fixture()
    reference_round, reference_arms = None, ()
    switch_days = (*R5_SWITCH_DAYS, *EARLY_SWITCH_DAYS)
    if prior is not None:
        if "switch_days" not in prior:
            reference_round, reference_arms = "r3", R3_ARMS
        elif set(prior["switch_days"]) == set(REFERENCE_SWITCH_DAYS):
            reference_round, reference_arms = "r4", R4_ARMS
        elif set(prior["switch_days"]) == set(R5_SWITCH_DAYS):
            reference_round, reference_arms = "r5", R5_ARMS
        else:
            raise ValueError("Unrecognized reference switch days")
        if reference_round != "r5":
            switch_days = R5_SWITCH_DAYS

    measured = {
        "baseline_commit": BASELINE,
        "day_zero": DAY_ZERO.isoformat(),
        "endpoint": (DAY_ZERO + timedelta(days=END_DAY)).isoformat(),
        "switch_days": sorted(switch_days),
        "decay_rate": RATE,
        "k": TOP_K,
        "confidence_gate": ranker.TAU_GATE,
        "graph": graph,
        "q_values": q_values,
        "queries": queries,
        "python_version": sys.version,
        "numpy_version": ranker.np.__version__,
    }
    if prior is not None:
        for key in ("baseline_commit", "day_zero", "endpoint", "decay_rate", "k",
                    "confidence_gate", "graph", "q_values", "queries"):
            if measured[key] != prior[key]:
                raise ValueError(f"{reference_round} input mismatch: {key}")
        for case, query in queries.items():
            if list(query["anchors"]) != list(prior["queries"][case]["anchors"]):
                raise ValueError(f"{reference_round} anchor order mismatch: {case}")
        if list(queries) != list(prior["queries"]):
            raise ValueError(f"{reference_round} query order mismatch")

    daily = range(1, END_DAY + 1)
    schedules = {
        "baseline_once": (None, {END_DAY: "legacy"}),
        "corrected_daily_clean": (None, dict.fromkeys(daily, "corrected")),
        "baseline_daily_full": (None, dict.fromkeys(daily, "legacy")),
        **{f"migrate_at_{switch}": (
            switch, {day: "legacy" if day < switch else "corrected" for day in daily},
        ) for switch in switch_days},
    }
    pre_switch_days = {day - 1 for day in switch_days if day > 1}
    arms = {}
    shared_history = {}
    matched = []
    legacy_pre_graphs = {}
    for name, (switch, schedule) in schedules.items():
        arm, pre_graphs = run_arm(schedule, switch, pre_switch_days, legacy_pre_graphs,
                                  operators, ranker, graph, q_values, queries)
        if switch is not None and switch > 1:
            shared_history[switch - 1] = True
        if name == "baseline_daily_full":
            legacy_pre_graphs = pre_graphs
        if name in reference_arms:
            endpoint = arm["snapshots"][END_DAY]
            expected = prior["arms"][name]["snapshots"][str(END_DAY)]
            if endpoint != expected:
                differing = [key for key in endpoint if endpoint[key] != expected.get(key)]
                raise ValueError(f"{reference_round} endpoint mismatch: {name}, "
                                 f"fields {differing}, keys {endpoint.keys() ^ expected.keys()}")
            matched.append(name)
        arms[name] = arm

    return {**measured, "shared_legacy_graph_bytes_equal": shared_history,
            "r3_matched_arms": matched if reference_round == "r3" else [],
            "r4_matched_arms": matched if reference_round == "r4" else [],
            "r5_matched_arms": matched if reference_round == "r5" else [],
            "arms": arms}


class DecaySwitchHorizonTest(unittest.TestCase):
    def test_switch_horizon_changes_inherited_weights_and_mirrored_top_1(self):
        data = run_experiment()
        self.assertEqual(data["shared_legacy_graph_bytes_equal"], dict.fromkeys(range(1, 23), True))
        arms = data["arms"]
        clean = arms["corrected_daily_clean"]["snapshots"][END_DAY]
        full = arms["baseline_daily_full"]["snapshots"][END_DAY]
        first_switch = arms["migrate_at_1"]
        self.assertIsNone(first_switch["pre_switch_day"])
        self.assertEqual(first_switch["schedule"], arms["corrected_daily_clean"]["schedule"])
        self.assertEqual(first_switch["snapshots"][END_DAY], clean)
        long_aged = {"kiln/registry", "beacon/field-slip"}
        lift = {"lift/order-17", "lift/order-18"}
        previous = clean
        for switch in data["switch_days"]:
            with self.subTest(switch=switch):
                states = arms[f"migrate_at_{switch}"]["snapshots"]
                before = data["graph"] if switch == 1 else states[switch - 1]
                first, final = states[switch], states[END_DAY]
                inherited = set() if switch == 1 else long_aged
                if switch == 23:
                    inherited = inherited | lift
                eligible = long_aged | lift if switch >= 22 else long_aged
                for old, corrected, edge, earlier, clean_edge, legacy_edge in zip(
                    before["edges"], first["edges"], final["edges"], previous["edges"],
                    clean["edges"], full["edges"],
                ):
                    with self.subTest(edge=edge["from"]):
                        if edge["from"] in eligible:
                            self.assertNotIn("decay_origin", old)
                            self.assertEqual(corrected["decay_origin"], {
                                "weight": old["weight"], "last_activation": old["last_activation"],
                            })
                            self.assertLess(corrected["weight"], old["weight"])
                            self.assertEqual(edge["decay_origin"], corrected["decay_origin"])
                        else:
                            self.assertEqual(corrected, old)
                        if edge["from"] in inherited:
                            self.assertLess(legacy_edge["weight"], edge["weight"])
                            self.assertLess(edge["weight"], earlier["weight"])
                        else:
                            self.assertEqual(edge, clean_edge)
                for row, reference, legacy in zip(final["cases"], clean["cases"], full["cases"]):
                    with self.subTest(case=row["case"]):
                        self.assertGreaterEqual(row["ranked"][0]["score"], data["confidence_gate"])
                        expected_membership = reference if switch < 5 else legacy
                        self.assertEqual(set(row["top_k"]), set(expected_membership["top_k"]))
                        expected_top_1 = legacy["top_1"] if switch >= 17 else reference["top_1"]
                        self.assertEqual(row["top_1"], expected_top_1)
                        if row["case"] in ("explicit_supersession", "uncontested_control"):
                            if row["case"] == "uncontested_control" or switch < 23:
                                self.assertEqual(row, reference)
                            else:
                                self.assertNotEqual(row["ranked"], reference["ranked"])
                previous = final


if __name__ == "__main__":
    network_events = []

    def block_network(event, _args):
        if event.startswith("socket."):
            network_events.append(event)
            raise RuntimeError(f"Unexpected socket audit event: {event}")

    sys.addaudithook(block_network)
    if sys.argv[1:2] == ["--json"]:
        prior = json.loads(Path(sys.argv[2]).read_bytes()) if len(sys.argv) == 3 else None
        data = run_experiment(prior)
        data["python_socket_audit_events"] = network_events
        print(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False))
        sys.exit(bool(network_events))
    suite = unittest.main(exit=False)
    print(f"Python socket audit events: {len(network_events)}", file=sys.stderr)
    sys.exit(not suite.result.wasSuccessful() or bool(network_events))
