"""Compare three decay switch days on the unchanged ranker-cadence fixture."""

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

SWITCH_DAYS = (8, 16, 23)
PRE_SWITCH_DAYS = tuple(day - 1 for day in SWITCH_DAYS)
R3_ARMS = ("baseline_once", "corrected_daily_clean", "baseline_daily_full", "migrate_at_16")


def run_arm(schedule, switch_day, operators, ranker, graph, q_values, queries):
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
            if day == END_DAY or (day in PRE_SWITCH_DAYS and
                                  (switch_day is None or day == switch_day - 1)):
                snapshots[day] = snapshot(graph_path, q_path, graph, q_payload, ranker, queries)
                if day != END_DAY:
                    pre_switch_graphs[day] = graph_path.read_bytes()
            if day == switch_day:
                snapshots[day] = {"edges": json.loads(graph_path.read_bytes())["edges"]}
    return {"schedule": schedule, "snapshots": snapshots}, pre_switch_graphs


def run_experiment(prior=None):
    sys.path.insert(0, str(ROOT))
    from src import q_value, spreading_activation as ranker

    operators = {"legacy": load_baseline(), "corrected": q_value}
    graph, q_values, queries = fixture()
    measured = {
        "baseline_commit": BASELINE,
        "day_zero": DAY_ZERO.isoformat(),
        "endpoint": (DAY_ZERO + timedelta(days=END_DAY)).isoformat(),
        "switch_days": SWITCH_DAYS,
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
                raise ValueError(f"Round 3 input mismatch: {key}")
        for case, query in queries.items():
            if list(query["anchors"]) != list(prior["queries"][case]["anchors"]):
                raise ValueError(f"Round 3 anchor order mismatch: {case}")
        if list(queries) != list(prior["queries"]):
            raise ValueError("Round 3 query order mismatch")

    daily = range(1, END_DAY + 1)
    schedules = {
        "baseline_once": (None, {END_DAY: "legacy"}),
        "corrected_daily_clean": (None, dict.fromkeys(daily, "corrected")),
        "baseline_daily_full": (None, dict.fromkeys(daily, "legacy")),
        **{f"migrate_at_{switch}": (
            switch, {day: "legacy" if day < switch else "corrected" for day in daily},
        ) for switch in (16, 8, 23)},
    }
    arms = {}
    shared_history = {}
    matched = []
    for name, (switch, schedule) in schedules.items():
        arm, pre_graphs = run_arm(schedule, switch, operators, ranker, graph, q_values, queries)
        if switch is not None:
            day = switch - 1
            shared_history[day] = pre_graphs[day] == legacy_pre_graphs[day]
            if not shared_history[day]:
                raise ValueError(f"{name} has different pre-switch bytes from full legacy")
        if name == "baseline_daily_full":
            legacy_pre_graphs = pre_graphs
        if prior is not None and name in R3_ARMS:
            endpoint = arm["snapshots"][END_DAY]
            expected = prior["arms"][name]["snapshots"][str(END_DAY)]
            for key in endpoint:
                if endpoint[key] != expected[key]:
                    raise ValueError(f"Round 3 endpoint mismatch: {name}, {key}")
            matched.append(name)
        arms[name] = arm

    return {**measured, "shared_legacy_graph_bytes_equal": shared_history,
            "r3_matched_arms": matched, "arms": arms}


class DecaySwitchHorizonTest(unittest.TestCase):
    def test_switch_horizon_changes_inherited_weights_and_mirrored_top_1(self):
        data = run_experiment()
        self.assertEqual(data["shared_legacy_graph_bytes_equal"], dict.fromkeys(PRE_SWITCH_DAYS, True))
        arms = data["arms"]
        clean = arms["corrected_daily_clean"]["snapshots"][END_DAY]
        full = arms["baseline_daily_full"]["snapshots"][END_DAY]
        long_aged = {"kiln/registry", "beacon/field-slip"}
        lift = {"lift/order-17", "lift/order-18"}
        previous = clean
        for switch in SWITCH_DAYS:
            with self.subTest(switch=switch):
                states = arms[f"migrate_at_{switch}"]["snapshots"]
                before, first, final = (states[day] for day in (switch - 1, switch, END_DAY))
                inherited = long_aged | lift if switch == 23 else long_aged
                for old, corrected, edge, earlier, clean_edge, legacy_edge in zip(
                    before["edges"], first["edges"], final["edges"], previous["edges"],
                    clean["edges"], full["edges"],
                ):
                    with self.subTest(edge=edge["from"]):
                        if edge["from"] in inherited:
                            self.assertNotIn("decay_origin", old)
                            self.assertEqual(corrected["decay_origin"], {
                                "weight": old["weight"], "last_activation": old["last_activation"],
                            })
                            self.assertLess(corrected["weight"], old["weight"])
                            self.assertEqual(edge["decay_origin"], corrected["decay_origin"])
                            self.assertLess(legacy_edge["weight"], edge["weight"])
                            self.assertLess(edge["weight"], earlier["weight"])
                        else:
                            self.assertEqual(corrected, old)
                            self.assertEqual(edge, clean_edge)
                for row, reference, legacy in zip(final["cases"], clean["cases"], full["cases"]):
                    with self.subTest(case=row["case"]):
                        self.assertGreaterEqual(row["ranked"][0]["score"], data["confidence_gate"])
                        self.assertEqual(set(row["top_k"]), set(legacy["top_k"]))
                        expected_top_1 = legacy["top_1"] if switch == 23 else reference["top_1"]
                        self.assertEqual(row["top_1"], expected_top_1)
                        if row["case"] in ("older_authoritative", "newer_authoritative"):
                            self.assertNotEqual(set(row["top_k"]), set(reference["top_k"]))
                        else:
                            self.assertEqual(set(row["top_k"]), set(reference["top_k"]))
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
