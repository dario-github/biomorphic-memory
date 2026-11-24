"""Exercise the edge-decay command on synthetic graph JSON."""

import io
import json
import math
import sys
import unittest
from contextlib import ExitStack, redirect_stdout
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import q_value


ACTIVATION = datetime(2030, 1, 1)
RATE = 0.02


def fixtures():
    return [
        ("light", 0.15, ACTIVATION.isoformat()),
        ("medium", 0.8, ACTIVATION.isoformat()),
        ("heavy", 1.0, ACTIVATION.isoformat()),
        ("zero", 0.0, ACTIVATION.isoformat()),
        ("future", 0.8, (ACTIVATION + timedelta(days=31)).isoformat()),
        ("no_timestamp", 0.8, None),
        ("invalid_timestamp", 0.8, "invalid-date"),
    ]


def run_case(days, schedule, run_days, reverse):
    inputs = fixtures()
    node_ids = [name for name, _, _ in inputs] + ["end"]
    edges = []
    for (name, weight, timestamp), target in zip(inputs, node_ids[1:]):
        edge = {"from": name, "to": target, "type": "association", "weight": weight}
        if timestamp is not None:
            edge["last_activation"] = timestamp
        edges.append(edge)
    if reverse:
        edges.reverse()
    graph = {"version": "1.0", "nodes": [{"id": name} for name in node_ids], "edges": edges}

    with TemporaryDirectory() as directory:
        path = Path(directory) / "graph.json"
        path.write_text(json.dumps(graph), encoding="utf-8")
        with (
            patch.multiple(q_value, GRAPH_FILE=path, EDGE_DECAY_RHO=RATE),
            patch.object(q_value, "datetime", wraps=datetime) as clock,
            redirect_stdout(io.StringIO()),
        ):
            for run_day in run_days:
                clock.now.return_value = ACTIVATION + timedelta(days=run_day)
                q_value.cmd_decay()
        result = json.loads(path.read_text(encoding="utf-8"))

    after = {edge["from"]: edge for edge in result["edges"]}
    records = {}
    for name, weight, timestamp in inputs:
        edge = after[name]
        expected = weight
        if timestamp == ACTIVATION.isoformat():
            expected = round(weight * math.exp(-RATE * days), 4)
        records[name] = {
            "initial_weight": weight,
            "final_weight": edge["weight"],
            "single_application_weight": expected,
            "last_activation_before": timestamp,
            "last_activation_after": edge.get("last_activation"),
        }

    identity_fields = ("from", "to", "type", "last_activation")
    before_identities = [tuple(edge.get(key) for key in identity_fields) for edge in edges]
    after_identities = [tuple(edge.get(key) for key in identity_fields) for edge in result["edges"]]
    return {
        "days": days,
        "schedule": schedule,
        "run_days": list(run_days),
        "edge_order": "reverse" if reverse else "forward",
        "graph_structure_unchanged": (
            result["nodes"] == graph["nodes"] and after_identities == before_identities
        ),
        "edges": records,
    }


def run_sweep():
    cases = []
    for days in (0, 1, 7, 30):
        schedules = {
            "once": (days,),
            "retry": (days, days),
            "daily": range(1, days + 1) if days else (0,),
        }
        for reverse in (False, True):
            for schedule, run_days in schedules.items():
                cases.append(run_case(days, schedule, run_days, reverse))
    return cases


class EdgeDecayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = run_sweep()

    def test_one_pass_and_negative_controls(self):
        for case in self.cases:
            self.assertTrue(case["graph_structure_unchanged"])
            for name, edge in case["edges"].items():
                if case["schedule"] == "once" or case["days"] == 0 or name in (
                    "zero", "future", "no_timestamp", "invalid_timestamp"
                ):
                    with self.subTest(days=case["days"], schedule=case["schedule"], edge=name):
                        self.assertEqual(edge["final_weight"], edge["single_application_weight"])

    def test_edge_order_does_not_change_results(self):
        by_order = {"forward": {}, "reverse": {}}
        for case in self.cases:
            by_order[case["edge_order"]][case["days"], case["schedule"]] = case["edges"]
        self.assertEqual(by_order["forward"], by_order["reverse"])

    def test_equal_age_gives_equal_weights_across_schedules(self):
        one_pass = {}
        for case in self.cases:
            if case["schedule"] == "once":
                one_pass[case["days"], case["edge_order"]] = case["edges"]
        for case in self.cases:
            days, order, schedule = case["days"], case["edge_order"], case["schedule"]
            if schedule == "once":
                continue
            with self.subTest(days=days, order=order, schedule=schedule):
                weights = {name: edge["final_weight"] for name, edge in case["edges"].items()}
                expected = {name: edge["final_weight"] for name, edge in one_pass[days, order].items()}
                self.assertEqual(weights, expected)


class EdgeDecayLifecycleTest(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.path = Path(stack.enter_context(TemporaryDirectory())) / "graph.json"
        stack.enter_context(patch.multiple(q_value, GRAPH_FILE=self.path, EDGE_DECAY_RHO=RATE))
        self.clock = stack.enter_context(patch.object(q_value, "datetime", wraps=datetime))
        stack.enter_context(redirect_stdout(io.StringIO()))
        self.write_edge(0.8)

    def write_edge(self, weight, activation=ACTIVATION, **fields):
        edge = {
            "from": "a", "to": "b", "type": "association", "weight": weight,
            "last_activation": activation.isoformat(), **fields,
        }
        self.path.write_text(json.dumps({"nodes": [{"id": "a"}, {"id": "b"}], "edges": [edge]}))

    def run_decay(self, day):
        self.clock.now.return_value = ACTIVATION + timedelta(days=day)
        q_value.cmd_decay()
        return json.loads(self.path.read_text())["edges"][0]

    def test_activation_timestamp_change_uses_current_weight(self):
        edge = self.run_decay(30)
        activation = ACTIVATION + timedelta(days=30)
        origin = {"decay_origin": edge["decay_origin"]} if "decay_origin" in edge else {}
        self.write_edge(edge["weight"], activation, **origin)
        self.assertEqual(self.run_decay(30)["weight"], edge["weight"])
        expected = round(edge["weight"] * math.exp(-RATE * 7), 4)
        self.assertEqual(self.run_decay(31)["last_activation"], activation.isoformat())
        self.assertEqual(self.run_decay(37)["weight"], expected)
        self.assertEqual(self.run_decay(37)["weight"], expected)

    def test_rebuilt_edge_starts_a_new_lifecycle(self):
        self.run_decay(30)
        self.write_edge(0.9, ACTIVATION + timedelta(days=30))
        self.assertEqual(self.run_decay(30)["weight"], 0.9)
        expected = round(0.9 * math.exp(-RATE), 4)
        self.assertEqual(self.run_decay(31)["weight"], expected)
        self.assertEqual(self.run_decay(31)["weight"], expected)

    def test_clock_rollback_does_not_raise_weight(self):
        older_weight = self.run_decay(30)["weight"]
        self.assertEqual(self.run_decay(7)["weight"], older_weight)
        self.assertEqual(self.run_decay(30)["weight"], older_weight)
        self.assertEqual(self.run_decay(31)["weight"], round(0.8 * math.exp(-RATE * 31), 4))

    def test_lower_rate_does_not_raise_weight(self):
        older_weight = self.run_decay(30)["weight"]
        with patch.object(q_value, "EDGE_DECAY_RHO", 0.01):
            self.assertEqual(self.run_decay(30)["weight"], older_weight)
            self.assertEqual(self.run_decay(90)["weight"], round(0.8 * math.exp(-0.01 * 90), 4))

    def test_zeroed_edge_is_not_restored_from_origin(self):
        edge = self.run_decay(30)
        origin = {"decay_origin": edge["decay_origin"]} if "decay_origin" in edge else {}
        self.write_edge(0.0, **origin)
        self.assertEqual(self.run_decay(31)["weight"], 0.0)

    def test_signed_weight_decays_toward_zero(self):
        self.write_edge(-0.8)
        expected = round(-0.8 * math.exp(-RATE * 30), 4)
        self.assertEqual(self.run_decay(30)["weight"], expected)
        self.assertEqual(self.run_decay(30)["weight"], expected)
        at_zero = self.run_decay(500)
        self.assertEqual(at_zero["weight"], 0.0)
        self.assertEqual(self.run_decay(500), at_zero)

    def test_legacy_weight_is_used_without_recovery(self):
        self.write_edge(0.2409)
        expected = round(0.2409 * math.exp(-RATE * 30), 4)
        self.assertEqual(self.run_decay(30)["weight"], expected)
        self.assertEqual(self.run_decay(30)["weight"], expected)

    def test_invalid_origin_falls_back_to_current_weight(self):
        for origin in (
            [], {}, {"last_activation": ACTIVATION.isoformat()},
            {"last_activation": ACTIVATION.isoformat(), "weight": "invalid"},
            {"last_activation": ACTIVATION.isoformat(), "weight": -0.2},
        ):
            with self.subTest(origin=origin):
                self.write_edge(0.8, decay_origin=origin)
                expected = round(0.8 * math.exp(-RATE), 4)
                self.assertEqual(self.run_decay(1)["weight"], expected)
                self.assertEqual(self.run_decay(1)["weight"], expected)


if __name__ == "__main__":
    if sys.argv[1:] == ["--json"]:
        print(json.dumps(run_sweep(), indent=2))
    else:
        unittest.main()
