"""Measure a mid-horizon decay switch on the frozen ranker-cadence fixture."""

import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

# Support running this file directly under Python -I as well as discovery.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_ranker_cadence import (BASELINE, DAY_ZERO, END_DAY, RATE, ROOT, TOP_K,
                                fixture, load_baseline, score_query)

SWITCH_DAY = 16
BEFORE_SWITCH = SWITCH_DAY - 1


def snapshot(graph_path, q_path, graph, q_payload, ranker, queries):
    maintained = json.loads(graph_path.read_text(encoding="utf-8"))
    with patch.multiple(ranker, GRAPH_FILE=graph_path, Q_VALUES_FILE=q_path):
        nodes, adjacency, _ = ranker.load_graph()
        q_values = ranker.load_q_values()
    return {
        "edges": maintained["edges"],
        "nodes_unchanged": maintained["nodes"] == graph["nodes"],
        "q_values_unchanged": q_path.read_text(encoding="utf-8") == q_payload
        and q_values == json.loads(q_payload),
        "cases": [score_query(ranker, case, query, nodes, adjacency, q_values)
                  for case, query in queries.items()],
    }


def run_experiment():
    sys.path.insert(0, str(ROOT))
    from src import q_value, spreading_activation as ranker

    operators = {"legacy": load_baseline(), "corrected": q_value}
    graph, q_values, queries = fixture()
    daily = range(1, END_DAY + 1)
    schedules = {
        "baseline_once": {END_DAY: "legacy"},
        "corrected_daily_clean": dict.fromkeys(daily, "corrected"),
        "baseline_daily_full": dict.fromkeys(daily, "legacy"),
        "migrate_at_16": {day: "legacy" if day < SWITCH_DAY else "corrected" for day in daily},
    }
    arms = {}
    day_15_graphs = {}
    for arm, schedule in schedules.items():
        with TemporaryDirectory() as directory:
            root = Path(directory)
            graph_path, q_path = root / "graph.json", root / "q-values.json"
            graph_path.write_text(json.dumps(graph), encoding="utf-8")
            q_payload = json.dumps(q_values)
            q_path.write_text(q_payload, encoding="utf-8")
            snapshots = {}
            for day, operator in schedule.items():
                decay = operators[operator]
                with (
                    patch.multiple(decay, GRAPH_FILE=graph_path, EDGE_DECAY_RHO=RATE),
                    patch.object(decay, "datetime", wraps=datetime) as clock,
                    redirect_stdout(io.StringIO()),
                ):
                    clock.now.return_value = DAY_ZERO + timedelta(days=day)
                    decay.cmd_decay()
                if day in (BEFORE_SWITCH, END_DAY):
                    snapshots[day] = snapshot(graph_path, q_path, graph, q_payload, ranker, queries)
                if day == BEFORE_SWITCH:
                    day_15_graphs[arm] = graph_path.read_bytes()
                if arm == "migrate_at_16" and day == SWITCH_DAY:
                    snapshots[day] = {
                        "edges": json.loads(graph_path.read_text(encoding="utf-8"))["edges"],
                    }
            arms[arm] = {"schedule": schedule, "snapshots": snapshots}

    return {
        "baseline_commit": BASELINE,
        "day_zero": DAY_ZERO.isoformat(),
        "endpoint": (DAY_ZERO + timedelta(days=END_DAY)).isoformat(),
        "switch_day": SWITCH_DAY,
        "decay_rate": RATE,
        "k": TOP_K,
        "confidence_gate": ranker.TAU_GATE,
        "python_version": sys.version,
        "numpy_version": ranker.np.__version__,
        "graph": graph,
        "q_values": q_values,
        "queries": queries,
        "day_15_legacy_graphs_byte_equal":
            day_15_graphs["baseline_daily_full"] == day_15_graphs["migrate_at_16"],
        "arms": arms,
    }


def exposure(rows):
    return [
        (row["top_1"], set(row["top_k"]), row["authoritative_current_at_1"],
         row["superseded_conflict_at_k"], row["uncontested_recall_at_k"])
        for row in rows
    ]


class DecayMigrationTest(unittest.TestCase):
    def test_switch_recovers_top_1_but_retains_legacy_top_2_membership(self):
        measured = run_experiment()
        self.assertTrue(measured["day_15_legacy_graphs_byte_equal"])
        arms = measured["arms"]
        once, clean, full, migration = (
            arms[name]["snapshots"][END_DAY] for name in (
                "baseline_once", "corrected_daily_clean", "baseline_daily_full", "migrate_at_16",
            )
        )
        before = arms["migrate_at_16"]["snapshots"][BEFORE_SWITCH]
        clean_before = arms["corrected_daily_clean"]["snapshots"][BEFORE_SWITCH]
        first_corrected = arms["migrate_at_16"]["snapshots"][SWITCH_DAY]
        self.assertEqual(before, arms["baseline_daily_full"]["snapshots"][BEFORE_SWITCH])
        self.assertEqual(once["cases"], clean["cases"])

        identity_keys = ("from", "to", "type", "last_activation")
        initial = measured["graph"]["edges"]
        identities = [tuple(edge.get(key) for key in identity_keys) for edge in initial]
        for name, arm in arms.items():
            for day, state in arm["snapshots"].items():
                with self.subTest(arm=name, day=day):
                    self.assertEqual([tuple(edge.get(key) for key in identity_keys)
                                      for edge in state["edges"]], identities)
                    if day == SWITCH_DAY:
                        continue  # This extra snapshot captures edges only.
                    self.assertTrue(state["nodes_unchanged"])
                    self.assertTrue(state["q_values_unchanged"])
                    for row in state["cases"]:
                        ranked = row["ranked"]
                        self.assertCountEqual([entry["node_id"] for entry in ranked],
                                              measured["queries"][row["case"]]["anchors"])
                        self.assertGreaterEqual(ranked[0]["score"], measured["confidence_gate"])

        inherited_edges = []
        for index, edge in enumerate(before["edges"]):
            with self.subTest(edge=edge["from"]):
                clean_weight = clean["edges"][index]["weight"]
                self.assertEqual(once["edges"][index]["weight"], clean_weight)
                first_edge = first_corrected["edges"][index]
                final_edge = migration["edges"][index]
                self.assertNotIn("decay_origin", edge)
                if edge["weight"] < clean_before["edges"][index]["weight"]:
                    inherited_edges.append(edge["from"])
                    self.assertEqual(first_edge["decay_origin"], {
                        "weight": edge["weight"], "last_activation": edge["last_activation"],
                    })
                    self.assertLess(first_edge["weight"], edge["weight"])
                    self.assertEqual(final_edge["decay_origin"], first_edge["decay_origin"])
                    self.assertLess(full["edges"][index]["weight"], final_edge["weight"])
                    self.assertLess(final_edge["weight"], clean_weight)
                else:
                    self.assertEqual(first_edge, edge)
                    self.assertEqual(final_edge, clean["edges"][index])
        self.assertEqual(inherited_edges, ["kiln/registry", "beacon/field-slip"])

        for migrated, reference, legacy in zip(
            migration["cases"][:2], clean["cases"][:2], full["cases"][:2],
        ):
            with self.subTest(case=migrated["case"]):
                self.assertEqual(migrated["top_1"], reference["top_1"])
                self.assertNotEqual(migrated["top_1"], legacy["top_1"])
                self.assertEqual(set(migrated["top_k"]), set(legacy["top_k"]))
                self.assertNotEqual(set(migrated["top_k"]), set(reference["top_k"]))
        self.assertEqual(exposure(migration["cases"])[2:], exposure(clean["cases"])[2:])
        self.assertEqual(migration["cases"][-1], clean["cases"][-1])


if __name__ == "__main__":
    network_events = []

    def block_network(event, _args):
        if event.startswith("socket."):
            network_events.append(event)
            raise RuntimeError(f"Unexpected socket audit event: {event}")

    sys.addaudithook(block_network)
    if sys.argv[1:] == ["--json"]:
        result = run_experiment()
        result["python_socket_audit_events"] = network_events
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        sys.exit(bool(network_events))
    suite = unittest.main(exit=False)
    print(f"Python socket audit events: {len(network_events)}", file=sys.stderr)
    sys.exit(not suite.result.wasSuccessful() or bool(network_events))
