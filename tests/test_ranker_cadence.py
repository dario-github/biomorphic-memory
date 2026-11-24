"""Exercise the production ranker at fixed anchors after three decay schedules."""

import io
import json
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
BASELINE = "7f11e26b81177d0679d91c926945e12ac2e3f30b"
DAY_ZERO = datetime(2030, 1, 1)
END_DAY = 30
ENDPOINT = DAY_ZERO + timedelta(days=END_DAY)
RATE = 0.02
TOP_K = 2

# In case and graph order. Each evidence tuple holds the node ID, source
# authority, document age, gold status, fixed anchor similarity, and text.
CASES = (
    (
        "older_authoritative", "What is the kiln maximum?",
        (
            ("kiln/registry", "high", 50, "current", 1.00,
             "Signed kiln register sets the maximum at 620 °C."),
            ("kiln/vendor", "low", 7, "superseded", 1.00,
             "Provisional vendor memo claims 680 °C. The standing register governs, "
             "and the provisional claim was voided at review on day 29."),
            ("kiln/units", "high", 50, "uncontested", 0.82,
             "Kiln panel displays temperatures in Celsius."),
        ),
    ),
    (
        "newer_authoritative", "What is the beacon interval?",
        (
            ("beacon/field-slip", "low", 50, "superseded", 1.00,
             "Field slip lists the beacon interval as 12 s."),
            ("beacon/registry", "high", 7, "current", 1.00,
             "Signed register replaces the old interval with 18 s."),
            ("beacon/units", "high", 50, "uncontested", 0.82,
             "Controller uses seconds for beacon intervals."),
        ),
    ),
    (
        "explicit_supersession", "Which bay is the lift pickup assigned to?",
        (
            ("lift/order-17", "high", 25, "superseded", 1.00,
             "Order 17 assigns lift pickup to bay C."),
            ("lift/order-18", "high", 12, "current", 0.93,
             "The same office replaces order 17 and assigns pickup to bay F."),
            ("lift/units", "high", 25, "uncontested", 0.84,
             "The lift board identifies pickup bays by letters."),
        ),
    ),
    (
        "uncontested_control", "When does greenhouse watering start?",
        (
            ("garden/watering", "high", 30, "uncontested", 1.00,
             "Greenhouse watering starts at 07:00."),
            ("garden/tool-rack", "high", 30, "off_topic", 0.55,
             "The greenhouse tool rack has four shelves."),
        ),
    ),
)

EDGES = (
    ("kiln/registry", "kiln/units", 0.90, 30),
    ("kiln/vendor", "kiln/units", 0.48, 5),
    ("beacon/field-slip", "beacon/units", 0.90, 30),
    ("beacon/registry", "beacon/units", 0.48, 5),
    ("lift/order-17", "lift/units", 0.55, 9),
    ("lift/order-18", "lift/units", 0.55, 9),
    ("garden/watering", "garden/tool-rack", 0.60, None),
)


def fixture():
    nodes = []
    queries = {}
    for case, query, evidence in CASES:
        gold = {status: nid for nid, _, _, status, _, _ in evidence}
        anchors = {nid: sim for nid, _, _, _, sim, _ in evidence}
        queries[case] = {"query": query, "anchors": anchors, "gold": gold}
        for nid, authority, age, status, _, description in evidence:
            node = {
                "id": nid, "path": f"memory/{nid}.md", "type": "semantic",
                "timestamp": (ENDPOINT - timedelta(days=age)).isoformat(),
                "pagerank": 0.0, "emotional_weight": 0.0,
                "source_authority": authority, "status": status, "evidence": description,
            }
            if status == "superseded":
                node["superseded_by"] = gold["current"]
            nodes.append(node)

    edges = []
    for source, target, weight, age in EDGES:
        edge = {"from": source, "to": target, "type": "association", "weight": weight}
        if age is not None:
            edge["last_activation"] = (ENDPOINT - timedelta(days=age)).isoformat()
        edges.append(edge)

    graph = {"version": "1.0", "nodes": nodes, "edges": edges}
    q_values = {node["path"]: {"q": 0.5, "n": 0} for node in nodes}
    return graph, q_values, queries


def load_baseline():
    source = subprocess.check_output(
        ["git", "show", f"{BASELINE}:src/q_value.py"], cwd=ROOT,
    )
    baseline = ModuleType("baseline_q_value")
    baseline.__file__ = str(ROOT / "src" / "q_value.py")
    exec(compile(source, f"{BASELINE}:src/q_value.py", "exec"), vars(baseline))
    return baseline


def score_query(ranker, case, query, nodes, adjacency, q_values):
    anchors = list(query["anchors"].items())
    activation = ranker.spreading_activation(anchors, nodes, adjacency)
    pagerank = ranker.personalized_pagerank(anchors, nodes, adjacency)
    ranked = ranker.hybrid_score(query["anchors"], activation, pagerank, q_values, nodes)
    for rank, row in enumerate(ranked, start=1):
        nid = row["node_id"]
        row.update(rank=rank, activation_raw=activation[nid], pagerank_raw=pagerank[nid],
                   emotional_weight=nodes[nid]["emotional_weight"])

    top_ids = [row["node_id"] for row in ranked[:TOP_K]]
    gold = query["gold"]
    return {
        "case": case,
        "top_1": top_ids[0] if top_ids else None,
        "top_k": top_ids,
        "authoritative_current_at_1": top_ids[:1] == [gold["current"]] if "current" in gold else None,
        "superseded_conflict_at_k": gold["superseded"] in top_ids if "superseded" in gold else None,
        "uncontested_recall_at_k": int(gold["uncontested"] in top_ids),
        "ranked": ranked,
    }


def run_experiment():
    from src import q_value, spreading_activation as ranker

    baseline = load_baseline()
    graph, q_values, queries = fixture()
    schedules = (
        ("baseline_once", baseline, (END_DAY,)),
        ("baseline_daily", baseline, tuple(range(1, END_DAY + 1))),
        ("corrected_daily", q_value, tuple(range(1, END_DAY + 1))),
    )
    arms = {}
    for arm, decay, days in schedules:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            graph_path, q_path = root / "graph.json", root / "q-values.json"
            graph_path.write_text(json.dumps(graph), encoding="utf-8")
            q_payload = json.dumps(q_values)
            q_path.write_text(q_payload, encoding="utf-8")
            with (
                patch.multiple(decay, GRAPH_FILE=graph_path, EDGE_DECAY_RHO=RATE),
                patch.object(decay, "datetime", wraps=datetime) as clock,
                redirect_stdout(io.StringIO()),
            ):
                for day in days:
                    clock.now.return_value = DAY_ZERO + timedelta(days=day)
                    decay.cmd_decay()

            maintained = json.loads(graph_path.read_text(encoding="utf-8"))
            with patch.multiple(ranker, GRAPH_FILE=graph_path, Q_VALUES_FILE=q_path):
                nodes, adjacency, _ = ranker.load_graph()
                loaded_q = ranker.load_q_values()
            rows = [score_query(ranker, case, query, nodes, adjacency, loaded_q)
                    for case, query in queries.items()]
            arms[arm] = {
                "run_days": list(days),
                "edges": maintained["edges"],
                "nodes_unchanged": maintained["nodes"] == graph["nodes"],
                "q_values_unchanged": loaded_q == q_values and q_path.read_text(encoding="utf-8") == q_payload,
                "cases": rows,
            }

    return {
        "baseline_commit": BASELINE,
        "day_zero": DAY_ZERO.isoformat(), "endpoint": ENDPOINT.isoformat(),
        "decay_rate": RATE, "k": TOP_K, "confidence_gate": ranker.TAU_GATE,
        "python_version": sys.version, "numpy_version": ranker.np.__version__,
        "graph": graph, "q_values": q_values, "queries": queries, "arms": arms,
    }


class RankerCadenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.measured = run_experiment()
        cls.arms = cls.measured["arms"]

    def test_fixture_reaches_ranker_with_fixed_metadata_and_q(self):
        identity_keys = ("from", "to", "type", "last_activation")
        initial = [tuple(edge.get(key) for key in identity_keys)
                   for edge in self.measured["graph"]["edges"]]
        for name, arm in self.arms.items():
            with self.subTest(arm=name):
                self.assertTrue(arm["nodes_unchanged"])
                self.assertTrue(arm["q_values_unchanged"])
                self.assertEqual([tuple(edge.get(key) for key in identity_keys)
                                  for edge in arm["edges"]], initial)
                for case in arm["cases"]:
                    ranked = case["ranked"]
                    with self.subTest(case=case["case"]):
                        self.assertCountEqual([row["node_id"] for row in ranked],
                                              self.measured["queries"][case["case"]]["anchors"])
                        self.assertGreaterEqual(ranked[0]["score"], self.measured["confidence_gate"])
                        self.assertTrue(all(row["q_value"] == 0.5 and row["q_n"] == 0 for row in ranked))

    def test_corrected_daily_matches_baseline_once(self):
        once, corrected = (self.arms[name] for name in ("baseline_once", "corrected_daily"))
        self.assertEqual([edge["weight"] for edge in once["edges"]],
                         [edge["weight"] for edge in corrected["edges"]])
        self.assertEqual(once["cases"], corrected["cases"])

    def test_legacy_cadence_exposure_and_negative_control(self):
        once, daily = (self.arms[name] for name in ("baseline_once", "baseline_daily"))
        once_weights = {edge["from"]: edge["weight"] for edge in once["edges"]}
        daily_weights = {edge["from"]: edge["weight"] for edge in daily["edges"]}
        for endpoint in ("kiln/registry", "kiln/vendor", "beacon/field-slip", "beacon/registry"):
            with self.subTest(edge=endpoint):
                self.assertLess(daily_weights[endpoint], once_weights[endpoint])
        daily_cases = {row["case"]: row for row in daily["cases"]}
        for before in once["cases"][:2]:
            with self.subTest(case=before["case"]):
                after = daily_cases[before["case"]]
                self.assertNotEqual(before["top_1"], after["top_1"])
                self.assertNotEqual(set(before["top_k"]), set(after["top_k"]))
        self.assertEqual(once["edges"][-1], self.measured["graph"]["edges"][-1])
        self.assertEqual(once["edges"][-1], daily["edges"][-1])
        self.assertEqual(once["cases"][-1], daily["cases"][-1])
        self.assertEqual(once["cases"][-1]["top_1"], "garden/watering")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
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
    else:
        suite = unittest.main(exit=False)
        print(f"Python socket audit events: {len(network_events)}", file=sys.stderr)
        sys.exit(not suite.result.wasSuccessful() or bool(network_events))
