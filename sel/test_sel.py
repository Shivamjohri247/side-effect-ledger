"""Deterministic self-checks for simulator, ledger and harness (no LLM). Run: python3 test_sel.py"""
import sim
from sim import World
from agent import Ledger, run_episode
from tasks import make_episode, Subtask, Episode

sim.SLIP_P = 0.0


def test_world_and_ledger():
    w = World({"mug": "cupboard", "cloth": "sink", "apple": "table"}, dirty_surfaces=["table"])
    L = Ledger(w.loc)
    seq = [("move_to", "cupboard"), ("pick", "mug"), ("open", "cupboard"), ("pick", "mug"),
           ("move_to", "table"), ("place", None)]
    rs = []
    for a, t in seq:
        r = w.step({"action": a, "target": t}); rs.append(r["ok"]); L.apply(r["effects"], 0)
    assert rs == [True, False, True, True, True, True], rs   # pick inside closed cupboard fails
    L.protect_completed([("at", "mug", "table")], 0)
    assert L.debts() == ["close the cupboard"], L.debts()
    # wipe must fail while the table holds objects
    for a, t in [("move_to", "sink"), ("pick", "cloth"), ("move_to", "table")]:
        r = w.step({"action": a, "target": t}); L.apply(r["effects"], 1)
    r = w.step({"action": "wipe", "target": "table"})
    assert r["ok"] and w.loc["mug"] == "counter" and w.loc["apple"] == "counter"   # swept aside
    L.apply(r["effects"], 1)
    assert any("mug on the table" in d for d in L.debts())                       # clobber visible to ledger
    w.loc["mug"] = "table"; L.loc["mug"] = "table"
    assert L.threat({"action": "pick", "target": "mug"}) is not None
    # coffee: mug at running machine fills
    w2 = World({"mug": "coffee_machine"})
    w2.agent = "coffee_machine"
    r = w2.step({"action": "toggle", "target": "coffee_machine"})
    assert w2.mug_coffee and ("made", "coffee") in r["effects"]
    assert w2.invariant_violations() == ["coffee_machine left on"]


class ScriptLLM:
    """Replays a fixed action list (ignores prompts) to exercise the harness."""
    primary_model = "script"

    def __init__(self, acts):
        self.acts = list(acts)

    def chat(self, msgs, **kw):
        import json
        return json.dumps(self.acts.pop(0)) if self.acts else '{"action":"done"}'


def test_harness_clobber_and_check():
    ep = Episode(0, {"mug": "shelf", "cloth": "sink"}, ["shelf"], [],
                 [Subtask("place", "Put the mug on the table.", [("at", "mug", "table")]),
                  Subtask("wipe", "Wipe the table clean.", [("clean_surface", "table")])])
    ep.init_loc = {"mug": "shelf", "cloth": "sink"}
    ep.dirty_surfaces = ["table"]
    s1 = [{"action": "move_to", "target": "shelf"}, {"action": "pick", "target": "mug"},
          {"action": "move_to", "target": "table"}, {"action": "place"}, {"action": "done"}]
    # myopic worker 2: moves mug to counter, wipes, puts cloth down, declares done
    s2 = [{"action": "move_to", "target": "table"}, {"action": "pick", "target": "mug"},
          {"action": "move_to", "target": "counter"}, {"action": "place"},
          {"action": "move_to", "target": "sink"}, {"action": "pick", "target": "cloth"},
          {"action": "move_to", "target": "table"}, {"action": "wipe", "target": "table"},
          {"action": "move_to", "target": "sink"}, {"action": "place"}, {"action": "done"}]
    r = run_episode(ep, "isolated", ScriptLLM(s1 + s2))
    assert r["clobbers"] == 1 and r["goal_success"] == 0, r
    # with the commit check the same myopic script is rejected; extra scripted repair restores the mug
    repair = [{"action": "move_to", "target": "counter"}, {"action": "pick", "target": "mug"},
              {"action": "move_to", "target": "table"}, {"action": "place"}, {"action": "done"}]
    r = run_episode(ep, "check", ScriptLLM(s1 + s2 + repair))
    assert r["check_rejects"] == 1 and r["warnings"] == 1, r
    assert r["clobbers"] == 0 and r["goal_success"] == 1 and r["clean_success"] == 1, r


def test_generator_feasible():
    for K in (4, 6, 8):
        for s in range(50):
            e = make_episode(s, K)
            assert len(e.subtasks) == K
            w = e.world()
            for o in e.dirty_objects:
                assert not w.obj_clean[o]


if __name__ == "__main__":
    test_world_and_ledger(); test_harness_clobber_and_check(); test_generator_feasible()
    print("all tests passed")
