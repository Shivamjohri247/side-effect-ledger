"""
Procedural generator of multi-subtask household episodes with seeded
cross-subtask interference.

Interference kinds (later subtask threatens an earlier one):
  clobber   : wipe(S) after an object was placed on S; wash(o)/coffee/toast after
              o (or mug/bread) was placed somewhere -> the later subtask must move
              it, and the earlier goal must be restored.
  relocate  : place(o, L) where o initially sat on a surface wiped earlier -> o is
              no longer where the shared initial layout says it is.
  lingering : coffee/toast switch an appliance on; containers get opened.
"""
from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import List

from sim import World, SURFACES

PLACE_DEST = ["table", "counter", "shelf", "cupboard", "fridge", "drawer"]
# Initial surface clutter kept small (<=~2 items per surface) so every subtask is
# solvable well within MAX_STEPS by a scripted oracle (see oracle.py).
INIT_CHOICES = {
    "mug": ["cupboard", "shelf"], "plate": ["cupboard", "shelf"], "bowl": ["cupboard"],
    "fork": ["drawer"], "spoon": ["drawer"], "apple": ["counter", "fridge"],
    "milk": ["fridge"], "bread": ["counter", "shelf"], "cloth": ["sink"],
}


@dataclass
class Subtask:
    kind: str
    text: str
    goals: List[tuple]
    args: tuple = ()


@dataclass
class Episode:
    seed: int
    init_loc: dict
    dirty_surfaces: List[str]
    dirty_objects: List[str]
    subtasks: List[Subtask]
    interference: dict = field(default_factory=dict)

    def world(self) -> World:
        return World(self.init_loc, self.dirty_surfaces, self.dirty_objects, seed=self.seed * 7919 + 1)

    def final_goals(self) -> List[tuple]:
        out = []
        for st in self.subtasks:
            for g in st.goals:
                if g not in out:
                    out.append(g)
        return out

    def layout_text(self) -> str:
        by = {}
        for o, l in self.init_loc.items():
            by.setdefault(l, []).append(o)
        return "; ".join(f"{l}: {', '.join(sorted(v))}" for l, v in sorted(by.items()))


def _prep(L):
    return "in" if L in ("cupboard", "fridge", "drawer", "sink") else "on"


def make_episode(seed: int, K: int) -> Episode:
    rng = random.Random(seed)
    init = {o: rng.choice(c) for o, c in INIT_CHOICES.items()}
    subs: List[Subtask] = []
    placed = {}            # object -> goal location
    wiped, washed = set(), set()
    dirty_s, dirty_o = set(), set()
    flags = {"coffee": False, "toast": False}
    inter = {"clobber": 0, "relocate": 0, "lingering": 0}

    def add_place(o, L, kind="place"):
        placed[o] = L
        subs.append(Subtask(kind, f"Put the {o} {_prep(L)} the {L}.", [("at", o, L)], (o, L)))

    def free_objects():
        return [o for o in init if o not in placed and o not in ("cloth",)]

    while len(subs) < K:
        cands = []
        # clobber candidates
        for S in ("table", "counter", "shelf"):  # fixed order matching the main run (set iteration depends on PYTHONHASHSEED)
            if S not in wiped and any(L == S for L in placed.values()):
                cands.append(("wipe", S))
        for o, L in placed.items():
            if o not in washed and L != "sink" and o != "milk" and o != "apple":
                cands.append(("wash", o))
        if "mug" in placed and not flags["coffee"]:
            cands.append(("coffee",))
        if "bread" in placed and not flags["toast"]:
            cands.append(("toast",))
        # relocation candidates: objects initially on a wiped surface, not yet placed
        for o in free_objects():
            if init[o] in wiped:
                cands.append(("fetch", o))
        if subs and cands and rng.random() < 0.75:
            c = rng.choice(cands)
            if c[0] == "wipe":
                S = c[1]
                wiped.add(S); dirty_s.add(S); inter["clobber"] += 1
                subs.append(Subtask("wipe", f"Wipe the {S} clean.", [("clean_surface", S)], (S,)))
            elif c[0] == "wash":
                o = c[1]
                washed.add(o); dirty_o.add(o); inter["clobber"] += 1
                subs.append(Subtask("wash", f"Wash the {o} at the sink.", [("clean_obj", o)], (o,)))
            elif c[0] == "coffee":
                flags["coffee"] = True; inter["clobber"] += 1; inter["lingering"] += 1
                subs.append(Subtask("coffee", "Make a coffee in the mug.", [("coffee",)]))
            elif c[0] == "toast":
                flags["toast"] = True; inter["clobber"] += 1; inter["lingering"] += 1
                subs.append(Subtask("toast", "Toast the bread.", [("toasted",)]))
            else:
                o = c[1]
                dests = [L for L in PLACE_DEST if L != init[o] and L not in wiped]
                inter["relocate"] += 1
                add_place(o, rng.choice(dests), kind="fetch")
            continue
        fo = free_objects()
        if not fo:
            break
        o = rng.choice(fo)
        if o in ("mug", "bread", "plate"):
            dests = [L for L in ("table", "counter", "shelf") if L != init[o]]
        else:
            dests = [L for L in PLACE_DEST if L != init[o]]
        # bias placements onto surfaces so later wipes interfere
        surf = [d for d in dests if d in SURFACES and d not in wiped]
        L = rng.choice(surf) if surf and rng.random() < 0.7 else rng.choice(dests)
        add_place(o, L)
    return Episode(seed, init, sorted(dirty_s), sorted(dirty_o), subs, inter)


if __name__ == "__main__":
    # self-check: generator is deterministic and goals are mutually consistent
    for K in (4, 6, 8):
        for s in range(200):
            e1, e2 = make_episode(s, K), make_episode(s, K)
            assert [x.text for x in e1.subtasks] == [x.text for x in e2.subtasks]
            locs = {}
            for g in e1.final_goals():
                if g[0] == "at":
                    assert g[1] not in locs, (s, K, g)
                    locs[g[1]] = g[2]
    e = make_episode(3, 6)
    print(e.layout_text())
    for st in e.subtasks:
        print(" -", st.kind, st.text, st.goals)
    print(e.interference, e.dirty_surfaces, e.dirty_objects)
    print("ok")
