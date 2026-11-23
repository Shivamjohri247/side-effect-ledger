"""
Symbolic household simulator for context-isolated multi-subtask execution.

Pure Python, deterministic given a seed. Partial observability: the agent sees
only objects at its current location (inside an openable container only while it
is open) plus what it holds. Grasps slip with probability SLIP_P.

Every successful action returns the list of *effects* it caused (fact deltas).
These are what the executing harness can observe about its own actions
(proprioception + action outcome); the side-effect ledger is built only from them.
"""
from __future__ import annotations
import copy
import random
from typing import Dict, List, Optional

LOCATIONS = ["counter", "table", "sink", "shelf", "cupboard", "fridge", "drawer",
             "coffee_machine", "toaster"]
OPENABLE = {"cupboard", "fridge", "drawer"}
APPLIANCES = {"coffee_machine", "toaster"}
SURFACES = {"counter", "table", "shelf"}
OBJECTS = ["mug", "plate", "bowl", "fork", "spoon", "apple", "milk", "bread", "cloth"]
SLIP_P = 0.1


class World:
    def __init__(self, loc: Dict[str, str], dirty_surfaces=(), dirty_objects=(), seed: int = 0):
        self.loc = dict(loc)                      # object -> location | "HELD"
        self.opened = {c: False for c in OPENABLE}
        self.on = {a: False for a in APPLIANCES}
        self.surface_dirty = {s: (s in dirty_surfaces) for s in SURFACES}
        self.obj_clean = {o: (o not in dirty_objects) for o in loc}
        self.mug_coffee = False
        self.bread_toasted = False
        self.agent = "counter"
        self.holding: Optional[str] = None
        self.rng = random.Random(seed)

    # ---------------- observation ----------------
    def visible(self) -> List[str]:
        if self.agent in OPENABLE and not self.opened[self.agent]:
            return []
        return sorted(o for o, l in self.loc.items() if l == self.agent)

    def obs_text(self) -> str:
        a = self.agent
        parts = [f"You are at the {a}."]
        if a in OPENABLE:
            parts.append(f"The {a} is {'open' if self.opened[a] else 'closed'}.")
        if a in APPLIANCES:
            parts.append(f"The {a} is {'ON' if self.on[a] else 'off'}.")
        if a in SURFACES:
            parts.append(f"The {a} surface is {'dirty' if self.surface_dirty[a] else 'clean'}.")
        vis = self.visible()
        if a in OPENABLE and not self.opened[a]:
            parts.append("You cannot see inside while it is closed.")
        else:
            parts.append("You see: " + (", ".join(self._desc(o) for o in vis) if vis else "nothing") + ".")
        parts.append(f"You are holding: {self._desc(self.holding) if self.holding else 'nothing'}.")
        return " ".join(parts)

    def short_obs(self) -> str:
        vis = self.visible()
        if self.agent in OPENABLE and not self.opened[self.agent]:
            seen = "(closed)"
        else:
            seen = ", ".join(vis) if vis else "nothing"
        return f"now at {self.agent}, see: {seen}, holding: {self.holding or 'nothing'}"

    def _desc(self, o: str) -> str:
        tags = []
        if not self.obj_clean[o]:
            tags.append("dirty")
        if o == "mug" and self.mug_coffee:
            tags.append("with coffee")
        if o == "bread" and self.bread_toasted:
            tags.append("toasted")
        return o + (f" ({', '.join(tags)})" if tags else "")

    # ---------------- facts (for evaluation) ----------------
    def fact(self, f: tuple) -> bool:
        kind = f[0]
        if kind == "at":
            return self.loc.get(f[1]) == f[2]
        if kind == "clean_surface":
            return not self.surface_dirty[f[1]]
        if kind == "clean_obj":
            return self.obj_clean[f[1]]
        if kind == "coffee":
            return self.mug_coffee
        if kind == "toasted":
            return self.bread_toasted
        raise ValueError(f)

    def invariant_violations(self) -> List[str]:
        v = [f"{c} left open" for c in sorted(OPENABLE) if self.opened[c]]
        v += [f"{a} left on" for a in sorted(APPLIANCES) if self.on[a]]
        if self.holding:
            v.append(f"still holding {self.holding}")
        return v

    # ---------------- transition ----------------
    def step(self, act: dict) -> dict:
        """Returns {ok, reason, effects}. effects: list of (kind, *args) tuples."""
        a = str(act.get("action", "")).strip().lower()
        t = act.get("target")
        t = str(t).strip().lower() if t is not None else None
        eff: List[tuple] = []

        def bad(msg):
            return {"ok": False, "reason": msg, "effects": []}

        if a == "move_to":
            if t not in LOCATIONS:
                return bad(f"unknown location '{t}'")
            self.agent = t
            return {"ok": True, "reason": "", "effects": []}
        if a in ("open", "close"):
            if t not in OPENABLE:
                return bad(f"'{t}' cannot be opened/closed")
            if self.agent != t:
                return bad(f"you must be at the {t}")
            want = a == "open"
            if self.opened[t] == want:
                return bad(f"the {t} is already {'open' if want else 'closed'}")
            self.opened[t] = want
            return {"ok": True, "reason": "", "effects": [("opened" if want else "closed", t)]}
        if a == "pick":
            if t not in self.loc:
                return bad(f"unknown object '{t}'")
            if self.holding:
                return bad(f"your hand is full (holding {self.holding})")
            if self.loc[t] != self.agent or t not in self.visible():
                return bad(f"you do not see the {t} here")
            if self.rng.random() < SLIP_P:
                return bad(f"the grasp on the {t} slipped; try again")
            src = self.loc[t]
            self.loc[t] = "HELD"
            self.holding = t
            return {"ok": True, "reason": "", "effects": [("moved", t, src, "HELD")]}
        if a == "place":
            if not self.holding:
                return bad("you are not holding anything")
            if t not in (None, "", self.holding, self.agent):
                if t in LOCATIONS:
                    return bad(f"place puts the held object at your current location ({self.agent}); "
                               f"move_to the {t} first")
                return bad(f"you are holding {self.holding}, not {t}")
            if self.agent in OPENABLE and not self.opened[self.agent]:
                return bad(f"the {self.agent} is closed")
            o = self.holding
            self.loc[o] = self.agent
            self.holding = None
            eff.append(("moved", o, "HELD", self.agent))
            if o == "mug" and self.agent == "coffee_machine" and self.on["coffee_machine"] and not self.mug_coffee:
                self.mug_coffee = True
                eff.append(("made", "coffee"))
            if o == "bread" and self.agent == "toaster" and self.on["toaster"] and not self.bread_toasted:
                self.bread_toasted = True
                eff.append(("made", "toast"))
            return {"ok": True, "reason": "", "effects": eff}
        if a == "toggle":
            if t not in APPLIANCES:
                return bad(f"'{t}' is not an appliance")
            if self.agent != t:
                return bad(f"you must be at the {t}")
            self.on[t] = not self.on[t]
            eff.append(("switched_on" if self.on[t] else "switched_off", t))
            if self.on[t] and t == "coffee_machine" and self.loc["mug"] == t and not self.mug_coffee:
                self.mug_coffee = True
                eff.append(("made", "coffee"))
            if self.on[t] and t == "toaster" and self.loc["bread"] == t and not self.bread_toasted:
                self.bread_toasted = True
                eff.append(("made", "toast"))
            return {"ok": True, "reason": "", "effects": eff}
        if a == "wipe":
            if t not in SURFACES:
                return bad(f"'{t}' is not a wipeable surface")
            if self.agent != t:
                return bad(f"you must be at the {t}")
            if self.holding != "cloth":
                return bad("you need to hold the cloth to wipe")
            # skill-level side effect: objects on the surface are swept aside first
            aside = "shelf" if t == "counter" else "counter"
            for o in sorted(o for o, l in self.loc.items() if l == t):
                self.loc[o] = aside
                eff.append(("moved", o, t, aside))
            self.surface_dirty[t] = False
            eff.append(("wiped", t))
            return {"ok": True, "reason": "", "effects": eff}
        if a == "wash":
            if self.agent != "sink":
                return bad("you must be at the sink")
            if not self.holding:
                return bad("hold the object you want to wash")
            if t not in (None, "", self.holding, "sink"):
                return bad(f"you are holding {self.holding}, not {t}")
            self.obj_clean[self.holding] = True
            return {"ok": True, "reason": "", "effects": [("washed", self.holding)]}
        return bad(f"unknown action '{a}'")


ACTION_HELP = """Actions (reply with exactly ONE JSON object):
{"action":"move_to","target":"<location>"}
{"action":"open","target":"<cupboard|fridge|drawer>"}   (must be at it)
{"action":"close","target":"<cupboard|fridge|drawer>"}  (must be at it)
{"action":"pick","target":"<object>"}     (object must be visible at your location; hand must be empty; grasps sometimes slip)
{"action":"place","target":"<your current location>"} (puts the held object down where you are; move_to the destination first; containers must be open)
{"action":"toggle","target":"<coffee_machine|toaster>"} (switches it on/off; a mug at a running coffee machine fills with coffee, bread in a running toaster gets toasted)
{"action":"wipe","target":"<counter|table|shelf>"} (hold the cloth; objects on the surface are moved aside to the counter -- or to the shelf when wiping the counter)
{"action":"wash"}                       (at the sink, washes the held object)
{"action":"done"}                       (when YOUR subtask is complete)
Locations: counter, table, sink, shelf, cupboard, fridge, drawer, coffee_machine, toaster."""
