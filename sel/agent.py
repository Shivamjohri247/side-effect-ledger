"""
Context-isolated subtask workers and the Side-Effect Ledger (SEL).

Every subtask is executed by a fresh LLM "worker" that sees the job plan, the
(possibly stale) initial layout, its own steps within the current subtask, and
the current partial observation. Conditions differ ONLY in the cross-subtask
channel:

  isolated     : nothing from earlier workers.
  summary      : free-text handoff notes written by each earlier worker (1 extra call/subtask).
  full_history : verbatim transcript of all earlier subtasks.
  ledger       : SEL rendered in context (no commit check).
  check        : SEL commit check only (no ledger in context).
  check_generic: control for the check -- the first "done" of each subtask is rejected with a
                 content-free reminder of the household rules (second chance, no information).
  sel          : ledger in context + commit check (full method).

The ledger is built only from the effects of actions the harness executed
(never from hidden simulator state); the commit check uses only the ledger.
"""
from __future__ import annotations
import json
import random
from typing import Dict, List, Optional

from llm import extract_json
from sim import ACTION_HELP, OPENABLE, APPLIANCES

CONDITIONS = ["isolated", "summary", "full_history", "ledger", "check", "check_generic", "sel"]
MAX_STEPS = 30

SYSTEM = (
    "You are a household robot executing ONE subtask of a larger job. Earlier subtasks were "
    "done by other workers and later subtasks will be done by other workers: do ONLY your own "
    "subtask, never start a later one. You only know about earlier subtasks what is shown to you.\n"
    "Household rules: when you declare done, (1) the results of all earlier subtasks must still hold, "
    "(2) cupboard, fridge and drawer must be closed, (3) appliances must be off, (4) your hand must be empty.\n"
    "You have ONE hand: to pick something up you must first place what you hold.\n"
    "As soon as your subtask is complete and the rules hold, reply {\"action\":\"done\"}.\n" + ACTION_HELP
)


def fact_text(f: tuple) -> str:
    if f[0] == "at":
        prep = "in" if f[2] in ("cupboard", "fridge", "drawer", "sink") else "on"
        return f"{f[1]} {prep} the {f[2]}"
    if f[0] == "clean_surface":
        return f"{f[1]} wiped clean"
    if f[0] == "clean_obj":
        return f"{f[1]} washed"
    return {"coffee": "coffee made in the mug", "toasted": "bread toasted"}[f[0]]


class Ledger:
    """Effect record maintained from executed-action effects only."""

    def __init__(self, init_loc: Dict[str, str]):
        self.init_loc = dict(init_loc)
        self.loc = dict(init_loc)
        self.moved_by: Dict[str, int] = {}
        self.opened: Dict[str, int] = {}     # container -> subtask that opened it (while open)
        self.on: Dict[str, int] = {}         # appliance -> subtask that switched it on (while on)
        self.done_facts = set()              # wiped / washed / made
        self.protected: List[tuple] = []     # (fact, subtask_idx)
        self.holding: Optional[str] = None

    def apply(self, effects: List[tuple], sub_idx: int):
        for e in effects:
            k = e[0]
            if k == "moved":
                _, o, _src, dst = e
                self.loc[o] = dst
                self.holding = o if dst == "HELD" else None
                self.moved_by[o] = sub_idx
            elif k == "opened":
                self.opened[e[1]] = sub_idx
            elif k == "closed":
                self.opened.pop(e[1], None)
            elif k == "switched_on":
                self.on[e[1]] = sub_idx
            elif k == "switched_off":
                self.on.pop(e[1], None)
            elif k == "wiped":
                self.done_facts.add(("clean_surface", e[1]))
            elif k == "washed":
                self.done_facts.add(("clean_obj", e[1]))
            elif k == "made":
                self.done_facts.add(("coffee",) if e[1] == "coffee" else ("toasted",))

    def holds(self, f: tuple) -> bool:
        if f[0] == "at":
            return self.loc.get(f[1]) == f[2]
        return f in self.done_facts

    def protect_completed(self, goals: List[tuple], sub_idx: int):
        """Protect a finished subtask's goals that the ledger believes were achieved."""
        for g in goals:
            if self.holds(g) and all(g != p for p, _ in self.protected):
                self.protected.append((g, sub_idx))

    def debts(self) -> List[str]:
        out = [f"restore: {fact_text(f)} (result of subtask {i + 1}; the {f[1]} is now "
               f"{'in your hand' if self.loc.get(f[1]) == 'HELD' else 'at the ' + str(self.loc.get(f[1]))})"
               if f[0] == "at" else f"restore: {fact_text(f)} (result of subtask {i + 1})"
               for f, i in self.protected if not self.holds(f)]
        out += [f"close the {c}" for c in sorted(self.opened)]
        out += [f"switch off the {a}" for a in sorted(self.on)]
        if self.holding:
            out.append(f"put down the {self.holding} somewhere sensible")
        return out

    def threat(self, act: dict) -> Optional[str]:
        a, t = str(act.get("action", "")).lower(), act.get("target")
        if a == "pick":
            for f, i in self.protected:
                if f[0] == "at" and f[1] == t and self.holds(f):
                    return (f"warning: '{fact_text(f)}' is a protected result of subtask {i + 1}; "
                            f"you must restore it before declaring done")
        return None

    def render(self) -> str:
        L = ["SIDE-EFFECT LEDGER (recorded automatically from all actions executed so far):"]
        if self.protected:
            L.append("  Protected results (must still hold when you finish):")
            for f, i in self.protected:
                L.append(f"    - {fact_text(f)} [subtask {i + 1}] -> "
                         f"{'currently OK' if self.holds(f) else 'currently VIOLATED'}")
        moved = [(o, self.init_loc[o], self.loc[o], i) for o, i in self.moved_by.items()
                 if self.loc[o] != self.init_loc[o]]
        if moved:
            L.append("  Objects no longer where the initial layout says:")
            for o, a, b, i in moved:
                L.append(f"    - {o}: {a} -> {'held by you' if b == 'HELD' else b} (last moved in subtask {i + 1})")
        if self.opened:
            L.append("  Containers currently open: " + ", ".join(f"{c} (opened in subtask {i + 1})" for c, i in self.opened.items()))
        if self.on:
            L.append("  Appliances currently ON: " + ", ".join(f"{a} (switched on in subtask {i + 1})" for a, i in self.on.items()))
        if len(L) == 1:
            L.append("  (no effects recorded yet)")
        return "\n".join(L)


def plan_text(subtasks, cur: int) -> str:
    rows = []
    for i, st in enumerate(subtasks):
        tag = "done by an earlier worker" if i < cur else ("<- YOUR SUBTASK" if i == cur else "later")
        rows.append(f"  {i + 1}. {st.text} [{tag}]")
    return "JOB PLAN:\n" + "\n".join(rows)


def run_episode(ep, cond: str, llm, log: Optional[list] = None, exo_p: float = 0.0) -> dict:
    assert cond in CONDITIONS
    world = ep.world()
    ledger = Ledger(ep.init_loc)
    use_ledger_ctx = cond in ("ledger", "sel")
    use_check = cond in ("check", "sel")
    notes: List[str] = []
    transcript: List[str] = []
    stats = {"llm_calls": 0, "prompt_chars": 0, "steps": 0, "invalid": 0, "parse_fail": 0,
             "check_rejects": 0, "warnings": 0, "step_cap_hits": 0, "exo_moves": 0}
    exo_rng = random.Random(ep.seed * 31 + 7)
    achieved_at_completion = []  # per subtask: goals that held right after the subtask

    def call(msgs):
        stats["llm_calls"] += 1
        stats["prompt_chars"] += sum(len(m["content"]) for m in msgs)
        try:
            return llm.chat(msgs, max_tokens=2048)
        except Exception as e:  # free-tier failure: treated as a no-op step, counted
            stats["llm_errors"] = stats.get("llm_errors", 0) + 1
            return ""

    for si, st in enumerate(ep.subtasks):
        if exo_p and si > 0 and exo_rng.random() < exo_p:
            # exogenous disturbance (another household member): invisible to every channel
            cands = [o for o, l in world.loc.items() if l != "HELD" and o != "cloth"]
            o = exo_rng.choice(sorted(cands))
            world.loc[o] = exo_rng.choice([L for L in ("counter", "table", "shelf", "sink") if L != world.loc[o]])
            stats["exo_moves"] += 1
        own: List[str] = []
        cur_tr: List[str] = []
        generic_used = False
        pending: List[str] = []  # outstanding items from the last check rejection
        for _ in range(MAX_STEPS):
            parts = [plan_text(ep.subtasks, si),
                     "INITIAL LAYOUT at the start of the job (may be outdated): " + ep.layout_text()]
            if cond == "summary" and notes:
                parts.append("HANDOFF NOTES FROM EARLIER WORKERS:\n" + "\n".join(notes))
            if cond == "full_history" and transcript:
                parts.append("FULL TRANSCRIPT OF EARLIER SUBTASKS:\n" + "\n".join(transcript))
            if use_ledger_ctx:
                parts.append(ledger.render())
            parts.append("YOUR STEPS SO FAR IN THIS SUBTASK:\n" + ("\n".join(own[-12:]) if own else "  (none)"))
            parts.append("CURRENT OBSERVATION: " + world.obs_text())
            if pending:
                parts.append("BEFORE YOU MAY DECLARE DONE, YOU MUST STILL: " + "; ".join(pending))
            parts.append(f"YOUR SUBTASK (subtask {si + 1} of {len(ep.subtasks)}): {st.text}\n"
                         "If it is complete and the household rules hold, reply done. Otherwise reply with ONE JSON "
                         'object only: {"thought": "<one short sentence: what you need next and why>", '
                         '"action": ..., "target": ...}.')
            msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n\n".join(parts)}]
            reply = call(msgs)
            act = extract_json(reply) if reply else None
            stats["steps"] += 1
            if not isinstance(act, dict) or "action" not in act:
                stats["parse_fail"] += 1
                own.append(f"  (unparseable reply) -> please answer with one JSON action")
                continue
            thought = str(act.get("thought", ""))[:120]
            act = {k: (str(v).strip().lower() if isinstance(v, str) else v) for k, v in act.items()
                   if k in ("action", "target")}
            a = str(act.get("action"))
            if a == "done":
                if cond == "check_generic" and not generic_used:
                    generic_used = True
                    stats["check_rejects"] += 1
                    own.append("  done -> please re-verify the household rules (earlier results still hold, "
                               "containers closed, appliances off, hand empty) and then declare done again")
                    pending = ["re-verify the household rules (earlier results still hold, containers closed, "
                               "appliances off, hand empty), then declare done again"]
                    continue
                if use_check:
                    d = ledger.debts()
                    if d:
                        stats["check_rejects"] += 1
                        own.append("  done -> REJECTED by ledger check; first: " + "; ".join(d))
                        pending = d
                        continue
                own.append("  done")
                break
            w = ledger.threat(act) if use_check else None  # evaluated before the effect is applied
            res = world.step(act)
            ledger.apply(res["effects"], si)
            msg = "ok" if res["ok"] else f"failed: {res['reason']}"
            if not res["ok"]:
                stats["invalid"] += 1
            if res["ok"]:
                if w:
                    stats["warnings"] += 1
                    msg += f" ({w})"
            line = (f"  (thought: {thought}) " if thought else "  ") + f"{json.dumps(act)} -> {msg} [{world.short_obs()}]"
            if pending and use_check and res["ok"]:
                pending = ledger.debts()  # keep the outstanding list current as items get fixed
            own.append(line)
            cur_tr.append(f"[subtask {si + 1}] {json.dumps(act)} -> {msg}. {world.obs_text()}")
        else:
            stats["step_cap_hits"] += 1
        transcript.extend(cur_tr)  # full_history only ever sees *earlier* subtasks
        ledger.protect_completed(st.goals, si)
        achieved_at_completion.append([g for g in st.goals if world.fact(g)])
        if cond == "summary":
            m = [{"role": "system", "content": "You are a household robot worker handing off to the next worker."},
                 {"role": "user", "content": f"You just worked on subtask {si + 1}: {st.text}\nYour steps:\n" +
                  "\n".join(own) + "\n\nWrite a short handoff note (max 60 words) for the next worker: what you "
                  "achieved, and anything you moved, opened, or switched on and left that way. Plain text only."}]
            notes.append(f"[after subtask {si + 1}] " + (call(m).strip()[:500] or "(no note)"))
        if log is not None:
            log.append({"subtask": si + 1, "text": st.text, "steps": own})

    goals = ep.final_goals()
    held = [g for g in goals if world.fact(g)]
    # clobbers: goals achieved when their subtask finished but false at the end
    clob = sum(1 for ach in achieved_at_completion for g in ach if not world.fact(g))
    inv = world.invariant_violations()
    return {
        "goal_success": int(len(held) == len(goals)),
        "goal_frac": len(held) / len(goals),
        "clean_success": int(len(held) == len(goals) and not inv),
        "clobbers": clob,
        "achieved_then": sum(len(a) for a in achieved_at_completion),
        "lingering": len(inv),
        "invariant_violations": inv,
        "subtask_end": [{"kind": st.kind, "held_at_end": all(world.fact(g) for g in st.goals)}
                        for st in ep.subtasks],
        **stats,
    }
