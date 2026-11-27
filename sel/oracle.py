"""Scripted full-state oracle: checks every generated episode is solvable within the
per-subtask step budget (feasibility upper bound; LLM workers have partial observability).
Run: python3 oracle.py"""
import math, collections
import sim; sim.SLIP_P=0.0
from sim import *
from tasks import make_episode
from agent import MAX_STEPS as MAXS
def pslip(p,slack):
    if slack<0: return 0.0
    return sum(math.comb(f+p-1,f)*0.9**p*0.1**f for f in range(slack+1)) if p else 1.0
class R:
    def __init__(s,w): s.w=w; s.n=0; s.picks=0
    def do(s,a,t=None):
        r=s.w.step({"action":a,"target":t}); assert r["ok"],(a,t,r,s.w.obs_text()); s.n+=1; s.picks+=a=="pick"
    def go(s,L):
        if s.w.agent!=L: s.do("move_to",L)
    def drop(s):
        if s.w.holding: s.go("sink"); s.do("place")
    def fetch(s,o):
        if s.w.holding==o: return
        s.drop(); l=s.w.loc[o]; s.go(l); op=False
        if l in OPENABLE and not s.w.opened[l]: s.do("open",l); op=True
        s.do("pick",o)
        if op: s.do("close",l)
    def put(s,L):
        s.go(L); op=False
        if L in OPENABLE and not s.w.opened[L]: s.do("open",L); op=True
        s.do("place")
        if op: s.do("close",L)
def solve_sub(r,st,prot):
    w=r.w; k=st.kind
    if k in("place","fetch"):
        o,L=st.args; r.fetch(o); r.put(L)
    elif k=="wash":
        o=st.args[0]; r.fetch(o); r.go("sink"); r.do("wash"); L=[g[2] for g in prot if g[0]=="at" and g[1]==o][0]; r.put(L)
    elif k=="wipe":
        S=st.args[0]
        items=[o for o,l in w.loc.items() if l==S and o!="cloth"]
        r.fetch("cloth"); r.go(S); r.do("wipe",S); r.do("place")  # wipe sweeps items aside
        for o in items:
            g=[g for g in prot if g[0]=="at" and g[1]==o]
            if g: r.fetch(o); r.put(g[0][2])
    elif k in("coffee","toast"):
        o,m,fl=("mug","coffee_machine","coffee") if k=="coffee" else ("bread","toaster","toasted")
        r.fetch(o); r.put(m); r.go(m)
        if not w.on[m]: r.do("toggle",m)
        r.do("toggle",m)
        g=[g for g in prot if g[0]=="at" and g[1]==o][0]; r.fetch(o); r.put(g[2])
    # restore violated
    for g in prot:
        if g[0]=="at" and not w.fact(g): r.fetch(g[1]); r.put(g[2])
    r.drop()
    for c in OPENABLE:
        if w.opened[c]: r.go(c); r.do("close",c)
    for a in APPLIANCES:
        if w.on[a]: r.go(a); r.do("toggle",a)
    r.n+=1  # done
import os, sys
SEED0 = int(sys.argv[1]) if len(sys.argv) > 1 else 1000  # 1000: main study; 5000: disturbance study
tot=collections.Counter(); bad=[]; allp=[]
for K in (4,6,8):
    nsub=0; nfeas=0; nep=0; epok=0; pep=[]
    for i in range(20):
        seed=SEED0+K*100+i; ep=make_episode(seed,K); w=ep.world(); prot=[]; eok=True; pe=1.0; row=[]
        for st in ep.subtasks:
            prot=prot+st.goals; r=R(w)
            try: solve_sub(r,st,prot)
            except Exception as e: row.append((st.kind,"ERR",str(e)[:80])); eok=False; pe=0; break
            ps=pslip(r.picks,MAXS-r.n); pe*=ps; nsub+=1; nfeas+=r.n<=MAXS
            row.append((st.kind,r.n,r.picks,round(ps,2)))
            if r.n>MAXS: eok=False
        okg=all(w.fact(g) for g in ep.final_goals()) and not w.invariant_violations()
        if not eok: print(K,seed,"det_ok",eok,"goals_ok",okg,"P_slip_ok=%.2f"%pe,row)
        epok+=eok; pep.append(pe)
    print(f"== K={K}: episodes det-feasible {epok}/20, mean P(solve|slip,greedy oracle)={sum(pep)/20:.2f}, subtasks <=budget: {nfeas}/{nsub}")

    assert epok == 20, f"K={K}: some episodes infeasible"
print("oracle: all episodes feasible")
