import json, math
from collections import defaultdict
import os
R=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","results","")
def load(f): return [json.loads(l) for l in open(R+f)]
main=load("main.jsonl")+load("ext.jsonl"); exo=load("exo.jsonl")
def idx(rows):
    d=defaultdict(dict)
    for r in rows: d[(r["seed"],r["K"])][r["cond"]]=r
    return d
def bcdf(k,n,p): return sum(math.comb(n,i)*p**i*(1-p)**(n-i) for i in range(k+1))
def bisect(f,lo=0.,hi=1.):  # f(lo) True, f(hi) False
    for _ in range(100):
        m=(lo+hi)/2
        if f(m): lo=m
        else: hi=m
    return lo
def cp(x,n):
    lo=0. if x==0 else bisect(lambda p: bcdf(x-1,n,p)>0.975)  # P(X>=x)=1-cdf(x-1) <0.025
    hi=1. if x==n else bisect(lambda p: bcdf(x,n,p)>0.025)
    return lo,hi
def sign(a,b):
    w=sum(x>y for x,y in zip(a,b)); l=sum(x<y for x,y in zip(a,b)); n=w+l
    p=min(1,2*sum(math.comb(n,i) for i in range(min(w,l)+1))/2**n) if n else 1
    return w,l,p
def holm(ps):
    o=sorted(range(len(ps)),key=lambda i:ps[i]); adj=[0]*len(ps); run=0
    for r,i in enumerate(o): run=max(run,min(1,(len(ps)-r)*ps[i])); adj[i]=run
    return adj
def report(rows,label):
    D=idx(rows); eps=sorted(D); conds=sorted({r["cond"] for r in rows})
    print(f"== {label}: {len(eps)} eps, conds {conds}, counts", {c:sum(1 for e in eps if c in D[e]) for c in conds})
    m=lambda c,f: sum(D[e][c].get(f,0) for e in eps)/len(eps)
    for c in conds:
        g=sum(D[e][c]["goal_success"] for e in eps); cl=sum(D[e][c]["clean_success"] for e in eps)
        lo,hi=cp(cl,len(eps))
        print(f"{c:14s} goal {g}/{len(eps)}={g/len(eps):.3f} clean {cl}={cl/len(eps):.3f} CP[{lo:.3f},{hi:.3f}] clob {m(c,'clobbers'):.3f} ling {m(c,'lingering'):.3f} ktok {sum(D[e][c]['prompt_chars'] for e in eps)/4000/len(eps):.1f} rej {m(c,'check_rejects'):.3f} cap {m(c,'step_cap_hits'):.3f} inv {m(c,'invalid'):.2f} calls {m(c,'llm_calls'):.1f} err {sum(D[e][c].get('llm_errors',0) for e in eps)}")
    tests={}
    for b in [c for c in conds if c!="sel"]:
        for f in ("goal_success","clean_success"):
            tests[(b,f)]=sign([D[e]["sel"][f] for e in eps],[D[e][b][f] for e in eps])
    adj=holm([t[2] for t in tests.values()])
    for (k,t),a in zip(tests.items(),adj): print(f"  sel vs {k}: {t[0]}:{t[1]} p={t[2]:.3g} holm={a:.3g}")
    Ks=sorted({e[1] for e in eps})
    for c in conds: print("  perK clean",c,[round(sum(D[e][c]["clean_success"] for e in eps if e[1]==K)/sum(1 for e in eps if e[1]==K),3) for K in Ks])
    for c in conds:
        kk=defaultdict(list)
        for e in eps:
            for s in D[e][c]["subtask_end"]: kk[s["kind"]].append(s["held_at_end"])
        print("  kinds",c,{k:(sum(v),len(v),round(sum(v)/len(v),3)) for k,v in sorted(kk.items())})
    tk=lambda c: sum(D[e][c]['prompt_chars'] for e in eps)
    if "isolated" in conds:
        print("  token ratios sel/iso %.3f sel/full %.3f"%(tk("sel")/tk("isolated"),tk("sel")/tk("full_history")), ("critic/sel %.3f full/iso %.3f"%(tk("critic")/tk("sel"),tk("full_history")/tk("isolated")) if "critic" in conds else ""))
        cheapest=min(conds,key=tk); print("  cheapest",cheapest)
    return D,eps
D,eps=report(main,"main")
# interference
nonfirst=sum(e[1]-1 for e in eps); inter=sum(D[e]["sel"]["interference"]["clobber"]+D[e]["sel"]["interference"]["relocate"] for e in eps)
print("interference",inter,"of",nonfirst, "consistent across conds:", all(len({json.dumps(D[e][c]["interference"],sort_keys=True) for c in D[e]})==1 for e in eps))
# SEL failures
for e in eps:
    r=D[e]["sel"]
    if not r["clean_success"]:
        print("SEL fail",e,"goal",r["goal_success"],"clob",r["clobbers"],"inv",r["invariant_violations"],"capHits",r["step_cap_hits"],"rej",r["check_rejects"])
        for st in r["trace"]:
            if st["steps"] and st["steps"][-1].strip()!="done": print("   subtask",st["subtask"],"len",len(st["steps"]),"last:",st["steps"][-1][:200])
report(exo,"exo")
