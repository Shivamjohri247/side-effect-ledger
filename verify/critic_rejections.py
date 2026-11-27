import json, re
from collections import Counter
import os
R=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","results","")
crit=[json.loads(l) for l in open(R+"ext.jsonl")]; crit=[r for r in crit if r["cond"]=="critic"]
C=Counter(); rejc=Counter(); n=0; samples={}
for r in crit:
    texts=[st["text"] for st in r["trace"]]
    opened=set(); on=set(); holding="nothing"
    for st in r["trace"]:
        i=st["subtask"]-1
        for s in st["steps"]:
            m=re.search(r'\{"action": "(\w+)"(?:, "target": "([\w ]+)")?\} -> ok',s)
            if m:
                a,t=m.groups()
                if a=="open": opened.add(t)
                if a=="close": opened.discard(t)
                if a=="toggle": on^={t}
            h=re.search(r'holding: ([\w]+)\]',s)
            if h: holding=h.group(1)
            if "REJECTED by verifier" in s:
                n+=1
                items=[x.strip().rstrip(".").lower() for x in s.split("first:",1)[1].split(";")]
                cats=set()
                for it in items:
                    lt=[t.rstrip(".").lower() for t in texts[i+1:]]
                    if any(t in it for t in lt): c="later_subtask_verbatim"
                    elif it.startswith("close"):
                        tgt=[x for x in ("cupboard","fridge","drawer") if x in it]
                        c="close_correct" if any(x in opened for x in tgt) else "close_spurious"
                    elif "hand" in it and "empty" in it or it.startswith("put down"):
                        c="hand_correct" if holding!="nothing" else "hand_spurious"
                    elif "appliance" in it or "turn off" in it or "switch off" in it:
                        c="appl_correct" if on else "appl_spurious"
                    elif texts[i].rstrip(".").lower() in it: c="own_subtask"
                    else: c="other"
                    C[c]+=1; cats.add(c); samples.setdefault(c,[]).append((texts[i],it))
                rejc[("any_correct" if any(c.endswith("_correct") for c in cats) else "no_correct_rule_item")]+=1
                if cats<= {"close_spurious","hand_spurious","appl_spurious","other","own_subtask"}: rejc["all_spurious_or_other"]+=1
                if "later_subtask_verbatim" in cats: rejc["has_later"]+=1
print("rejections",n); print("item cats",C); print("rej cats",rejc)
for c in ("other","own_subtask"):
    import random; random.seed(0)
    for x in random.sample(samples[c],min(15,len(samples[c]))): print(c,"|",x)
