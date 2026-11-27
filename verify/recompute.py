import json, collections
from math import comb
import os; D=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results') + os.sep
def load(f): return [json.loads(l) for l in open(D+f) if l.strip()]
def mean(x): x=list(x); return sum(x)/len(x) if x else float('nan')
def sign(w,l):
    n=w+l
    if n==0: return 1.0
    k=min(w,l)
    return min(1.0,2*sum(comb(n,i) for i in range(k+1))/2**n)
def g(r,k,d=0):
    return r.get(k,r.get('stats',{}).get(k,d)) if k in r or 'stats' in r else d
for name in ['main.jsonl','exo.jsonl']:
    R=load(name); print('=====',name,len(R))
    c=collections.Counter((r['seed'],r['K'],r['cond']) for r in R)
    print('dups',[k for k,v in c.items() if v>1])
    conds=sorted({r['cond'] for r in R}); eps=sorted({(r['seed'],r['K']) for r in R})
    print('episodes',len(eps),'conds',conds)
    print('missing',[(e,cd) for e in eps for cd in conds if (e[0],e[1],cd) not in c])
    print('llm_errors>0',[(r['seed'],r['K'],r['cond']) for r in R if g(r,'llm_errors')>0])
    print('models',{r['model'] for r in R}, 'model_counts keys',{k for r in R for k in r['model_counts']})
    print('clean>goal',[(r['seed'],r['K'],r['cond']) for r in R if r['clean_success']>r['goal_success']])
    print('goal=1 & clobbers>0',[(r['seed'],r['K'],r['cond'],r['clobbers']) for r in R if r['goal_success'] and r['clobbers']>0])
    print('cond n goal clean clob ling steps calls ktok(c/4) ptok_k rej cap')
    by=collections.defaultdict(list)
    for r in R: by[r['cond']].append(r)
    for cd in conds:
        x=by[cd]
        print(cd,len(x),*[round(mean(g(r,k) for r in x),3) for k in ['goal_success','clean_success','clobbers','lingering','steps','llm_calls']],
          round(mean(g(r,'prompt_chars')/4/1000 for r in x),1), round(mean(g(r,'prompt_tokens')/1000 for r in x),1),
          round(mean(g(r,'check_rejects') for r in x),2), round(mean(g(r,'step_cap_hits') for r in x),2))
    print('clean by K')
    for cd in conds:
        print(cd,{K:round(mean(r['clean_success'] for r in by[cd] if r['K']==K),3) for K in sorted({r['K'] for r in R})})
    if 'sel' in by:
        for m in ['goal_success','clean_success']:
            for cd in conds:
                if cd=='sel': continue
                a={(r['seed'],r['K']):r[m] for r in by['sel']}; b={(r['seed'],r['K']):r[m] for r in by[cd]}
                w=sum(a[e]>b[e] for e in a if e in b); l=sum(a[e]<b[e] for e in a if e in b)
                both=sum(a[e]==b[e]==1 for e in a if e in b); neither=sum(a[e]==b[e]==0 for e in a if e in b)
                print(m,'sel vs',cd,f'W{w} L{l} p={sign(w,l):.4f} both1={both} both0={neither}')
        print('SEL clean=0:')
        for r in by['sel']:
            if not r['clean_success']:
                print(r['seed'],r['K'],'goal',r['goal_success'],'clob',r['clobbers'],'ling',r['lingering'],'cap',r['step_cap_hits'],'viol',r['invariant_violations'],'frac',r['goal_frac'])
    if name=='main.jsonl':
        for e in eps[:0]: pass
        e=(by['sel'][0]['seed'],by['sel'][0]['K'])
        for cd in ['sel','isolated']:
            r=[r for r in by[cd] if (r['seed'],r['K'])==e][0]
            print('TRACE',cd,e,'goal',r['goal_success'],'clean',r['clean_success'],'clob',r['clobbers'],'ling',r['lingering'],r['invariant_violations'])
            for t in r['trace']:
                print(' sub',t.get('subtask'),t.get('text'))
                for s in t['steps'][-3:]: print('   ',s[:160])
                print('   keys',[k for k in t if k not in('steps',)] )
