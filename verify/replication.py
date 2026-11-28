"""Pre-registered replication analysis (replication/PREREG.md). Independent of sel/analyze.py."""
import json, os, re
from math import comb
from collections import Counter
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', '')
load = lambda p: [json.loads(l) for l in open(R + p)]
M = load('rep_main.jsonl'); O = load('main.jsonl') + load('ext.jsonl')
C = ['isolated', 'check', 'check_rules', 'sel']
I = {(r['seed'], r['cond']): r for r in M}; IO = {(r['seed'], r['cond']): r for r in O}
seeds = sorted({r['seed'] for r in M}); oseeds = sorted({r['seed'] for r in O})
assert len(M) == 96 and len(I) == 96 and seeds == [9000 + K * 100 + i for K in (4, 6, 8) for i in range(8)]
def p2(x, n): return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(x, n - x) + 1)) / 2 ** n)
def holm(ps):
    o = sorted(range(len(ps)), key=lambda i: ps[i]); adj = [0] * len(ps); run = 0
    for r, i in enumerate(o): run = max(run, min(1, (len(ps) - r) * ps[i])); adj[i] = run
    return adj
def sign(ix, ss, a, b, f):
    x = sum(ix[(s, a)][f] > ix[(s, b)][f] for s in ss); y = sum(ix[(s, a)][f] < ix[(s, b)][f] for s in ss); return x, y, p2(x, x + y)
def cp(k, n):
    cdf = lambda k, n, p: sum(comb(n, i) * p**i * (1 - p)**(n - i) for i in range(k + 1))
    def bis(f):
        lo, hi = 0.0, 1.0
        for _ in range(100):
            m = (lo + hi) / 2; lo, hi = (lo, m) if f(m) else (m, hi)
        return (lo + hi) / 2
    return '[%.2f,%.2f]' % (0 if k == 0 else bis(lambda p: 1 - cdf(k - 1, n, p) >= .025), 1 if k == n else bis(lambda p: cdf(k, n, p) <= .025))
for f in ['clean_success', 'goal_success']:
    res = [sign(I, seeds, 'sel', b, f) for b in C[:3]]; adj = holm([r[2] for r in res])
    ores = [sign(IO, oseeds, 'sel', b, f) for b in C[:3]]
    for b, r, h, o in zip(C[:3], res, adj, ores):
        print('PRIMARY' if f == 'clean_success' else 'SEC', f, 'sel vs', b, '%d:%d p=%.3g holm=%.3g' % (r[0], r[1], r[2], h), '| reported %d:%d' % o[:2], 'same direction' if (r[0] - r[1]) * (o[0] - o[1]) > 0 else 'NOT same')
def held_bug(r):
    holding = 'nothing'; bad = 0
    for t in r['trace']:
        for s in t['steps']:
            h = re.search(r'holding: (\w+)\]', s)
            if h: holding = h.group(1)
            if s.strip().startswith('done') and 'REJECTED' not in s and holding != 'nothing': bad += 1
    return bad
for c in C:
    rs = [I[(s, c)] for s in seeds]; os_ = [IO[(s, c)] for s in oseeds]; n = len(rs)
    cs, gs = sum(r['clean_success'] for r in rs), sum(r['goal_success'] for r in rs)
    hb = [held_bug(r) for r in rs]
    print('%-11s clean %d/%d %s goal %d/%d %s clob/ep %.2f ling/ep %.2f ktok %.0f capped %d parse_fail %d err %d | held-bug dones %d in %d eps | reported clean %d/24 goal %d/24 clob %.2f ktok %.0f' % (
        c, cs, n, cp(cs, n), gs, n, cp(gs, n), sum(r['clobbers'] for r in rs) / n, sum(r['lingering'] for r in rs) / n, sum(r['prompt_chars'] for r in rs) / 4000 / n,
        sum(r['step_cap_hits'] for r in rs), sum(r['parse_fail'] for r in rs), sum(bool(r.get('err') or r.get('llm_errors')) for r in rs), sum(hb), sum(b > 0 for b in hb),
        sum(r['clean_success'] for r in os_), sum(r['goal_success'] for r in os_), sum(r['clobbers'] for r in os_) / 24, sum(r['prompt_chars'] for r in os_) / 4000 / 24))
    print('   per-K clean', {K: sum(I[(s, c)]['clean_success'] for s in seeds if I[(s, c)]['K'] == K) for K in (4, 6, 8)}, 'models', Counter(r['model'] for r in rs))
print('row keys with err:', sorted({k for r in M for k in r if 'err' in k.lower()}))
