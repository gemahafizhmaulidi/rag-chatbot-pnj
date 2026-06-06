"""eval_router_prod.py — akurasi router.py PRODUKSI (LLM-router) di routing_testset."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import router

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cases = json.load(open(os.path.join(BASE, 'data', 'routing_testset.json'), encoding='utf-8'))['cases']
LABELS = router.ROUTES

rows, ok = [], 0
for c in cases:
    d = router.route_query(c['query'])
    got = d['route']
    correct = got == c['expected']
    ok += correct
    rows.append((c['id'], c['expected'], got, correct, d.get('search_query', ''), d.get('stat_subtype', ''), c['query']))
    if not correct:
        print(f'  X {c["id"]:<6} exp={c["expected"]:<14} got={got:<14} | {c["query"]}')

print(f'\nAkurasi router.py (produksi): {ok}/{len(cases)} = {ok/len(cases):.1%}')
for lab in LABELS:
    s = [r for r in rows if r[1] == lab]
    if s:
        a = sum(r[3] for r in s)
        print(f'  {lab:<16}: {a}/{len(s)} = {a/len(s):.0%}')
