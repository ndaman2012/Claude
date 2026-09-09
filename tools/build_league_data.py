"""Join the league roster/contract export to the projection pool.

Reads the dynasty league CSV (rosters, salaries, contract years) and the
projection payload built by build_projection_data.py, matches players by a
normalised name (accents, suffixes, IR prefixes and initials removed, with a
fuzzy fallback), and writes league.json for embedding in the HTML app.
"""
import pandas as pd, json, unicodedata, re, difflib, math, sys

LEAGUE_CSV = sys.argv[1] if len(sys.argv) > 1 else 'leagueplayers202627.csv'
PROJ_JSON  = sys.argv[2] if len(sys.argv) > 2 else 'data.json'
OUT        = sys.argv[3] if len(sys.argv) > 3 else 'league.json'
ME         = 'N. Daman'

d = pd.read_csv(LEAGUE_CSV, encoding='utf-8-sig')
proj = json.load(open(PROJ_JSON))

def norm(s):
    s = unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode()
    s = re.sub(r'^ir[- ]', '', s.strip(), flags=re.I).lower()
    s = re.sub(r'\b(jr|sr|ii|iii|iv)\b', '', s)
    return re.sub(r'[^a-z]', '', s)

pmap = {}
for p in proj['players']:
    pmap.setdefault(norm(p['name']), p['id'])
keys = list(pmap)

def match(n):
    if n in pmap:
        return pmap[n]
    c = difflib.get_close_matches(n, keys, n=1, cutoff=0.9)
    return pmap[c[0]] if c else None

d['pid'] = d.player.map(norm).map(match)

def num(v):
    try:
        f = float(v)
        return None if math.isnan(f) else round(f, 2)
    except (TypeError, ValueError):
        return None

rows = [{
    'name': str(r.player).replace('IR-', '').strip(),
    'club': None if pd.isna(r.club) else str(r.club),
    'pos': None if pd.isna(r.pos) else str(r.pos),
    'pid': None if pd.isna(r.pid) else int(r.pid),
    's25': num(r['salary_2025-26']), 's26': num(r['salary_2026-27']),
    's27': num(r['salary_2027-28']), 's28': num(r['salary_2028-29']),
    'yrs': num(r.years_left), 'opt': None if pd.isna(r.option) else str(r.option),
    'acq': num(r.acquired), 'rating': num(r.rating),
    'ir': 1 if (isinstance(r.ir, str) and r.ir.strip()) else 0,
    'blk': 1 if (isinstance(r.on_block, str) and r.on_block.strip()) else 0,
    'fa': 1 if r.status == 'free agent' else 0,
} for _, r in d.iterrows()]

json.dump({'clubs': sorted({x['club'] for x in rows if x['club']}), 'me': ME, 'rows': rows},
          open(OUT, 'w'), separators=(',', ':'))
print(f"{d.pid.notna().sum()}/{len(d)} matched -> {OUT}")
