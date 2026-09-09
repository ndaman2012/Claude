import pandas as pd, numpy as np, json, math

SRC='/root/.claude/uploads/f9486914-e5f8-53a6-a977-32670ff1eb07/129687de-nba_player_season_totals.csv'
d=pd.read_csv(SRC)
d=d[d.SEASON_TYPE=='Regular Season'].copy()
d['YR']=d.SEASON.str[:4].astype(int)

# ---- per-game / per-36 rate construction -------------------------------
COUNT=['PTS','REB','OREB','DREB','AST','TOV','STL','BLK','FG3M','FG3A','FGM','FGA','FTM','FTA','PF']
d['GP']=d.GP.astype(float)
d['MPG']=d.MIN/d.GP
for c in COUNT:
    d[c]=d[c].astype(float)
    d[c+'_36']=np.where(d.MIN>0, d[c]/d.MIN*36.0, 0.0)
    d[c+'_G']=d[c]/d.GP
# efficiency
d['TS']=np.where((d.FGA+0.44*d.FTA)>0, d.PTS/(2*(d.FGA+0.44*d.FTA)), 0.0)
d['EFG']=np.where(d.FGA>0,(d.FGM+0.5*d.FG3M)/d.FGA,0.0)
d['FT_PCT']=np.where(d.FTA>0, d.FTM/d.FTA, 0.0)
d['FG3_PCT']=np.where(d.FG3A>0, d.FG3M/d.FG3A, 0.0)
d['FG_PCT']=np.where(d.FGA>0, d.FGM/d.FGA, 0.0)
d['USG']=np.where(d.MIN>0,(d.FGA+0.44*d.FTA+d.TOV)/d.MIN*36.0,0.0)  # usage proxy per36
d['PM_36']=np.where(d.MIN>0, d.PLUS_MINUS/d.MIN*36.0, 0.0)

d=d.sort_values(['PLAYER_ID','YR'])

# league games available per season (for GP context)
LG_GP={2019:72,2020:72}  # covid-shortened
def sched(y): return LG_GP.get(y,82)

# ---- age curve: empirical YoY multiplier of per-36 production by age ----
RATE36=['PTS_36','REB_36','AST_36','STL_36','BLK_36','FG3M_36','TOV_36','FGA_36','FTA_36','FG3A_36','OREB_36','DREB_36','FGM_36','FTM_36']
nxt=d.copy()
nxt['YR']=nxt.YR-1
pairs=d.merge(nxt, on=['PLAYER_ID','YR'], suffixes=('','_N'))
pairs=pairs[(pairs.MIN>=300)&(pairs.MIN_N>=300)]

age_curve={}
for a in range(19,42):
    sub=pairs[pairs.AGE.round()==a]
    if len(sub)<15:
        age_curve[a]=None; continue
    row={}
    for c in RATE36+['TS','MPG']:
        cur=sub[c].values; nx=sub[c+'_N'].values
        m=cur>0.05 if c not in('TS','MPG') else cur>0
        if m.sum()<10: row[c]=1.0; continue
        r=np.clip(nx[m]/cur[m],0.4,2.0)
        row[c]=float(np.median(r))
    age_curve[a]=row
# fill gaps
keys=[k for k,v in age_curve.items() if v]
for a in range(19,42):
    if not age_curve[a]:
        near=min(keys,key=lambda k:abs(k-a)); age_curve[a]=age_curve[near]

# ---- similarity feature space -------------------------------------------
FEAT=['MPG','PTS_36','REB_36','AST_36','TOV_36','STL_36','BLK_36','FG3A_36','FTA_36','FGA_36','TS','FG3_PCT','FT_PCT','USG']
pool=d[(d.MIN>=400)&(d.GP>=20)].copy()
mu=pool[FEAT].mean(); sd=pool[FEAT].std().replace(0,1)
for c in FEAT: d['z_'+c]=(d[c]-mu[c])/sd[c]
ZF=['z_'+c for c in FEAT]

# feature weights for the distance metric
W=np.array([1.2,1.5,1.2,1.2,0.7,0.7,0.8,1.0,0.8,1.0,1.0,0.6,0.5,1.0])
W=W/W.sum()*len(W)

# historical comp pool: season s with a season s+1 available
hist=d[(d.YR<=2023)&(d.MIN>=400)&(d.GP>=20)].copy()
nx2=d.set_index(['PLAYER_ID','YR'])
def get_next(pid,yr):
    try: return nx2.loc[(pid,yr+1)]
    except KeyError: return None

hist=hist.reset_index(drop=True)
hidx={}
for i,r in hist.iterrows(): hidx[i]=(r.PLAYER_ID,r.YR)
HZ=hist[ZF].values
HAGE=hist.AGE.values

# next-season records for the comp pool
key=set(zip(d.PLAYER_ID,d.YR))
nxt_map=d.set_index(['PLAYER_ID','YR'])
has_next=np.array([ (p,y+1) in key for p,y in zip(hist.PLAYER_ID,hist.YR) ])
hist2=hist[has_next].reset_index(drop=True)
HZ=hist2[ZF].values; HAGE=hist2.AGE.values

nrows=nxt_map.loc[list(zip(hist2.PLAYER_ID,hist2.YR+1))]
DELTA=['MPG','PTS_36','REB_36','AST_36','STL_36','BLK_36','FG3M_36','FG3A_36','TOV_36','FGA_36','FTA_36','FGM_36','FTM_36','OREB_36','DREB_36','TS','FG3_PCT','FT_PCT','FG_PCT']
ratios={}
for c in DELTA:
    cur=hist2[c].values.astype(float); nx=nrows[c].values.astype(float)
    with np.errstate(divide='ignore',invalid='ignore'):
        r=np.where(cur>1e-6, nx/cur, 1.0)
    r=np.clip(np.nan_to_num(r,nan=1.0,posinf=1.0),0.25,3.0)
    ratios[c]=r
# durability: next-season games played fraction of schedule
gp_next=nrows.GP.values.astype(float)
sch_next=np.array([sched(y+1) for y in hist2.YR])
gp_frac=np.clip(gp_next/sch_next,0,1)
gp_frac_cur=np.clip(hist2.GP.values/np.array([sched(y) for y in hist2.YR]),0,1)

# ---- targets: each player's most recent season (2024-25 or 2025-26) ------
recent=d[(d.YR>=2024)&(d.GP>=10)&(d.MIN>=200)].copy()
cur=recent.sort_values('YR').groupby('PLAYER_ID').tail(1).reset_index(drop=True)
print('targets',len(cur),'from 2025-26:',(cur.YR==2025).sum(),'from 2024-25:',(cur.YR==2024).sum())

TZ=cur[ZF].values
K_STORE=25
out_players=[]
for i in range(len(cur)):
    r=cur.iloc[i]
    diff=(HZ-TZ[i])*W
    dist=np.sqrt((diff**2).sum(axis=1))
    agepen=np.abs(HAGE-r.AGE)
    # exclude the player's own seasons
    mask=(hist2.PLAYER_ID.values!=r.PLAYER_ID)&(agepen<=2.5)
    dtot=dist+0.35*agepen
    dtot=np.where(mask,dtot,1e9)
    order=np.argsort(dtot)[:K_STORE]
    comps=[]
    for j in order:
        if dtot[j]>=1e8: continue
        sim=round(min(99.0,float(100*math.exp(-max(0.0,dtot[j]-1.1)/2.2))),1)
        c=hist2.iloc[j]
        comps.append({
          'n':c.PLAYER_NAME,'s':c.SEASON,'a':round(float(c.AGE)),'sim':sim,
          'd':{k:round(float(ratios[k][j]),3) for k in DELTA},
          'gpf':round(float(gp_frac[j]),3),'gpfc':round(float(gp_frac_cur[j]),3),
          'now':[round(float(c.MPG),1),round(float(c.PTS_36),1),round(float(c.REB_36),1),round(float(c.AST_36),1)],
          'nxt':[round(float(nrows.iloc[j].MPG),1),round(float(nrows.iloc[j].PTS_36),1),round(float(nrows.iloc[j].REB_36),1),round(float(nrows.iloc[j].AST_36),1)]
        })
    # last 3 seasons of the target
    lastYr=int(r.YR)
    histrows=d[(d.PLAYER_ID==r.PLAYER_ID)&(d.YR>=lastYr-2)&(d.YR<=lastYr)].sort_values('YR')
    seasons=[]
    for _,h in histrows.iterrows():
        seasons.append({
          'season':h.SEASON,'yr':int(h.YR),'age':round(float(h.AGE),1),'gp':int(h.GP),
          'mpg':round(float(h.MPG),2),'team':h.TEAM_ABBREVIATION,
          'r36':{k:round(float(h[k]),3) for k in DELTA if k!='MPG'},
          'pg':{k.replace('_36','_G'):round(float(h[k.replace('_36','_G')]),3) for k in DELTA if k not in('MPG','TS','FG3_PCT','FT_PCT','FG_PCT')},
          'gpfrac':round(float(min(h.GP/sched(int(h.YR)),1)),3)
        })
    out_players.append({
      'id':int(r.PLAYER_ID),'name':r.PLAYER_NAME,'team':r.TEAM_ABBREVIATION,
      'lastYr':lastYr,'gap':2025-lastYr,
      'age26':round(float(r.AGE)+(2026-lastYr),1),'age':round(float(r.AGE),1),
      'seasons':seasons,'comps':comps
    })

payload={'generated':'2026-09','ageCurve':{str(k):{kk:round(vv,4) for kk,vv in v.items()} for k,v in age_curve.items()},
         'players':sorted(out_players,key=lambda p:p['name']),
         'stats':DELTA}
with open('/tmp/claude-0/-home-user-Claude/f9486914-e5f8-53a6-a977-32670ff1eb07/scratchpad/data.json','w') as f:
    json.dump(payload,f,separators=(',',':'))
import os
print('json MB', os.path.getsize('/tmp/claude-0/-home-user-Claude/f9486914-e5f8-53a6-a977-32670ff1eb07/scratchpad/data.json')/1e6)
print({a:age_curve[a]['PTS_36'] for a in [21,24,27,30,33,36]})
