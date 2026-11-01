import json,csv,sys,numpy as np,matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
D=sys.argv[1] if len(sys.argv)>1 else 'fr7_raw'  # mid.jsonl from ~/ippo_runs/exp_FR7/logs
M=[json.loads(l) for l in open(D+'/mid.jsonl')]
S=['kickoff','random','replay','air_reset','reset_curriculum']
with open('data/fr7_mid.csv','w') as f:
  w=csv.writer(f);w.writerow(['steps','episodes','resets','uses','goals_after_reset']+[f'{s}_{k}' for s in S for k in ('resets','uses','goals_after_reset')]+['h2h_FR4_200'])
  for m in M:
    r=m['rates'];w.writerow([m['steps']]+[r['all'][k] for k in ('episodes','resets','uses','goals_after_reset')]+[r[s][k] for s in S for k in ('resets','uses','goals_after_reset')]+[m['h2h_FR4_200']])
x=np.array([m['steps']/1e6 for m in M])
fig,ax=plt.subplots(2,1,figsize=(9,6),sharex=True)
ax[0].plot(x,[m['h2h_FR4_200']*100 for m in M],'o-');ax[0].axhline(50,color='k',ls=':');ax[0].axhline(48,color='r',ls=':',label='keep bar 48%')
ax[0].axhline(47.2,color='C2',ls='--',label='final, 1000 games: 47.2%');ax[0].set_ylabel('h2h vs FR4-final (%)\n200 games, ±3.5');ax[0].legend(fontsize=8);ax[0].grid(alpha=.3)
play=[sum(m['rates'][s]['uses'] for s in ('kickoff','random','replay')) for m in M]
drill=[sum(m['rates'][s]['uses'] for s in ('air_reset','reset_curriculum')) for m in M]
ax[1].bar(x-8,[m['rates']['all']['resets'] for m in M],8,label='resets (all setters)')
ax[1].bar(x,drill,8,label='uses, drill setters');ax[1].bar(x+8,play,8,label='uses, kickoff/random/replay')
for xi,m in zip(x,M):
  if m['rates']['all']['goals_after_reset']:ax[1].annotate('goal',(xi,m['rates']['all']['resets']+.2),ha='center',fontsize=8,color='C3')
ax[1].set_ylabel('count per 150k steps\n(rates_fr7.toml)');ax[1].set_xlabel('steps (M)');ax[1].legend(fontsize=8);ax[1].grid(alpha=.3)
ax[0].set_title('FR7 mid-run checks after each 50M chunk + BC refresh',fontsize=10)
plt.tight_layout();plt.savefig('figures/fr7_mid.png',dpi=120)
