import json,csv,sys,numpy as np,matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
D=sys.argv[1] if len(sys.argv)>1 else 'fr6_raw'  # metrics.jsonl + mid.jsonl from ~/ippo_runs/exp_FR6/logs, curr.jsonl from ~/ippo_runs/fr6_final
R=[json.loads(l) for l in open(D+'/metrics.jsonl')]
M=[json.loads(l) for l in open(D+'/mid.jsonl')]
C=[json.loads(l) for l in open(D+'/curr.jsonl')]
keys=sorted({k for r in R for k in r if k.startswith(('Events/','Curriculum/','Setters/','Player/','Loss/','RewardAbs/','Throughput/'))})
with open('data/metrics_exp_FR6.csv','w') as f:
  w=csv.writer(f);w.writerow(['steps']+keys)
  for i in range(0,len(R)-9,10):
    c=R[i:i+10];w.writerow([int(c[-1]['Cumulative/steps'])]+[f"{np.nanmean([r.get(k,np.nan) for r in c]):.6g}" for k in keys])
with open('data/fr6_mid.csv','w') as f:
  w=csv.writer(f);w.writerow(['steps','episodes','resets','uses','goals_after_reset','h2h_FR4_200'])
  for m in M:r=m['rates'];w.writerow([m['steps'],r['episodes'],r['resets'],r['uses'],r['goals_after_reset'],m['h2h_FR4_200']])
x=[m['steps']/1e6 for m in M]
fig,ax=plt.subplots(2,1,figsize=(9,6),sharex=True)
ax[0].plot(x,[m['h2h_FR4_200']*100 for m in M],'o-');ax[0].axhline(50,color='k',ls=':');ax[0].axhline(48,color='r',ls=':',label='keep bar 48%')
ax[0].axhline(48.0,color='C2',ls='--',label='final, 1000 games: 48.0%');ax[0].set_ylabel('h2h vs FR4-final (%)\n200 games, ±3.5');ax[0].legend(fontsize=8);ax[0].grid(alpha=.3)
ax[1].bar(np.array(x)-6,[m['rates']['resets'] for m in M],12,label='resets');ax[1].bar(np.array(x)+6,[m['rates']['uses'] for m in M],12,label='uses');ax[1].set_ylabel('count per 100k steps\n(rates_fr3.toml)');ax[1].set_xlabel('steps (M)');ax[1].legend(fontsize=8);ax[1].grid(alpha=.3)
ax[0].set_title('FR6 mid-run checks after each 50M chunk + BC refresh (0 goals after a reset in every check)',fontsize=10)
plt.tight_layout();plt.savefig('figures/fr6_mid.png',dpi=120)
g={(c['level'],c['policy'].split(':')[0]):c['success']*100 for c in C}
exp=[53.5,44.5,15.4];noin=[17.9,10.2,3.1];fr5=[27.4,18.4,8.5]
xs=np.arange(3);w=.18;plt.figure(figsize=(7.5,4))
for i,(v,l) in enumerate(((exp,'scripted expert'),([g[(k,'greedy')] for k in range(3)],'FR6-final greedy'),([g[(k,'sample')] for k in range(3)],'FR6-final sampled'),(fr5,'FR5-final greedy'),(noin,'no input'))):
  b=plt.bar(xs+(i-2)*w,v,w,label=l);plt.bar_label(b,fmt='%.1f',fontsize=6)
plt.xticks(xs,['L0','L1','L2']);plt.ylabel('earned reset success (%)');plt.title('FR6-final (2631.6M) by curriculum level, 960 eps each');plt.legend(fontsize=7);plt.tight_layout();plt.savefig('figures/fr6_levels.png',dpi=120)
def sm(k,w=40):
  s=np.array([r['Cumulative/steps'] for r in R])/1e6;y=np.array([r.get(k,np.nan) for r in R],float);return s[w-1:],np.convolve(y,np.ones(w)/w,'valid')
ks=['Curriculum/success','Events/Flip Resets per min','Loss/entropy','Events/Goals per min','Player/In Air Ratio','Throughput/overall']
fig,ax=plt.subplots(2,3,figsize=(11,5.5))
for a,k in zip(ax.flat,ks):
  a.plot(*sm(k));a.set_title(k,fontsize=9);a.grid(alpha=.3)
  for m in M:a.axvline(m['steps']/1e6,color='gray',lw=.4,alpha=.4)
fig.suptitle('FR6 training (40-iteration means; grey = BC refresh + mid eval)');plt.tight_layout();plt.savefig('figures/fr6_training.png',dpi=120)
