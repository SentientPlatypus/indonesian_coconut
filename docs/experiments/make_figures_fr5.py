import json,csv,numpy as np,matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
import sys
D=sys.argv[1] if len(sys.argv)>1 else 'fr5_raw'  # dir with metrics.jsonl, bc_refresh.jsonl from ~/ippo_runs/exp_FR5/logs
R=[json.loads(l) for l in open(D+'/metrics.jsonl')]
S=[r['Cumulative/steps'] for r in R]
cut=next(i for i in range(1,len(S)) if S[i]<S[i-1]); A,B=R[:cut],R[cut:]
print('attempt1',S[0],S[cut-1],'restart',S[cut],S[-1],len(B))
def sm(rs,k,w=25):
  x=np.array([r['Cumulative/steps'] for r in rs])/1e6;y=np.array([r.get(k,np.nan) for r in rs],float)
  return x[w-1:],np.convolve(y,np.ones(w)/w,'valid')
keys=sorted({k for r in R for k in r if k.startswith(('Events/','Curriculum/','Setters/','Player/','Loss/','RewardAbs/','Throughput/'))})
with open('data/metrics_exp_FR5.csv','w') as f:
  w=csv.writer(f);w.writerow(['attempt','steps']+keys)
  for name,rs in (('plain',A),('chunked',B)):
    for i in range(0,len(rs)-4,5):
      c=rs[i:i+5];w.writerow([name,int(c[-1]['Cumulative/steps'])]+[f"{np.nanmean([r.get(k,np.nan) for r in c]):.6g}" for k in keys])
bc=[json.loads(l) for l in open(D+'/bc_refresh.jsonl')]
with open('data/fr5_bc_refresh.csv','w') as f:
  w=csv.writer(f);w.writerow(['steps','actor','policy','episodes','success','used_per_success'])
  for d in bc:w.writerow([d['steps_total'],d['actor'],d['policy'].split(':')[0],d['episodes'],round(d['success'],4),round(d['used_per_success'],4)])
plt.figure(figsize=(9,4.8))
for a,c in (('ppo','C0'),('bc','C1')):
  for p,ls in (('greedy','-'),('sample','--')):
    pts=[(d['steps_total']/1e6,d['success']*100) for d in bc if d['actor']==a and d['policy'].startswith(p)]
    plt.plot(*zip(*pts),ls,color=c,marker='o',ms=3,label=f'{a.upper()} actor, {p}')
plt.axhline(17.9,color='gray',ls=':',label='no input 17.9%');plt.axhline(53.5,color='k',ls=':',label='expert 52-55%');plt.axhline(1.9,color='r',ls=':',label='FR4-final greedy 1.9%')
plt.xlabel('steps (M)');plt.ylabel('level-0 earned reset success (%), 480 eps');plt.title('FR5: PPO actor after each 25M chunk vs after the 1500-step BC refresh',fontsize=10);plt.legend(fontsize=7,ncol=2);plt.grid(alpha=.3);plt.tight_layout();plt.savefig('figures/fr5_bc_refresh.png',dpi=120)
fig,ax=plt.subplots(3,1,figsize=(8,8),sharex=True)
for rs,lab,c in ((A,'plain PPO (stopped)','C3'),(B,'chunked PPO + BC refresh','C0')):
  ax[0].plot(*sm(rs,'Curriculum/success @L0'),color=c,label=lab);ax[1].plot(*sm(rs,'Loss/entropy'),color=c);ax[2].plot(*sm(rs,'Setters/reset_curriculum share'),color=c)
for x in sorted({d['steps_total'] for d in bc}):
  for a in ax:a.axvline(x/1e6,color='gray',lw=.5,alpha=.5)
ax[0].set_ylabel('train L0 success\n(sampled, 25-it mean)');ax[0].legend(fontsize=8);ax[1].set_ylabel('policy entropy');ax[2].set_ylabel('reset_curriculum share');ax[2].set_xlabel('steps (M); grey = BC refresh');ax[0].set_title('FR5 training: curriculum level-0 success (level never left 0)')
for a in ax:a.grid(alpha=.3)
plt.tight_layout();plt.savefig('figures/fr5_training_curriculum.png',dpi=120)
exp=[53.5,44.5,15.4];noin=[17.9,10.2,3.1];g=[27.4,18.4,8.5];s=[19.1,12.3,4.6]
x=np.arange(3);w=.2;plt.figure(figsize=(7,4))
for i,(v,l) in enumerate(((exp,'scripted expert'),(g,'FR5-final greedy'),(s,'FR5-final sampled'),(noin,'no input'))):
  b=plt.bar(x+(i-1.5)*w,v,w,label=l);plt.bar_label(b,fmt='%.1f',fontsize=7)
plt.xticks(x,['L0','L1','L2']);plt.ylabel('earned reset success (%)');plt.title('FR5-final (1731.6M) by curriculum level, 960 eps each');plt.legend(fontsize=8);plt.tight_layout();plt.savefig('figures/fr5_levels.png',dpi=120)
last=B[-20:];ab={k[10:]:np.mean([r[k] for r in last]) for k in last[0] if k.startswith('RewardAbs/')};tot=sum(ab.values())
it=sorted(ab.items(),key=lambda t:-t[1])[:18]
with open('data/reward_shares_FR5.csv','w') as f:
  w=csv.writer(f);w.writerow(['term','share_last20']);[w.writerow([k,f"{v/tot:.5f}"]) for k,v in sorted(ab.items(),key=lambda t:-t[1])]
plt.figure(figsize=(7,5));plt.barh([k for k,_ in it][::-1],[v/tot*100 for _,v in it][::-1]);plt.xlabel('% of |reward|, last 20 iterations');plt.title('FR5 reward share');plt.tight_layout();plt.savefig('figures/reward_share_exp_FR5.png',dpi=120)
print({k:round(v/tot*100,2) for k,v in it})
ks=['Events/Goals per min','Events/Aerial Goals per min','Events/Air Dribble Bumps per min','Events/Bumps per min','Events/Demos per min','Player/In Air Ratio']
fig,ax=plt.subplots(2,3,figsize=(11,5.5))
for a,k in zip(ax.flat,ks):
  for rs,c in ((A,'C3'),(B,'C0')):a.plot(*sm(rs,k),color=c)
  a.set_title(k,fontsize=9);a.grid(alpha=.3)
fig.suptitle('FR5 training events (red: plain PPO attempt, blue: chunked run)');plt.tight_layout();plt.savefig('figures/fr5_events.png',dpi=120)
l100=[r for r in B if r['Cumulative/steps']>=1631.6e6]
print({k:round(np.mean([r[k] for r in l100]),3) for k in ks+['Curriculum/success @L0','Throughput/overall','Setters/reset_curriculum share','Loss/entropy']})
