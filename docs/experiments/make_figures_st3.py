import json,csv,sys,numpy as np,matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
# mid.jsonl + mid_galactus.jsonl from ~/ippo_runs/exp_ST3/logs; final pool/bp5 from st3_final and st2_final (st2_*.jsonl)
D=sys.argv[1] if len(sys.argv)>1 else 'st3_raw'
M=[json.loads(l) for l in open(D+'/mid.jsonl')]
G={}
for l in open(D+'/mid_galactus.jsonl'):
  if l.strip():d=json.loads(l);G[d['steps']]=d['galactus_200']
with open('data/st3_mid.csv','w') as f:
  w=csv.writer(f);w.writerow(['steps','h2h_BEST_200','h2h_FR4_200','galactus_200'])
  for m in M:w.writerow([m['steps'],m['h2h_BEST_200'],m['h2h_FR4_200'],G.get(m['steps'],'')])
def pool(p):
  r={}
  for l in open(p):
    if l.strip():d=json.loads(l);r[d['opponent']]=100*d['win_rate_decisive']
  return r
x=np.array([m['steps']/1e6 for m in M])
fig,ax=plt.subplots(1,2,figsize=(13,4.5),gridspec_kw={'width_ratios':[3,2]})
ax[0].plot(x,[100*m['h2h_BEST_200'] for m in M],'o-',label='vs ST2 5511.6M (BEST)')
ax[0].plot(x,[100*m['h2h_FR4_200'] for m in M],'s-',label='vs FR4-final')
gx=sorted(G);ax[0].plot([s/1e6 for s in gx],[100*G[s] for s in gx],'^-',color='C3',label='vs Galactus')
ax[0].axhline(50,color='k',ls=':');ax[0].axhline(52,color='r',ls=':',label='promotion bar 52%')
ax[0].axvline(5771.6,color='C2',ls='--',label='finals ckpt 5771.6M: 53.4% / 69.1% / Galactus 20.4% (1000 g)')
ax[0].set_xlabel('steps (M)');ax[0].set_ylabel('win % (200 games, ±3.5)');ax[0].legend(fontsize=8);ax[0].grid(alpha=.3)
ax[0].set_title('ST3 mid-run h2h after each 50M chunk',fontsize=10)
a=pool(D+'/st2_pool400.jsonl');a.update(pool(D+'/st2_bp5.jsonl'));b=pool(D+'/pool400.jsonl');b.update(pool(D+'/bp5.jsonl'))
k=list(a);i=np.arange(len(k))
ax[1].bar(i-.2,[a[o] for o in k],.4,label='ST2 5511.6M');ax[1].bar(i+.2,[b[o] for o in k],.4,label='ST3 5771.6M')
ax[1].set_xticks(i);ax[1].set_xticklabels(k,rotation=45,fontsize=8);ax[1].set_ylabel('win % (400 games)');ax[1].legend(fontsize=8);ax[1].grid(alpha=.3,axis='y')
ax[1].set_title('pool400 + botpack5',fontsize=10)
plt.tight_layout();plt.savefig('figures/st3_mid.png',dpi=120)
