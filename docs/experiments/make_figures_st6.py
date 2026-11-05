import json,csv,sys,numpy as np,matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
# mid.jsonl from ~/ippo_runs/exp_ST6/logs; final pool/bp5 from st6_final and st5_final (st5_*.jsonl)
D=sys.argv[1] if len(sys.argv)>1 else 'st6_raw'
M=[json.loads(l) for l in open(D+'/mid.jsonl') if l.strip()]
with open('data/st6_mid.csv','w') as f:
  w=csv.writer(f);w.writerow(['steps','h2h_BEST_200','h2h_FR4_200','galactus_200'])
  for m in M:w.writerow([m['steps'],m.get('h2h_BEST_200',''),m.get('h2h_FR4_200',''),m.get('galactus_200','')])
def pool(p):
  r={}
  for l in open(p):
    if l.strip():d=json.loads(l);r[d['opponent']]=100*d['win_rate_decisive']
  return r
def s(k):
  P=[(m['steps']/1e6,100*m[k]) for m in M if m.get(k) is not None];return [p[0] for p in P],[p[1] for p in P]
fig,ax=plt.subplots(1,2,figsize=(13,4.5),gridspec_kw={'width_ratios':[3,2]})
ax[0].plot(*s('h2h_BEST_200'),'o-',label='vs ST5 6192.8M (BEST)')
ax[0].plot(*s('h2h_FR4_200'),'s-',label='vs FR4-final')
ax[0].plot(*s('galactus_200'),'^-',color='C3',label='vs Galactus')
ax[0].axhline(50,color='k',ls=':');ax[0].axhline(52,color='r',ls=':',label='promotion bar 52%')
ax[0].axvline(6954.5,color='C2',ls='--',label='finals ckpt 6954.5M: 56.2% / 71.2% / Galactus 23.8% (500 g)')
ax[0].set_xlabel('steps (M)');ax[0].set_ylabel('win % (200 games, ±3.5)');ax[0].legend(fontsize=8);ax[0].grid(alpha=.3)
ax[0].set_title('ST6 mid-run h2h after each 50M chunk',fontsize=10)
a=pool(D+'/st5_pool400.jsonl');a.update(pool(D+'/st5_bp5.jsonl'));b=pool(D+'/pool400.jsonl');b.update(pool(D+'/bp5.jsonl'))
k=list(a);i=np.arange(len(k))
ax[1].bar(i-.2,[a[o] for o in k],.4,label='ST5 6192.8M');ax[1].bar(i+.2,[b.get(o,0) for o in k],.4,label='ST6 6954.5M')
ax[1].set_xticks(i);ax[1].set_xticklabels(k,rotation=45,fontsize=8);ax[1].set_ylabel('win % (400 games)');ax[1].legend(fontsize=8);ax[1].grid(alpha=.3,axis='y')
ax[1].set_title('pool400 + botpack5',fontsize=10)
plt.tight_layout();plt.savefig('figures/st6_mid.png',dpi=120)
