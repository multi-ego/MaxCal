"""Walkthrough: are low-barrier folding times Poissonian, what does the tilt do to the
KS test, and what does (and does not) fix lambda?  Run from the repository root:
    python examples/walkthrough_ks_lambda.py
"""
import sys; import os
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0]=[ROOT, os.path.join(ROOT,'tests')]
import numpy as np, maxcal as mp, langevin as L
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
OUT='walkthrough_ks_lambda.png'
QU,QF,QTS=0.3,0.8,0.40
KW=dict(dt=0.002,sigma=0.06,stride=10)
def trajs(Qs,dt):
    out=[]
    for i,q in enumerate(Qs):
        f,fa=mp.parse_traj(q,QU,QF); out.append(dict(path=str(i),q=q,dt=dt,n=q.size,fold=f,fails=fa))
    return out
Qm,dt=L.run_many(200,1,dB=0.0,**KW); Qt,_=L.run_many(200,2,dB=3.0,**KW)
M,Tr=trajs(Qm,dt),trajs(Qt,dt)
A=mp.attempts(M,QTS,0,False); k,T,d=A['k'],A['T'],A['done']; p=mp.success_prob(k,d)
At=mp.attempts(Tr,QTS,0,False); pt=mp.success_prob(At['k'],At['done'])
lam_true=np.log(p/(1-p))-np.log(pt/(1-pt))
def ksp(t): return mp.lilliefors_p(mp.D_rows(np.asarray(t)[None,:])[0], len(t))
print(f"STEP 1  model folding times: N={T.size} CV={T.std()/T.mean():.2f} KS p={ksp(T):.4f}")
print(f"        p per attempt at Q‡={QTS}: {p:.3f}   (truth: {pt:.3f} -> true lambda {lam_true:.2f} kT)")
lams=np.round(np.arange(0,3.01,0.25),2)
rng=np.random.default_rng(0)
rows=[]
print("\nSTEP 2  lambda | reweighting: CV  N_eff  KS p   | stitching: p'    CV   median KS p")
for lam in lams:
    w=np.exp(mp.log_weights(lam,k,d,p)[0]); neff=1/np.sum(w**2)
    cv,_=mp.cv_curve(np.array([lam]),k,T,d,p)
    Dw,_=mp.ks_exp_weighted(T,d,w); kw=mp.lilliefors_p(Dw,round(neff))
    pp=mp.tilted_p(p,lam); ts=mp.stitch(pp,A['fail_pool'],A['succ_pool'],20000,rng)
    sub=rng.choice(ts,size=(300,T.size)); null=mp.null_D(T.size)
    pv=(np.sum(null[None,:]>=mp.D_rows(sub)[:,None],axis=1)+1)/(null.size+1)
    rows.append((lam,cv[0],neff,kw,pp,ts.std()/ts.mean(),np.median(pv)))
    print(f"        {lam:5.2f}  |          {cv[0]:.2f}  {neff:5.1f}  {kw:.4f} |          {pp:.3f}  {ts.std()/ts.mean():.2f}   {np.median(pv):.3f}")
R=np.array(rows)
ok=(R[:,6]>=0.05)&(np.abs(R[:,5]-1)<=0.1)
lam_min=R[np.argmax(ok),0] if ok.any() else np.nan
print(f"\nlambda_min (first passing) = {lam_min}  vs true lambda = {lam_true:.2f}")
np.save('/tmp/walk.npy',R)
# figure
fig,axs=plt.subplots(2,2,figsize=(12,8.5)); axs=axs.ravel()
x=np.linspace(0,5,200)
def surv(ax,t,lab,**kw):
    s=np.sort(t)/np.mean(t); ax.step(s,1-np.arange(1,s.size+1)/s.size,where='post',label=lab,**kw)
surv(axs[0],T,f'model (low barrier): KS p = {ksp(T):.3f}')
axs[0].semilogy(x,np.exp(-x),'k--',lw=1,label='exponential (Poisson)')
axs[0].set_ylim(1e-2,1.05);axs[0].set_xlim(0,5);axs[0].set_xlabel('t / ⟨t⟩');axs[0].set_ylabel('survival');axs[0].legend(fontsize=8)
axs[0].set_title('Step 1: raw folding times are not Poissonian')
axs[1].plot(R[:,0],R[:,3],'o-',label='reweighting (KS p)')
axs[1].plot(R[:,0],R[:,6],'s-',label='stitching (median KS p)')
axs[1].axhline(0.05,color='r',ls=':',lw=1,label='α = 0.05')
axs[1].axvline(lam_min,color='C2',ls='--',label=f'λ_min = {lam_min:.2f}')
axs[1].axvline(lam_true,color='k',ls='-',lw=1,label=f'true λ = {lam_true:.2f}')
axs[1].set_yscale('log');axs[1].set_ylim(1e-4,1.2);axs[1].set_xlabel('λ (kT)');axs[1].set_ylabel('KS p-value');axs[1].legend(fontsize=8)
axs[1].set_title('Step 2: KS p-value as λ increases')
surv(axs[2],T,'model, λ = 0',color='0.6')
for lam,c in [(lam_min,'C2'),(lam_true,'C3')]:
    ts=mp.stitch(mp.tilted_p(p,lam),A['fail_pool'],A['succ_pool'],20000,np.random.default_rng(1))
    surv(axs[2],ts,f'stitched, λ = {lam:.2f}',color=c)
surv(axs[2],At['T'][At['done']],'truth (higher barrier)',color='k',lw=1.5)
axs[2].semilogy(x,np.exp(-x),'k--',lw=1)
axs[2].set_ylim(1e-2,1.05);axs[2].set_xlim(0,5);axs[2].set_xlabel('t / ⟨t⟩');axs[2].legend(fontsize=8)
axs[2].set_title('Step 3a: shape (t/⟨t⟩) — every λ ≥ λ_min is Poissonian')
Tt0=At['T'][At['done']]
def surv_abs(ax,t,lab,**kw):
    s_=np.sort(t); ax.step(s_,1-np.arange(1,s_.size+1)/s_.size,where='post',label=lab,**kw)
surv_abs(axs[3],T,f'model, λ = 0 (⟨t⟩={T.mean():.0f})',color='0.6')
for lam,c in [(lam_min,'C2'),(lam_true,'C3')]:
    ts=mp.stitch(mp.tilted_p(p,lam),A['fail_pool'],A['succ_pool'],20000,np.random.default_rng(1))
    surv_abs(axs[3],ts,f'stitched, λ = {lam:.2f} (⟨t⟩={ts.mean():.0f})',color=c)
surv_abs(axs[3],Tt0,f'truth (⟨t⟩={Tt0.mean():.0f})',color='k',lw=1.5)
axs[3].set_yscale('log');axs[3].set_ylim(1e-2,1.05);axs[3].set_xlim(0,150)
axs[3].set_xlabel('folding time (model units)');axs[3].set_ylabel('survival');axs[3].legend(fontsize=8)
axs[3].set_title('Step 3b: absolute time — only the true λ matches the truth')
plt.tight_layout(); plt.savefig(OUT,dpi=130)
# absolute times, to show what lambda changes
Tt=At['T'][At['done']]
for lam in [lam_min,lam_true]:
    ts=mp.stitch(mp.tilted_p(p,lam),A['fail_pool'],A['succ_pool'],20000,np.random.default_rng(1))
    print(f"mean folding time: stitched λ={lam:.2f}: {ts.mean():.1f}   truth: {Tt.mean():.1f}   model: {T.mean():.1f}")
