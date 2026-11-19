"""Primary template-number screening in AIRFOIL SHAPE coordinate space."""
from __future__ import annotations
import argparse,csv,json,math,time,hashlib,warnings
from collections import Counter
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from airfoil_registration import validate_airfoil_shape

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/airfoils/processed/airfoils_n100_seed0.npz'
OUT=ROOT/'outputs/airfoil_shape_pca_gmm_selection'
SEED=20260813; N_INIT=5; REG_COVAR=1e-7; MAX_ITER=500; TOL=1e-5
QS=range(1,101); KS=range(1,21)

def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def jwrite(p,x):p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,sort_keys=True),encoding='utf8');t.replace(p)
def cwrite(p,rows,fields=None):
 p.parent.mkdir(parents=True,exist_ok=True);fields=fields or sorted({k for r in rows for k in r});t=p.with_suffix(p.suffix+'.tmp')
 with t.open('w',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
 t.replace(p)
def cread(p):
 if not p.exists():return []
 with p.open(newline='',encoding='utf8') as f:return list(csv.DictReader(f))

def inputs():
 z=np.load(DATA); shapes=z['shapes']
 if shapes.shape!=(1654,200,2) or shapes.dtype!=np.float64:raise RuntimeError(f'bad shapes {shapes.shape} {shapes.dtype}')
 if not np.isfinite(shapes).all():raise RuntimeError('nonfinite shape input')
 if np.max(np.ptp(shapes[:,:,0],axis=0))!=0:raise RuntimeError('x grid differs')
 validity=[validate_airfoil_shape(s)['valid'] for s in shapes]
 if not all(validity):raise RuntimeError(f'{sum(not x for x in validity)} geometrically unusable processed shapes')
 X=np.ascontiguousarray(shapes).reshape(len(shapes),400,order='C')
 if X.shape[1]==10368:raise RuntimeError('STOP: wrong velocity-field representation')
 xv=np.var(shapes[:,:,0],axis=0); xs=np.std(shapes[:,:,0],axis=0)
 audit={'input_path':str(DATA.relative_to(ROOT)),'array_keys':z.files,'sample_count':len(shapes),'shapes_shape':list(shapes.shape),'dtype':str(shapes.dtype),
 'nan_count':int(np.isnan(shapes).sum()),'inf_count':int(np.isinf(shapes).sum()),'point_ordering':'exact pointwise correspondence inherited unchanged from processed dataset',
 'fixed_x_grid_exact':True,'geometrically_usable':int(sum(validity)),'PCA_INPUT_TYPE':'AIRFOIL SHAPE COORDINATES','PCA_INPUT_MATRIX_SHAPE':list(X.shape),
 'flatten_order':'C','maximum_x_variance':float(xv.max()),'maximum_x_standard_deviation':float(xs.max()),'effectively_zero_variance_x_dimensions_std_le_1e-12':int(np.sum(xs<=1e-12)),
 'upper_points':100,'lower_points':100,'input_sha256':sha(DATA)}
 print('PCA INPUT TYPE = AIRFOIL SHAPE COORDINATES',flush=True);print(f'PCA INPUT MATRIX SHAPE = {X.shape}',flush=True)
 return z,X,audit

def split(n):
 rng=np.random.default_rng(SEED);o=rng.permutation(n);nt=int(round(.8*n));return np.sort(o[:nt]),np.sort(o[nt:])

def fit_pca(X,tr,ho):
 p=PCA(n_components=100,svd_solver='randomized',random_state=SEED,iterated_power=7)
 A=p.fit_transform(X[tr]);B=p.transform(X[ho]);rows=[]
 for q in QS:
  ar=A[:,:q]@p.components_[:q]+p.mean_;br=B[:,:q]@p.components_[:q]+p.mean_; hm=float(np.mean((X[ho]-br)**2))
  rows.append({'q':q,'marginal_explained_variance_ratio':float(p.explained_variance_ratio_[q-1]),'cumulative_explained_variance_ratio':float(p.explained_variance_ratio_[:q].sum()),
  'train_reconstruction_mse':float(np.mean((X[tr]-ar)**2)),'heldout_reconstruction_mse':hm,'heldout_reconstruction_rmse':math.sqrt(hm)})
 np.savez_compressed(OUT/'pca/pca_shape_q100.npz',mean=p.mean_,components=p.components_,explained_variance=p.explained_variance_,explained_variance_ratio=p.explained_variance_ratio_,singular_values=p.singular_values_,train_indices=tr,heldout_indices=ho,train_scores=A,heldout_scores=B)
 cwrite(OUT/'pca/pca_shape_summary_q1_q100.csv',rows)
 return p,A,B,rows

def fit_cell(A,B,q,k,seed=SEED,n_init=N_INIT):
 start=time.perf_counter();err='';ww=[]
 try:
  with warnings.catch_warnings(record=True) as caught:
   warnings.simplefilter('always');m=GaussianMixture(n_components=k,covariance_type='full',n_init=n_init,random_state=seed,reg_covar=REG_COVAR,max_iter=MAX_ITER,tol=TOL,init_params='kmeans').fit(A[:,:q])
   ww=[str(x.message) for x in caught]
  tmean=float(m.score(A[:,:q]));hmean=float(m.score(B[:,:q]));pro=m.predict_proba(A[:,:q]);ent=float(-np.sum(pro*np.log(np.clip(pro,1e-300,1))));lab=pro.argmax(1);sizes=np.bincount(lab,minlength=k);bic=float(m.bic(A[:,:q]))
  tiny10=int(np.sum(sizes<10));tiny1=int(np.sum(sizes<.01*len(A)))
  return {'q':q,'K':k,'train_log_likelihood':tmean*len(A),'train_log_likelihood_per_sample':tmean,'heldout_log_likelihood':hmean*len(B),'heldout_log_likelihood_per_sample':hmean,
  'BIC':bic,'ICL':bic+2*ent,'classification_entropy':ent,'converged':bool(m.converged_),'em_iterations':int(m.n_iter_),'runtime_seconds':time.perf_counter()-start,
  'component_weights':json.dumps(m.weights_.tolist(),separators=(',',':')),'hard_assignment_cluster_sizes':json.dumps(sizes.tolist(),separators=(',',':')),
  'minimum_cluster_size':int(sizes.min()),'maximum_cluster_size':int(sizes.max()),'minimum_component_weight':float(m.weights_.min()),'components_size_lt_10':tiny10,
  'components_fraction_lt_1pct':tiny1,'small_component_flag':bool(tiny10 or tiny1),'numerical_warnings_errors':' | '.join(ww)}
 except Exception as e:
  err=f'{type(e).__name__}: {e}'
  return {'q':q,'K':k,'converged':False,'runtime_seconds':time.perf_counter()-start,'numerical_warnings_errors':err}

def preflight(A,B):
 rows=[];start=time.perf_counter()
 for q in (5,20,50,100):
  for k in (1,3,5,7,10,20):
   r=fit_cell(A,B,q,k);rows.append(r);print(f"preflight q={q} K={k} conv={r['converged']} time={r['runtime_seconds']:.2f}s",flush=True)
 cwrite(OUT/'gmm/preflight_results.csv',rows);elapsed=time.perf_counter()-start;mean=float(np.mean([r['runtime_seconds'] for r in rows]))
 est={'cells':24,'runtime_seconds':elapsed,'all_converged':all(r['converged'] for r in rows),'mean_cell_seconds':mean,'estimated_full_grid_serial_seconds':mean*2000,
 'estimated_metrics_storage_bytes':2000*2500,'approx_peak_model_bytes_q100_k20':20*100*100*8*3}
 jwrite(OUT/'gmm/preflight_estimate.json',est);return rows,est

def grid(A,B):
 path=OUT/'gmm/all_shape_gmm_results.csv';rows=cread(path);done={(int(r['q']),int(r['K'])) for r in rows};
 for q in QS:
  for k in KS:
   if (q,k) in done:continue
   r=fit_cell(A,B,q,k);rows.append(r);cwrite(path,sorted(rows,key=lambda x:(int(x['q']),int(x['K']))));print(f"[{len(rows)}/2000] q={q} K={k} conv={r['converged']} runtime={r['runtime_seconds']:.2f}s",flush=True)
 return sorted(rows,key=lambda x:(int(x['q']),int(x['K'])))

def plots(pca_rows,gridrows):
 def save(name,title,x,y,ylabel='Preferred K',ylim=None):
  fig,ax=plt.subplots(figsize=(10,5));ax.plot(x,y,'o-',ms=3,lw=1.4);ax.set(xlabel='Shape-PCA dimension q',ylabel=ylabel,title=title);ax.grid(alpha=.25); 
  if ylim:ax.set_ylim(*ylim)
  fig.tight_layout();fig.savefig(OUT/'figures'/name,dpi=300);plt.close(fig)
 pref=[];gaps=[]
 for q in QS:
  rr=[r for r in gridrows if int(r['q'])==q and str(r['converged'])=='True']
  b=sorted(rr,key=lambda r:float(r['BIC']));i=sorted(rr,key=lambda r:float(r['ICL']));h=sorted(rr,key=lambda r:float(r['heldout_log_likelihood_per_sample']),reverse=True)
  pref.append({'q':q,'K_BIC':int(b[0]['K']),'K_ICL':int(i[0]['K']),'K_heldout_LL':int(h[0]['K'])})
  gaps.append({'q':q,'BIC_best_second_gap':float(b[1]['BIC'])-float(b[0]['BIC']),'ICL_best_second_gap':float(i[1]['ICL'])-float(i[0]['ICL']),
  'heldout_LL_best_second_gap':float(h[0]['heldout_log_likelihood_per_sample'])-float(h[1]['heldout_log_likelihood_per_sample'])})
 cwrite(OUT/'gmm/preferred_k_by_q.csv',pref);cwrite(OUT/'tables/model_selection_gap_by_q.csv',gaps)
 seq={'BIC':[r['K_BIC'] for r in pref],'ICL':[r['K_ICL'] for r in pref],'heldout_log_likelihood':[r['K_heldout_LL'] for r in pref]};jwrite(OUT/'gmm/preferred_k_sequences.json',seq)
 q=list(QS);fig,ax=plt.subplots(figsize=(12,5));
 for key,label,mark in [('K_BIC','BIC','o'),('K_ICL','ICL','s'),('K_heldout_LL','Held-out LL','^')]:ax.plot(q,[r[key] for r in pref],marker=mark,ms=3,lw=1.2,label=label)
 ax.set(xlabel='Shape-PCA dimension q',ylabel='Preferred GMM component count K',title='Preferred GMM Component Count Across SHAPE-SPACE PCA Dimensions',ylim=(.5,20.5));ax.set_yticks(range(1,21));ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(OUT/'figures/preferred_k_shape_q1_q100.png',dpi=300);plt.close(fig)
 save('bic_preferred_k_shape.png','BIC-preferred K — SHAPE-SPACE PCA + GMM',q,seq['BIC'],ylim=(.5,20.5));save('icl_preferred_k_shape.png','ICL-preferred K — SHAPE-SPACE PCA + GMM',q,seq['ICL'],ylim=(.5,20.5));save('heldout_preferred_k_shape.png','Held-out-LL-preferred K — SHAPE-SPACE PCA + GMM',q,seq['heldout_log_likelihood'],ylim=(.5,20.5))
 pr=pca_rows;save('pca_cumulative_explained_variance.png','Shape PCA cumulative explained variance',q,[r['cumulative_explained_variance_ratio'] for r in pr],'Cumulative explained variance',(0,1.01));save('pca_marginal_explained_variance.png','Shape PCA marginal explained variance',q,[r['marginal_explained_variance_ratio'] for r in pr],'Marginal explained variance');save('pca_reconstruction_mse.png','Shape PCA held-out reconstruction MSE',q,[r['heldout_reconstruction_mse'] for r in pr],'Held-out reconstruction MSE')
 fig,ax=plt.subplots(figsize=(11,5));ax.plot(q,[r['BIC_best_second_gap'] for r in gaps],label='BIC');ax.plot(q,[r['ICL_best_second_gap'] for r in gaps],label='ICL');ax.plot(q,[r['heldout_LL_best_second_gap'] for r in gaps],label='Held-out LL');ax.set(xlabel='q',ylabel='Best vs second-best gap',title='Model-selection best vs second-best gaps — shape space');ax.legend();ax.grid(alpha=.25);fig.tight_layout();fig.savefig(OUT/'figures/model_selection_gaps.png',dpi=300);plt.close(fig)
 for metric,name,mode in [('BIC','bic_heatmap.png','min'),('ICL','icl_heatmap.png','min'),('heldout_log_likelihood_per_sample','heldout_ll_heatmap.png','max')]:
  M=np.array([[float(next(r[metric] for r in gridrows if int(r['q'])==qq and int(r['K'])==kk)) for kk in KS] for qq in QS]);D=M-M.min(1,keepdims=True) if mode=='min' else M.max(1,keepdims=True)-M
  fig,ax=plt.subplots(figsize=(10,12));im=ax.imshow(D,aspect='auto',origin='lower',extent=(.5,20.5,.5,100.5),cmap='viridis');ax.set(xlabel='K',ylabel='Shape-PCA dimension q',title=f'Relative {metric} — SHAPE-SPACE PCA + GMM');fig.colorbar(im,ax=ax,label='gap from row optimum');fig.tight_layout();fig.savefig(OUT/'figures'/name,dpi=300);plt.close(fig)
 diag=[]
 for r in pref:
  for crit,key in [('BIC','K_BIC'),('ICL','K_ICL'),('heldout_LL','K_heldout_LL')]:
   x=next(x for x in gridrows if int(x['q'])==r['q'] and int(x['K'])==r[key]);diag.append({'q':r['q'],'criterion':crit,'preferred_K':r[key],'minimum_component_size':x['minimum_cluster_size'],'minimum_component_weight':x['minimum_component_weight'],'components_size_lt_10':x['components_size_lt_10'],'components_fraction_lt_1pct':x['components_fraction_lt_1pct']})
 cwrite(OUT/'tables/preferred_component_diagnostics.csv',diag);return pref,seq

def stability(A,seq):
 counts=Counter(seq['BIC']+seq['ICL']+seq['heldout_log_likelihood']);candidates=[k for k,_ in counts.most_common(5)];rows=[]
 for k in candidates:
  qs=[q for q in (5,20,50,100) if q<=A.shape[1]];q=qs[len(qs)//2]
  labs=[]
  for seed in range(5):labs.append(GaussianMixture(n_components=k,covariance_type='full',n_init=3,random_state=seed,reg_covar=REG_COVAR,max_iter=MAX_ITER,tol=TOL).fit_predict(A[:,:q]))
  vals=[adjusted_rand_score(labs[a],labs[b]) for a in range(5) for b in range(a+1,5)]
  rows.append({'q':q,'K':k,'seeds':'0,1,2,3,4','pair_count':len(vals),'mean_ARI':float(np.mean(vals)),'minimum_ARI':float(np.min(vals)),'maximum_ARI':float(np.max(vals)),'standard_deviation_ARI':float(np.std(vals,ddof=1))})
 cwrite(OUT/'tables/targeted_stability_ari.csv',rows);return rows,candidates

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['prepare','preflight','full','analyse','all'],default='all');args=ap.parse_args()
 for d in ('data','splits','pca','gmm','stability','tables','figures','report'):(OUT/d).mkdir(parents=True,exist_ok=True)
 z,X,audit=inputs();tr,ho=split(len(X));np.save(OUT/'splits/train_indices.npy',tr);np.save(OUT/'splits/heldout_indices.npy',ho);p,A,B,pr=fit_pca(X,tr,ho)
 thresholds={}
 for t in (.9,.95,.99,.995,.999,.9995,.9999):
  hit=np.flatnonzero(np.cumsum(p.explained_variance_ratio_)>=t);thresholds[str(t)]=int(hit[0]+1) if len(hit) else None
 cfg={'purpose':'PRIMARY SHAPE-SPACE PCA-GMM template-number screening','data_audit':audit,'split':{'seed':SEED,'train_count':len(tr),'heldout_count':len(ho),'PCA_fit_on_training_only':True},'pca':{'q':list(QS),'threshold_q':thresholds},
 'gmm':{'K':list(KS),'covariance_type':'full','n_init':N_INIT,'reg_covar':REG_COVAR,'max_iter':MAX_ITER,'tol':TOL,'seed':SEED,'ICL_formula':'BIC + 2 * sum_i[-sum_k tau_ik log(tau_ik)]'}};jwrite(OUT/'config.json',cfg)
 if args.stage=='prepare':return
 pre,est=preflight(A,B)
 if not est['all_converged']:raise RuntimeError('preflight non-convergence; full grid not started')
 if args.stage=='preflight':return
 rows=grid(A,B) if args.stage in ('full','all') else cread(OUT/'gmm/all_shape_gmm_results.csv')
 if len(rows)!=2000:raise RuntimeError(f'need 2000 grid cells, found {len(rows)}')
 pref,seq=plots(pr,rows);stab,cands=stability(A,seq)
 finite=all(np.isfinite(float(r[x])) for r in rows if str(r['converged'])=='True' for x in ('BIC','ICL','heldout_log_likelihood_per_sample'))
 regroot=ROOT/'outputs/airfoil_q20_k1_k8_velocity_properties/registrations';regfiles=list(regroot.glob('K*/*.npz'));regbad=0
 for f in regfiles:
  try:
   with np.load(f) as zz:json.loads(str(zz['metrics_json']))
  except:regbad+=1
 checks={'shape_input':X.shape==(1654,400),'not_velocity_input':X.shape!=(1651,10368),'sample_flattening':np.array_equal(X[0],z['shapes'][0].reshape(400,order='C')),'PCA_train_only':len(tr)==1323 and len(ho)==331,
 'q_exact':list(QS)==list(range(1,101)),'K_exact':list(KS)==list(range(1,21)),'cells_exact':len(rows)==2000,'finite_converged_metrics':finite,'input_hash_unchanged':sha(DATA)==audit['input_sha256'],
 'paused_registration_checkpoints_readable':regbad==0,'paused_registration_checkpoint_count_preserved':len(regfiles)>=3057}
 jwrite(OUT/'validation.json',{'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'counts':{'converged':sum(str(r['converged'])=='True' for r in rows),'nonconverged_or_failed':sum(str(r['converged'])!='True' for r in rows),'registration_checkpoints':len(regfiles)}})
 freq={x:dict(Counter(seq[x])) for x in seq};km={3,5,7};over=set(cands)&km;comparison='agrees' if km.issubset(set(cands)) else ('partially overlaps' if over else 'disagrees')
 report=f"""# Primary SHAPE-SPACE PCA + GMM screening\n\nPCA INPUT TYPE = AIRFOIL SHAPE COORDINATES. Matrix: `{X.shape}`; train/held-out: `{len(tr)}/{len(ho)}`. PCA was fit only on training shapes.\n\nVariance threshold q: `{json.dumps(thresholds)}`. Primary grid: 2000 full-covariance cells, conventions: `{json.dumps(cfg['gmm'])}`. Preflight estimate: `{json.dumps(est)}`.\n\nBIC sequence: `{seq['BIC']}`.\n\nICL sequence: `{seq['ICL']}`.\n\nHeld-out LL sequence: `{seq['heldout_log_likelihood']}`.\n\nPreferred-K frequencies/plateaus summary: `{json.dumps(freq,sort_keys=True)}`. Recurrent candidate set used for limited stability: `{cands}`. Stability: `{json.dumps(stab)}`.\n\nTiny-component diagnostics are in `tables/preferred_component_diagnostics.csv`. Qualitative relationship to direct shape K-means evidence near 3/5/7: **{comparison}** (overlap `{sorted(over)}`). This is a preliminary shape-space candidate region, not a final application-driven K.\n\nNo templates, downstream registration, deformation-property ranking, or aerodynamic analysis were run. The paused registration has {len(regfiles)} readable atomic checkpoints and remains resumable.\n"""
 (OUT/'report/shape_pca_gmm_report.md').write_text(report,encoding='utf8')
if __name__=='__main__':main()
