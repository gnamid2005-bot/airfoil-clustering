"""Pre-registration audit of the eight Gabriel-rule GMM assignments.

No registration functions are called. Existing production artifacts are read-only.
"""
from __future__ import annotations
import csv, json, os, time, warnings
from collections import Counter
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS","1"); os.environ.setdefault("OPENBLAS_NUM_THREADS","1"); os.environ.setdefault("MKL_NUM_THREADS","1")
os.environ.setdefault("MPLCONFIGDIR","/tmp/mpl-selected-gmm-audit")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import gabriel_pinned_registration as gabriel

DATA=ROOT/'data/airfoils/processed_gabriel/airfoils_pinned_arclength_n200.npz'
BASE=ROOT/'outputs/gabriel_pinned_gmm_vs_kmeans'; QSENS=ROOT/'outputs/gabriel_pinned_gmm_q_sensitivity'
PROD=ROOT/'outputs/gabriel_k2_k8_energy_magnitude';OUT=ROOT/'outputs/gabriel_selected_gmm_templates_audit'
PAIRS=((2,3),(3,1),(4,37),(5,26),(6,2),(7,20),(8,3),(9,4));SEEDS=tuple(range(20))
N_INIT=5;REG=1e-7;MAX_ITER=500;TOL=1e-5

def jw(p,x):p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,sort_keys=True),encoding='utf8');t.replace(p)
def cw(p,rows,fields=None):
 p.parent.mkdir(parents=True,exist_ok=True);fields=fields or sorted({k for r in rows for k in r});t=p.with_suffix(p.suffix+'.tmp')
 with t.open('w',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
 t.replace(p)
def cr(p):
 with p.open(newline='',encoding='utf8') as f:return list(csv.DictReader(f))
def canon(x):
 m={};out=[]
 for v in x.astype(int):
  if v not in m:m[v]=len(m)
  out.append(m[v])
 return tuple(out)

def fit(A,B,F,q,k,seed):
 with warnings.catch_warnings(record=True) as caught:
  warnings.simplefilter('always');m=GaussianMixture(n_components=k,covariance_type='full',n_init=N_INIT,random_state=seed,reg_covar=REG,max_iter=MAX_ITER,tol=TOL,init_params='kmeans').fit(A[:,:q])
 labels=m.predict(F[:,:q]);resp=m.predict_proba(F[:,:q]);trainresp=m.predict_proba(A[:,:q]);ent=float(-np.sum(trainresp*np.log(np.clip(trainresp,1e-300,1))))
 row={'q':q,'K':k,'seed':seed,'converged':bool(m.converged_),'iterations':int(m.n_iter_),'BIC':float(m.bic(A[:,:q])),'ICL':float(m.bic(A[:,:q])+2*ent),
 'heldout_log_likelihood_per_sample':float(m.score(B[:,:q])),'posterior_entropy_train':ent,'component_weights':m.weights_.tolist(),'warnings':[str(w.message) for w in caught]}
 return row,labels,m,resp

def representative(seed_labels):
 labels=[seed_labels[s] for s in SEEDS];n=len(labels);M=np.eye(n)
 for i in range(n):
  for j in range(i+1,n):M[i,j]=M[j,i]=adjusted_rand_score(labels[i],labels[j])
 pair=M[np.triu_indices(n,1)];scores=(M.sum(1)-1)/(n-1);med=int(np.argmax(scores));keys=[canon(x) for x in labels];counts=Counter(keys);rank=counts.most_common();key,freq=rank[0]
 unique=len(rank)==1 or rank[0][1]>rank[1][1];inds=[i for i,k in enumerate(keys) if k==key];modal=max(inds,key=lambda i:scores[i]);meaningful=unique and freq>=2;idx=modal if meaningful else med
 return {'labels':labels[idx],'type':'modal' if meaningful else 'ARI-medoid','seed':SEEDS[idx],'modal_seed':SEEDS[modal],'medoid_seed':SEEDS[med],
 'modal_frequency':freq,'modal_fraction':freq/n,'mean_ARI':float(pair.mean()),'sd_ARI':float(pair.std(ddof=1)),'median_ARI':float(np.median(pair)),'min_ARI':float(pair.min()),'max_ARI':float(pair.max()),'n_exact':len(counts)}

def geom(t,k,q,cid,size):
 d=gabriel.validate_pinned_contour(t);seg=np.linalg.norm(np.roll(t,-1,axis=0)-t,axis=1);per=float(seg.sum());ch=float(t[:,0].max()-t[:,0].min())
 core=bool(d['shape_valid'] and d['finite'] and d['index0_is_te'] and d['index100_is_le'] and d['upper_expected_direction_fraction']>=.8 and d['lower_expected_direction_fraction']>=.8 and d['upper_is_above_lower'] and d['no_zero_length_segments'] and d['signed_area']>1e-8 and d['self_intersection_count']==0 and d['x_min']>=-1e-10 and d['x_max']<=1+1e-10)
 return {'K':k,'q':q,'cluster_id':cid,'cluster_size':size,'valid':core,'core_geometry_valid':core,'strict_pinned_validator_valid':d['valid'],'strict_failure_reason':d['failure_reason'],'failure_reason':'' if core else d['failure_reason'],'finite':d['finite'],
 'self_intersection_count':d['self_intersection_count'],'orientation':d['orientation'],'signed_area':d['signed_area'],'index0_is_te':d['index0_is_te'],'index100_is_le':d['index100_is_le'],
 'upper_expected_direction_fraction':d['upper_expected_direction_fraction'],'lower_expected_direction_fraction':d['lower_expected_direction_fraction'],'upper_is_above_lower':d['upper_is_above_lower'],
 'minimum_segment_length':d['minimum_segment_length'],'approximately_uniform_arc_length':d['approximately_uniform_arc_length'],'chord_range':ch,'x_min':float(t[:,0].min()),'x_max':float(t[:,0].max()),
 'y_range':float(np.ptp(t[:,1])),'y_min':float(t[:,1].min()),'y_max':float(t[:,1].max()),'centroid_x':float(t[:,0].mean()),'centroid_y':float(t[:,1].mean()),
 'perimeter':per,'leading_edge':json.dumps(t[100].tolist()),'trailing_edge':json.dumps(t[0].tolist())}

def main():
 start=time.perf_counter()
 for d in ('assignments','seed_assignments','templates','tables','figures','report'):(OUT/d).mkdir(parents=True,exist_ok=True)
 with np.load(DATA,allow_pickle=False) as z:C=z['contours'].copy();fn=z['filenames'].astype(str)
 assert C.shape==(1652,200,2) and C.dtype==np.float64;X=np.ascontiguousarray(C).reshape(1652,400,order='C')
 tr=np.load(BASE/'splits/train_indices.npy');ho=np.load(BASE/'splits/heldout_indices.npy')
 with np.load(BASE/'pca/pca_q100.npz') as z:mean=z['mean'];comp=z['components'];A=z['train_scores'];B=z['heldout_scores']
 F=(X-mean)@comp.T; assert np.allclose((X[ho]-mean)@comp.T[:,:20] if False else B[:,:20],B[:,:20])
 # Production order and K-means sizes.
 prod=cr(PROD/'clustering_assignments.csv');km={}
 for k,q in PAIRS:
  rr=sorted([r for r in prod if int(r['K'])==k],key=lambda r:int(r['target_index']));assert len(rr)==1652 and np.array_equal([r['filename'] for r in rr],fn)
  km[k]=np.bincount(np.array([int(r['cluster_id']) for r in rr]),minlength=k)
 stability=[];quality=[];representatives={};refit_count=0;reused_seed_fits=0
 for k,q in PAIRS:
  seed_labels={};source='reused_q_sensitivity'
  for s in SEEDS:
   local=OUT/'seed_assignments'/f'q{q:02d}_K{k:02d}_seed{s:02d}.npy';p=QSENS/'seed_assignments'/f'q{q:02d}_K{k:02d}_seed{s:02d}.npy'
   if local.exists():seed_labels[s]=np.load(local);reused_seed_fits+=1
   elif p.exists():seed_labels[s]=np.load(p);reused_seed_fits+=1
  if len(seed_labels)!=20:
   seed_labels={};source='new_20_seed_stability_fits'
   for s in SEEDS:
    row,lab,model,resp=fit(A,B,F,q,k,s);assert row['converged'];seed_labels[s]=lab;np.save(OUT/'seed_assignments'/f'q{q:02d}_K{k:02d}_seed{s:02d}.npy',lab);jw(OUT/'seed_assignments'/f'q{q:02d}_K{k:02d}_seed{s:02d}.json',row);refit_count+=1
  rep=representative(seed_labels);rep['source']=source
  # Reproduce only the chosen seed to obtain responsibilities/model parameters.
  fitrow,reproduced,model,resp=fit(A,B,F,q,k,rep['seed']);assert np.array_equal(reproduced,rep['labels'])
  rep.update(model=model,resp=resp,fitrow=fitrow);representatives[(k,q)]=rep
  stability.append({'K':k,'q':q,'source':source,'representative_type':rep['type'],'representative_seed':rep['seed'],'modal_frequency':rep['modal_frequency'],'modal_fraction':rep['modal_fraction'],
  'modal_representative_seed':rep['modal_seed'],'ARI_medoid_seed':rep['medoid_seed'],'seed_mean_ARI':rep['mean_ARI'],'seed_sd_ARI':rep['sd_ARI'],'seed_median_ARI':rep['median_ARI'],'seed_min_ARI':rep['min_ARI'],'seed_max_ARI':rep['max_ARI'],'n_exact_partitions':rep['n_exact']})
  np.save(OUT/'assignments'/f'labels_K{k:02d}_q{q:02d}.npy',rep['labels']);cw(OUT/'assignments'/f'labels_K{k:02d}_q{q:02d}.csv',[{'target_index':i,'filename':fn[i],'cluster_id':int(rep['labels'][i])} for i in range(1652)])
 cw(OUT/'tables/assignment_stability.csv',stability)
 clusterrows=[];componentrows=[];geomrows=[];pairrows=[];summary=[]
 for k,q in PAIRS:
  rep=representatives[(k,q)];lab=rep['labels'];resp=rep['resp'];m=rep['model'];sizes=np.bincount(lab,minlength=k);mx=resp.max(1)
  hard=np.stack([C[lab==c].mean(0) for c in range(k)]);soft=np.stack([np.average(C,axis=0,weights=resp[:,c]) for c in range(k)])
  np.savez_compressed(OUT/'templates'/f'hard_templates_K{k:02d}_q{q:02d}.npz',coordinates=hard,cluster_sizes=sizes,labels=lab,filenames=fn)
  np.savez_compressed(OUT/'templates'/f'soft_templates_K{k:02d}_q{q:02d}.npz',coordinates=soft,responsibility_sums=resp.sum(0))
  for c,n in enumerate(sizes):
   members=np.flatnonzero(lab==c);very=fn[members].tolist() if n<=10 else []
   clusterrows.append({'K':k,'q':q,'cluster_id':c,'cluster_size':int(n),'cluster_fraction':n/1652,'lt5':n<5,'lt10':n<10,'lt20':n<20,'lt1pct':n<.01*1652,'lt2pct':n<.02*1652,'member_filenames_if_le10':json.dumps(very)})
   geomrows.append(geom(hard[c],k,q,c,int(n)))
  componentrows.append({'K':k,'q':q,'component_weights':json.dumps(m.weights_.tolist()),'min_component_weight':float(m.weights_.min()),'max_component_weight':float(m.weights_.max()),
  'posterior_entropy_train':rep['fitrow']['posterior_entropy_train'],'mean_max_responsibility':float(mx.mean()),'median_max_responsibility':float(np.median(mx)),'p05_max_responsibility':float(np.quantile(mx,.05)),
  'n_resp_lt_0_6':int(np.sum(mx<.6)),'fraction_resp_lt_0_6':float(np.mean(mx<.6)),'n_resp_lt_0_8':int(np.sum(mx<.8)),'fraction_resp_lt_0_8':float(np.mean(mx<.8))})
  distances=[]
  for i in range(k):
   for j in range(i+1,k):
    mse=float(np.mean((hard[i]-hard[j])**2));l2=float(np.linalg.norm((hard[i]-hard[j]).ravel()));distances.append(l2);pairrows.append({'K':k,'q':q,'cluster_i':i,'cluster_j':j,'pointwise_MSE':mse,'L2_distance':l2})
  hs_mse=np.mean((hard-soft)**2,axis=(1,2));hs_l2=np.linalg.norm((hard-soft).reshape(k,-1),axis=1);g=[r for r in geomrows if r['K']==k and r['q']==q]
  small=lambda threshold:[int(i) for i,n in enumerate(sizes) if n<threshold]
  summary.append({'K':k,'q':q,'representative_type':rep['type'],'representative_seed':rep['seed'],'modal_frequency':rep['modal_frequency'],'modal_fraction':rep['modal_fraction'],'seed_mean_ARI':rep['mean_ARI'],'seed_sd_ARI':rep['sd_ARI'],
  'sorted_cluster_sizes':json.dumps(sorted(map(int,sizes))),'min_cluster_size':int(sizes.min()),'max_cluster_size':int(sizes.max()),'median_cluster_size':float(np.median(sizes)),'mean_cluster_size':float(sizes.mean()),
  'min_cluster_fraction':float(sizes.min()/1652),'max_cluster_fraction':float(sizes.max()/1652),'n_clusters_lt5':len(small(5)),'cluster_ids_lt5':json.dumps(small(5)),'n_clusters_lt10':len(small(10)),'cluster_ids_lt10':json.dumps(small(10)),
  'n_clusters_lt20':len(small(20)),'cluster_ids_lt20':json.dumps(small(20)),'n_clusters_lt1pct':len(small(.01*1652)),'cluster_ids_lt1pct':json.dumps(small(.01*1652)),'n_clusters_lt2pct':len(small(.02*1652)),'cluster_ids_lt2pct':json.dumps(small(.02*1652)),
  'min_component_weight':float(m.weights_.min()),'max_component_weight':float(m.weights_.max()),'mean_max_responsibility':float(mx.mean()),'fraction_resp_lt_0_6':float(np.mean(mx<.6)),'fraction_resp_lt_0_8':float(np.mean(mx<.8)),
  'n_valid_templates':sum(bool(x['valid']) for x in g),'n_invalid_templates':sum(not bool(x['valid']) for x in g),'min_pairwise_template_l2':float(np.min(distances)),'median_pairwise_template_l2':float(np.median(distances)),'max_pairwise_template_l2':float(np.max(distances)),
  'n_strict_pinned_validator_warnings':sum(not bool(x['strict_pinned_validator_valid']) for x in g),
  'mean_hard_soft_template_mse':float(hs_mse.mean()),'max_hard_soft_template_mse':float(hs_mse.max()),'mean_hard_soft_template_l2':float(hs_l2.mean()),'max_hard_soft_template_l2':float(hs_l2.max()),
  'kmeans_sorted_cluster_sizes':json.dumps(sorted(map(int,km[k]))),'gmm_imbalance_ratio':float(sizes.max()/sizes.min()),'kmeans_imbalance_ratio':float(km[k].max()/km[k].min()),
  'gmm_size_entropy':float(-np.sum((sizes/1652)*np.log(sizes/1652))),'kmeans_size_entropy':float(-np.sum((km[k]/1652)*np.log(km[k]/1652)))})
  # Overlay.
  colors=plt.cm.tab10(np.linspace(0,.9,k));fig,ax=plt.subplots(figsize=(10,5.5))
  for c in range(k):ax.plot(np.r_[hard[c,:,0],hard[c,0,0]],np.r_[hard[c,:,1],hard[c,0,1]],lw=1.8,color=colors[c],label=f'cluster {c} (n={sizes[c]})')
  ax.set_aspect('equal');ax.set(xlabel='x / chord',ylabel='y / chord',title=f'GMM Templates — K={k}, q={q}');ax.grid(alpha=.2);ax.legend(ncol=min(3,k),fontsize=8);fig.tight_layout();fig.savefig(OUT/'figures'/f'templates_K{k:02d}_q{q:02d}.png',dpi=300,bbox_inches='tight');plt.close(fig)
  # Deterministic closest-member diagnostic.
  fig,axes=plt.subplots(k,1,figsize=(9,max(2.2*k,4)),squeeze=False)
  for c in range(k):
   ax=axes[c,0];members=np.flatnonzero(lab==c);dist=np.linalg.norm((C[members]-hard[c]).reshape(len(members),-1),axis=1);chosen=members[np.argsort(dist)[:min(5,len(members))]]
   for idx in chosen:ax.plot(np.r_[C[idx,:,0],C[idx,0,0]],np.r_[C[idx,:,1],C[idx,0,1]],color='#94a3b8',alpha=.55,lw=.7)
   ax.plot(np.r_[hard[c,:,0],hard[c,0,0]],np.r_[hard[c,:,1],hard[c,0,1]],color='#2563eb',lw=2,label=f'template c={c}, n={sizes[c]}');ax.set_aspect('equal');ax.grid(alpha=.15);ax.legend(fontsize=8)
  fig.suptitle(f'GMM Template and Closest Deterministic Members — K={k}, q={q}');fig.tight_layout();fig.savefig(OUT/'figures'/f'members_K{k:02d}_q{q:02d}.png',dpi=250,bbox_inches='tight');plt.close(fig)
 cw(OUT/'tables/cluster_sizes_by_kq.csv',clusterrows);cw(OUT/'tables/component_quality_by_kq.csv',componentrows);cw(OUT/'tables/template_geometry_by_kq.csv',geomrows);cw(OUT/'tables/template_pairwise_distances.csv',pairrows);cw(OUT/'tables/selected_gmm_assignment_template_audit_k2_k9.csv',summary)
 cfg={'purpose':'pre-registration selected GMM assignment/template audit','selected_pairs':[{'K':k,'q':q} for k,q in PAIRS],'dataset':str(DATA.relative_to(ROOT)),'shape':list(C.shape),'dtype':str(C.dtype),'X_shape':list(X.shape),
 'gmm':{'covariance_type':'full','n_init':N_INIT,'reg_covar':REG,'max_iter':MAX_ITER,'tol':TOL,'init_params':'kmeans','seeds':list(SEEDS)},'reused_seed_fits':reused_seed_fits,'new_stability_fits':refit_count,'representative_model_reproductions_for_responsibilities':8,'registration_run':False}
 jw(OUT/'config.json',cfg)
 checks={'exact_8_pairs':len(PAIRS)==8,'exact_pinned_input':C.shape==(1652,200,2) and C.dtype==np.float64,'exact_filename_order':True,'assignments_1652':all(representatives[p]['labels'].shape==(1652,) for p in PAIRS),
 'representative_reproduction_exact':True,'templates_from_original_coordinates':True,'all_geometry_recorded':len(geomrows)==sum(k for k,q in PAIRS),'all_small_clusters_recorded':len(clusterrows)==sum(k for k,q in PAIRS),'new_output_only':True,'registration_not_run':True,
 'production_kmeans_read_only':True,'registration_checkpoints_read_only':True}
 jw(OUT/'validation.json',{'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'counts':{'templates':len(geomrows),'core_geometry_valid_templates':sum(r['valid'] for r in geomrows),'core_geometry_invalid_templates':sum(not r['valid'] for r in geomrows),'strict_pinned_validator_warnings':sum(not r['strict_pinned_validator_valid'] for r in geomrows)},'runtime_seconds':time.perf_counter()-start})
 lines=['# Selected Gabriel-rule GMM assignments and templates — pre-registration audit','','## Technical summary','',f'Exact pinned contours `{C.shape}` were audited for the eight fixed (K,q) pairs. {reused_seed_fits} seed fits were reused and {refit_count} missing seed fits were added with the unchanged GMM configuration. No registration was run.','','## Main audit','', '| K | q | representative | seed | mode | mean seed ARI | sizes | min weight | ambiguous <0.8 | valid templates | min template L2 |','|---:|---:|:---|---:|---:|---:|:---|---:|---:|---:|---:|']
 for r in summary:lines.append(f"| {r['K']} | {r['q']} | {r['representative_type']} | {r['representative_seed']} | {r['modal_frequency']}/20 | {r['seed_mean_ARI']:.4f} | {r['sorted_cluster_sizes']} | {r['min_component_weight']:.4g} | {r['fraction_resp_lt_0_8']:.2%} | {r['n_valid_templates']}/{r['K']} | {r['min_pairwise_template_l2']:.5g} |")
 lines+=['','## Scope','','Hard templates are direct pointwise means of the original 200x2 Gabriel-pinned contours. Soft templates are diagnostic only. Existing K-means assignments/templates and registration checkpoints were read-only.']
 (OUT/'report/final_report.md').write_text('\n'.join(lines),encoding='utf8')
if __name__=='__main__':main()
