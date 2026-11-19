"""R=200 K-means stability and modal assignments on exact Gabriel pinned input."""

from __future__ import annotations

import csv, hashlib, json, platform, time, warnings
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/airfoils/processed_gabriel/airfoils_pinned_arclength_n200.npz'
PROD=ROOT/'outputs/gabriel_k2_k8_energy_magnitude'
OLD_R200=ROOT/'outputs/airfoil_kmeans_stability_r200'
OLD_MODAL=ROOT/'outputs/airfoil_kmeans_modal_assignments_r200'
OUT=ROOT/'outputs/gabriel_pinned_kmeans_r200'
KS=list(range(2,21)); SEEDS=list(range(200))

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def cread(p):
 with p.open(newline='',encoding='utf8') as f:return list(csv.DictReader(f))
def cwrite(p,rows):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def canonical(x):
 out=np.empty(len(x),np.int16);m={};n=0
 for i,v in enumerate(x):
  v=int(v)
  if v not in m:m[v]=n;n+=1
  out[i]=m[v]
 return out
def exact_partition(a,b):return bool(np.array_equal(canonical(a),canonical(b)))
def label_diff(a,b,k):
 tab=np.zeros((k,k),int);np.add.at(tab,(a,b),1);r,c=linear_sum_assignment(-tab);mp={int(y):int(x) for x,y in zip(r,c)};bb=np.array([mp[int(v)] for v in b]);n=int(np.count_nonzero(a!=bb));return n,n/len(a)
def model(k,seed):return KMeans(n_clusters=k,init='k-means++',n_init=50,max_iter=300,tol=1e-4,random_state=seed,algorithm='lloyd',copy_x=True,verbose=0)
def style(ax,title,ylabel):ax.set(title=title,xlabel='Number of clusters K',ylabel=ylabel);ax.set_xticks(KS);ax.grid(alpha=.22,color='#9CA3AF');ax.spines[['top','right']].set_visible(False)

def main():
 OUT.mkdir(parents=True,exist_ok=True);(OUT/'assignments').mkdir(exist_ok=True);(OUT/'checkpoints').mkdir(exist_ok=True);(OUT/'figures').mkdir(exist_ok=True)
 started=time.perf_counter();z=np.load(DATA,allow_pickle=False);C=z['contours'].astype(np.float64);names=z['filenames'].astype(str);X=np.ascontiguousarray(C).reshape(1652,400,order='C')
 prodrows=cread(PROD/'clustering_assignments.csv');prod={k:np.array([int(r['cluster_id']) for r in sorted((x for x in prodrows if int(x['K'])==k),key=lambda x:int(x['target_index']))],np.int16) for k in range(2,10)}
 # Critical reproduction gate before the sweep.
 repro=[]
 for k in range(2,10):
  lab=model(k,0).fit_predict(X);ari=float(adjusted_rand_score(prod[k],lab));eq=exact_partition(prod[k],lab);repro.append({'K':k,'ARI':ari,'exact_partition_equality':eq})
  if ari!=1.0 or not eq:raise RuntimeError(f'production reproduction failed K={k}: ARI={ari}, exact={eq}')
 cwrite(OUT/'production_seed0_reproduction_k2_k9.csv',repro)
 fits=skips=warnings_count=0;fit_start=time.perf_counter()
 for k in KS:
  p=OUT/'checkpoints'/f'labels_K{k:02d}.npz'
  if p.exists():
   q=np.load(p,allow_pickle=False)
   if q['labels'].shape==(200,1652) and np.array_equal(q['seeds'],SEEDS):skips+=200;print(f'K={k} checkpoint reused',flush=True);continue
  labels=np.empty((200,1652),np.int16);inertia=np.empty(200);nit=np.empty(200,np.int16);conv=np.empty(200,bool)
  for seed in SEEDS:
   km=model(k,seed)
   with warnings.catch_warnings(record=True) as caught:warnings.simplefilter('always');labels[seed]=km.fit_predict(X)
   cw=[x for x in caught if issubclass(x.category,ConvergenceWarning)];warnings_count+=len(cw);inertia[seed]=km.inertia_;nit[seed]=km.n_iter_;conv[seed]=not cw and km.n_iter_<km.max_iter;fits+=1
  tmp=p.with_suffix('.tmp.npz');np.savez_compressed(tmp,labels=labels,seeds=np.array(SEEDS),inertia=inertia,n_iterations=nit,converged=conv);tmp.replace(p);print(f'K={k} completed fits={fits}',flush=True)
 fit_seconds=time.perf_counter()-fit_start
 tests={'permutation_equivalence':np.array_equal(canonical([5,5,2,2,8,8]),canonical([1,1,0,0,2,2])),'different_partitions_distinct':not np.array_equal(canonical([0,0,1,1]),canonical([0,1,0,1])),'idempotent':np.array_equal(canonical(canonical([9,2,9,4,2])),canonical([9,2,9,4,2]))}
 if not all(tests.values()):raise AssertionError(tests)
 pairrows=[];stability=[];modal=[];perseed=[];labelsets={};pair_start=time.perf_counter()
 for k in KS:
  labels=np.load(OUT/'checkpoints'/f'labels_K{k:02d}.npz',allow_pickle=False)['labels'];labelsets[k]=labels
  sums=np.zeros(200);vals=[]
  for a,b in combinations(range(200),2):
   v=float(adjusted_rand_score(labels[a],labels[b]));vals.append(v);sums[a]+=v;sums[b]+=v;pairrows.append({'K':k,'seed_a':a,'seed_b':b,'ARI':v})
  av=np.array(vals);means=sums/199;se=float(av.std(ddof=0)/np.sqrt(len(av)))
  stability.append({'K':k,'n_runs':200,'n_pairwise':len(av),'mean_ARI':av.mean(),'SD_ARI':av.std(ddof=0),'SE_mean_ARI':se,'median_ARI':np.median(av),'min_ARI':av.min(),'p05_ARI':np.percentile(av,5),'p95_ARI':np.percentile(av,95),'max_ARI':av.max()})
  can=np.stack([canonical(x) for x in labels]);groups=defaultdict(list);rep={}
  for s in SEEDS:key=can[s].tobytes();groups[key].append(s);rep[key]=can[s]
  mf=max(map(len,groups.values()));ties=sorted((min(v),key,v) for key,v in groups.items() if len(v)==mf);ms,key,mseeds=ties[0];med=int(np.argmax(means));modearr=rep[key];medarr=can[med]
  np.save(OUT/'assignments'/f'modal_assignment_K{k:02d}.npy',modearr);np.save(OUT/'assignments'/f'ari_medoid_assignment_K{k:02d}.npy',medarr)
  tieinfo={'K':k,'modal_frequency':mf,'n_tied_modal_partitions':len(ties),'selected_representative_seed':ms,'tied_partitions':[{'representative_seed':x[0],'seeds':x[2],'canonical_sha256':hashlib.sha256(rep[x[1]].tobytes()).hexdigest()} for x in ties]}
  (OUT/'assignments'/f'modal_ties_K{k:02d}.json').write_text(json.dumps(tieinfo,indent=2),encoding='utf8')
  modal.append({'K':k,'n_runs':200,'n_unique_partitions':len(groups),'modal_frequency':mf,'modal_fraction':mf/200,'modal_representative_seed':ms,'modal_seed_list':json.dumps(mseeds),'n_tied_modal_partitions':len(ties),'mode_tied':len(ties)>1,'ari_medoid_seed':med,'ari_medoid_mean_ARI':means[med],'mode_vs_medoid_ARI':adjusted_rand_score(modearr,medarr),'mode_equals_medoid':exact_partition(modearr,medarr)})
  perseed.extend({'K':k,'seed':s,'mean_ARI_to_other_199':means[s]} for s in SEEDS)
 pair_seconds=time.perf_counter()-pair_start
 cwrite(OUT/'pinned_kmeans_stability_summary_k2_k20.csv',stability);cwrite(OUT/'pinned_modal_assignment_summary_k2_k20.csv',modal);cwrite(OUT/'per_seed_mean_ari_k2_k20.csv',perseed);cwrite(OUT/'pairwise_ari_k2_k20.csv',pairrows)
 prodcomp=[]
 for k in range(2,10):
  m=modal[k-2];modearr=np.load(OUT/'assignments'/f'modal_assignment_K{k:02d}.npy');medarr=np.load(OUT/'assignments'/f'ari_medoid_assignment_K{k:02d}.npy');n,fr=label_diff(prod[k],modearr,k);nm,fm=label_diff(prod[k],medarr,k)
  prodcomp.append({'K':k,'production_seed':0,'modal_representative_seed':m['modal_representative_seed'],'modal_frequency':m['modal_frequency'],'modal_fraction':m['modal_fraction'],'n_unique_partitions':m['n_unique_partitions'],'production_vs_mode_ARI':adjusted_rand_score(prod[k],modearr),'production_vs_mode_NMI':normalized_mutual_info_score(prod[k],modearr),'production_equals_mode':exact_partition(prod[k],modearr),'production_vs_mode_n_different':n,'production_vs_mode_fraction_different':fr,'ari_medoid_seed':m['ari_medoid_seed'],'production_vs_medoid_ARI':adjusted_rand_score(prod[k],medarr),'production_vs_medoid_NMI':normalized_mutual_info_score(prod[k],medarr),'production_equals_medoid':exact_partition(prod[k],medarr),'production_vs_medoid_n_different':nm,'production_vs_medoid_fraction_different':fm,'mode_vs_medoid_ARI':m['mode_vs_medoid_ARI']})
 cwrite(OUT/'production_vs_pinned_modal_assignment_k2_k9.csv',prodcomp)
 oldstab={int(r['k']):r for r in cread(OLD_R200/'tables/pairwise_ari_summary_r200.csv')};oldmod={int(r['K']):r for r in cread(OLD_MODAL/'modal_assignment_summary_k2_k20.csv')};compare=[]
 for s,m in zip(stability,modal):
  k=s['K'];compare.append({'K':k,'new_dataset':'Gabriel pinned N=1652, 200x2','old_dataset':'direct shape N=1654, 200x2','pinned_mean_ARI':s['mean_ARI'],'old_direct_mean_ARI':oldstab[k]['mean_pairwise_ari'],'difference_mean_ARI':s['mean_ARI']-float(oldstab[k]['mean_pairwise_ari']),'pinned_SD_ARI':s['SD_ARI'],'old_direct_SD_ARI':oldstab[k]['std_pairwise_ari'],'difference_SD_ARI':s['SD_ARI']-float(oldstab[k]['std_pairwise_ari']),'pinned_unique_partitions':m['n_unique_partitions'],'old_direct_unique_partitions':oldmod[k]['n_unique_partitions'],'pinned_modal_fraction':m['modal_fraction'],'old_direct_modal_fraction':oldmod[k]['modal_fraction']})
 cwrite(OUT/'pinned_vs_old_direct_summary_k2_k20.csv',compare)
 k=np.array(KS);mean=np.array([x['mean_ARI'] for x in stability]);sd=np.array([x['SD_ARI'] for x in stability]);mf=np.array([x['modal_fraction'] for x in modal]);un=np.array([x['n_unique_partitions'] for x in modal]);mm=np.array([x['mode_vs_medoid_ARI'] for x in modal])
 fig,ax=plt.subplots(figsize=(12,7));ax.fill_between(k,np.clip(mean-sd,-1,1),np.clip(mean+sd,-1,1),color='#93C5FD',alpha=.42,label='mean ± SD');ax.plot(k,mean,'o-',color='#1D4ED8',lw=2,label='mean pairwise ARI');style(ax,'Gabriel Pinned K-means Stability — 200 Seeds','Adjusted Rand Index');ax.set_ylim(-.05,1.05);ax.legend(frameon=False);fig.tight_layout();fig.savefig(OUT/'figures/pinned_kmeans_stability_vs_k_r200.png',dpi=300);plt.close(fig)
 for y,title,ylabel,name,ylim in [(mf,'Gabriel Pinned Exact Modal Partition Fraction','Modal fraction','pinned_modal_partition_fraction_vs_k.png',(0,1.02)),(un,'Gabriel Pinned Unique Exact Partitions','Unique partitions among 200 seeds','pinned_unique_partitions_vs_k.png',None),(mm,'Gabriel Pinned Mode vs ARI Medoid','ARI(mode, medoid)','pinned_mode_vs_medoid_ari_vs_k.png',(0,1.02))]:
  fig,ax=plt.subplots(figsize=(11,6.4));ax.plot(k,y,'o-',color='#1D4ED8',lw=2);style(ax,title,ylabel);ax.set_ylim(ylim) if ylim else None;fig.tight_layout();fig.savefig(OUT/'figures'/name,dpi=300);plt.close(fig)
 fig,ax=plt.subplots(figsize=(10,6));pk=np.arange(2,10);ax.plot(pk,[x['production_vs_mode_ARI'] for x in prodcomp],'o-',color='#1D4ED8',lw=2);ax.set_xticks(pk);ax.set_ylim(0,1.02);ax.set(title='Production Assignment vs Gabriel-Pinned Modal Partition',xlabel='Number of clusters K',ylabel='Adjusted Rand Index');ax.grid(alpha=.22);ax.spines[['top','right']].set_visible(False);fig.tight_layout();fig.savefig(OUT/'figures/production_vs_pinned_mode_ari_k2_k9.png',dpi=300);plt.close(fig)
 fig,ax=plt.subplots(figsize=(12,7));ax.plot(k,mean,'o-',color='#1D4ED8',lw=2,label='Gabriel pinned N=1652');ax.plot(k,[float(oldstab[x]['mean_pairwise_ari']) for x in KS],'s--',color='#C2410C',lw=1.7,label='Old direct N=1654');style(ax,'K-means Stability: Gabriel Pinned vs Old Direct Shape','Mean pairwise ARI');ax.legend(frameon=False);fig.tight_layout();fig.savefig(OUT/'figures/pinned_vs_direct_stability_comparison.png',dpi=300);plt.close(fig)
 runtime=time.perf_counter()-started;cfg={'dataset':str(DATA.relative_to(ROOT)),'dataset_sha256':sha(DATA),'shape':list(C.shape),'dtype':str(C.dtype),'flattened_shape':list(X.shape),'flatten_order':'NumPy C order; both x and y included','sample_order':'persisted filenames array, unchanged','filenames_sha256':hashlib.sha256(names.tobytes()).hexdigest(),'scaling':'none beyond persisted Gabriel preprocessing','K_values':KS,'seeds':SEEDS,'kmeans':model(2,0).get_params(),'sklearn_version':sklearn.__version__,'canonicalization':'first occurrence relabeling','old_direct_comparison':'summary-level only; different dataset and representation'};(OUT/'experiment_config.json').write_text(json.dumps(cfg,indent=2,sort_keys=True),encoding='utf8')
 val={'production_seed0_reproduction_all_k2_k9':all(x['ARI']==1 and x['exact_partition_equality'] for x in repro),'production_seed0_reproduction':repro,'total_fits_expected':3800,'fits_executed_this_run':fits,'fits_reused':skips,'all_K_present':all((OUT/'checkpoints'/f'labels_K{k:02d}.npz').exists() for k in KS),'exactly_200_labels_each_K':all(np.load(OUT/'checkpoints'/f'labels_K{k:02d}.npz')['labels'].shape==(200,1652) for k in KS),'all_pairwise_finite':bool(np.isfinite([x['ARI'] for x in pairrows]).all()),'exactly_19900_pairs_each_K':all(sum(x['K']==k for x in pairrows)==19900 for k in KS),'total_pairwise_ARI':len(pairrows),'canonicalization_tests':tests,'convergence_warning_count':warnings_count,'all_runs_converged':all(np.load(OUT/'checkpoints'/f'labels_K{k:02d}.npz')['converged'].all() for k in KS),'fit_runtime_seconds':fit_seconds,'pairwise_and_analysis_runtime_seconds':pair_seconds,'total_runtime_seconds':runtime,'registration_run':False,'production_files_modified':False};(OUT/'validation.json').write_text(json.dumps(val,indent=2,sort_keys=True),encoding='utf8');print(json.dumps(val,indent=2),flush=True)
if __name__=='__main__':main()
