"""Append K=9 to the existing Gabriel pinned-shape clustering experiment."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import run_gabriel_k2_k8_energy_magnitude as production
import gabriel_pinned_registration as gabriel

K=9
def main():
 _,c,fn,_=production.load_data();X=np.ascontiguousarray(c).reshape(len(c),400,order='C')
 assignments=production.cread(production.OUT/'clustering_assignments.csv');summary=production.cread(production.OUT/'template_summary.csv')
 existing_a=[r for r in assignments if int(r['K'])==K];existing_t=[r for r in summary if int(r['K'])==K]
 if existing_a or existing_t:
  if len(existing_a)!=len(c) or len(existing_t)!=K:raise RuntimeError('partial K=9 clustering artifacts exist')
  print('K=9 clustering artifacts already complete',flush=True);return
 km=KMeans(n_clusters=K,init='k-means++',n_init=50,random_state=0,algorithm='lloyd').fit(X);labels=km.labels_;d=production.OUT/'templates'/f'K{K:02d}';d.mkdir(parents=True,exist_ok=True)
 fig,ax=plt.subplots(figsize=(10,4.8));new_summary=[]
 for cluster in range(K):
  members=np.flatnonzero(labels==cluster);mean=c[members].mean(0);diag=gabriel.validate_pinned_contour(mean);geom={'finite':bool(np.isfinite(mean).all()),'signed_area':gabriel.signed_polygon_area(mean),'self_intersection_count':gabriel.self_intersection_count(mean)};geom['geometrically_valid']=bool(geom['finite'] and geom['signed_area']>1e-8 and geom['self_intersection_count']==0)
  if not geom['geometrically_valid']:raise RuntimeError(f'invalid K=9 template {cluster}: {geom}')
  tid=f'K{K:02d}_cluster{cluster:02d}';gabriel.atomic_npz(d/f'{tid}.npz',coordinates=mean,member_indices=members,member_filenames=fn[members],pinned_validation_json=np.asarray(json.dumps(gabriel.jsonable(diag),sort_keys=True)),geometry_json=np.asarray(json.dumps(geom,sort_keys=True)))
  new_summary.append({'K':K,'cluster_id':cluster,'template_id':tid,'cluster_size':len(members),'template_geometrically_valid':True,'signed_area':geom['signed_area'],'self_intersection_count':0,'pinned_validator_valid':diag['valid'],'coordinates_sha256':hashlib.sha256(mean.tobytes()).hexdigest()});ax.plot(np.r_[mean[:,0],mean[0,0]],np.r_[mean[:,1],mean[0,1]],lw=1.2,label=f'cluster {cluster} (n={len(members)})')
 assignments.extend({'K':K,'target_index':i,'filename':fn[i],'cluster_id':int(labels[i])} for i in range(len(c)));summary.extend(new_summary);production.cwrite(production.OUT/'clustering_assignments.csv',assignments);production.cwrite(production.OUT/'template_summary.csv',summary)
 ax.set_aspect('equal');ax.set(xlabel='x / chord',ylabel='y / chord',title='Gabriel pinned pointwise templates — K=9');ax.grid(alpha=.2);ax.legend(ncol=3,fontsize=7);fig.tight_layout();fig.savefig(production.OUT/'figures/template_overlay_K09.png',dpi=250);plt.close(fig)
 config=json.loads((production.OUT/'config.json').read_text());config['K']=list(range(2,10));config['K9_extension']={'kmeans':{'init':'k-means++','n_init':50,'random_state':0,'algorithm':'lloyd'},'assignment':'single deterministic fitted solution, identical convention to K=2..8'};gabriel.atomic_json(production.OUT/'config.json',config);print('K=9 clustering/templates appended',flush=True)
if __name__=='__main__':main()
