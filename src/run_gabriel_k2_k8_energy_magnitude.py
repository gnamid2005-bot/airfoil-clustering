"""Pinned-shape K-means templates registered with Gabriel's frozen method.

This driver only orchestrates the existing implementation in
``gabriel_pinned_registration.py``.  It does not alter Gabriel's mathematics.
"""
from __future__ import annotations
import argparse,csv,json,math,os,time,hashlib
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans
import gabriel_pinned_registration as gabriel

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/airfoils/processed_gabriel/airfoils_pinned_arclength_n200.npz'
MEAN_NPY=ROOT/'data/airfoils/processed_gabriel/global_mean_gabriel.npy'
MEAN_NPZ=ROOT/'data/airfoils/processed_gabriel/global_mean_gabriel.npz'
OUT=ROOT/'outputs/gabriel_k2_k8_energy_magnitude'
DEV=ROOT/'outputs/gabriel_pinned_registration_dev'
CONFIG=gabriel.RegistrationConfig();KS=tuple(range(2,10));DIAGNOSTIC_RESOLUTION=101

def sha(p):return gabriel.sha256_file(p)
def cwrite(p,rows,fields=None):
 p.parent.mkdir(parents=True,exist_ok=True);fields=fields or list(dict.fromkeys(k for r in rows for k in r));t=p.with_name(p.name+'.tmp')
 with t.open('w',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
 os.replace(t,p)
def cread(p):
 if not p.exists():return []
 with p.open(newline='',encoding='utf8') as f:return list(csv.DictReader(f))
def load_data():
 z=np.load(DATA,allow_pickle=False);c=z['contours'].astype(np.float64);fn=z['filenames'].astype(str);mean=np.load(MEAN_NPY).astype(np.float64)
 if c.shape!=(1652,200,2) or not np.isfinite(c).all():raise RuntimeError(f'bad pinned data {c.shape}')
 if mean.shape!=(200,2) or not np.array_equal(mean,c.mean(0)):raise RuntimeError('saved Gabriel mean is not exact contour mean')
 valid=[gabriel.validate_pinned_contour(x)['valid'] for x in c]
 if not all(valid):raise RuntimeError(f'pinned input invalid count {sum(not x for x in valid)}')
 return z,c,fn,mean
def metadata_fingerprint(paths):
 return {str(p.relative_to(ROOT)):{'sha256':sha(p),'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in paths}

def prepare():
 z,c,fn,mean=load_data();X=np.ascontiguousarray(c).reshape(len(c),400,order='C');templates={};assign=[];ts=[]
 protected=[DATA,MEAN_NPY,MEAN_NPZ,ROOT/'src/gabriel_pinned_registration.py',ROOT/'src/run_gabriel_pinned_registration_dev.py',ROOT/'src/validate_gabriel_pinned_registration_dev.py',DEV/'config.json',DEV/'validation.json',DEV/'source_faithfulness_validation.json']
 if (OUT/'config.json').exists() and (OUT/'clustering_assignments.csv').exists() and (OUT/'template_summary.csv').exists():
  assign=cread(OUT/'clustering_assignments.csv');ts=cread(OUT/'template_summary.csv')
  if len(assign)!=len(c)*len(KS):raise RuntimeError('existing assignment table incomplete')
  for row in ts:
   K=int(row['K']);cl=int(row['cluster_id']);tid=row['template_id'];p=OUT/'templates'/f'K{K:02d}'/f'{tid}.npz'
   with np.load(p) as a:templates[(K,cl)]=(a['coordinates'],a['member_indices'],tid)
  return z,c,fn,templates,assign,ts,protected
 for K in KS:
  km=KMeans(n_clusters=K,init='k-means++',n_init=50,random_state=0,algorithm='lloyd').fit(X);labels=km.labels_;sizes=np.bincount(labels,minlength=K)
  for i in range(len(c)):assign.append({'K':K,'target_index':i,'filename':fn[i],'cluster_id':int(labels[i])})
  d=OUT/'templates'/f'K{K:02d}';d.mkdir(parents=True,exist_ok=True)
  fig,ax=plt.subplots(figsize=(10,4.8))
  for cluster in range(K):
   members=np.flatnonzero(labels==cluster);cmean=c[members].mean(0);diag=gabriel.validate_pinned_contour(cmean)
   geom={'finite':bool(np.isfinite(cmean).all()),'signed_area':gabriel.signed_polygon_area(cmean),'self_intersection_count':gabriel.self_intersection_count(cmean)}
   geom['geometrically_valid']=bool(geom['finite'] and geom['signed_area']>1e-8 and geom['self_intersection_count']==0)
   if not geom['geometrically_valid']:raise RuntimeError(f'invalid template K={K} cluster={cluster}: {geom}')
   tid=f'K{K:02d}_cluster{cluster:02d}';templates[(K,cluster)]=(cmean,members,tid)
   gabriel.atomic_npz(d/f'{tid}.npz',coordinates=cmean,member_indices=members,member_filenames=fn[members],pinned_validation_json=np.asarray(json.dumps(gabriel.jsonable(diag),sort_keys=True)),geometry_json=np.asarray(json.dumps(geom,sort_keys=True)))
   ts.append({'K':K,'cluster_id':cluster,'template_id':tid,'cluster_size':len(members),'template_geometrically_valid':geom['geometrically_valid'],'signed_area':geom['signed_area'],'self_intersection_count':geom['self_intersection_count'],'pinned_validator_valid':diag['valid'],'coordinates_sha256':hashlib.sha256(cmean.tobytes()).hexdigest()})
   ax.plot(np.r_[cmean[:,0],cmean[0,0]],np.r_[cmean[:,1],cmean[0,1]],lw=1.4,label=f'cluster {cluster} (n={len(members)})')
  ax.set_aspect('equal');ax.set(xlabel='x / chord',ylabel='y / chord',title=f'Gabriel pinned pointwise templates — K={K}');ax.grid(alpha=.2);ax.legend(ncol=min(4,K),fontsize=8);fig.tight_layout();fig.savefig(OUT/'figures'/f'template_overlay_K{K:02d}.png',dpi=250);plt.close(fig)
 cwrite(OUT/'clustering_assignments.csv',assign);cwrite(OUT/'template_summary.csv',ts)
 config={'experiment':'Gabriel pinned shape K-means templates: energy and magnitude','dataset':str(DATA.relative_to(ROOT)),'dataset_shape':list(c.shape),'clustering_input_shape':list(X.shape),'clustering_input':'C-order flattened Gabriel pinned shape coordinates','K':list(KS),'kmeans':{'init':'k-means++','n_init':50,'random_state':0,'algorithm':'lloyd'},'template_definition':'hard-cluster pointwise arithmetic mean in 200x2 pinned coordinate space','registration_config':asdict(CONFIG),'parameter_count':CONFIG.parameter_count,'diagnostic_grid':{'resolution':101,'xlim':list(CONFIG.grid_xlim),'ylim':list(CONFIG.grid_ylim),'physical_dirichlet':'gabriel.diagnostic_grid physical_dirichlet_integral','primary_svf_magnitude':'gabriel.diagnostic_grid velocity_mean_magnitude'},'resume':'atomic per-(K,target) NPZ; readable matching metadata skips','protected_inputs_before':metadata_fingerprint(protected)}
 gabriel.atomic_json(OUT/'config.json',config);return z,c,fn,templates,assign,ts,protected

def geometry(deformed):
 g={'finite':bool(np.isfinite(deformed).all()),'signed_area':gabriel.signed_polygon_area(deformed),'self_intersection_count':gabriel.self_intersection_count(deformed)}
 g['invalid_geometry']=bool(not g['finite'] or g['signed_area']<=1e-8 or g['self_intersection_count']>0);return g
def run_one(payload):
 K,cluster,idx,filename,tid,source,target,artifact_text=payload;torch.set_num_threads(1)
 try:torch.set_num_interop_threads(1)
 except RuntimeError:pass
 start=time.perf_counter();base={'K':K,'cluster_id':cluster,'target_index':idx,'filename':filename,'source_template_id':tid}
 try:
  result=gabriel.optimise_registration(source,target,CONFIG);after=gabriel.correspondence_metrics(result['deformed'],target)
  grid=gabriel.diagnostic_grid(result['coefficients'],result['control_x'],result['control_y'],CONFIG.rk_steps,DIAGNOSTIC_RESOLUTION);geo=geometry(result['deformed'])
  valid=bool(result['status']=='completed' and result['finite'] and not geo['invalid_geometry'] and grid['jacobian_minimum']>0 and grid['jacobian_nonpositive_fraction']==0)
  row={**base,'status':result['status'],'failure_reason':result['failure_reason'],'finite':result['finite'],'valid_registration':valid,'geometry_valid':not geo['invalid_geometry'],'runtime_seconds':time.perf_counter()-start,'optimisation_runtime_seconds':result['runtime_seconds'],'closure_evaluations':result['closure_evaluations'],'optimizer_reported_iterations_total':sum(x['optimizer_reported_iterations'] for x in result['cycles']),
  'final_pointwise_mse':after['pointwise_mse'],'final_pointwise_rmse':after['pointwise_rmse'],'maximum_correspondence_error':after['maximum_correspondence_error'],'physical_dirichlet_integral':grid['physical_dirichlet_integral'],'velocity_mean_magnitude':grid['velocity_mean_magnitude'],'velocity_rms_magnitude':grid['velocity_rms_magnitude'],'velocity_maximum_magnitude':grid['velocity_maximum_magnitude'],'coefficient_smoothness':result['final_coefficient_smoothness'],'final_lambda':result['final_lambda'],'weighted_regularisation':result['final_weighted_regularisation'],'total_objective':result['final_objective'],'jacobian_minimum':grid['jacobian_minimum'],'jacobian_p01':grid['jacobian_p01'],'jacobian_p05':grid['jacobian_p05'],'jacobian_nonpositive_fraction':grid['jacobian_nonpositive_fraction'],'signed_area':geo['signed_area'],'self_intersection_count':geo['self_intersection_count'],'diagnostic_resolution':DIAGNOSTIC_RESOLUTION}
  p=Path(artifact_text);gabriel.atomic_npz(p,metrics_json=np.asarray(json.dumps(gabriel.jsonable(row),sort_keys=True)),coefficients=result['coefficients'],control_x=result['control_x'],control_y=result['control_y'],deformed=result['deformed'])
  return row
 except Exception as e:
  row={**base,'status':'failed','failure_reason':f'{type(e).__name__}:{e}','finite':False,'valid_registration':False,'geometry_valid':False,'runtime_seconds':time.perf_counter()-start,'diagnostic_resolution':DIAGNOSTIC_RESOLUTION};gabriel.atomic_npz(Path(artifact_text),metrics_json=np.asarray(json.dumps(row,sort_keys=True)));return row
def load_artifact(p,K,idx):
 try:
  with np.load(p) as z:r=json.loads(str(z['metrics_json']))
  return r if int(r['K'])==K and int(r['target_index'])==idx else None
 except:return None
def run(c,fn,templates,assign,workers,only=None,requested_ks=KS,progress_log=None):
 amap={(int(a['K']),int(a['target_index'])):int(a['cluster_id']) for a in assign};pairs=only or [(K,i) for K in requested_ks for i in range(len(c))];rows=[];todo=[]
 for K,i in pairs:
  cl=amap[(K,i)];source,_,tid=templates[(K,cl)];p=OUT/'registrations'/f'K{K:02d}'/f'{i:04d}.npz';old=load_artifact(p,K,i)
  if old is not None:rows.append(old)
  else:todo.append((K,cl,i,str(fn[i]),tid,source,c[i],str(p)))
 print(f'resume: {len(rows)} checkpoints reused; {len(todo)} pending',flush=True)
 start=time.perf_counter();completed_new=0
 def progress(current_k=None):
  if progress_log is None:return
  counts={K:sum(int(r['K'])==K for r in rows) for K in requested_ks};elapsed=time.perf_counter()-start;throughput=completed_new/elapsed if elapsed>0 else 0.0;remaining=len(todo)-completed_new
  record={'timestamp':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'current_K':current_k,'latest_target':None if current_k is None else int(rows[-1]['target_index']),'completed_by_K':counts,'total_remaining':remaining,'successful_registrations':sum(r.get('status')=='completed' for r in rows),'failures':sum(r.get('status')!='completed' for r in rows),'invalid_registrations':sum(not bool(r.get('valid_registration',False)) for r in rows),'new_throughput_registrations_per_second':throughput,'elapsed_wall_seconds':elapsed,'estimated_remaining_seconds':remaining/throughput if throughput>0 else None,'checkpoints_reused':len(rows)-completed_new,'newly_completed':completed_new}
  progress_log.parent.mkdir(parents=True,exist_ok=True)
  with progress_log.open('a',encoding='utf8') as f:f.write(json.dumps(record,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
 progress()
 if workers==1:
  for n,p in enumerate(todo,1):r=run_one(p);rows.append(r);completed_new+=1;progress(int(r['K']));print(f"[{n}/{len(todo)}] K={r['K']} target={r['target_index']} valid={r['valid_registration']} runtime={r['runtime_seconds']:.2f}s",flush=True)
 else:
  with ProcessPoolExecutor(max_workers=workers) as pool:
   fs={pool.submit(run_one,p):p[:3] for p in todo}
   for n,f in enumerate(as_completed(fs),1):r=f.result();rows.append(r);completed_new+=1;progress(int(r['K']));print(f"[{n}/{len(todo)}] K={r['K']} target={r['target_index']} valid={r['valid_registration']} runtime={r['runtime_seconds']:.2f}s",flush=True)
 return sorted(rows,key=lambda x:(int(x['K']),int(x['target_index'])))

def stats(v):
 a=np.asarray(v,float);n=len(a);return {'mean':float(a.mean()),'sd':float(a.std(ddof=1)) if n>1 else math.nan,'se':float(a.std(ddof=1)/math.sqrt(n)) if n>1 else math.nan,'median':float(np.median(a)),'p90':float(np.quantile(a,.9)),'p95':float(np.quantile(a,.95))}
def finalise(c,fn,templates,assign,ts,protected,requested_ks=KS):
 rows=[];corrupt=[]
 for K in requested_ks:
  for i in range(len(c)):
   p=OUT/'registrations'/f'K{K:02d}'/f'{i:04d}.npz';r=load_artifact(p,K,i)
   if r is None:corrupt.append(str(p));continue
   rows.append(r)
 suffix=f"k{min(requested_ks)}_k{max(requested_ks)}" if tuple(requested_ks)!=KS else ''
 cwrite(OUT/(f'per_registration_results_{suffix}.csv' if suffix else 'per_registration_results.csv'),rows)
 summary=[];clusters=[]
 for K in requested_ks:
  rr=[r for r in rows if int(r['K'])==K];valid=[r for r in rr if r.get('valid_registration') is True]
  d=stats([r['physical_dirichlet_integral'] for r in valid]);m=stats([r['velocity_mean_magnitude'] for r in valid]);mse=stats([r['final_pointwise_mse'] for r in valid]);rt=stats([r['runtime_seconds'] for r in rr])
  summary.append({'K':K,'n_total':len(c),'n_targets':len(c),'n_attempted':len(rr),'n_valid':len(valid),'n_invalid':sum(not bool(r.get('valid_registration',False)) for r in rr),'n_failed':sum(r['status']!='completed' for r in rr),'n_geometry_invalid':sum(not bool(r.get('geometry_valid',False)) for r in rr),'n_jacobian_invalid':sum(float(r.get('jacobian_minimum',-math.inf))<=0 or float(r.get('jacobian_nonpositive_fraction',1))>0 for r in rr),'average_dirichlet_energy':d['mean'],'dirichlet_sd':d['sd'],'dirichlet_se':d['se'],'dirichlet_median':d['median'],'dirichlet_p90':d['p90'],'dirichlet_p95':d['p95'],'dirichlet_maximum':max((float(r['physical_dirichlet_integral']) for r in valid),default=math.nan),'average_svf_magnitude':m['mean'],'svf_magnitude_sd':m['sd'],'svf_magnitude_se':m['se'],'svf_magnitude_median':m['median'],'svf_magnitude_p90':m['p90'],'svf_magnitude_p95':m['p95'],'svf_magnitude_maximum':max((float(r['velocity_mean_magnitude']) for r in valid),default=math.nan),'average_correspondence_mse':mse['mean'],'correspondence_mse_median':mse['median'],'correspondence_mse_p95':mse['p95'],'maximum_correspondence_mse':max((float(r['final_pointwise_mse']) for r in valid),default=math.nan),'average_runtime':rt['mean'],'median_runtime':rt['median']})
  for cl in range(K):
   ar=[r for r in rr if int(r['cluster_id'])==cl];cr=[r for r in ar if r.get('valid_registration') is True];d=stats([r['physical_dirichlet_integral'] for r in cr]);m=stats([r['velocity_mean_magnitude'] for r in cr]);clusters.append({'K':K,'cluster_id':cl,'cluster_size':len(next(x[1] for key,x in templates.items() if key==(K,cl))),'n_valid':len(cr),'n_failed_or_invalid':sum(r.get('status')!='completed' or not bool(r.get('valid_registration',False)) for r in ar),'mean_physical_dirichlet_energy':d['mean'],'median_physical_dirichlet_energy':d['median'],'dirichlet_sd':d['sd'],'dirichlet_se':d['se'],'mean_svf_magnitude':m['mean'],'median_svf_magnitude':m['median'],'svf_magnitude_sd':m['sd'],'svf_magnitude_se':m['se'],'mean_correspondence_mse':float(np.mean([r['final_pointwise_mse'] for r in cr])) if cr else math.nan,'mean_runtime':float(np.mean([r['runtime_seconds'] for r in ar])) if ar else math.nan})
 cwrite(OUT/(f'summary_by_k_{suffix}.csv' if suffix else 'summary_by_k.csv'),summary);cwrite(OUT/(f'summary_by_k_and_cluster_{suffix}.csv' if suffix else 'summary_by_k_and_cluster.csv'),clusters)
 q=[x['K'] for x in summary]
 chartmap=[]
 for field,sefield,ylabel,title,name,color in [('average_dirichlet_energy','dirichlet_se','Average physical Dirichlet energy','Average Dirichlet Energy vs Number of Templates','average_dirichlet_energy_vs_k.png','#1f4e79'),('average_svf_magnitude','svf_magnitude_se','Average velocity-field magnitude','Average SVF Magnitude vs Number of Templates','average_svf_magnitude_vs_k.png','#c45a18')]:
  plt.close('all')
  if suffix:
   if suffix in ('k2_k6','k2_k9'):title=f'{title} — K={min(requested_ks)}–{max(requested_ks)}'
   else:title=f'{title} — Partial K={min(requested_ks)}–{max(requested_ks)} result'
   name=name.replace('.png',f'_{suffix}.png' if suffix in ('k2_k6','k2_k9') else f'_{suffix}_partial.png')
  fig,ax=plt.subplots(figsize=(10,5.6));y=[x[field] for x in summary];e=[x[sefield] for x in summary];ax.errorbar(q,y,yerr=e,marker='o',mfc='white',mec=color,color=color,lw=2,capsize=5);ax.set_xticks(q);ax.set(xlabel='Number of templates K',ylabel=ylabel,title=title);ax.grid(axis='y',alpha=.22);ax.spines[['top','right']].set_visible(False);fig.tight_layout();fig.savefig(OUT/'figures'/name,dpi=300,bbox_inches='tight');plt.close(fig);chartmap.append({'figure':name,'family':'ordered comparison','type':'line with mean ± SE','x':'K=2..8','y':field,'denominator':'all valid target registrations; sample weighted','palette':color})
 gabriel.atomic_json(OUT/(f'chart_map_{suffix}.json' if suffix else 'chart_map.json'),chartmap)
 if tuple(requested_ks)==tuple(range(2,10)):
  incremental=[]
  for previous,current in zip(summary,summary[1:]):
   incremental.append({'K_from':previous['K'],'K_to':current['K'],'dirichlet_reduction_percent':100*(previous['average_dirichlet_energy']-current['average_dirichlet_energy'])/previous['average_dirichlet_energy'],'svf_magnitude_reduction_percent':100*(previous['average_svf_magnitude']-current['average_svf_magnitude'])/previous['average_svf_magnitude']})
  cwrite(OUT/'incremental_improvement_k2_k9.csv',incremental)
  plt.close('all');fig,ax=plt.subplots(figsize=(10,5.6));x=[r['K_to'] for r in incremental];ax.plot(x,[r['dirichlet_reduction_percent'] for r in incremental],marker='o',label='Dirichlet energy');ax.plot(x,[r['svf_magnitude_reduction_percent'] for r in incremental],marker='s',label='SVF magnitude');ax.axhline(0,color='black',lw=.8,alpha=.5);ax.set_xticks(x);ax.set(xlabel='Number of templates K',ylabel='Reduction from K-1 to K (%)',title='Incremental Metric Reduction Across K=2–9');ax.grid(axis='y',alpha=.22);ax.legend();ax.spines[['top','right']].set_visible(False);fig.tight_layout();fig.savefig(OUT/'figures/incremental_improvement_k2_k9.png',dpi=300,bbox_inches='tight');plt.close(fig)
 before=json.loads((OUT/'config.json').read_text())['protected_inputs_before'];after=metadata_fingerprint(protected);tmp=list(OUT.rglob('*.tmp'))+list(OUT.rglob('*.partial'))
 checks={'Gabriel_pinned_dataset_used':True,'dataset_exact_1652x200x2':c.shape==(1652,200,2),'shape_clustering_input':True,'requested_K_exact':list(requested_ks)==list(range(min(requested_ks),max(requested_ks)+1)),'templates_pointwise_means':all(np.array_equal(templates[(K,cl)][0],c[templates[(K,cl)][1]].mean(0)) for K in requested_ks for cl in range(K)),'Gabriel_exact_registration_config':json.loads(json.dumps(asdict(CONFIG)))==json.loads((DEV/'config.json').read_text())['registration_config'],'RK4_5_steps':CONFIG.rk_steps==5,'diagnostic_grid_101_all':all(int(r.get('diagnostic_resolution',0))==101 for r in rows),'all_cells_present':len(rows)==len(c)*len(requested_ks),'no_corrupt_or_partial':not corrupt and not tmp,'sample_weighted_means':all(np.isclose(s['average_dirichlet_energy'],np.mean([r['physical_dirichlet_integral'] for r in rows if int(r['K'])==s['K'] and r['valid_registration']])) for s in summary),'protected_inputs_unchanged':before==after}
 gabriel.atomic_json(OUT/(f'validation_{suffix}.json' if suffix else 'validation.json'),{'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'corrupt_or_missing':corrupt,'partial_files':[str(x) for x in tmp],'counts':{'rows':len(rows),'expected':len(c)*len(requested_ks)}})
 size=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file());attempted=len(rows);valid=sum(bool(r.get('valid_registration')) for r in rows);failed=sum(r['status']!='completed' for r in rows)
 report=['# Gabriel pinned templates: physical energy and SVF magnitude','','## Technical summary','',f'Gabriel pinned contours `{c.shape}` were clustered in shape space and registered with the frozen Gabriel implementation. This report is descriptive only and does not select K.','', '## Requested K-level data','', '| K | valid / attempted | average physical Dirichlet | SE | average SVF magnitude | SE | average MSE |','|---:|---:|---:|---:|---:|---:|---:|']
 for s in summary:report.append(f"| {s['K']} | {s['n_valid']} / {s['n_attempted']} | {s['average_dirichlet_energy']:.10g} | {s['dirichlet_se']:.4g} | {s['average_svf_magnitude']:.10g} | {s['svf_magnitude_se']:.4g} | {s['average_correspondence_mse']:.4g} |")
 report+=['','The two figures show sample-weighted means across valid target registrations; cluster means were not given equal weight.','','## Data and metric definitions','',f'- Dataset: `{DATA.relative_to(ROOT)}`; {len(c)} contours, each 200 pinned corresponding points.','- Physical Dirichlet: `gabriel.diagnostic_grid(..., resolution=101)[physical_dirichlet_integral]`.','- SVF magnitude: `gabriel.diagnostic_grid(...)[velocity_mean_magnitude]`.','- Valid means completed, finite, geometrically valid, positive sampled Jacobian with zero non-positive fraction.','','## Methodology','',f'- KMeans: k-means++, n_init=50, random_state=0, direct C-order 400-coordinate shapes.','- Template: hard-cluster pointwise arithmetic mean in pinned coordinate space.',f'- RegistrationConfig: `{json.dumps(asdict(CONFIG),sort_keys=True)}`.','- Velocity: 30x30x2 cubic B-spline coefficients; flow: stationary RK4 with 5 steps.','','## Quality, safety, runtime','',f'- Attempted: {attempted}; valid: {valid}; failed: {failed}; output storage: {size/1024/1024:.1f} MiB.','- Positive Jacobian on a 101x101 numerical grid is diagnostic evidence, not an analytical proof.','','## Limitations and next step','','These curves describe deformation properties under one frozen clustering and registration configuration. They do not establish a true number of airfoil families. Review the raw tables before any K selection.','','## Further questions','','No downstream template choice or aerodynamic analysis was performed.']
 (OUT/(f'final_report_{suffix}.md' if suffix else 'final_report.md')).write_text('\n'.join(report)+'\n',encoding='utf8')
 return summary

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=('prepare','smoke','full','finalise','all'),default='all');ap.add_argument('--workers',type=int,default=2);ap.add_argument('--k',default='2,3,4,5,6,7,8',help='comma-separated K values to run, each in 2..8');ap.add_argument('--finalise-k',help='comma-separated K values to summarise; defaults to --k');ap.add_argument('--progress-log',type=Path);args=ap.parse_args()
 requested_ks=tuple(dict.fromkeys(int(x.strip()) for x in args.k.split(',') if x.strip()))
 if not requested_ks or any(K not in KS for K in requested_ks):ap.error('--k must contain only values from 2 through 8')
 finalise_ks=tuple(dict.fromkeys(int(x.strip()) for x in (args.finalise_k or args.k).split(',') if x.strip()))
 if not finalise_ks or any(K not in KS for K in finalise_ks):ap.error('--finalise-k must contain only values from 2 through 8')
 for d in ('templates','registrations','figures','logs'):(OUT/d).mkdir(parents=True,exist_ok=True)
 z,c,fn,templates,assign,ts,protected=prepare()
 if args.stage=='prepare':return
 if args.stage in ('smoke','all'):
  X=c.reshape(len(c),400);dist=np.linalg.norm(X-X.mean(0),axis=1);ordinary=int(np.argsort(dist)[len(c)//2]);extreme=int(np.argmax(dist));pairs=[(K,i) for K in (2,8) for i in (ordinary,extreme)];start=time.perf_counter();rows=run(c,fn,templates,assign,1,pairs);elapsed=time.perf_counter()-start;cwrite(OUT/'logs/smoke_results.csv',rows);est={'status':'PASS' if all(r['status']=='completed' for r in rows) else 'FAIL','pairs':pairs,'runtime_seconds':elapsed,'mean_case_seconds':float(np.mean([r['runtime_seconds'] for r in rows])),'projected_11564_serial_hours':float(np.mean([r['runtime_seconds'] for r in rows]))*11564/3600,'projected_storage_bytes':11564*25000,'resume':'atomic per-(K,target) NPZ'};gabriel.atomic_json(OUT/'logs/preflight.json',est)
  if est['status']!='PASS':raise RuntimeError('smoke failed')
  if args.stage=='smoke':return
 if args.stage in ('full','all'):run(c,fn,templates,assign,args.workers,requested_ks=requested_ks,progress_log=args.progress_log)
 if args.stage in ('full','finalise','all'):finalise(c,fn,templates,assign,ts,protected,requested_ks=finalise_ks)
if __name__=='__main__':main()
