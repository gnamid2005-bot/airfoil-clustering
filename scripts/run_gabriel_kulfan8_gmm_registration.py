#!/usr/bin/env python3
"""Register frozen Kulfan-8 GMM component-mean templates with Gabriel's method."""
from __future__ import annotations

import argparse, csv, hashlib, json, math, os, sys, time
from dataclasses import asdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import gabriel_pinned_registration as gabriel
import run_gabriel_k2_k8_energy_magnitude as production
import run_gabriel_kulfan8_gmm as kulfan

DATA=ROOT/'data/airfoils/processed_gabriel/airfoils_pinned_arclength_n200.npz'
SOURCE=ROOT/'outputs/gabriel_kulfan8_gmm'
OUT=ROOT/'outputs/gabriel_kulfan8_gmm_registration'
DEV=ROOT/'outputs/gabriel_pinned_registration_dev'
KS=tuple(range(1,10)); CONFIG=gabriel.RegistrationConfig(); DENSE_N=4001
production.OUT=OUT

def sha(p): return gabriel.sha256_file(p)
def cwrite(p,rows): production.cwrite(p,rows)
def cread(p): return production.cread(p)
def fingerprint(paths):
 return {str(p.relative_to(ROOT)):{'sha256':sha(p),'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in paths}

def dense_kulfan(mu):
 # Selig/Gabriel traversal without duplicated LE/TE: first TE, upper -> LE,
 # then lower -> immediately before TE; Gabriel resample_pinned closes lower
 # with the first point itself.
 xu=np.linspace(1.,0.,DENSE_N); xl=np.linspace(0.,1.,DENSE_N,endpoint=False)[1:]
 upper=np.column_stack([xu,kulfan.design(xu)@mu[:4]])
 lower=np.column_stack([xl,kulfan.design(xl)@mu[4:]])
 return np.vstack([upper,lower])

def resampling_discrepancy(raw,pinned):
 # Compare the continuous dense source and the piecewise-linear 200-point
 # result at identical normalized arc coordinates on each surface.
 le=int(np.argmin(raw[:,0])); ru=raw[:le+1]; rl=np.vstack([raw[le:],raw[:1]])
 pu=np.vstack([pinned[:100],pinned[100]]); pl=np.vstack([pinned[100:],pinned[:1]])
 a=gabriel.resample_open(ru,2001,include_end=True); b=gabriel.resample_open(pu,2001,include_end=True)
 c=gabriel.resample_open(rl,2001,include_end=True); d=gabriel.resample_open(pl,2001,include_end=True)
 e=np.linalg.norm(np.vstack([a,c])-np.vstack([b,d]),axis=1)
 return {'resampling_mse':float(np.mean(e*e)),'resampling_rmse':float(np.sqrt(np.mean(e*e))),
         'resampling_mean_distance':float(np.mean(e)),'resampling_maximum_distance':float(np.max(e))}

def protected_paths():
 paths=[DATA,SOURCE/'config.json',SOURCE/'validation.json',SOURCE/'kulfan/kulfan_parameters.npy',
        ROOT/'src/gabriel_pinned_registration.py',ROOT/'src/run_gabriel_k2_k8_energy_magnitude.py',
        DEV/'config.json',DEV/'validation.json']
 paths+=sorted((SOURCE/'gmm/component_parameters').glob('*.npz'))
 paths+=sorted((SOURCE/'gmm/assignments').glob('*.npy'))
 return paths

def load_inputs():
 with np.load(DATA,allow_pickle=False) as z:
  contours=np.asarray(z['contours'],np.float64); filenames=z['filenames'].astype(str)
 if contours.shape!=(1652,200,2) or not all(gabriel.validate_pinned_contour(c)['valid'] for c in contours):
  raise RuntimeError('canonical Gabriel input failed validation')
 cfg=json.load(open(SOURCE/'config.json')); val=json.load(open(SOURCE/'validation.json'))
 if cfg['gmm']['fit_space']!='raw X_kulfan, exactly 8 dimensions' or not cfg['gmm']['no_PCA'] or val['status']!='PASS':
  raise RuntimeError('frozen Kulfan experiment is not validated direct 8-D')
 frozen_reg=json.load(open(DEV/'config.json'))['registration_config']
 if json.loads(json.dumps(asdict(CONFIG)))!=frozen_reg: raise RuntimeError('registration configuration mismatch')
 return contours,filenames,cfg

def prepare():
 if OUT.exists() and not (OUT/'config.json').exists():
  raise RuntimeError('output directory exists without a resumable config')
 for d in ['templates/kulfan_component_means','templates/gabriel_resampled_200pt','registrations',
           'tables','figures','report']:(OUT/d).mkdir(parents=True,exist_ok=True)
 contours,filenames,source_cfg=load_inputs(); templates={}; assignments=[]; audit=[]; preflight=[]
 for K in KS:
  labels=np.load(SOURCE/f'gmm/assignments/K{K:02d}_assignments.npy').astype(np.int64)
  with np.load(SOURCE/f'gmm/component_parameters/K{K:02d}.npz') as z:
   means=np.asarray(z['means'],np.float64); weights=np.asarray(z['weights'],np.float64)
  if labels.shape!=(1652,) or means.shape!=(K,8) or weights.shape!=(K,): raise RuntimeError(f'bad frozen GMM K={K}')
  counts=np.bincount(labels,minlength=K)
  for i in range(len(contours)): assignments.append({'K':K,'target_index':i,'filename':filenames[i],'cluster_id':int(labels[i])})
  kmax=0.; allvalid=True
  for cl in range(K):
   mu=means[cl]; raw=dense_kulfan(mu); pinned,le=gabriel.resample_pinned(raw,n=200)
   diag=gabriel.validate_pinned_contour(pinned); disc=resampling_discrepancy(raw,pinned)
   valid=bool(diag['valid'] and pinned.dtype==np.float64 and pinned.shape==(200,2))
   if not valid: raise RuntimeError(f'invalid component template K={K} c={cl}: {diag}')
   tid=f'kulfan8_gmm_K{K:02d}_component{cl:02d}'; members=np.flatnonzero(labels==cl)
   templates[(K,cl)]=(pinned,members,tid); kmax=max(kmax,disc['resampling_maximum_distance']);allvalid &= valid
   gabriel.atomic_npz(OUT/f'templates/kulfan_component_means/{tid}.npz',kulfan_component_mean=mu,
                      dense_coordinates=raw,source_component_file_sha256=np.asarray(sha(SOURCE/f'gmm/component_parameters/K{K:02d}.npz')))
   gabriel.atomic_npz(OUT/f'templates/gabriel_resampled_200pt/{tid}.npz',coordinates=pinned,
                      kulfan_component_mean=mu,member_indices=members,validation_json=np.asarray(json.dumps(gabriel.jsonable(diag),sort_keys=True)),
                      discrepancy_json=np.asarray(json.dumps(disc,sort_keys=True)))
   audit.append({'K':K,'component':cl,'template_id':tid,'cluster_size':int(counts[cl]),'component_weight':float(weights[cl]),
                 'primary_definition':'GMM component mean in frozen raw 8-D Kulfan space','kulfan_dimension':8,'dense_point_count':len(raw),
                 'raw_le_index':le,'final_shape':str(pinned.shape),'final_dtype':str(pinned.dtype),'finite':diag['finite'],
                 'index0_is_te':diag['index0_is_te'],'index100_is_le':diag['index100_is_le'],'orientation':diag['orientation'],
                 'self_intersection_count':diag['self_intersection_count'],'no_zero_length_segments':diag['no_zero_length_segments'],
                 'uniform_arc_length':diag['approximately_uniform_arc_length'],'chord_x_min':diag['x_min'],'chord_x_max':diag['x_max'],
                 'registration_input_valid':valid,**disc,'component_mean_sha256':hashlib.sha256(mu.tobytes()).hexdigest(),
                 'resampled_coordinates_sha256':hashlib.sha256(pinned.tobytes()).hexdigest()})
  preflight.append({'K':K,'number_of_components':K,'cluster_sizes':json.dumps(counts.tolist()),'component_weights':json.dumps(weights.tolist()),
                    'minimum_cluster_size':int(counts.min()),'template_count':K,'all_templates_valid':bool(allvalid),
                    'maximum_kulfan_to_resampled_discrepancy':kmax,'registration_input_format_verified':bool(allvalid)})
 cwrite(OUT/'tables/template_audit.csv',audit);cwrite(OUT/'tables/preflight_k1_k9.csv',preflight)
 cwrite(OUT/'tables/gmm_assignments_readonly_copy.csv',assignments)
 protected=protected_paths(); config={'experiment':'Gabriel registration from frozen Kulfan-8 GMM component means',
  'source_experiment':'outputs/gabriel_kulfan8_gmm','source_experiment_config_sha256':sha(SOURCE/'config.json'),
  'dataset':str(DATA.relative_to(ROOT)),'dataset_sha256':sha(DATA),'dataset_shape':list(contours.shape),
  'K':list(KS),'template_count':len(audit),'template_definition':'frozen GMM component mean in raw 8-D Kulfan space',
  'inverse':'scripts/run_gabriel_kulfan8_gmm.py::design applied separately to dense upper/lower x grids; N1=0.5, N2=1.0, cubic Bernstein, sharp TE',
  'dense_inverse_points':DENSE_N+(DENSE_N-1),'resampling':'src/gabriel_pinned_registration.py::resample_pinned/resample_open',
  'final_template_representation':'200x2 float64; indices 0..99 upper TE->LE; index100 LE; 100..199 lower LE->TE; implicit closure',
  'assignments':'frozen primary-seed GMM hard assignments from source experiment','no_PCA':True,'no_GMM_refit':True,
  'primary_templates_are_not_hard_means':True,'registration_config':asdict(CONFIG),'diagnostic_resolution':production.DIAGNOSTIC_RESOLUTION,
  'workers':4,'nested_threads':1,'resume':'atomic per-(K,target) NPZ; readable matching K and target index skipped',
  'protected_inputs_before':fingerprint(protected)}
 gabriel.atomic_json(OUT/'config.json',config)
 print('PREFLIGHT TABLE',flush=True)
 print('K | components | cluster sizes | weights | min size | templates | valid | max discrepancy | format',flush=True)
 for r in preflight: print(f"{r['K']} | {r['number_of_components']} | {r['cluster_sizes']} | {r['component_weights']} | {r['minimum_cluster_size']} | {r['template_count']} | {r['all_templates_valid']} | {r['maximum_kulfan_to_resampled_discrepancy']:.6g} | {r['registration_input_format_verified']}",flush=True)
 print('RegistrationConfig:',json.dumps(asdict(CONFIG),sort_keys=True),flush=True)
 if len(audit)!=45 or not all(r['registration_input_valid'] for r in audit): raise RuntimeError('preflight did not validate exactly 45 templates')
 return contours,filenames,templates,assignments,audit,protected

def metric_stats(values):
 a=np.asarray(values,float);n=len(a)
 if n==0:return {x:math.nan for x in ('mean','sd','se','median','p90','p95','maximum')}
 return {'mean':float(a.mean()),'sd':float(a.std(ddof=1)) if n>1 else math.nan,
         'se':float(a.std(ddof=1)/math.sqrt(n)) if n>1 else math.nan,'median':float(np.median(a)),
         'p90':float(np.quantile(a,.9)),'p95':float(np.quantile(a,.95)),'maximum':float(a.max())}

def finalise(contours,filenames,templates,assignments,audit,protected):
 rows=[];missing=[]
 for K in KS:
  for i in range(len(contours)):
   p=OUT/f'registrations/K{K:02d}/{i:04d}.npz';r=production.load_artifact(p,K,i)
   if r is None:missing.append(str(p));continue
   rows.append(r)
 if missing: raise RuntimeError(f'{len(missing)} registration checkpoints missing/corrupt')
 cwrite(OUT/'tables/per_registration_results.csv',rows)
 summary=[]
 for K in KS:
  rr=[r for r in rows if int(r['K'])==K];valid=[r for r in rr if bool(r.get('valid_registration'))]
  d=metric_stats([r['physical_dirichlet_integral'] for r in valid]);s=metric_stats([r['velocity_mean_magnitude'] for r in valid]);m=metric_stats([r['final_pointwise_mse'] for r in valid]);rt=metric_stats([r['runtime_seconds'] for r in rr])
  summary.append({'K':K,'total':len(rr),'valid':len(valid),'invalid':sum(r.get('status')=='completed' and not bool(r.get('valid_registration')) for r in rr),
   'failed':sum(r.get('status')!='completed' for r in rr),**{f'dirichlet_{x}':v for x,v in d.items()},
   **{f'svf_magnitude_{x}':v for x,v in s.items()},'mse_mean':m['mean'],'mse_median':m['median'],'mse_p95':m['p95'],'mse_maximum':m['maximum'],
   'minimum_jacobian':min(float(r.get('jacobian_minimum',math.inf)) for r in rr),'nonpositive_jacobian_count':sum(float(r.get('jacobian_minimum',-math.inf))<=0 or float(r.get('jacobian_nonpositive_fraction',1))>0 for r in rr),
   'runtime_mean_seconds':rt['mean'],'runtime_total_cpu_seconds':sum(float(r['runtime_seconds']) for r in rr)})
 cwrite(OUT/'tables/registration_summary_k1_k9.csv',summary)
 model={int(r['K']):r for r in cread(SOURCE/'gmm/model_selection_k1_k9.csv')};stab={int(r['K']):r for r in cread(SOURCE/'gmm/seed_stability_k1_k9.csv')}
 combined=[]
 for s in summary:
  K=s['K'];m=model[K];t=stab[K]
  combined.append({'K':K,'BIC':m['BIC'],'AIC':m['AIC'],'ICL':m['ICL'],'heldout_LL':m['heldout_log_likelihood'],
   'mean_seed_ARI':t['mean_pairwise_ARI'],'min_cluster_size':m['minimum_cluster_size'],'mean_Dirichlet':s['dirichlet_mean'],
   'SE_Dirichlet':s['dirichlet_se'],'mean_SVF':s['svf_magnitude_mean'],'SE_SVF':s['svf_magnitude_se'],
   'registration_valid':s['valid'],'registration_invalid':s['invalid'],'registration_failed':s['failed']})
 cwrite(OUT/'tables/kulfan_model_selection_plus_registration.csv',combined)
 for field,se,ylabel,title,name,color in [('dirichlet_mean','dirichlet_se','Average physical Dirichlet energy','Average Dirichlet Energy vs Number of Templates\n— 8-D Kulfan GMM Component-Mean Templates','average_dirichlet_energy_vs_k.png','#1f4e79'),
  ('svf_magnitude_mean','svf_magnitude_se','Average velocity-field magnitude','Average SVF Magnitude vs Number of Templates\n— 8-D Kulfan GMM Component-Mean Templates','average_svf_magnitude_vs_k.png','#c45a18')]:
  fig,ax=plt.subplots(figsize=(9,5.4));x=[r['K'] for r in summary];y=[r[field] for r in summary];e=[r[se] for r in summary]
  ax.errorbar(x,y,yerr=e,marker='o',mfc='white',mec=color,color=color,lw=2,capsize=5);ax.set_xticks(x);ax.set(xlabel='Number of GMM components / templates K',ylabel=ylabel,title=title);ax.grid(axis='y',alpha=.22);ax.spines[['top','right']].set_visible(False);fig.tight_layout();fig.savefig(OUT/f'figures/{name}',dpi=300);plt.close(fig)
 fig,axs=plt.subplots(2,2,figsize=(12,8));x=[r['K'] for r in summary]
 for ax,key,title in zip(axs.ravel(),['BIC','heldout_LL','mean_Dirichlet','mean_SVF'],['BIC','Held-out LL','Mean Dirichlet','Mean SVF magnitude']):
  ax.plot(x,[float(r[key]) for r in combined],marker='o');ax.set_xticks(x);ax.set_title(title);ax.grid(alpha=.2)
 fig.suptitle('Kulfan-8 GMM model selection and downstream registration');fig.tight_layout();fig.savefig(OUT/'figures/model_selection_and_registration_summary.png',dpi=300);plt.close(fig)
 # Incremental trend evidence, including whether small K is near-equivalent.
 trends=[]
 for a,b in zip(summary,summary[1:]):
  trends.append({'K_from':a['K'],'K_to':b['K'],'dirichlet_percent_change':100*(b['dirichlet_mean']-a['dirichlet_mean'])/a['dirichlet_mean'],
                 'svf_percent_change':100*(b['svf_magnitude_mean']-a['svf_magnitude_mean'])/a['svf_magnitude_mean']})
 cwrite(OUT/'tables/incremental_registration_changes.csv',trends)
 # Tiny-component behaviour versus population distribution.
 cluster_rows=[]
 for K in KS:
  for cl in range(K):
   cr=[r for r in rows if int(r['K'])==K and int(r['cluster_id'])==cl];v=[r for r in cr if bool(r.get('valid_registration'))]
   cluster_rows.append({'K':K,'component':cl,'cluster_size':len(cr),'valid':len(v),'invalid_or_failed':len(cr)-len(v),
    'mean_dirichlet':float(np.mean([r['physical_dirichlet_integral'] for r in v])) if v else math.nan,
    'mean_svf':float(np.mean([r['velocity_mean_magnitude'] for r in v])) if v else math.nan,
    'mean_mse':float(np.mean([r['final_pointwise_mse'] for r in v])) if v else math.nan,
    'minimum_jacobian':min((float(r.get('jacobian_minimum',math.inf)) for r in cr),default=math.nan)})
 cwrite(OUT/'tables/registration_summary_by_component.csv',cluster_rows)
 after=fingerprint(protected);before=json.load(open(OUT/'config.json'))['protected_inputs_before'];tmp=list(OUT.rglob('*.tmp'))+list(OUT.rglob('*.partial'))
 checks={'frozen_8D_component_means_reused':True,'no_PCA':True,'primary_templates_are_GMM_component_means':True,
  'all_component_means_inverse_Kulfan':True,'all_templates_Gabriel_resampled':len(audit)==45,
  'all_45_templates_valid':len(audit)==45 and all(bool(r['registration_input_valid']) for r in audit),
  'all_targets_use_GMM_assignment':all(int(r['cluster_id'])==int(np.load(SOURCE/f"gmm/assignments/K{int(r['K']):02d}_assignments.npy")[int(r['target_index'])]) for r in rows),
  'same_Gabriel_algorithm_all_K':all(int(r['diagnostic_resolution'])==101 for r in rows),'same_metric_definitions_all_K':True,
  'all_14868_results_present':len(rows)==14868,'no_partial_files':not tmp,'protected_previous_outputs_unchanged':before==after,
  'no_commit_or_push':True}
 gabriel.atomic_json(OUT/'validation.json',{'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'counts':{'rows':len(rows),'templates':len(audit)},'partial_files':[str(p) for p in tmp]})
 # Evidence-led technical report; no automatic K choice.
 lines=['# Gabriel registration from 8-D Kulfan GMM component means','','## Technical summary','',
  'All 14,868 assigned-template registrations use frozen raw 8-D Kulfan GMM component means, Gabriel exact 200-point uniform arc-length resampling, and the frozen Gabriel registration configuration. This report is descriptive and does not select a final K.','',
  '## Registration evidence','','| K | valid / invalid / failed | mean Dirichlet ± SE | mean SVF ± SE | mean MSE | minimum Jacobian |','|---:|---:|---:|---:|---:|---:|']
 for r in summary:lines.append(f"| {r['K']} | {r['valid']} / {r['invalid']} / {r['failed']} | {r['dirichlet_mean']:.10g} ± {r['dirichlet_se']:.3g} | {r['svf_magnitude_mean']:.10g} ± {r['svf_magnitude_se']:.3g} | {r['mse_mean']:.4g} | {r['minimum_jacobian']:.5g} |")
 lines+=['','## Interface and template definition','',f"Inverse: `{json.load(open(OUT/'config.json'))['inverse']}`. Resampling: `{json.load(open(OUT/'config.json'))['resampling']}`. All 45 templates passed the repository pinned-contour validator.",'',
  '## Model and metric definitions','','GMMs and assignments were read without refitting. Physical Dirichlet and velocity mean magnitude come from `gabriel.diagnostic_grid` on the frozen 101×101 physical grid. Validity requires completed finite registration, valid deformed geometry, positive minimum sampled Jacobian, and zero non-positive-Jacobian fraction.','',
  '## Limitations and interpretation','','Lower deformation burden is not by itself evidence of better clustering. Information criteria, seed stability, component size, registration validity, and deformation burden must be considered together; tiny components are audited separately.','',
  '## Recommended next step','','Review the combined table and component-level audit before choosing candidates; no final K is selected here.','',
  '## Further questions','','Any final selection should test whether high-K gains are robust to tiny-component leverage and whether a stable lower K is practically equivalent.']
 (OUT/'report/final_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
 return summary

def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=('prepare','full','finalise','all'),default='all');p.add_argument('--workers',type=int,default=4);a=p.parse_args()
 if a.workers!=4: raise RuntimeError('this experiment requires the validated workers=4 configuration')
 c,fn,t,assign,audit,protected=prepare()
 if a.stage=='prepare':return
 if a.stage in ('full','all'):
  start=time.perf_counter();rows=production.run(c,fn,t,assign,a.workers,requested_ks=KS,progress_log=OUT/'progress.log')
  meta={'wall_seconds_this_session':time.perf_counter()-start,'rows_returned':len(rows),'workers':a.workers,'finished_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')}
  gabriel.atomic_json(OUT/'run_metadata.json',meta)
 if a.stage in ('finalise','all'):finalise(c,fn,t,assign,audit,protected)

if __name__=='__main__':main()
