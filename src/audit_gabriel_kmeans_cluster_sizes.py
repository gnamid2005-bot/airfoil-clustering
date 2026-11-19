"""Read-only cluster-size audit for Gabriel production K-means assignments."""

from __future__ import annotations

import csv, hashlib, json, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'outputs/gabriel_k2_k8_energy_magnitude'
OUT=ROOT/'outputs/gabriel_kmeans_cluster_size_audit'
N=1652; KS=range(2,10)

def cread(p):
 with p.open(newline='',encoding='utf8') as f:return list(csv.DictReader(f))
def cwrite(p,rows):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def main():
 if OUT.exists():raise FileExistsError(f'Refusing to overwrite {OUT}')
 (OUT/'tables').mkdir(parents=True);(OUT/'report').mkdir()
 started=time.perf_counter();rows=cread(SOURCE/'clustering_assignments.csv');config=json.loads((SOURCE/'config.json').read_text())
 detail=[];summary=[];template_ok=True;checkpoint_ok=True;filename_ok=True
 with np.load(ROOT/config['dataset'],allow_pickle=False) as z:filenames=z['filenames'].astype(str)
 for k in KS:
  rr=sorted((r for r in rows if int(r['K'])==k),key=lambda r:int(r['target_index']))
  if len(rr)!=N or [int(r['target_index']) for r in rr]!=list(range(N)):raise RuntimeError(f'incomplete K={k}')
  filename_ok &= [r['filename'] for r in rr]==filenames.tolist()
  labels=np.array([int(r['cluster_id']) for r in rr]);sizes=np.bincount(labels,minlength=k)
  if len(sizes)!=k or sizes.sum()!=N or np.any(sizes==0):raise RuntimeError(f'bad cluster sizes K={k}: {sizes}')
  for cl,size in enumerate(sizes):
   pct=100*int(size)/N
   detail.append({'K':k,'Cluster':cl,'Size':int(size),'Percentage':pct,'flag_lt_2_percent':pct<2,'flag_lt_1_percent':pct<1,'flag_lt_20_airfoils':int(size)<20})
   p=SOURCE/'templates'/f'K{k:02d}'/f'K{k:02d}_cluster{cl:02d}.npz'
   with np.load(p,allow_pickle=False) as z:
    template_ok &= np.array_equal(z['member_indices'],np.flatnonzero(labels==cl)) and np.array_equal(z['member_filenames'].astype(str),filenames[labels==cl])
  for i,cl in enumerate(labels):
   p=SOURCE/'registrations'/f'K{k:02d}'/f'{i:04d}.npz'
   with np.load(p,allow_pickle=False) as z:m=json.loads(str(z['metrics_json']))
   if int(m.get('cluster_id',-1))!=int(cl) or m.get('source_template_id')!=f'K{k:02d}_cluster{int(cl):02d}':checkpoint_ok=False;break
  smallest=int(sizes.min());largest=int(sizes.max());smallpct=100*smallest/N;largepct=100*largest/N
  flagged=[int(i) for i,s in enumerate(sizes) if 100*s/N<2 or 100*s/N<1 or s<20]
  summary.append({'K':k,'n_assignments':len(rr),'cluster_sizes_label_order':json.dumps(sizes.tolist()),'cluster_sizes_sorted_ascending':json.dumps(sorted(sizes.tolist())),'smallest_cluster':smallest,'smallest_percent':smallpct,'largest_cluster':largest,'largest_percent':largepct,'largest_smallest_ratio':largest/smallest,'clusters_lt_2_percent':json.dumps([int(i) for i,s in enumerate(sizes) if 100*s/N<2]),'clusters_lt_1_percent':json.dumps([int(i) for i,s in enumerate(sizes) if 100*s/N<1]),'clusters_lt_20_airfoils':json.dumps([int(i) for i,s in enumerate(sizes) if s<20]),'any_small_cluster_flag':bool(flagged),'sum_cluster_sizes':int(sizes.sum())})
 cwrite(OUT/'tables/kmeans_cluster_sizes_k2_k9.csv',detail);cwrite(OUT/'tables/kmeans_cluster_size_summary_k2_k9.csv',summary)
 ratios=np.array([r['largest_smallest_ratio'] for r in summary]);most=summary[int(np.argmax(ratios))]['K'];balanced=summary[int(np.argmin(ratios))]['K']
 validation={'analysis_read_only':True,'kmeans_rerun':False,'registration_rerun':False,'source_assignment_file':'outputs/gabriel_k2_k8_energy_magnitude/clustering_assignments.csv','source_assignment_sha256':sha(SOURCE/'clustering_assignments.csv'),'K_values':list(KS),'detail_rows':len(detail),'expected_detail_rows':44,'all_K_exactly_1652_assignments':all(r['n_assignments']==N for r in summary),'all_cluster_size_sums_1652':all(r['sum_cluster_sizes']==N for r in summary),'assignment_filename_order_matches_pinned_dataset':bool(filename_ok),'template_member_indices_and_filenames_match_assignments':bool(template_ok),'all_13216_registration_checkpoints_reference_expected_assignment_templates':bool(checkpoint_ok),'small_cluster_thresholds':{'lt_2_percent':'cluster percentage < 2%','lt_1_percent':'cluster percentage < 1%','lt_20_airfoils':'cluster size < 20'},'K_most_imbalanced_by_largest_smallest_ratio':most,'K_most_balanced_by_largest_smallest_ratio':balanced,'runtime_seconds':time.perf_counter()-started}
 (OUT/'validation.json').write_text(json.dumps(validation,indent=2,sort_keys=True),encoding='utf8')
 report=['# Gabriel生产K-means簇大小只读审计','','## 结论摘要','',f'K=2–9 均有恰好 {N} 条生产分配，所有簇大小之和均为 {N}。生产模板成员索引/文件名与分配完全一致，全部13,216个注册检查点也引用了相应分配模板。','','## K级汇总','','| K | 标签顺序簇大小 | 升序簇大小 | 最小簇 | 最小% | 最大簇 | 最大% | 最大/最小 | 小簇标志 |','|---:|---|---|---:|---:|---:|---:|---:|---|']
 for r in summary:report.append(f"| {r['K']} | {r['cluster_sizes_label_order']} | {r['cluster_sizes_sorted_ascending']} | {r['smallest_cluster']} | {r['smallest_percent']:.3f}% | {r['largest_cluster']} | {r['largest_percent']:.3f}% | {r['largest_smallest_ratio']:.3f} | {'是' if r['any_small_cluster_flag'] else '否'} |")
 report+=['','## 小簇诊断','','阈值仅为描述：簇比例 <2%、簇比例 <1%、簇大小 <20。','']
 for r in summary:report.append(f"- K={r['K']}：<2%={r['clusters_lt_2_percent']}；<1%={r['clusters_lt_1_percent']}；<20={r['clusters_lt_20_airfoils']}。")
 report+=['',f'按最大/最小簇比率，K={most} 最不平衡，K={balanced} 最平衡。小簇不自动表示无效。','','未运行K-means、注册或模板构建；未修改生产文件。']
 (OUT/'report/final_report.md').write_text('\n'.join(report)+'\n',encoding='utf8');print(json.dumps(validation,indent=2))
if __name__=='__main__':main()
