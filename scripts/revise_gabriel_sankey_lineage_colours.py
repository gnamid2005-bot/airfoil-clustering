"""Revise only the visual encoding of the saved Gabriel K=8..11 Sankey."""
from __future__ import annotations
import csv, hashlib, json, os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR","/tmp/mpl-gabriel-sankey-lineage")
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mc
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/"outputs/gabriel_big_registration_k8_k11"
OUT=ROOT/"outputs/gabriel_cross_k_cluster_evolution"
FIG=OUT/"figures";TAB=OUT/"tables"
PAIRS={8:27,9:14,10:17,11:15};ADJ=((8,9),(9,10),(10,11));N=1652
BASE=["#2B6CB0","#D97706","#2F855A","#C2415D","#7C3AED","#0F766E","#A16207","#64748B"]

def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def write_csv(p,rows):
 rows=list(rows);p.parent.mkdir(parents=True,exist_ok=True);fields=list(dict.fromkeys(k for r in rows for k in r));t=p.with_suffix(p.suffix+".tmp")
 with t.open("w",newline="",encoding="utf8") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 os.replace(t,p)
def lighten(hexcolor,amount):
 rgb=np.array(mc.to_rgb(hexcolor));return mc.to_hex(rgb+(1-rgb)*amount)
def main():
 labels={};srcs=[]
 for k,q in PAIRS.items():
  p=SRC/"assignments"/f"K{k:02d}_q{q:02d}_labels.npy";labels[k]=np.load(p).astype(int);srcs.append(p)
  if labels[k].shape!=(N,) or set(np.unique(labels[k]))!=set(range(k)):raise RuntimeError(f"assignment mismatch K={k}")
 before={str(p.relative_to(ROOT)):sha(p) for p in srcs}
 matrices={};flows=[]
 for a,b in ADJ:
  m=np.zeros((a,b),int);np.add.at(m,(labels[a],labels[b]),1);matrices[(a,b)]=m
  old=np.loadtxt(TAB/f"contingency_K{a}_K{b}.csv",delimiter=",",skiprows=1,usecols=range(1,b+1))
  if not np.array_equal(m,old):raise RuntimeError(f"flow count drift K{a}->K{b}")
  for i in range(a):
   for j in range(b):
    if m[i,j]:flows.append({"source_K":a,"source_raw_cluster":i,"target_K":b,"target_raw_cluster":j,"count":int(m[i,j])})
 # Lineage propagation. Each child takes its largest contributor as parent.
 lineage={8:{c:c for c in range(8)}};color={8:{c:BASE[c] for c in range(8)}};parent={8:{c:"" for c in range(8)}};variant={8:{c:"base" for c in range(8)}}
 mappings=[]
 for a,b in ADJ:
  m=matrices[(a,b)];parent[b]={j:int(np.argmax(m[:,j])) for j in range(b)};lineage[b]={j:lineage[a][parent[b][j]] for j in range(b)};color[b]={};variant[b]={}
  children={i:[j for j in range(b) if parent[b][j]==i] for i in range(a)}
  dominant_child={i:max(js,key=lambda j:(m[i,j],-j)) for i,js in children.items() if js}
  for i,js in children.items():
   secondary_rank=0
   for j in sorted(js,key=lambda x:(-m[i,x],x)):
    if j==dominant_child[i]:color[b][j]=color[a][i];variant[b][j]="exact parent colour (dominant child)"
    else:
     secondary_rank+=1;color[b][j]=lighten(color[a][i],min(.18+.13*(secondary_rank-1),.48));variant[b][j]=f"lighter lineage shade {secondary_rank} (secondary child)"
  for j in range(b):
   i=parent[b][j];col=m[:,j];order=np.argsort(col)[::-1];second=int(order[1])
   mappings.append({"K":b,"raw_cluster_label":j,"dominant_parent_K":a,"dominant_parent":i,"dominant_parent_n":int(col[i]),"cluster_size":int(col.sum()),"dominant_parent_fraction":float(col[i]/col.sum()),"second_parent":second,"second_parent_n":int(col[second]),"second_parent_fraction":float(col[second]/col.sum()),"lineage_id":f"L{lineage[b][j]} (root K8:C{lineage[b][j]})","assigned_colour":color[b][j],"colour_variant_rule":variant[b][j],"merge_colour_rule":"largest-source parent; no blending"})
 # Strict size-descending visual order, with raw label only as deterministic tie breaker.
 orders={};size_rows=[]
 for k in PAIRS:
  sizes=np.bincount(labels[k],minlength=k);orders[k]=sorted(range(k),key=lambda c:(-sizes[c],c))
  if any(sizes[orders[k][i]]<sizes[orders[k][i+1]] for i in range(k-1)):raise RuntimeError("size order failure")
  for rank,c in enumerate(orders[k],1):size_rows.append({"K":k,"raw_cluster_label":c,"cluster_size":int(sizes[c]),"vertical_rank":rank,"lineage_id":f"L{lineage[k][c]} (root K8:C{lineage[k][c]})","dominant_parent":"" if k==8 else parent[k][c],"assigned_colour_family":BASE[lineage[k][c]],"assigned_node_colour":color[k][c],"shade_rule":variant[k][c]})
 write_csv(TAB/"sankey_size_order.csv",size_rows);write_csv(TAB/"sankey_lineage_mapping.csv",mappings)
 # Draw alluvial with source-node ribbon colours.
 fig,ax=plt.subplots(figsize=(15.5,9.5));ax.set_xlim(-.18,3.18);ax.set_ylim(-.015,1.075);ax.axis("off");node={};gap=.007
 for xi,k in enumerate(PAIRS):
  sizes=np.bincount(labels[k],minlength=k);usable=1-gap*(k-1);y=1;small=[]
  for c in orders[k]:
   h=sizes[c]/N*usable;y-=h;node[(k,c)]=(xi,y,h)
   ax.add_patch(Rectangle((xi-.032,y),.064,h,facecolor=color[k][c],edgecolor="#243044",lw=.75,zorder=4))
   if h>=.032:ax.text(xi,y+h/2,f"C{c}\nn={sizes[c]}",ha="center",va="center",fontsize=6.6,color="#111827",zorder=5)
   else:small.append((c,y+h/2,int(sizes[c])))
   y-=gap
  # Place labels for very small bottom nodes outside the rectangles with
  # collision-free fixed spacing; this changes labels only, never node order.
  if small:
   desired=[z[1] for z in small];placed=[]
   for yc in desired:placed.append(min(yc,placed[-1]-.027) if placed else yc)
   if placed[-1]<.015:
    shift=.015-placed[-1];placed=[v+shift for v in placed]
   side=1 if xi<3 else -1;tx=xi+side*.075
   for (c,yc,n),yl in zip(small,placed):
    ax.plot([xi+side*.032,tx-side*.008],[yc,yl],color="#4B5563",lw=.55,zorder=6)
    ax.text(tx,yl,f"C{c}  n={n}",ha="left" if side>0 else "right",va="center",fontsize=6.1,color="#111827",zorder=7)
 outoff={z:0. for z in node};inoff={z:0. for z in node}
 for a,b in ADJ:
  m=matrices[(a,b)];usable_a=1-gap*(a-1);usable_b=1-gap*(b-1)
  # Traverse in displayed order for reproducible ribbon stacking.
  for i in orders[a]:
   for j in orders[b]:
    n=int(m[i,j])
    if not n:continue
    xa,ya,_=node[(a,i)];xb,yb,_=node[(b,j)];h0=n/N*usable_a;h1=n/N*usable_b;y0=ya+outoff[(a,i)];y1=yb+inoff[(b,j)];outoff[(a,i)]+=h0;inoff[(b,j)]+=h1
    verts=[(xa+.032,y0),(xa+.39,y0),(xb-.39,y1),(xb-.032,y1),(xb-.032,y1+h1),(xb-.39,y1+h1),(xa+.39,y0+h0),(xa+.032,y0+h0),(xa+.032,y0)]
    codes=[MplPath.MOVETO,MplPath.CURVE4,MplPath.CURVE4,MplPath.CURVE4,MplPath.LINETO,MplPath.CURVE4,MplPath.CURVE4,MplPath.CURVE4,MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(verts,codes),facecolor=color[a][i],alpha=.28,edgecolor=color[a][i],lw=.18,zorder=1))
 for x,(k,q) in enumerate(PAIRS.items()):ax.text(x,1.035,f"K={k}, q={q}",ha="center",va="bottom",fontsize=12,weight="bold",color="#172033")
 fig.suptitle("Airfoil Cluster Evolution Across K",y=.985,fontsize=19,weight="semibold",color="#172033")
 ax.text(1.5,1.067,"Flow width = number of identical airfoils",ha="center",fontsize=10.5,color="#4B5563")
 ax.text(1.5,-.007,"Nodes are ordered by cluster size; colours track dominant cluster lineage. Secondary split children use lighter shades.",ha="center",va="top",fontsize=9,color="#4B5563")
 png=FIG/"cluster_evolution_sankey_lineage_colours.png";pdf=FIG/"cluster_evolution_sankey_lineage_colours.pdf";fig.savefig(png,dpi=300,bbox_inches="tight",facecolor="white");fig.savefig(pdf,bbox_inches="tight",facecolor="white");plt.close(fig)
 after={str(p.relative_to(ROOT)):sha(p) for p in srcs};checks={"cluster_sizes_unchanged":True,"flow_counts_unchanged":True,"adjacent_flow_totals_1652":all(m.sum()==N for m in matrices.values()),"raw_labels_unchanged":before==after,"columns_size_descending":all(all(np.bincount(labels[k],minlength=k)[orders[k][i]]>=np.bincount(labels[k],minlength=k)[orders[k][i+1]] for i in range(k-1)) for k in PAIRS),"lineage_from_overlap_not_labels":True,"dominant_continuations_same_colour":all(color[b][j]==color[a][parent[b][j]] for a,b in ADJ for i,js in {x:[y for y in range(b) if parent[b][y]==x] for x in range(a)}.items() if js for j in [max(js,key=lambda y:(matrices[(a,b)][i,y],-y))]),"no_PCA_GMM_registration":True,"original_sankey_not_overwritten":(FIG/"cluster_evolution_sankey.png").exists()}
 (OUT/"audit/sankey_lineage_revision_validation.json").write_text(json.dumps({"status":"PASS" if all(checks.values()) else "FAIL","checks":checks,"source_assignment_hashes":before,"flow_totals":{f"K{a}_K{b}":int(matrices[(a,b)].sum()) for a,b in ADJ},"colour_rule":"K8 distinct base colours; child inherits dominant-parent lineage; largest-flow child gets exact parent colour; additional children get progressively lighter shades; merges use largest-source parent with no blending","ordering_rule":"cluster size descending within every K; raw label breaks exact ties"},indent=2)+"\n")
 print(json.dumps({"status":"PASS" if all(checks.values()) else "FAIL","orders":{str(k):[(c,int(np.bincount(labels[k],minlength=k)[c])) for c in orders[k]] for k in PAIRS},"lineage_mapping":mappings,"outputs":[str(png),str(pdf)]},indent=2))
if __name__=="__main__":main()
