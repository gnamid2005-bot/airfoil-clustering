"""Persistent sequential coordinator for the Gabriel K=2..7 extension."""
from __future__ import annotations
import csv, json, os, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs/gabriel_big_registration_k2_k7_full_leftmost"
RUNNER=ROOT/"scripts/run_gabriel_big_registration_k2_k7_full_leftmost.py"
PY=ROOT/".venv/bin/python"
PAIRS=((2,1),(3,89),(4,5),(5,3),(6,2),(7,5))
STATE=OUT/"runtime/status.json"; PROGRESS=OUT/"progress.log"

def atomic_json(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True); tmp=p.with_suffix(p.suffix+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n"); os.replace(tmp,p)
def event(message,**extra):
    now=datetime.now(timezone.utc).isoformat(); row={"timestamp":now,"message":message,**extra}
    with PROGRESS.open("a",encoding="utf8") as f:f.write(json.dumps(row,sort_keys=True)+"\n")
    atomic_json(STATE,{"coordinator_pid":os.getpid(),"updated_at":now,**extra,"message":message})
def call(args):
    with (OUT/"runtime/coordinator_commands.log").open("a",encoding="utf8") as f:f.write(" ".join(map(str,args))+"\n")
    subprocess.run([str(x) for x in args],cwd=ROOT,check=True)
def final_tables():
    rows=[]
    for K,q in PAIRS:
        bid=f"K{K:02d}_q{q:02d}"; s=json.loads((OUT/"tables"/f"summary_{bid}.json").read_text())
        with (OUT/"tables"/f"per_registration_{bid}.csv").open() as f: rr=[r for r in csv.DictReader(f) if r["valid_registration"]=="True"]
        def ext(key):
            import numpy as np
            a=np.array([float(r[key]) for r in rr]); sd=float(a.std(ddof=1))
            return dict(mean=float(a.mean()),sd=sd,se=sd/len(a)**.5,median=float(np.median(a)),p95=float(np.quantile(a,.95)),maximum=float(a.max()))
        d=ext("physical_dirichlet_energy");v=ext("velocity_mean_magnitude");m=ext("correspondence_mse")
        rows.append({"K":K,"q":q,"q_selection_rule":"full-range leftmost BIC-or-ICL","total":s["total"],"valid":s["valid"],"invalid":s["invalid"],"failed":s["failed"],**{f"dirichlet_{k}":x for k,x in d.items()},**{f"svf_{k}":x for k,x in v.items()},"mean_MSE":m["mean"],"median_MSE":m["median"],"p95_MSE":m["p95"],"minimum_Jacobian":s["minimum_Jacobian"]})
    p=OUT/"tables/registration_summary_k2_k7_full_leftmost.csv"
    with p.open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    # Preserve the methodological difference explicitly in a combined table.
    combined=list(rows)
    old=ROOT/"outputs/gabriel_big_registration_k8_k11"
    for K,q in ((8,27),(9,14),(10,17),(11,15)):
        s=json.loads((old/"tables"/f"summary_K{K:02d}_q{q:02d}.json").read_text())
        combined.append({"K":K,"q":q,"q_selection_rule":"previous post-spike further-left rule","valid":s["valid"],"dirichlet_mean":s["mean_Dirichlet"],"dirichlet_se":s["SE_Dirichlet"],"svf_mean":s["mean_SVF_magnitude"],"svf_se":s["SE_SVF_magnitude"],"mean_MSE":s["mean_MSE"],"minimum_Jacobian":s["minimum_Jacobian"]})
    fields=list(dict.fromkeys(k for r in combined for k in r));p=OUT/"tables/combined_registration_summary_k2_k11.csv"
    with p.open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(combined)
    return rows,combined
def figures(combined):
    import matplotlib;matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ks=[r["K"] for r in combined];qs=[r["q"] for r in combined]
    fig,ax=plt.subplots(1,2,figsize=(11,4.4))
    for a,key,se,title,y in ((ax[0],"dirichlet_mean","dirichlet_se","Average physical Dirichlet energy","Physical Dirichlet energy"),(ax[1],"svf_mean","svf_se","Average SVF magnitude","SVF mean magnitude")):
        vals=[float(r[key]) for r in combined];errs=[float(r[se]) for r in combined];a.errorbar(ks,vals,yerr=errs,marker="o",capsize=3)
        for x,z,q in zip(ks,vals,qs):a.annotate(f"q={q}",(x,z),xytext=(0,7),textcoords="offset points",ha="center",fontsize=7)
        a.set(title=title,xlabel="Number of templates K",ylabel=y,xticks=ks);a.grid(alpha=.25)
    fig.suptitle("K=2–7: full-range leftmost; K=8–11: previous post-spike rule",fontsize=10);fig.tight_layout();
    fd=OUT/"figures";fd.mkdir(exist_ok=True);fig.savefig(fd/"dirichlet_svf_k2_k11.png",dpi=280);plt.close(fig)
def main():
    (OUT/"runtime").mkdir(parents=True,exist_ok=True)
    (OUT/"runtime/coordinator.pid").write_text(str(os.getpid())+"\n")
    started=time.time();event("coordinator_started",phase="registration",current_K=2,current_q=1,completed=0,total=9912,workers=4,started_epoch=started)
    for branch_no,(K,q) in enumerate(PAIRS):
        event("branch_started",phase="registration",current_K=K,current_q=q,completed=branch_no*1652,total=9912,workers=4,started_epoch=started)
        call([PY,RUNNER,"--stage","run","--workers","4","--branch",f"K{K:02d}_q{q:02d}"])
        event("branch_complete",phase="registration",current_K=K,current_q=q,completed=(branch_no+1)*1652,total=9912,workers=4,started_epoch=started)
    call([PY,RUNNER,"--stage","finalise"]);rows,combined=final_tables();figures(combined)
    event("all_complete",phase="complete",current_K=None,current_q=None,completed=9912,total=9912,workers=0,started_epoch=started,status="PASS")
if __name__=="__main__":main()
