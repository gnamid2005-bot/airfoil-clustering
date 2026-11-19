"""Read-only status view for the detached Gabriel K=2..7 coordinator."""
import json, os, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"outputs/gabriel_big_registration_k2_k7_full_leftmost";R=OUT/"runtime"
state=json.loads((R/"status.json").read_text()) if (R/"status.json").exists() else {}
pid=int((R/"coordinator.pid").read_text()) if (R/"coordinator.pid").exists() else int(state.get("coordinator_pid",0) or 0)
try:os.kill(pid,0);alive=True
except (OSError,ValueError):alive=False
K=state.get("current_K");q=state.get("current_q");base=int(state.get("completed",0));valid=invalid=failed=0;latest=None
if K:
 d=OUT/"registrations"/f"K{K:02d}_q{q:02d}"
 files=sorted(d.glob("*.npz")) if d.exists() else []
 for p in files:
  try:
   import numpy as np
   with np.load(p,allow_pickle=False) as z:r=json.loads(str(z["metrics_json"]))
   if r.get("status")=="completed" and r.get("valid_registration"):valid+=1
   elif r.get("status")=="completed":invalid+=1
   else:failed+=1
  except Exception:failed+=1
  latest=max(latest or 0,p.stat().st_mtime)
 completed=base+len(files)
else:completed=base
elapsed=time.time()-float(state.get("started_epoch",time.time()));rate=completed/elapsed if elapsed>0 else 0;eta=(9912-completed)/rate if rate>0 else None
print(json.dumps({"alive":alive,"pid":pid,"current_K":K,"current_q":q,"completed":completed,"total":9912,"current_branch_valid":valid,"current_branch_invalid":invalid,"current_branch_failed_or_corrupt":failed,"latest_checkpoint_epoch":latest,"elapsed_seconds":elapsed,"estimated_remaining_seconds":eta,"message":state.get("message")},indent=2))
