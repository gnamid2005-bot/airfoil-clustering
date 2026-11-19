"""Persistent sequential coordinator: finish K=6, then run K=7,8,9."""
from __future__ import annotations
import json,os,re,subprocess,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/gabriel_k2_k8_energy_magnitude';LOG=OUT/'resume_k6_k9.log';STDOUT=OUT/'resume_k6_k9_stdout.log'
ENV={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','MPLCONFIGDIR':'/tmp/gabriel-production-k6-k9-mpl'}
def log(event,**fields):
 with LOG.open('a',encoding='utf8') as f:f.write(json.dumps({'timestamp':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'event':event,**fields},sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
def count(k):
 good=0
 for p in (OUT/'registrations'/f'K{k:02d}').glob('*.npz'):
  try:
   with np.load(p,allow_pickle=False) as z:r=json.loads(str(np.asarray(z['metrics_json']).item()))
   good+=int(int(r['K'])==k and int(r['target_index'])==int(p.stem))
  except Exception:pass
 return good
def screen_session_names(screen_output):
 return set(re.findall(r'^\s*\d+\.([^\s]+)\s+\(',screen_output,re.MULTILINE))
def k6_screen_present(screen_output):
 return 'gabriel_k6' in screen_session_names(screen_output)
def k6_screen_alive():
 r=subprocess.run(['/usr/bin/screen','-ls'],capture_output=True,text=True);return k6_screen_present(r.stdout+r.stderr)
def run(args):
 with STDOUT.open('a',encoding='utf8') as f:return subprocess.run(args,cwd=ROOT,env=ENV,stdout=f,stderr=subprocess.STDOUT).returncode
def main():
 while k6_screen_alive():
  n=count(6);log('waiting_for_existing_k6',current_K=6,completed=n,remaining=1652-n);time.sleep(30)
 n=count(6);log('existing_k6_screen_ended',completed=n,remaining=1652-n)
 if n<1652:
  rc=run([str(ROOT/'.venv/bin/python'),'src/run_gabriel_k2_k8_energy_magnitude.py','--stage','full','--k','6','--finalise-k','2,3,4,5,6','--workers','4','--progress-log',str(LOG)]);log('k6_resume_exit',returncode=rc,completed=count(6))
  if rc or count(6)!=1652:raise SystemExit('K=6 did not complete')
 rc=run([str(ROOT/'.venv/bin/python'),'src/prepare_gabriel_k9_extension.py']);log('k9_prepare_exit',returncode=rc)
 if rc:raise SystemExit('K=9 preparation failed')
 rc=run([str(ROOT/'.venv/bin/python'),'src/run_gabriel_k2_k8_energy_magnitude.py','--stage','full','--k','7,8,9','--finalise-k','2,3,4,5,6,7,8,9','--workers','4','--progress-log',str(LOG)]);log('k7_k9_exit',returncode=rc,counts={str(k):count(k) for k in range(2,10)})
 if rc:raise SystemExit(rc)
if __name__=='__main__':main()
