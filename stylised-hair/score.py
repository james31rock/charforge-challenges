"""Score data-only full-rate hair vertices; never import submission code."""
import argparse,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent))
from common import blender,scratch,finish

def main():
    p=argparse.ArgumentParser();p.add_argument('submission');p.add_argument('--control',action='store_true');p.add_argument('--json');a=p.parse_args()
    try:
        submission=Path(a.submission).resolve()
        if not submission.is_dir():raise ValueError('Submission must be a directory with walk.npz and jump.npz')
        with scratch() as temp:
            report=Path(temp)/'report.json'
            blender(HERE/'measure.py',[submission,HERE/('tests/control/assets' if a.control else 'assets'),report],temp)
            result=json.loads(report.read_text())
        result['scope']='synthetic-control' if a.control else 'pippa-bob-walk-jump'
    except Exception:
        # Avoid leaking workstation paths or Blender diagnostics to shared reports.
        result=dict(status='FAIL',error='Invalid input or measurement failed; require finite full-rate hair arrays, unchanged topology, and Blender 5.x.')
    return finish(result,a.json)
if __name__=='__main__':sys.exit(main())
