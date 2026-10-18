"""Reproduce the eight paired baseline/control results without executing entries."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CASES=[('hand-plant','baseline/motion.npz',[],1),('hand-plant','tests/control/poses.npz',['--control'],0),
       ('one-brow','baseline',[],1),('one-brow','tests/bruno',['--control-bruno'],0),
       ('garment-separation','baseline/parts.npz',[],1),('garment-separation','tests/good.npz',[],0),
       ('stylised-hair','baseline',[],1),('stylised-hair','tests/control',['--control'],0)]
def main():
    p=argparse.ArgumentParser();p.add_argument('--json');a=p.parse_args();rows=[]
    for name,path,flags,expected in CASES:
        result=subprocess.run([sys.executable,str(ROOT/name/'score.py'),str(ROOT/name/path),*flags],capture_output=True,text=True,timeout=900)
        try:metrics=json.loads(result.stdout[result.stdout.index('{'):])
        except (ValueError,json.JSONDecodeError):metrics={'status':'ERROR'}
        valid=result.returncode==expected and metrics.get('status')==('PASS' if expected==0 else 'FAIL') and 'error' not in metrics
        rows.append({'challenge':name,'case':path,'expected_exit':expected,'actual_exit':result.returncode,'verified':valid,'metrics':metrics})
        print(name,path,'OK' if valid else 'ERROR',flush=True)
    report={'verified':all(r['verified'] for r in rows),'cases':rows}
    if a.json:Path(a.json).write_text(json.dumps(report,indent=2))
    return 0 if report['verified'] else 1
if __name__=='__main__':sys.exit(main())
