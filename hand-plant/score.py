"""Score pose data against the fixed native skinned rig; never run submissions."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import blender,scratch,read,finish,sha,load_npz
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor'))
from hands import evaluate_hands,DEFAULTS

def score(path,control=False):
    import numpy as np
    pose=load_npz(path)
    ref=load_npz(ROOT/('tests/control/poses.npz' if control else 'baseline/motion.npz'))
    for k in ('names','basis','world','times'):
        if k not in pose or pose[k].shape!=ref[k].shape:raise ValueError('Wrong full-rate pose layout: '+k)
    if not np.array_equal(pose['names'],ref['names']) or not np.allclose(pose['times'],ref['times'],atol=1e-8,rtol=0):raise ValueError('Bone names and source sample times must be preserved')
    for k in ('basis','world'):
        a=pose[k]
        if not np.allclose(a[...,-1,:],[0,0,0,1],atol=1e-5):raise ValueError('Nonaffine matrix')
        rot=a[...,:3,:3]
        if not np.allclose(rot@np.swapaxes(rot,-1,-2),np.eye(3),atol=2e-4) or not np.allclose(np.linalg.det(rot),1,atol=2e-4):raise ValueError('Only rigid transforms allowed; no scale or reflection')
    folder=ROOT/'tests/control' if control else ROOT/'assets'
    with scratch() as temp:
        out=Path(temp)/'trace.json'
        blender(ROOT/'extract.py',[folder,Path(path).resolve(),out],temp)
        trace=read(out)
    checks,coverage=evaluate_hands(trace,DEFAULTS)
    if not control:
        gap=trace['limb_attachment_error_height']
        checks.append({'id':'rig.limb_attachment','status':'PASS' if gap<=.0001 else 'FAIL','value':gap,'threshold':.0001,'unit':'height','detail':'Arm/leg joint-head distances must preserve the fixed rig lengths'})
    return {'status':'PASS' if all(c['status']=='PASS' for c in checks) else 'FAIL',
            'challenge':'hand-plant','scope':'synthetic extractor control' if control else 'C7 native hand support; full motion acceptance requires review',
            'eligible_submission':not control,'submission_sha256':sha(path),'checks':checks,'coverage':coverage,
            'height_m':trace['height_m'],'rig_sha256':sha(folder/'rig.blend')}

def main():
    p=argparse.ArgumentParser();p.add_argument('submission');p.add_argument('--control',action='store_true');p.add_argument('--json');a=p.parse_args()
    try:r=score(a.submission,a.control)
    except Exception as e:r={'status':'FAIL','error':str(e),'challenge':'hand-plant'}
    return finish(r,a.json)
if __name__=='__main__':sys.exit(main())

