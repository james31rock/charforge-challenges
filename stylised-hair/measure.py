"""Trusted headless geometry adapter for toolkit C2; inputs contain no code."""
import sys,json
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from common import load_npz,write,sha
from toolkit.qa.core import evaluate
from toolkit.qa.geometry import tree,penetration
from toolkit.qa.hair import strip_spacing

def measure(submission,assets):
    fixture=load_npz(assets/'fixture.npz');h=float(fixture['height_m']);fps=float(fixture['fps']);ids=fixture['hair_ids']
    tips=fixture['tip_ids'];side=fixture['tip_sides'].astype(bool);w=fixture['strip_weights'];probes=fixture['collision_ids']
    rest=fixture['rest'][ids];target_keys=sorted(k for k in fixture if k.startswith('target_triangles_'))
    if not target_keys or not len(probes) or not side.any() or side.all():raise ValueError('Incomplete fixed coverage')
    clips={}
    for clip in ['walk','jump']:
        drive=load_npz(assets/(clip+'.npz'));data=load_npz(submission/(clip+'.npz'))
        if set(data)!={'hair'}:raise ValueError('Only the hair array is accepted')
        hair=data['hair'];n=len(drive['head'])
        if hair.shape!=(n,len(ids),3):raise ValueError('Wrong topology or frame count')
        trace=dict(schema='qa-trace/1',units='metres-z-up',height_m=h,hip_width_m=h*.15,leg_length_m=h*.4,fps=fps,intent=clip,floor_m=0.,coverage=dict(hair_tips=True,hair_volume=True),frames=[])
        depths=[];centres=[];amplitudes=[]
        for f in range(n):
            head=drive['head'][f];inv=np.linalg.inv(head);local=hair[f]@inv[:3,:3].T+inv[:3,3]
            patches=np.array([local[tips][mask].mean(0) for mask in (side,~side)])
            centres.append(w.T@local);amplitudes.append(patches);points=hair[f,probes];depth=0.;witness=None
            for key in target_keys:
                index=key.removeprefix('target_triangles_');tri=fixture[key]
                if 'target_ids_'+index in fixture:
                    v=drive['vertices'][f,fixture['target_ids_'+index]]
                    lengths=fixture['target_loop_lengths_'+index];flat=fixture['target_loops_'+index];ends=np.r_[0,np.cumsum(lengths)]
                    if len(lengths):
                        centres_of_caps=[v[flat[ends[i]:ends[i+1]]].mean(0) for i in range(len(lengths))]
                        v=np.vstack([v,centres_of_caps])
                else:v=drive['target_'+index][f]
                bbox=v[np.unique(tri)]
                mask=np.all((points>=bbox.min(0)-h*.03)&(points<=bbox.max(0)+h*.03),axis=1)
                if mask.any():
                    d,hit=penetration(points[mask],tree(v,tri),h*2,closed=True)
                    if d>depth:depth=d;witness=hit
            depths.append(depth)
            trace['frames'].append(dict(frame=f+1,points=dict(head=head[:3,3].tolist()),hair_depth_m=depth,hair_relative_tips=patches.tolist()))
        checks=[c for c in evaluate(trace)['checks'] if c['id'].startswith('C2.')]
        # Distances are rigid-frame invariant, so bind-space rest centres need no transform.
        checks+=strip_spacing(w.T@rest,np.array(centres))
        roots=fixture['root_ids'];root_d=float(np.linalg.norm(hair[:,roots]-drive['vertices'][:,ids[roots]],axis=-1).max(initial=0))
        excursion=float(np.linalg.norm(hair-drive['vertices'][:,ids],axis=-1).max())
        for name,value,limit in [('root_attachment',root_d,.001*h),('bounded_excursion',excursion,.35*h)]:
            checks.append(dict(id='C2.'+name,status='PASS' if value<=limit else 'FAIL',value=value,threshold=limit,unit='m'))
        accel=np.linalg.norm(np.diff(drive['head'][:,:3,3],n=2,axis=0)*fps**2,axis=1)
        amplitude=float(np.sqrt(np.mean(np.sum(np.ptp(amplitudes,axis=0)**2,axis=-1))))
        clips[clip]=dict(status='PASS' if all(c['status']=='PASS' for c in checks) else 'FAIL',frames=n,checks=checks,
                         tip_amplitude_m=amplitude,head_acceleration_rms_m_s2=float(np.sqrt(np.mean(accel**2))),worst_depth_frame=int(np.argmax(depths))+1)
    return dict(schema='hair-challenge-score/1',status='PASS' if all(c['status']=='PASS' for c in clips.values()) else 'FAIL',clips=clips,
                code_sha256={name:sha(ROOT/name) for name in ['toolkit/qa/core.py','toolkit/qa/geometry.py','toolkit/qa/hair.py','stylised-hair/measure.py']},
                fixed_sha256={name:sha(assets/name) for name in ['fixture.npz','walk.npz','jump.npz']},
                submission_sha256={name:sha(submission/name) for name in ['walk.npz','jump.npz']},
                limitations=['Sampled fixed hair vertices versus closed body envelopes; not continuous triangle collision certification.',
                             'Strip spacing bands are provisional shape guards, not professional-rig calibration or a visual certificate.'])

if __name__=='__main__':
    submission,assets,output=map(Path,sys.argv[sys.argv.index('--')+1:]);write(output,measure(submission,assets))
