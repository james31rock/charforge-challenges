"""Closed parts, hidden-body completeness, semantic shape, silhouette fidelity."""
import argparse,sys
from pathlib import Path
import numpy as np
import trimesh
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent));sys.path.insert(0,str(ROOT/'vendor'))
from common import load_npz,finish,sha
from metrics import topology,mask,iou,vertical_coverage

LABELS=('body','garment','hair')
def mesh(data,label):
    v=data[label+'_vertices'];f=data[label+'_faces']
    if v.ndim!=2 or v.shape[1]!=3 or not len(v) or len(v)>200000:raise ValueError('Invalid vertices')
    if f.ndim!=2 or f.shape[1]!=3 or not len(f) or len(f)>400000 or f.dtype.kind not in 'iu' or f.min()<0 or f.max()>=len(v):raise ValueError('Invalid triangle indices')
    m=trimesh.Trimesh(v,f,process=False)
    if (m.area_faces<1e-14).any():raise ValueError('Degenerate triangles')
    return m
def samples(m):
    # Fixed deterministic surface sampling, weighted by triangle area.
    rng=np.random.default_rng(314159);ids=rng.choice(len(m.faces),4096,p=m.area_faces/m.area)
    u=rng.random(4096);v=rng.random(4096);u=np.sqrt(u)
    return m.triangles[ids,0]*(1-u[:,None])+m.triangles[ids,1]*(u*(1-v))[:,None]+m.triangles[ids,2]*(u*v)[:,None]
def score(path):
    data=load_npz(path);oracle=load_npz(ROOT/'assets/oracle.npz');source=load_npz(ROOT/'assets/fused.npz')
    src=trimesh.Trimesh(source['vertices'],source['faces'],process=False);height=float(np.ptp(src.vertices[:,2]))
    checks=[];meshes=[];details={}
    for label in LABELS:
        if label+'_vertices' not in data or label+'_faces' not in data:
            checks.append({'id':label+'.present','pass':False});continue
        m=mesh(data,label);ref=mesh(oracle,label);meshes.append(m);t=topology(m)
        # Components: the garment has a disconnected inside boundary by design.
        topo=t['watertight'] and t['winding_consistent'] and t['volume']>0 and t['nonmanifold_edges']==0
        checks.append({'id':label+'.closed','pass':topo,'value':t})
        to_candidate=trimesh.proximity.closest_point(m,samples(ref))[1]/height
        to_oracle=trimesh.proximity.closest_point(ref,samples(m))[1]/height
        p95=float(max(np.quantile(to_candidate,.95),np.quantile(to_oracle,.95)))
        maxdist=float(max(to_candidate.max(),to_oracle.max()))
        checks.append({'id':label+'.semantic_surface','pass':p95<=.012 and maxdist<=.035,'p95_height':p95,'max_height':maxdist,'limits':[.012,.035]})
        if label=='body':
            # Every spatial octant and every height bin must retain the reference
            # body. This rejects a closed torso blob with a good outer silhouette.
            points=samples(ref);distance=trimesh.proximity.closest_point(m,points)[1]/height
            coverage=float((distance<=.015).mean())
            bins=np.clip(((points[:,2]-src.bounds[0,2])/height*16).astype(int),0,15)
            worst=min(float((distance[bins==i]<=.015).mean()) for i in np.unique(bins))
            checks.append({'id':'body.hidden_completeness','pass':coverage>=.98 and worst>=.90,'fraction':coverage,'worst_height_bin':worst,'limits':[.98,.90]})
            details['body_vertical_diagnostic']=vertical_coverage([m],height)
    for view in ('front','side'):
        metric=iou(mask([src],src.bounds,view),mask(meshes,src.bounds,view)) if meshes else 0
        checks.append({'id':'silhouette.'+view,'pass':metric>=.97,'value':metric,'minimum':.97})
    return {'status':'PASS' if all(c['pass'] for c in checks) else 'FAIL','challenge':'garment-separation','submission_sha256':sha(path),'checks':checks,**details,'scope':'Self-made reconstruction benchmark; sampled surface completeness, not anatomical certification'}
def main():
    p=argparse.ArgumentParser();p.add_argument('submission');p.add_argument('--json');a=p.parse_args()
    try:r=score(a.submission)
    except Exception as e:r={'status':'FAIL','error':str(e),'challenge':'garment-separation'}
    return finish(r,a.json)
if __name__=='__main__':sys.exit(main())
