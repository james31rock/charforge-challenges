"""Build an original analytic scorer control, not a solved Pippa hairstyle."""
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent

def sphere(radius,centre):
    rings=10;segments=20;v=[[0,0,radius]]
    for i in range(1,rings):
        for j in range(segments):
            a=np.pi*i/rings;b=2*np.pi*j/segments
            v.append([radius*np.sin(a)*np.cos(b),radius*np.sin(a)*np.sin(b),radius*np.cos(a)])
    v.append([0,0,-radius]);t=[];end=len(v)-1
    for j in range(segments):
        t.append([0,1+j,1+(j+1)%segments]);a=1+(rings-2)*segments
        t.append([end,a+(j+1)%segments,a+j])
    for i in range(rings-2):
        for j in range(segments):
            a=1+i*segments+j;b=1+i*segments+(j+1)%segments;c=b+segments;d=a+segments
            t.extend([[a,d,c],[a,c,b]])
    return np.array(v)+centre,np.array(t)

def build():
    base=HERE/'tests/control';assets=base/'assets';assets.mkdir(parents=True,exist_ok=True)
    v=[];tri=[];roots=[];tips=[];w=np.zeros((8*7*3,8));u=[]
    for s,angle in enumerate(np.linspace(-2.3,2.3,8)):
        for level in range(7):
            for width in [-.012,0,.012]:
                b=angle+width;v.append([.18*np.sin(b),.18*np.cos(b),.9-.2*level/6]);u.append((level/6)**2)
                index=len(v)-1
                if level==0:roots.append(index)
                if level==6:tips.append(index);w[index,s]=1/3
            if level:
                a=s*21+(level-1)*3
                for j in range(2):tri.extend([[a+j,a+j+3,a+j+4],[a+j,a+j+4,a+j+1]])
    v=np.array(v);tri=np.array(tri);u=np.array(u);tips=np.array(tips)
    body,bt=sphere(.10,np.array([0,0,.87]));n=len(v)
    fixture=dict(rest=v,triangles=tri,hair_ids=np.arange(n),collision_ids=np.flatnonzero(u>0),tip_ids=tips,root_ids=np.array(roots),strip_weights=w,
                 colours=np.tile([.16,.10,.30],(n,1)),height_m=np.array(1.),fps=np.array(24.),tip_sides=(v[tips,0]>0).astype(int),target_triangles_0=bt)
    np.savez_compressed(assets/'fixture.npz',**fixture)
    for clip in ['walk','jump']:
        time=np.arange(97)/24;delta=np.zeros((len(time),3));axis=0 if clip=='walk' else 2
        delta[:,axis]=(.02 if clip=='walk' else .05)*np.sin(2*np.pi*time)
        head=np.tile(np.eye(4),(len(time),1,1));head[:,:3,3]=delta+np.array([0,0,.87])
        drive=v[None,:,:]+delta[:,None,:];hair=drive.copy()
        hair[:,:,0]+=(.025 if clip=='walk' else .04)*np.sin(2*np.pi*time-.5)[:,None]*u
        np.savez_compressed(assets/(clip+'.npz'),head=head,vertices=drive,target_0=body[None,:,:]+delta[:,None,:])
        np.savez_compressed(base/(clip+'.npz'),hair=hair)
if __name__=='__main__':build()
