import math
import numpy as np
import cv2

def sample(im,x,y):
    return cv2.remap(im,np.asarray(x,dtype=np.float32),np.asarray(y,dtype=np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)

def project(mesh,size=512,yaw=0,frame=None,forward=-1):
    v=mesh['vertices'];n=mesh['normals'];tri=mesh['triangles'];uv=mesh['uv'];a=math.radians(yaw)
    # Canonical S3 front is -Y. All coordinates are rest-space, camera independent.
    right=np.array([-math.cos(a)*forward,math.sin(a),0]);toward=np.array([math.sin(a),math.cos(a)*forward,0])
    if frame is None:
        height=np.ptp(v[:,2]);frame={'center_x':0,'center_z':float(v[:,2].max()-.21*height),'span':float(.49*height)}
    span=frame['span'];sx=(v@right-frame['center_x'])/span*size+size/2;sy=(frame['center_z']-v[:,2])/span*size+size/2;depth=v@toward
    pos=np.stack((sx,sy),axis=1);zbuf=np.full((size,size),-np.inf,dtype=np.float32);world=np.zeros((size,size,3),np.float32);normal=world.copy();tex=np.zeros((size,size,2),np.float32);ids=np.full((size,size),-1,np.int32)
    for ti,indices in enumerate(tri):
        pts=pos[indices];mn=np.maximum(np.floor(pts.min(axis=0)).astype(int),0);mx=np.minimum(np.ceil(pts.max(axis=0)).astype(int),size-1)
        if (mn>mx).any():continue
        x0,y0=mn;x1,y1=mx;a0,b0,c0=pts;den=(b0[1]-c0[1])*(a0[0]-c0[0])+(c0[0]-b0[0])*(a0[1]-c0[1])
        if abs(den)<1e-8:continue
        yy,xx=np.mgrid[y0:y1+1,x0:x1+1];xx=xx+.5;yy=yy+.5
        b1=((b0[1]-c0[1])*(xx-c0[0])+(c0[0]-b0[0])*(yy-c0[1]))/den
        b2=((c0[1]-a0[1])*(xx-c0[0])+(a0[0]-c0[0])*(yy-c0[1]))/den;b3=1-b1-b2
        weights=np.stack((b1,b2,b3),axis=-1);dep=weights@depth[indices]
        patch=zbuf[y0:y1+1,x0:x1+1];ok=(weights.min(axis=-1)>=-1e-6)&(dep>patch)
        if not ok.any():continue
        patch[ok]=dep[ok];world[y0:y1+1,x0:x1+1][ok]=(weights@v[indices])[ok];normal[y0:y1+1,x0:x1+1][ok]=(weights@n[indices])[ok];tex[y0:y1+1,x0:x1+1][ok]=(weights@uv[ti])[ok];ids[y0:y1+1,x0:x1+1][ok]=ti
    normal/=np.maximum(np.linalg.norm(normal,axis=-1,keepdims=True),1e-8)
    texture=mesh['texture'];color=sample(texture,tex[:,:,0]*(texture.shape[1]-1),(1-tex[:,:,1])*(texture.shape[0]-1));color[ids<0]=[237,238,241]
    return {'color':color,'world':world,'normal':normal,'ids':ids,'frame':frame,'yaw':yaw,'forward':forward}
