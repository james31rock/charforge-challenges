"""Rebuild immutable region transports from the bundled test head."""
import sys,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent));sys.path.insert(0,str(ROOT/'vendor'))
from common import read,load_npz
from raster import project
from layers import coordinates
assets=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'assets'
b=read(assets/'reference/binding.json');mesh=load_npz(assets/'geometry.npz');mesh['texture']=np.zeros((2,2,3),np.uint8)
out=assets/'maps';out.mkdir(exist_ok=True)
for yaw in np.arange(0,35.01,2.5):
    r=project(mesh,yaw=yaw,frame=b['camera']);sx,sy=coordinates(r,b)
    a=math.radians(yaw);depth=r['world'][:,:,0]*math.sin(a)-r['world'][:,:,1]*math.cos(a)
    np.savez_compressed(out/f'map-{yaw:g}.npz',sx=sx,sy=sy,valid=(r['ids']>=0)&(r['world'][:,:,1]<0),world=r['world'],depth=depth,occupied=r['ids']>=0)
print('MAPS_BUILT')
