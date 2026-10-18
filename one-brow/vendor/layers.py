import numpy as np
from PIL import Image
def read(p): return np.array(Image.open(p).convert("RGB"))

def load_groups(f,binding):
    out={};w,h=binding['canvas']
    for k,v in binding['atlases'].items():
        a=np.array(Image.open(f/v['path']).convert('RGBA'));out[k]={}
        cw,ch=v.get('tile_size',[w,h]);x0,y0,x1,y1=v.get('source_rect',[0,0,w,h])
        for i,name in enumerate(v['states']):
            p=np.zeros((h,w,4),np.uint8);p[y0:y1,x0:x1]=a[(i//v['cols'])*ch:(i//v['cols']+1)*ch,(i%v['cols'])*cw:(i%v['cols']+1)*cw];out[k][name]=p
    if binding.get('visibility_override'):out['_visibility_override']=read(f/binding['visibility_override'])[:,:,0]/255
    return out

def coordinates(r,b):
    world=r['world'];camera=b['camera'];w,h=b['canvas'];fx=(world[:,:,0]-camera['center_x'])/camera['span']*w+w/2;fy=(camera['center_z']-world[:,:,2])/camera['span']*h+h/2;m=np.array(b['front_to_source'])
    return fx*m[0,0]+fy*m[0,1]+m[0,2],fx*m[1,0]+fy*m[1,1]+m[1,2]
