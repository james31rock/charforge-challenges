"""C8/5 frozen pixel rules, independently rendered static and expression views."""
import argparse,sys,json
from pathlib import Path
import cv2,numpy as np
cv2.setNumThreads(2)
from PIL import Image
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent));sys.path.insert(0,str(ROOT/'vendor'))
from common import blender,scratch,read,finish,sha,load_npz
from pixels import CONFIG,regions,masks,ink,assess_features
from raster import sample
from layers import load_groups
def rgb(path):return np.array(Image.open(path).convert('RGB'))
def context(assets=None):
    assets=assets or ROOT/'assets';folder=assets/'reference';b=read(folder/'binding.json');g=load_groups(folder,b)
    maps={yaw:load_npz(assets/f'maps/map-{yaw:g}.npz') for yaw in np.arange(0,35.01,2.5)}
    return b,g,regions(g,b),maps,rgb(folder/'feature-free.png')
def features(im,yaw,row,ctx):
    b,g,reg,maps,skin=ctx;m=maps[yaw]
    return assess_features(im,masks(m,reg),m,g,row,b,skin,maps[0],yaw)
def okay(row):return all(r['pass'] for r in row.values())
def evaluate_images(folder,video,track,ctx):
    b,g,reg,maps,skin=ctx;static=[];expressions=[]
    for row in read(folder/'renders.json'):
        im=rgb(folder/row['path'])
        if im.shape!=(512,512,3):raise ValueError('Unexpected render size')
        if 'static' in row['mode']:
            rs=masks(maps[row['yaw']],reg);counts={k:int((ink(im)&v).sum()) for k,v in rs.items()};static.append({**row,'feature_pixels':counts,'pass':sum(counts.values())==0})
        else:
            ex=['neutral','happy','worried','surprised','sad','determined'][row['index']] if row['mode']=='expressions' else 'neutral'
            state={'expression':ex,'brows':ex,'eyes':['open','half','closed'][row['index']] if row['mode']=='blink' else 'open'}
            f=features(im,row['yaw'],state,ctx);expressions.append({**row,'features':f,'pass':okay(f)})
    if len(static)!=4 or len(expressions)!=18:raise ValueError('Missing mandatory views')
    cap=cv2.VideoCapture(str(video));failed=[];count=0
    if abs(cap.get(cv2.CAP_PROP_FPS)-24)>.01:raise ValueError('Movie must be 24 fps')
    while True:
        ok,bgr=cap.read()
        if not ok:break
        if count>=240:raise ValueError('Movie must have exactly 240 frames')
        im=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        if im.shape[0]<568 or im.shape[1]!=1024:raise ValueError('Movie requires original two 512px view layout')
        t=count/24;row=track['frames'][count//track.get('exposure_frames',1)]
        yaw=0 if t<6.7 else round(max(0,min((t-6.7)/1.25,(9.6-t)/1.2,1))*35/2.5)*2.5 if t<9.6 else 0
        for i,angle in enumerate((yaw,35)):
            f=features(im[56:568,i*512:(i+1)*512],angle,row,ctx)
            if not okay(f):failed.append({'frame':count+1,'yaw':angle,'features':f})
        count+=1
    cap.release()
    return {'static':static,'expressions':expressions,'movie_frames':count,'movie_failed_frames':len({r['frame'] for r in failed}),'movie_failures':failed[:12],
            'pass':all(r['pass'] for r in static+expressions) and count==240 and not failed}
def score(folder,control=False,review=None):
    assets=ROOT/'tests/bruno/assets' if control else ROOT/'assets'
    folder=Path(folder).resolve();ctx=context(assets)
    for name in ('face.blend','feature-free.png','face-demo.mp4','binding.json'):
        if not (folder/name).is_file():raise ValueError('Missing '+name)
    if (folder/'face.blend').stat().st_size>64*1024**2:raise ValueError('Oversized head file')
    b=read(folder/'binding.json')
    for key in ('camera','canvas','front_to_source','source_landmarks'):
        if b[key]!=ctx[0][key]:raise ValueError('Fixed registration changed: '+key)
    for atlas in b['atlases'].values():
        path=(folder/atlas['path']).resolve()
        if not path.is_relative_to(folder):raise ValueError('Atlas outside submission')
        with Image.open(path) as im:
            if max(im.size)>16384 or im.width*im.height>32*1024**2:raise ValueError('Oversized atlas')
    current=load_groups(folder,b)
    # Freeze regions from the original drawings; use submitted active ink for
    # depth proof exactly as C8/6 does. Registration/geometry cannot be moved.
    ctx=(b,current,ctx[2],ctx[3],ctx[4])
    with scratch() as tmp:
        out=Path(tmp);blender(ROOT/'render.py',[folder/'face.blend',out,assets],tmp)
        result=evaluate_images(out,folder/'face-demo.mp4',read(assets/'reference/track.json'),ctx)
        if review:
            from PIL import ImageDraw
            sheet=Image.new('RGB',(1024,820),'white');draw=ImageDraw.Draw(sheet)
            selected=['raw-static-0-0.png','composite-static-0-35.png','expressions-3-0.png','expressions-3-35.png','expressions-2-0.png','expressions-2-35.png']
            for i,name in enumerate(selected):
                im=Image.open(out/name).convert('RGB');x=(i%2)*512;y=(i//2)*270
                # Tight face crop is a review aid; all scoring uses full resolution.
                im=im.crop((60,75,452,350)).resize((392,240));sheet.paste(im,(x+60,y+25));draw.text((x+12,y+5),name,fill='black')
            sheet.save(review)
    plate=rgb(folder/'feature-free.png');reg=ctx[2]
    if plate.shape!=(512,512,3):raise ValueError('Plate must be 512x512')
    pixels={k:int((ink(plate)&(v>.5)).sum()) for k,v in reg.items()}
    result['plate_feature_pixels']=pixels;result['pass'] &= sum(pixels.values())==0
    return {'status':'PASS' if result.pop('pass') else 'FAIL','challenge':'one-brow','config':CONFIG,'eligible_submission':not control,
            'head_sha256':sha(folder/'face.blend'),'movie_sha256':sha(folder/'face-demo.mp4'),**result,
            'scope':'C8 pixel gate; output movie must be bound to the submitted recipe/head during review; global likeness and temporal smoothness not certified'}
def main():
    p=argparse.ArgumentParser();p.add_argument('submission');p.add_argument('--json');p.add_argument('--review');p.add_argument('--control-bruno',action='store_true');a=p.parse_args()
    try:r=score(a.submission,a.control_bruno,a.review)
    except Exception as e:r={'status':'FAIL','error':str(e),'challenge':'one-brow'}
    return finish(r,a.json)
if __name__=='__main__':sys.exit(main())

