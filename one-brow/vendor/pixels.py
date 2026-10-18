import math
import cv2
import numpy as np
from raster import sample
CONFIG={'version': 'C8/6', 'ink_luminance': 120, 'alpha_core': 0.12, 'region_dilation_interocular': 0.045, 'raster_tolerance_px': 2, 'minimum_component_area_interocular_squared': 0.0005, 'minimum_stroke_width_interocular': 0.07, 'static_feature_pixels_allowed': 0, 'active_feature_regions_exclusive': True, 'eye_skin_colour_distance': 35, 'assign_active_blush_separately': True, 'eye_join_interocular': 0.01, 'occlusion_depth_tolerance_pixels': 1.5, 'occlusion_minimum_hidden_fraction': 0.5, 'missing_core_pixels_allowed': 4, 'oracle_edge_erosion_px': 1, 'camera_motion': 'smoothstep/1 or historical 2.5 degree steps'}
FEATURES=['brow-left', 'brow-right', 'eye-left', 'eye-right', 'mouth']

def ink(im):return cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)<CONFIG['ink_luminance']

def eye_evidence(im,skin):
    # Sclera connects iris and liner into one anatomical eye. Counting only
    # black pixels incorrectly treats a detached outer liner as another eye.
    return np.linalg.norm(im.astype(float)-skin.astype(float),axis=-1)>CONFIG['eye_skin_colour_distance']

def components(mask,d,eye=False):
    mask=np.uint8(mask)
    if eye:
        radius=max(1,round(d*CONFIG['eye_join_interocular']));mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1)))
    n,lab,stats,centers=cv2.connectedComponentsWithStats(mask)
    return [dict(x=int(x),y=int(y),w=int(w),h=int(h),area=int(area)) for x,y,w,h,area in stats[1:] if area>=max(4,d*d*CONFIG['minimum_component_area_interocular_squared']) and w>=max(3,d*CONFIG['minimum_stroke_width_interocular'])]

def regions(groups,b):
    d=b['source_landmarks']['interocular_px'];radius=max(3,round(d*CONFIG['region_dilation_interocular']));kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(radius*2+1,radius*2+1));out={}
    for key in FEATURES:
        union=np.maximum.reduce([p[:,:,3] for p in groups[key].values()]);out[key]=cv2.dilate(np.uint8(union>255*CONFIG['alpha_core']),kernel).astype(float)
    return out

def state(key,row,b):
    if key=='mouth':return row.get('mouth_sample','mouth-'+row['mouth'])
    if key.startswith('eye-') and row['eyes']!='open' and row['expression'] not in b.get('native_closed_expressions',{}).get(key,[]):return 'eyes-'+row['eyes']
    return 'expression-'+row['brows' if key.startswith('brow-') else 'expression']

def masks(mapping_,reg):return {k:(sample(v,mapping_['sx'],mapping_['sy'])>.5)&mapping_['valid'] for k,v in reg.items()}

def active_masks(m,groups,row,b):
    out={};tolerance=CONFIG['raster_tolerance_px']
    for key in FEATURES:
        if key=='mouth':continue
        p=groups[key][state(key,row,b)];expected=sample(np.uint8(ink(p[:,:,:3])&(p[:,:,3]>30))*255,m['sx'],m['sy'])>20
        out[key]=cv2.dilate(np.uint8(expected),np.ones((tolerance*2+1,tolerance*2+1),np.uint8))>0
    for key in ('blush-left','blush-right'):
        p=groups[key]['expression-'+row['expression']];expected=sample(p[:,:,3],m['sx'],m['sy'])>30
        out[key]=cv2.dilate(np.uint8(expected),np.ones((tolerance*2+1,tolerance*2+1),np.uint8))>0
    return out

def exclusive(roi,key,expected):
    # Union regions overlap when a lowered brow approaches an upper eyelid.
    # Assign the currently ACTIVE other feature first; it is not a duplicate.
    other=np.logical_or.reduce([value for name,value in expected.items() if name!=key]);return roi&~other

def camera_yaw(time,smooth=False):
    u=max(0,min((time-6.7)/1.25,(9.6-time)/1.2,1))
    return 35*u*u*(3-2*u) if smooth else round(35*u/2.5)*2.5

def brow_oracle(key,row,b,groups,m,front,yaw):
    """Project selected source ink through the ORIGINAL mesh z-buffer.

    Hidden fraction uses forward-transported front surface points and a second
    depth test, rather than concluding occlusion from a missing output stroke.
    """
    p=groups[key][state(key,row,b)];source=ink(p[:,:,:3])&(p[:,:,3]>127)
    expected=(sample(np.uint8(source)*255,m['sx'],m['sy'])>127)&m['valid']
    front_mask=(sample(np.uint8(source)*255,front['sx'],front['sy'])>127)&front['valid']
    points=front['world'][front_mask];a=math.radians(yaw);cam=b['camera'];size=512
    if len(points):
        x=(points[:,0]*math.cos(a)+points[:,1]*math.sin(a)-cam['center_x'])/cam['span']*size+size/2-.5
        y=(cam['center_z']-points[:,2])/cam['span']*size+size/2-.5
        depth=points[:,0]*math.sin(a)-points[:,1]*math.cos(a)
        surface_depth=sample(m['depth'],x.reshape(1,-1),y.reshape(1,-1)).ravel()
        hidden=surface_depth-depth>cam['span']/size*CONFIG['occlusion_depth_tolerance_pixels']
        fraction=float(hidden.mean())
    else:fraction=0
    d=b['source_landmarks']['interocular_px']
    proof={'source_strokes':len(components(source,d)),'front_strokes':len(components(front_mask,d)),'front_pixels':int(front_mask.sum()),'predicted_visible_pixels':int(expected.sum()),'predicted_visible_strokes':len(components(expected,d)),'depth_hidden_fraction':fraction}
    return expected,proof

def assess_brow(observed,expected,d,proof):
    """Same physical brow may be sub-resolution; absence needs depth proof."""
    kernel=np.ones((5,5),np.uint8);cc=components(observed,d)
    extra=components(observed&~cv2.dilate(np.uint8(expected),kernel).astype(bool),d)
    # The independent CPU raster and Blender use different subpixel filtering.
    # Test the interior, not the one-pixel antialias fringe. Tiny strokes still
    # have the explicit nonempty-observation check immediately below.
    core=cv2.erode(np.uint8(expected),np.ones((3,3),np.uint8)).astype(bool)
    missing=core&~cv2.dilate(np.uint8(observed),kernel).astype(bool)
    support_matches=int(missing.sum())<=CONFIG['missing_core_pixels_allowed']
    # Do not silently accept an empty drawing when the oracle predicts ink.
    if expected.sum()>=4 and not observed.any():support_matches=False
    occluded=(proof['source_strokes']==1 and proof['front_strokes']==1 and proof['predicted_visible_strokes']==0 and proof['depth_hidden_fraction']>=CONFIG['occlusion_minimum_hidden_fraction'])
    passed=(len(cc)==1 or (len(cc)==0 and occluded)) and not extra and support_matches
    return {'connected_groups':len(cc),'components':cc,'extra_ink_groups':len(extra),'extra_components':extra,'missing_core_pixels':int(missing.sum()),'occlusion_exception_used':len(cc)==0 and occluded,'occlusion_proof':proof,'pass':bool(passed)}

def assess_features(crop,rs,m,groups,row,b,skin_plate,front,yaw):
    d=b['source_landmarks']['interocular_px'];col=ink(crop);expected_masks=active_masks(m,groups,row,b);features={}
    for key,roi in rs.items():
        if key=='mouth':continue
        roi=exclusive(roi,key,expected_masks)
        if key.startswith('brow-'):
            expected,proof=brow_oracle(key,row,b,groups,m,front,yaw)
            features[key]=assess_brow(col&roi,expected&roi,d,proof)
        else:
            evidence=eye_evidence(crop,sample(skin_plate,m['sx'],m['sy']))
            cc=components(evidence&roi,d,True);extra=components(col&roi&~expected_masks[key],d)
            features[key]={'connected_groups':len(cc),'components':cc,'extra_ink_groups':len(extra),'extra_components':extra,'pass':len(cc)==1 and not extra}
    return features
