import numpy as np
from hands import anatomy, surface_metrics, unit

def measured_anatomy(arm,mp,fingers):
    result={}
    def head(name):return np.array(arm.matrix_world@arm.pose.bones[name].head)
    def tail(name):return np.array(arm.matrix_world@arm.pose.bones[name].tail)
    for s,chains in fingers.items():
        if s not in ('L','R') or any(n not in arm.pose.bones for chain in chains for n in chain):continue
        try:
            result[s]=anatomy(head(mp[s+'elbow']),head(mp[s+'wrist']),tail(mp[s+'wrist']),
                head(chains[0][0]),head(chains[3][0]),
                [[tail(n)-head(n) for n in chain] for chain in chains],s)
        except ValueError:continue
    return result

def select_patches(rest,arm,mp,fingers,neutral):
    """Choose palm-side skin in four spatial cells, then freeze memberships.

    Selection uses bone-labelled hand/palm vertices between the wrist and MCP
    row. Fingers, cuffs and the dorsal side are excluded from palm contact.
    All hand/finger vertices remain in the ground-penetration measurement.
    """
    selection={}
    for s,anatomical in neutral.items():
        wrist=np.array(arm.matrix_world@arm.pose.bones[mp[s+'wrist']].head)
        normal=np.array(anatomical['palm_normal']);forward=np.array(anatomical['forward'])
        lateral=unit(np.cross(normal,forward))
        bases=[np.array(arm.matrix_world@arm.pose.bones[c[0]].head) for c in fingers[s]]
        length=float(np.median([(p-wrist)@forward for p in bases]));width=float(np.linalg.norm(bases[0]-bases[-1]))
        if min(length,width)<1e-6:continue
        hand=arm.data.bones[mp[s+'wrist']];names={hand.name}|{b.name for b in hand.children_recursive}
        finger_names={n for chain in fingers[s] for n in chain}
        palm_names={n for n in names if n==hand.name or 'palm' in n.lower()}
        available=[];all_hand={}
        for name,g in rest.items():
            dominant=np.asarray(g['dominant']);all_hand[name]=np.flatnonzero(np.isin(dominant,list(names))).tolist()
            v=g['v'];local=v-wrist;along=local@forward;across=local@lateral;depth=local@normal
            mask=np.isin(dominant,list(palm_names-finger_names))&(along>length*.12)&(along<length*.95)&(abs(across)<width*.70)
            for i in np.flatnonzero(mask):available.append((name,int(i),float(along[i]),float(across[i]),float(depth[i])))
        patches=[]
        for dist in (False,True):
            for outer in (False,True):
                cell=[r for r in available if (r[2]>=length*.53)==dist and (r[3]>=0)==outer]
                if len(cell)<3:break
                cutoff=float(np.quantile([r[4] for r in cell],.75))
                patch=[(name,i) for name,i,_,_,z in cell if z>=cutoff]
                patches.append(patch)
        if len(patches)==4:selection[s]=dict(patches=patches,all_hand=all_hand)
    return selection

def measure_surfaces(ev,selection,anatomical,scale,floor):
    result={s:dict(row) for s,row in anatomical.items()}
    for s,sel in selection.items():
        if s not in result:continue
        patches=np.array([np.mean([ev[name]['v'][i] for name,i in patch],axis=0) for patch in sel['patches']])*scale
        vertices=np.concatenate([ev[name]['v'][ids] for name,ids in sel['all_hand'].items() if ids])*scale
        try:result[s].update(surface_metrics(patches,result[s]['palm_normal'],vertices,floor))
        except ValueError:continue
    return result
