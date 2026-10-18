"""Material-strip spacing checks for the stylised-hair challenge (MIT).

Spacing is normalized pair-by-pair to bind geometry. Variance alone misses a
uniform collapse; the minimum and maximum ratios are mandatory companion gates.
"""
import numpy as np

SPACING_LIMITS = {'variance': .04, 'minimum': .65, 'maximum': 1.6}

def strip_spacing(rest_centres, posed_centres):
    rest=np.linalg.norm(np.diff(rest_centres,axis=0),axis=-1)
    posed=np.linalg.norm(np.diff(posed_centres,axis=1),axis=-1)
    if len(rest)<2 or np.any(rest<1e-6):
        raise ValueError('At least three distinct fixed strip patches required')
    ratios=posed/rest
    values={'variance':np.var(ratios,axis=1), 'minimum':ratios.min(axis=1), 'maximum':ratios.max(axis=1)}
    result=[]
    for name,v in values.items():
        lower=name=='minimum';limit=SPACING_LIMITS[name];bad=v<limit if lower else v>limit
        result.append(dict(id='C2.strip_spacing_'+name,status='FAIL' if bad.any() else 'PASS',
                           value=float(v.min() if lower else v.max()),threshold=limit,unit='ratio²' if name=='variance' else 'ratio',
                           frames=(np.flatnonzero(bad)+1).tolist()))
    return result
