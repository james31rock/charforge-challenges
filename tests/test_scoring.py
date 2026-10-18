"""Adversarial controls for the public scorer kernels and data boundary."""
import copy,importlib.util,json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'hand-plant/vendor'))
from common import load_npz
from hands import evaluate_hands,DEFAULTS
sys.path.insert(0,str(ROOT/'one-brow/vendor'))
from pixels import assess_brow

class Controls(unittest.TestCase):
    def test_npz_rejects_object_and_nan(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'x.npz'
            for value in (np.array([None],object),np.array([float('nan')])):
                np.savez_compressed(p,x=value)
                with self.assertRaises(ValueError):load_npz(p)
    def test_hand_each_metric_rejects_seed(self):
        hand=dict(wrist_neutral_deg=0.,finger_bend_deg=0.,wrist_extension_deg=80.,palm_plane_deg=0.,palm_gap_m=0.,ground_depth_m=0.,palm_points=[[0,0,0],[.01,0,0],[0,.01,0],[.01,.01,0]])
        trace=dict(height_m=1.,fps=24,source=dict(clip='cartwheel',cache_sha256='target'),hand_support_required=True,
                   rest_hands={s:copy.deepcopy(hand) for s in 'LR'},
                   hand_support=dict(schema='qa-hand-support/1',bound=True,fps=24,frame_count=3,source_sha256='source',cache_sha256='target',hands={s:dict(intervals=[[0,3]]) for s in 'LR'}),
                   frames=[dict(frame=i,hands={s:copy.deepcopy(hand) for s in 'LR'}) for i in (1,2,3)])
        self.assertTrue(all(c['status']=='PASS' for c in evaluate_hands(trace,DEFAULTS)[0]))
        for key,value,cid in [('palm_plane_deg',30,'C7.palm_plane'),('palm_gap_m',.02,'C7.palm_contact'),('wrist_extension_deg',0,'C7.wrist_extension'),('ground_depth_m',.02,'C7.hand_ground_penetration'),('finger_bend_deg',40,'C7.support_fingers')]:
            bad=copy.deepcopy(trace);bad['frames'][1]['hands']['L'][key]=value
            self.assertEqual(next(c['status'] for c in evaluate_hands(bad,DEFAULTS)[0] if c['id']==cid),'FAIL')
        bad=copy.deepcopy(trace);bad['frames'][1]['hands']['L']['palm_points'][0][0]=.1
        self.assertEqual(next(c['status'] for c in evaluate_hands(bad,DEFAULTS)[0] if c['id']=='C7.hand_skate'),'FAIL')
        for key,cid in [('wrist_neutral_deg','C7.rest_wrist'),('finger_bend_deg','C7.rest_fingers')]:
            bad=copy.deepcopy(trace);bad['rest_hands']['L'][key]=45.
            self.assertEqual(next(c['status'] for c in evaluate_hands(bad,DEFAULTS)[0] if c['id']==cid),'FAIL')
        bad=copy.deepcopy(trace);bad['frames']=bad['frames'][:2]
        self.assertFalse(all(c['status']=='PASS' for c in evaluate_hands(bad,DEFAULTS)[0]))
    def test_brow_duplicate_and_unproven_disappearance(self):
        expected=np.zeros((64,64),bool);expected[20:25,10:40]=True
        proof=dict(source_strokes=1,front_strokes=1,predicted_visible_strokes=1,depth_hidden_fraction=0.)
        self.assertTrue(assess_brow(expected,expected,100,proof)['pass'])
        duplicate=expected.copy();duplicate[32:37,10:40]=True
        self.assertFalse(assess_brow(duplicate,expected,100,proof)['pass'])
        self.assertFalse(assess_brow(np.zeros_like(expected),expected,100,proof)['pass'])
        tiny=np.zeros_like(expected);tiny[20:22,10:12]=True
        hidden=dict(source_strokes=1,front_strokes=1,predicted_visible_strokes=0,depth_hidden_fraction=.9)
        self.assertTrue(assess_brow(tiny,tiny,100,hidden)['pass'])
        self.assertFalse(assess_brow(np.zeros_like(tiny),tiny,100,hidden)['pass'])
        self.assertFalse(assess_brow(tiny,tiny,100,proof)['pass'])

if __name__=='__main__':unittest.main()
