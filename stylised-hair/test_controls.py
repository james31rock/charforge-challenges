"""Independent counterexamples for contact, response, clumping and data scope."""
import json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent))
from toolkit.qa.hair import strip_spacing

class HairControls(unittest.TestCase):
    def run_case(self,edit):
        with tempfile.TemporaryDirectory(dir=os.environ.get('CHARFORGE_SCRATCH')) as t:
            for clip in ['walk','jump']:
                with np.load(HERE/'tests/control'/f'{clip}.npz') as z:hair=z['hair'].copy()
                with np.load(HERE/'tests/control/assets'/f'{clip}.npz') as z:drive={k:z[k] for k in z.files}
                edit(hair,drive);np.savez_compressed(Path(t)/f'{clip}.npz',hair=hair)
            r=subprocess.run([sys.executable,str(HERE/'score.py'),t,'--control'],capture_output=True,text=True,timeout=180)
            report=json.loads(r.stdout[r.stdout.index('{'):]);self.assertEqual(r.returncode,1);self.assertNotIn('error',report)
            return {c['id']:c for c in report['clips']['walk']['checks']}
    def test_frozen_hair_fails_response(self):
        c=self.run_case(lambda h,d:h.__setitem__(slice(None),d['vertices']))
        self.assertEqual(c['C2.hair_response']['status'],'FAIL')
    def test_inside_surface_fails_contact(self):
        c=self.run_case(lambda h,d:h.__setitem__((0,slice(3,None)),d['head'][0,:3,3]+[.01,0,0]))
        self.assertEqual(c['C2.hair_penetration']['status'],'FAIL')
    def test_detached_roots_fail(self):
        c=self.run_case(lambda h,d:h.__iadd__(np.array([.02,0,0])))
        self.assertEqual(c['C2.root_attachment']['status'],'FAIL')
    def test_uniform_clump_is_not_hidden_by_zero_variance(self):
        rest=np.array([[i,0,0] for i in range(8)],float);posed=rest[None]*.1
        c={x['id']:x for x in strip_spacing(rest,posed)}
        self.assertEqual(c['C2.strip_spacing_variance']['status'],'PASS')
        self.assertEqual(c['C2.strip_spacing_minimum']['status'],'FAIL')
    def test_uneven_spacing_fails_variance(self):
        rest=np.array([[i,0,0] for i in range(8)],float);posed=rest[None].copy();posed[0,3,0]+=2
        c={x['id']:x for x in strip_spacing(rest,posed)}
        self.assertEqual(c['C2.strip_spacing_variance']['status'],'FAIL')
    def test_baked_clump_fails_minimum_spacing(self):
        def clump(hair,drive):
            hair[:,:,:2]=drive['head'][:,None,:2,3]+.1*(hair[:,:,:2]-drive['head'][:,None,:2,3])
        c=self.run_case(clump)
        self.assertEqual(c['C2.strip_spacing_minimum']['status'],'FAIL')

if __name__=='__main__':unittest.main()
