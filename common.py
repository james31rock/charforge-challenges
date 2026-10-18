"""Local data-only scoring utilities. Proposed MIT; see LICENSE."""
import hashlib, json, os, shutil, subprocess, tempfile, zipfile
from pathlib import Path

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf8'))
def write(path, data): Path(path).write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf8')
def scratch():
    base=os.environ.get('CHARFORGE_SCRATCH')
    if base:
        Path(base).mkdir(parents=True,exist_ok=True)
    return tempfile.TemporaryDirectory(prefix='charforge-score-',dir=base)
def blender(script, args, directory):
    exe=os.environ.get('BLENDER') or shutil.which('blender')
    if not exe: raise ValueError('Set BLENDER to a Blender 5.x executable, or add Blender to PATH')
    p=subprocess.run([exe,'-b','--factory-startup','-Y','-t','2','--python-exit-code','3','--python',str(script),'--',*map(str,args)],capture_output=True,text=True,timeout=840,cwd=directory)
    if p.returncode:
        raise ValueError('Headless extraction failed: '+p.stdout[-1500:]+p.stderr[-500:])
def load_npz(path, limit=128*1024**2):
    import numpy as np
    path=Path(path)
    if path.stat().st_size>limit:raise ValueError('Input too large')
    with zipfile.ZipFile(path) as z:
        if sum(x.file_size for x in z.infolist())>limit:raise ValueError('Expanded input too large')
    with np.load(path,allow_pickle=False) as z: data={k:z[k] for k in z.files}
    for a in data.values():
        if a.dtype.kind not in 'iufbUS':raise ValueError('Unsupported array dtype')
        if a.dtype.kind in 'f' and not np.isfinite(a).all():raise ValueError('Nonfinite input')
    return data
def finish(report, output=None):
    print(report['status'])
    print(json.dumps(report,indent=2,allow_nan=False))
    if output:write(output,report)
    return 0 if report['status']=='PASS' else 1

