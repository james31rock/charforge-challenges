"""Caller-selected temporary output; never write generated media into the package."""
import os
import shutil
import tempfile
from pathlib import Path


def output_root():
    root = Path(os.environ.get('CHARFORGE_SCRATCH', Path(tempfile.gettempdir()) / 'charforge')).resolve()
    if root.is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError('Choose an output root outside the distribution')
    root.mkdir(parents=True, exist_ok=True)
    return root


def safe_output(path):
    out = Path(path).resolve()
    root = output_root()
    if out == root or not out.is_relative_to(root):
        raise ValueError('Output must be a child of CHARFORGE_SCRATCH')
    if shutil.disk_usage(root).free < 40 * 1024**3:
        raise RuntimeError('Insufficient disk: 40 GiB free required')
    return out
