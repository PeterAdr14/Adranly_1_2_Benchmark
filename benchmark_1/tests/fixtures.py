"""Offline test world: fake tool checkouts, a fake Hugging Face download, a fake `osra`, and tiny datasets.

Both models are stand-ins that read their answer from a sidecar file next to each image (``x.pred`` for MolGlyph,
``x.pred2`` for MolNexTR; a missing sidecar makes the tool fail on that image). MolNexTR runs in a *real* worker
subprocess so the JSON protocol is exercised.
"""

from __future__ import annotations

import os
import stat
import sys
import types
import zipfile
from pathlib import Path
from unittest import mock

from .helpers import make_png


def fake_torch() -> types.ModuleType:
    module = types.ModuleType("torch")
    module.device = lambda s: f"device({s})"
    module.__version__ = "0.0-fake"
    module.cuda = types.SimpleNamespace(is_available=lambda: False)
    module.manual_seed = lambda s: None
    return module


def fake_hub(real_zip: Path) -> types.ModuleType:
    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = mock.Mock(side_effect=lambda repo_id, filename, repo_type=None, token=None: str(real_zip))

    def offline(*args, **kwargs):
        raise OSError("offline")

    hub.get_hf_file_metadata, hub.hf_hub_url = offline, lambda *a, **k: "url"
    hub.HfApi = lambda: types.SimpleNamespace(list_repo_files=offline)
    return hub


def write(path: Path, text: str = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if text is None:
        make_png(path)
    else:
        path.write_text(text)


def build_world(tmp: Path) -> dict:
    """Create every fixture under `tmp` and return their locations."""
    # --- fake BioMiner checkout (real layout; the model returns plain SMILES strings) ---
    bm = tmp / "fake_biominer"
    pkg = bm / "BioMiner" / "MolScribe" / "molscribe"
    write(pkg / "__init__.py", "raise RuntimeError('package __init__ must not run')\n")
    write(pkg / "model.py", "class Encoder: ...\nclass Decoder: ...\n")
    write(pkg / "interface_molglyph.py", '''
from pathlib import Path
from .model import Encoder, Decoder
class MolGlyph:
    def __init__(self, model_path, device=None, num_workers=1): pass
    def predict_image_files(self, image_files, return_atoms_bonds=False, return_confidence=False):
        out = []
        for f in image_files:
            side = Path(f).with_suffix(".pred")
            if not side.exists():
                raise FileNotFoundError(f)
            out.append(side.read_text().strip())
        return out
''')
    write(bm / "BioMiner" / "commons" / "molglyph_utils.py", '''
class Translator:
    @staticmethod
    def refactor(s):
        if s == "EXPLODE":
            raise ValueError("cannot translate")
        return ("t", s.split("<sep>")[0])
def get_refactor(res): return res[1]
''')
    # --- fake MolNexTR checkout + a fake torch for the *worker* process ---
    mn = tmp / "fake_molnextr"
    write(mn / "MolNexTR" / "__init__.py", '''
from pathlib import Path
class molnextr:
    def __init__(self, model_path, device=None): self.model_path = model_path
    def predict_final_results(self, image_path, return_atoms_bonds=False, return_confidence=False):
        side = Path(image_path).with_suffix(".pred2")
        if not side.exists():
            raise FileNotFoundError(image_path)
        return {"predicted_smiles": side.read_text().strip(), "predicted_molfile": "x",
                "atom_sets": [{"symbol": "C", "conf": 0.9}, {"symbol": "C", "conf": 0.7}], "bonds_sets": []}
''')
    write(mn / "requirements.txt", "numpy\n")
    write(tmp / "fake_site" / "torch" / "__init__.py", '__version__ = "0.0-fake"\ndef device(s): return s\n'
          'class _C:\n    @staticmethod\n    def is_available(): return False\ncuda = _C()\n')
    (tmp / "ckpt.bin").write_bytes(b"x")
    # --- fake osra program ---
    bin_dir = tmp / "bin"
    bin_dir.mkdir()
    osra = bin_dir / "osra"
    osra.write_text('#!/bin/sh\nf="$3"; side="${f%.png}.pred"; [ -f "$side" ] && cat "$side"; true\n')
    osra.chmod(osra.stat().st_mode | stat.S_IEXEC)
    # --- fake real.zip: MolScribe-style CSVs, image names collide across datasets ---
    stage = tmp / "stage"
    clef = [("CCO", "CCO", "CCO"), ("CCN", "CCN", "CCC"), ("CCC", "CCO", "CCC")]       # (gt, MolGlyph, MolNexTR)
    uob = [("CCCC", "CCCC", None), ("C[C@H](N)O", "C[C@@H](N)O", "C[C@H](N)O")]         # None: MolNexTR fails
    write(stage / "real" / "CLEF.csv", "image_id,file_path,SMILES\n" + "".join(f"c{i},real/CLEF/{i+1}.png,{g}\n" for i, (g, _, _) in enumerate(clef)))
    write(stage / "real" / "UOB.csv", "image_id,file_path,SMILES\n" + "".join(f"u{i},UOB/{i+1}.png,{g}\n" for i, (g, _, _) in enumerate(uob)))
    write(stage / "real" / "ACS.csv", "image_id,file_path,SMILES\na1,ACS/1.png,CC\n")
    write(stage / "real" / "ACS" / "1.png")
    for folder, rows in (("CLEF", clef), ("UOB", uob)):
        for i, (gt, glyph, nexus) in enumerate(rows):
            write(stage / "real" / folder / f"{i+1}.png")
            write(stage / "real" / folder / f"{i+1}.pred", glyph)
            if nexus is not None:
                write(stage / "real" / folder / f"{i+1}.pred2", nexus)
    real_zip = tmp / "real.zip"
    with zipfile.ZipFile(real_zip, "w") as z:
        for p in sorted(stage.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(stage).as_posix())
    # --- BioVista-style dataset ---
    rows = ["pdf_name,index,smiles"]
    for i, (gt, glyph, nexus) in enumerate([("CCO", "CCO", "CCO"), ("CCN", "CCN", "CCN")]):
        name = f"{40 + i}_6s8a"
        p = make_png(tmp / "bv" / "ocsr" / "data" / f"{name}_image_merge_bbox_3-{i}.png")
        p.with_suffix(".pred").write_text(glyph)
        p.with_suffix(".pred2").write_text(nexus)
        rows.append(f"{name},{i},{gt}")
    write(tmp / "bv" / "ocsr" / "ocsr_gd.csv", "\n".join(rows) + "\n")
    return {"biominer": bm, "molnextr": mn, "site": tmp / "fake_site", "bin": bin_dir, "real_zip": real_zip, "bv": tmp / "bv"}


def make_config(tmp: Path, world: dict, work: Path = None) -> dict:
    """A full configuration (same shape as configs/default.yaml) pointing at the fixtures."""
    work = work or tmp / "work"
    param = lambda **kw: kw
    archive = dict(hf_repo="x/y", hf_filename="real.zip", repo_type="dataset", extract_name="real")
    return {
        "run": {"name": "t", "output_dir": str(tmp / "out"), "data_dir": str(work / "data"), "seed": 7, "max_samples": None,
                "batch_size": 2, "device": "auto", "resume": True},
        "metrics": {"fingerprint_radius": 2, "fingerprint_bits": 2048, "bootstrap_samples": 60, "confidence_bins": 10,
                    "openbabel_cross_check": False},
        "datasets": [
            {"name": "clef", "kind": "hf_archive", "params": param(**archive, labels_match="clef")},
            {"name": "uob", "kind": "hf_archive", "params": param(**archive, labels_match="uob")},
            {"name": "biovista_ocsr", "kind": "local_dir", "params": param(path=str(world["bv"]))},
        ],
        "tools": [
            {"name": "biominer_molglyph", "kind": "molglyph", "params": param(
                checkpoint_path=str(tmp / "ckpt.bin"), biominer_repo_path=str(world["biominer"]))},
            {"name": "molnextr", "kind": "molnextr", "params": param(
                checkpoint_path=str(tmp / "ckpt.bin"), repo_path=str(world["molnextr"]), python_executable=sys.executable,
                worker_env={"PYTHONPATH": str(world["site"])})},
            {"name": "osra", "kind": "osra", "params": param(timeout_s=30)},
        ],
    }
