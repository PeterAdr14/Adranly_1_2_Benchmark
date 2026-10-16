"""Chemistry helpers: SMILES canonicalization, InChIKeys and fingerprints (RDKit; optional Open Babel)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

# ----------------------------------------------------------------------------------------------
# Chemistry layer. RDKit is the primary backend; Open Babel is an optional second opinion used only
# for the `exact_match_openbabel` cross-check column. Heavy imports are lazy.
# ----------------------------------------------------------------------------------------------
_MARKUSH_RE = re.compile(r"\*|\[R\d*[a-z]?\]")  # wildcard atoms or generic R-group labels


@dataclass(frozen=True)
class MolInfo:
    """Everything the scorer needs to know about one parsed molecule."""
    canonical: str       # canonical isomeric SMILES
    canonical_flat: str  # canonical SMILES with stereochemistry removed
    inchikey: str        # standard InChIKey, "" if InChI generation failed
    fingerprint: Any     # backend-specific fingerprint object
    has_stereo: bool     # any assigned stereocenter or stereo double bond

    @property
    def inchikey_connectivity(self) -> str:
        """First InChIKey block (connectivity layer; ignores stereo and isotopes)."""
        return self.inchikey.split("-")[0] if self.inchikey else ""


def strip_extended_smiles(smiles: Any) -> str:
    """Drop an E-SMILES style '<sep>...' suffix. Non-strings (None, pandas NaN) give ''."""
    if not isinstance(smiles, str) or not smiles:
        return ""
    return smiles.split("<sep>")[0].strip()


def is_markush(smiles: str) -> bool:
    """Heuristic: does this ground-truth SMILES contain wildcard / R-group atoms?"""
    return bool(_MARKUSH_RE.search(smiles or ""))


class RDKitBackend:
    """Canonicalization, InChIKey and Morgan-fingerprint similarity via RDKit."""

    def __init__(self, radius: int = 2, n_bits: int = 2048) -> None:
        from rdkit import Chem, DataStructs, RDLogger
        from rdkit.Chem import rdFingerprintGenerator
        RDLogger.DisableLog("rdApp.*")
        self._chem, self._data_structs = Chem, DataStructs
        self._fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)

    def analyze(self, smiles: Optional[str]) -> Optional[MolInfo]:
        """Parse `smiles`; return None if it is empty or chemically invalid."""
        if smiles is None or not smiles.strip():
            return None
        Chem = self._chem
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None or mol.GetNumAtoms() == 0:
            return None
        try:
            inchikey = Chem.MolToInchiKey(mol) or ""
        except Exception:  # InChI cannot represent every structure (e.g. wildcard atoms)
            inchikey = ""
        stereo_atoms = Chem.FindMolChiralCenters(mol, includeUnassigned=False)
        stereo_bonds = any(b.GetStereo() != Chem.BondStereo.STEREONONE for b in mol.GetBonds())
        return MolInfo(
            canonical=Chem.MolToSmiles(mol, isomericSmiles=True),
            canonical_flat=Chem.MolToSmiles(mol, isomericSmiles=False),
            inchikey=inchikey,
            fingerprint=self._fp_gen.GetFingerprint(mol),
            has_stereo=bool(stereo_atoms) or stereo_bonds,
        )

    def similarity(self, a: MolInfo, b: MolInfo) -> float:
        """Tanimoto similarity of the Morgan fingerprints."""
        return float(self._data_structs.TanimotoSimilarity(a.fingerprint, b.fingerprint))


def molblock_to_smiles(molblock: str) -> Optional[str]:
    """Convert an MDL molfile string to canonical SMILES (None on failure)."""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromMolBlock(molblock, sanitize=True, removeHs=True)
    return Chem.MolToSmiles(mol) if mol is not None else None


def openbabel_available() -> bool:
    try:
        from openbabel import pybel  # noqa: F401
    except ImportError:
        return False
    return True


def openbabel_canonical(smiles: Optional[str]) -> Optional[str]:
    """Open Babel canonical SMILES, or None if the input cannot be parsed."""
    if smiles is None or not smiles.strip():
        return None
    from openbabel import openbabel as ob
    from openbabel import pybel
    ob.obErrorLog.SetOutputLevel(0)
    try:
        return pybel.readstring("smi", smiles.strip()).write("can").split()[0]
    except (OSError, IndexError, ValueError):
        return None
