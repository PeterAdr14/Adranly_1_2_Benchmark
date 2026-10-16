"""Shared test fixtures: a chemistry-free stub backend and tiny image factories."""

from __future__ import annotations

import zlib
from pathlib import Path
from typing import Optional

from PIL import Image

from ocsr_bench.chem import MolInfo


class StubBackend:
    """Deterministic stand-in for RDKit so orchestration logic is testable anywhere.

    A "molecule" is just its SMILES string; stereo marks are ``@ / \\``.
    Strings containing ``BAD`` or empty strings are invalid. Similarity is the
    Jaccard index of the character sets.
    """

    def analyze(self, smiles: Optional[str]) -> Optional[MolInfo]:
        if smiles is None or not smiles.strip() or "BAD" in smiles:
            return None
        text = smiles.strip()
        flat = text.replace("@", "").replace("/", "").replace("\\", "")
        return MolInfo(
            canonical=text,
            canonical_flat=flat,
            inchikey=f"{zlib.crc32(flat.encode()):08X}-{zlib.crc32(text.encode()):010X}-N",
            fingerprint=frozenset(flat),
            has_stereo="@" in text,
        )

    def similarity(self, a: MolInfo, b: MolInfo) -> float:
        union = a.fingerprint | b.fingerprint
        return len(a.fingerprint & b.fingerprint) / len(union) if union else 1.0


def make_png(path: Path, size: int = 8) -> Path:
    """Write a tiny white PNG, creating parent folders."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (size, size), "white").save(path)
    return path
