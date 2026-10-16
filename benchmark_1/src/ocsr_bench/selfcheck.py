"""Chemistry sanity check with known answers (catches a broken RDKit install or a scoring regression)."""

from __future__ import annotations

from .chem import RDKitBackend
from .metrics import Scorer


def run_selfcheck() -> None:
    """Raise AssertionError if the chemistry stack does not behave as expected."""
    _backend = RDKitBackend()
    _scorer = Scorer(_backend)

    assert _backend.analyze("OCC").canonical == _backend.analyze("C(O)C").canonical          # same molecule, different strings
    assert _backend.analyze("CCO").inchikey == "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"                  # ethanol
    assert _backend.analyze("C(C") is None and _backend.analyze("") is None                   # invalid / empty
    l_ala, d_ala = _backend.analyze("C[C@H](N)C(=O)O"), _backend.analyze("C[C@@H](N)C(=O)O")
    assert l_ala.canonical != d_ala.canonical and l_ala.canonical_flat == d_ala.canonical_flat  # stereo is respected
    row = _scorer.score("C[C@H](N)C(=O)O", "C[C@@H](N)C(=O)O")
    assert (row["exact_match"], row["exact_match_no_stereo"], row["gt_category"]) == (0.0, 1.0, "chiral")
    assert _scorer.score("CCO", "CCO")["tanimoto"] == 1.0 and _scorer.score("CCO", None)["exact_match"] == 0.0
    assert _scorer.score("C*N", "C*N")["gt_category"] == "markush"
    print("Chemistry sanity checks passed.")
