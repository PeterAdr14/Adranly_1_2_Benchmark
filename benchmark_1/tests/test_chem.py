"""RDKit-backed chemistry tests (skipped automatically if RDKit is not installed)."""

import unittest

from ocsr_bench import chem

try:
    import rdkit  # noqa: F401

    HAVE_RDKIT = True
except ImportError:
    HAVE_RDKIT = False


class StringHelperTests(unittest.TestCase):
    def test_markush_detection(self):
        self.assertTrue(chem.is_markush("C*N"))
        self.assertTrue(chem.is_markush("[R1]CC"))
        self.assertFalse(chem.is_markush("CCO"))
        self.assertFalse(chem.is_markush(""))

    def test_strip_extended_smiles(self):
        self.assertEqual(chem.strip_extended_smiles("CCO<sep>a:1"), "CCO")
        self.assertEqual(chem.strip_extended_smiles("  CCO "), "CCO")
        self.assertEqual(chem.strip_extended_smiles(None), "")


@unittest.skipUnless(HAVE_RDKIT, "RDKit not installed")
class RDKitBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = chem.RDKitBackend(radius=2, n_bits=2048)

    def test_equivalent_smiles_canonicalize_identically(self):
        a, b = self.backend.analyze("OCC"), self.backend.analyze("C(O)C")
        self.assertEqual(a.canonical, b.canonical)

    def test_stereo_handling(self):
        l_ala = self.backend.analyze("C[C@H](N)C(=O)O")
        d_ala = self.backend.analyze("C[C@@H](N)C(=O)O")
        self.assertNotEqual(l_ala.canonical, d_ala.canonical)
        self.assertEqual(l_ala.canonical_flat, d_ala.canonical_flat)
        self.assertEqual(l_ala.inchikey_connectivity, d_ala.inchikey_connectivity)
        self.assertNotEqual(l_ala.inchikey, d_ala.inchikey)
        self.assertTrue(l_ala.has_stereo)
        self.assertFalse(self.backend.analyze("CCO").has_stereo)

    def test_double_bond_stereo_counts_as_stereo(self):
        self.assertTrue(self.backend.analyze("C/C=C/C").has_stereo)

    def test_invalid_inputs(self):
        for bad in (None, "", "   ", "C(C", "not a smiles"):
            self.assertIsNone(self.backend.analyze(bad), bad)

    def test_ethanol_inchikey(self):
        self.assertEqual(self.backend.analyze("CCO").inchikey, "LFQSCWFLJHTTHZ-UHFFFAOYSA-N")

    def test_similarity(self):
        ethanol, benzene = self.backend.analyze("CCO"), self.backend.analyze("c1ccccc1")
        self.assertAlmostEqual(self.backend.similarity(ethanol, ethanol), 1.0)
        self.assertLess(self.backend.similarity(ethanol, benzene), 0.5)

    def test_wildcard_atoms_still_parse(self):
        self.assertIsNotNone(self.backend.analyze("C*N"))


@unittest.skipUnless(chem.openbabel_available(), "Open Babel not installed")
class OpenBabelTests(unittest.TestCase):
    def test_canonical_equivalence_and_invalid(self):
        self.assertEqual(chem.openbabel_canonical("OCC"), chem.openbabel_canonical("C(O)C"))
        self.assertIsNone(chem.openbabel_canonical(""))


if __name__ == "__main__":
    unittest.main()
