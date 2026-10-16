import math
import unittest

import numpy as np
import pandas as pd

from ocsr_bench.metrics import Scorer, bootstrap_ci, calibration_table, summarize

from .helpers import StubBackend


class ScorerTests(unittest.TestCase):
    def setUp(self):
        self.scorer = Scorer(StubBackend())

    def test_exact_match(self):
        row = self.scorer.score("CCO", "CCO")
        self.assertEqual((row["exact_match"], row["tanimoto"], row["gt_category"]), (1.0, 1.0, "plain"))

    def test_stereo_error_is_caught_by_strict_but_not_loose_metrics(self):
        row = self.scorer.score("C[C@H](N)O", "C[C@@H](N)O")
        self.assertEqual(row["gt_category"], "chiral")
        self.assertEqual(row["exact_match"], 0.0)
        self.assertEqual(row["exact_match_no_stereo"], 1.0)
        self.assertEqual(row["inchikey_connectivity_match"], 1.0)
        self.assertEqual(row["inchikey_match"], 0.0)

    def test_invalid_prediction_counts_as_wrong(self):
        for pred in (None, "", "   ", "BAD(("):
            row = self.scorer.score("CCO", pred)
            self.assertFalse(row["pred_valid"])
            self.assertEqual((row["exact_match"], row["tanimoto"]), (0.0, 0.0))

    def test_unparseable_ground_truth_is_excluded(self):
        row = self.scorer.score("BAD", "CCO")
        self.assertFalse(row["gt_valid"])
        self.assertEqual(row["gt_category"], "unparseable")
        self.assertTrue(math.isnan(row["exact_match"]))

    def test_markush_category(self):
        self.assertEqual(self.scorer.score("C*N", "C*N")["gt_category"], "markush")

    def test_extended_smiles_suffix_is_ignored(self):
        self.assertEqual(self.scorer.score("CCO", "CCO<sep>a:b")["exact_match"], 1.0)

    def test_openbabel_cross_check_column(self):
        scorer = Scorer(StubBackend(), openbabel_fn=lambda s: s.strip() if s and s.strip() else None)
        self.assertEqual(scorer.score("CCO", "CCO")["exact_match_openbabel"], 1.0)
        self.assertEqual(scorer.score("CCO", "CCN")["exact_match_openbabel"], 0.0)
        self.assertTrue(math.isnan(self.scorer.score("CCO", "CCO")["exact_match_openbabel"]))


class AggregationTests(unittest.TestCase):
    def test_bootstrap_ci_is_deterministic_and_brackets_mean(self):
        values = np.array([1.0] * 70 + [0.0] * 30)
        low, high = bootstrap_ci(values, 500, seed=1)
        self.assertEqual((low, high), bootstrap_ci(values, 500, seed=1))
        self.assertLess(low, 0.7)
        self.assertGreater(high, 0.7)
        self.assertEqual(bootstrap_ci(np.ones(10), 100, seed=0), (1.0, 1.0))
        self.assertTrue(all(math.isnan(x) for x in bootstrap_ci(np.array([]), 100, seed=0)))

    def test_calibration_and_ece(self):
        conf = np.array([0.9, 0.9, 0.1, 0.1])
        correct = np.array([1, 1, 0, 0])
        table, ece = calibration_table(conf, correct, n_bins=10)
        self.assertAlmostEqual(ece, 0.1)
        self.assertEqual(sum(b["n"] for b in table), 4)

    def test_calibration_without_confidence(self):
        table, ece = calibration_table(np.array([np.nan, np.nan]), np.array([1, 0]), 5)
        self.assertEqual(table, [])
        self.assertTrue(math.isnan(ece))

    def test_summarize_overall_and_by_category(self):
        scorer = Scorer(StubBackend())
        cases = [("CCO", "CCO"), ("CCN", "CCO"), ("C[C@H](N)O", "C[C@@H](N)O"), ("C*N", "C*N"), ("BAD", "CCO")]
        rows = []
        for i, (gt, pred) in enumerate(cases):
            rows.append({"gt_smiles": gt, "pred_smiles": pred, "confidence": 0.8, "runtime_s": 0.5,
                         "error": "boom" if i == 1 else None, **scorer.score(gt, pred)})  # fmt: skip
        result = summarize(pd.DataFrame(rows), {"bootstrap_samples": 50, "confidence_bins": 10}, seed=0)
        overall = result["categories"]["all"]
        self.assertEqual((overall["n_total"], overall["n_scored"], overall["n_gt_unparseable"]), (5, 4, 1))
        self.assertAlmostEqual(overall["exact_match"], 0.5)  # CCO and C*N correct, 2 of 4 scorable
        self.assertAlmostEqual(overall["exact_match_no_stereo"], 0.75)
        self.assertEqual(overall["n_tool_errors"], 1)
        self.assertAlmostEqual(overall["throughput_img_per_s"], 2.0)
        cats = result["categories"]
        self.assertEqual((cats["chiral"]["n_scored"], cats["markush"]["n_scored"]), (1, 1))
        self.assertEqual(cats["chiral"]["exact_match"], 0.0)
        self.assertIn("ece", result["calibration"])


if __name__ == "__main__":
    unittest.main()


class FailureAndPairwiseTests(unittest.TestCase):
    def setUp(self):
        from ocsr_bench.metrics import classify_failure
        self.classify = classify_failure
        self.scorer = Scorer(StubBackend())

    def row(self, gt, pred, error=None):
        return pd.Series({**self.scorer.score(gt, pred), "error": error})

    def test_failure_types(self):
        cases = {
            ("CCO", "CCO"): "correct",
            ("C[C@H](N)O", "C[C@@H](N)O"): "stereo_only",
            ("CCC", "CCO"): "near_miss" if False else "wrong_structure",   # Jaccard 0.5 < 0.85
            ("CCO", None): "invalid_output",
            ("C*N", "CCN"): "markush_mismatch",
            ("BAD", "CCO"): "gt_unparseable",
        }
        for (gt, pred), expected in cases.items():
            self.assertEqual(self.classify(self.row(gt, pred)), expected, (gt, pred))
        self.assertEqual(self.classify(self.row("CCO", None, error="boom")), "tool_error")

    def test_pairwise_comparison_pairs_images_and_reports_ci(self):
        from ocsr_bench.metrics import pairwise_comparison
        rows = []
        for tool, hits in (("a", [1, 1, 1, 0]), ("b", [1, 0, 0, 0])):
            for i, h in enumerate(hits):
                rows.append({"tool": tool, "dataset": "d", "sample_id": f"s{i}", "gt_valid": True, "exact_match": float(h)})
        table = pairwise_comparison(pd.DataFrame(rows), {"bootstrap_samples": 200}, seed=1)
        self.assertEqual(len(table), 1)
        first = table.iloc[0]
        self.assertEqual((first["tool_a"], first["tool_b"], first["n_paired"]), ("a", "b", 4))
        self.assertAlmostEqual(first["difference_a_minus_b"], 0.5)
        self.assertLessEqual(first["ci_low"], 0.5)
        self.assertGreaterEqual(first["ci_high"], 0.5)
