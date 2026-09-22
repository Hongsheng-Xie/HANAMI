"""Regression tests for the released Figure 5 inputs and reconstruction.

Run from the repository root:
python -m unittest discover -s analysis/clinical_concordance/tests -v
"""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build_mean_rank_percentile_source as means
import build_figure5_errorbar_data as cases_builder


class CurrentFigure5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest, cls.cohort, cls.cases, cls.percentiles, cls.checks = means.reconstruct()

    def test_ties_rank_over_full_pool(self):
        result = means.rank_percentiles(np.array([[9., 3., 3., 0.]]))
        np.testing.assert_array_equal(result, [[25., 62.5, 62.5, 100.]])

    def test_invalid_scores_rejected(self):
        for array in [np.array([1., 2.]), np.array([[1., np.nan]]), np.empty((1, 0))]:
            with self.assertRaises(ValueError):
                means.rank_percentiles(array)

    def test_frozen_cohort_counts_and_cases(self):
        self.assertEqual(len(self.cohort), 1630)
        self.assertEqual(self.cohort.candidate_index.nunique(), 1630)
        self.assertEqual(len(self.cohort[["disease_local_id", "drug_local_id"]].drop_duplicates()), 785)
        self.assertEqual(self.cohort.disease_mesh_id.nunique(), 109)
        self.assertEqual(self.cases.candidate_index.tolist(), [3999, 14585, 40511, 6118, 35674])
        self.assertEqual(self.cases.panel.tolist(), list("cdefg"))
        self.assertTrue(self.cohort.supporting_trials.notna().all())

    def test_released_means_reproduced(self):
        released = pd.read_csv(means.DEFAULT_INPUT / "validated_gene_star_motifs_1630_mean10.tsv", sep="\t")
        for key, column in means.METHOD_COLUMNS.items():
            # Nonzero tolerance is only for the release table's decimal rounding.
            np.testing.assert_allclose(self.cohort[column], released[column], rtol=0, atol=1e-8)
            expected = self.percentiles[key].mean(axis=0)[self.cohort.candidate_index]
            np.testing.assert_allclose(self.cohort[column], expected, rtol=0, atol=1e-12)
        self.assertFalse(self.manifest["median_used"])
        self.assertEqual(self.manifest["seeds"], list(range(0, 100, 10)))

    def test_frozen_inputs_never_written(self):
        files = list(means.DEFAULT_INPUT.rglob("*"))
        before = {path: means.sha256(path) for path in files if path.is_file()}
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            table = means.build(output_dir=output_dir)
            case_file = cases_builder.build(output_dir=output_dir)
            self.assertTrue(table.exists())
            result = pd.read_csv(case_file)
            self.assertEqual(len(result), 250)
            self.assertFalse(result.duplicated(["panel", "method", "seed"]).any())
            self.assertTrue((result.groupby(["panel", "method"]).size() == 10).all())
            for case in self.cases.itertuples(index=False):
                source = self.cohort.set_index("candidate_index").loc[case.candidate_index]
                for key, column in means.METHOD_COLUMNS.items():
                    values = result.loc[(result.panel == case.panel) & (result.method == means.METHOD_NAMES[key]), "rank_percentile"]
                    self.assertAlmostEqual(values.mean(), source[column], places=10)
        after = {path: means.sha256(path) for path in before}
        self.assertEqual(before, after)
        with self.assertRaises(ValueError):
            means.ensure_output_directory(means.DEFAULT_INPUT, means.DEFAULT_INPUT)

    def test_mismatched_checksum_rejected(self):
        with self.assertRaises(ValueError):
            means.checked_path(means.DEFAULT_INPUT, {"path": "cohort_metadata.tsv", "sha256": "0" * 64})


if __name__ == "__main__":
    unittest.main()
