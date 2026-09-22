import json
from pathlib import Path
import tempfile
import unittest

from import_records import redact
from rebuild_recorded_tables import build, summarize_tasks


class RecordedTablesTest(unittest.TestCase):
    def test_path_redaction_preserves_numeric_values(self):
        value = {r"C:\local\data": {"seconds": 123.456, "seed": 1,
                 "nested": [r"C:\local\output", None, True]}}
        revised = redact(value, [(r"C:\local", "<ROOT>")])
        self.assertEqual(revised, {r"<ROOT>\data": {"seconds": 123.456, "seed": 1,
                         "nested": [r"<ROOT>\output", None, True]}})

    def test_incomplete_and_duplicate_tasks_cannot_become_complete_seed(self):
        rows = [dict(method="HANAMI", seed=30, task=str(i),
                     stages={"train": {"total_seconds": 60}},
                     inference_mean_seconds=0.01, peak_process_ram_mib=1024,
                     peak_gpu_allocated_mib=2048) for i in range(7)]
        self.assertEqual(summarize_tasks(rows[:2]), [])
        self.assertEqual(summarize_tasks(rows[:6] + [rows[0]]), [])
        self.assertEqual(summarize_tasks(rows)[0]["training_minutes"], 7)

    def test_archives_and_drkg_table(self):
        repo = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as temporary:
            result = build(repo / "data/computational_cost", Path(temporary))
            drkg = result["drkg_seed1_full_20260912_190355"]["summary"]
            expected = {
                "RF": [570.38, 1766.22, 15.90, 0],
                "TriNet": [0.82, 16.68, 1.98, 1.81],
                "N2V-MLP": [9.40, 49.06, 2.39, 3.56],
                "TriMoGCL": [85.66, 157.80, 3.59, 7.21],
                "HANAMI": [265.93, 389.18, 4.69, 4.66],
            }
            for row in drkg:
                actual = [row[key] for key in ("training_minutes", "inference_ms", "peak_ram_gib", "peak_gpu_gib")]
                for i in range(3 if row["method"] == "RF" else 4):
                    self.assertEqual(round(actual[i], 2), expected[row["method"]][i])
                self.assertEqual(row["seeds"], [1])
            self.assertEqual(result["ms_hanami_seed30_clean_20260914"]["jobs"], [])
            original = result["ms_ten_seeds_20260911"]["summary"]
            self.assertTrue(all(row["n_seeds"] == 10 for row in original))
            # Retain known timing flags; do not silently clean the original run.
            self.assertGreater(sum(row["timing_flags"] for row in original), 0)
            reviewed = result["cost_timing_verification_20260911"]["summary"]
            trimogcl = next(row for row in reviewed if row["method"] == "TriMoGCL")
            self.assertEqual(round(trimogcl["training_minutes"], 2), 12.96)
            self.assertEqual(trimogcl["n_seeds"], 1)


if __name__ == "__main__":
    unittest.main()
