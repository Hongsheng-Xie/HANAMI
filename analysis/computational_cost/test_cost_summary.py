import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from run_ten_seed_suite import save, summarize


class SummaryTest(unittest.TestCase):
    def test_drkg_label_is_not_reported_as_ms(self):
        with tempfile.TemporaryDirectory() as tmp, patch("run_ten_seed_suite.SEEDS", [1]), patch("run_ten_seed_suite.DATASET", "drkg"):
            output = Path(tmp)
            summarize(output)
            self.assertIn("DRKG computational cost", (output/"cost_table.md").read_text())
            self.assertIn("0/35 tasks and 0/5", (output/"summary.md").read_text())

    def test_single_seed_table_and_totals(self):
        with tempfile.TemporaryDirectory() as tmp, patch("run_ten_seed_suite.SEEDS", [10]):
            output = Path(tmp)
            path = output / "seed_10" / "HANAMI"
            path.mkdir(parents=True)
            rows = [dict(event="end", seed=10, method="HANAMI", task=str(i),
                         stages={"train": {"total_seconds": 60}},
                         inference_mean_seconds=0.01, peak_process_ram_mib=1024,
                         peak_gpu_allocated_mib=2048, test_motifs=10, timing_flags=[])
                    for i in range(7)]
            (path/"events.jsonl").write_text("\n".join(map(json.dumps, rows)))
            (path/"success.json").write_text("{}")
            self.assertEqual(summarize(output), (7, 1, 0))
            self.assertIn("7/35 tasks and 1/5", (output/"summary.md").read_text())
            self.assertIn("| HANAMI | 7.00 min | 2.00 GiB |", (output/"cost_table.md").read_text())

    def test_save_retries_transient_file_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "status.json"
            target.write_text('{"old": true}')
            original_replace = Path.replace
            attempts = []

            def locked_once(source, destination):
                attempts.append(1)
                if len(attempts) == 1:
                    raise PermissionError("simulated Windows file lock")
                return original_replace(source, destination)

            with patch.object(Path, "replace", locked_once), patch("run_ten_seed_suite.time.sleep"):
                save(target, {"new": True})
            self.assertEqual(len(attempts), 2)
            self.assertEqual(json.loads(target.read_text()), {"new": True})

    def test_save_preserves_existing_file_on_persistent_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "status.json"
            target.write_text('{"old": true}')
            with patch.object(Path, "replace", side_effect=PermissionError("locked")), patch("run_ten_seed_suite.time.sleep"):
                with self.assertRaises(PermissionError):
                    save(target, {"new": True})
            self.assertEqual(json.loads(target.read_text()), {"old": True})

    def test_aggregate_tasks_within_seeds_then_summarize_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            for seed, seconds in ((1, 60), (10, 120)):
                path = output / f"seed_{seed}" / "HANAMI"
                path.mkdir(parents=True)
                rows = [dict(event="end", seed=seed, method="HANAMI", task=str(i),
                             stages={"train": {"total_seconds": seconds}},
                             inference_mean_seconds=0.01, peak_process_ram_mib=1024+i,
                             peak_gpu_allocated_mib=2048+i, test_motifs=10, timing_flags=[])
                        for i in range(7)]
                (path/"events.jsonl").write_text("\n".join(map(json.dumps, rows)))
                (path/"success.json").write_text("{}")
            self.assertEqual(summarize(output), (14, 2, 0))
            costs = json.loads((output/"seed_level_costs.json").read_text())
            self.assertEqual([row["training_minutes"] for row in costs], [7, 14])
            self.assertAlmostEqual(costs[0]["inference_ms"], 70)
            self.assertEqual(costs[0]["peak_gpu_gib"], 2054/1024)
            summary = (output/"summary.md").read_text(encoding="utf-8")
            self.assertIn("10.50 ± 4.95", summary)

    def test_incomplete_job_not_in_seed_average(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(summarize(Path(tmp)), (0, 0, 0))


if __name__ == "__main__":
    unittest.main()
