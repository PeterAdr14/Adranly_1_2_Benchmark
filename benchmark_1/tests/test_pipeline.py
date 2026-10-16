"""End-to-end tests with fake models: full pipeline, resume, worker process, CLI commands."""

import io
import json
import os
import re
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import pandas as pd
import yaml

from ocsr_bench import cli
from ocsr_bench.config import get_specs
from ocsr_bench.metrics import Scorer
from ocsr_bench.pipeline import PipelineError, run_benchmark
from ocsr_bench.tools import ToolSetupError
from ocsr_bench.tools.molnextr import MolNexTRTool
from ocsr_bench.config import Spec

from .fixtures import build_world, fake_hub, fake_torch, make_config
from .helpers import StubBackend


def stub_scorer(*args, **kwargs):
    return Scorer(StubBackend())


class OfflinePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)
        cls.world = build_world(cls.tmp)
        cls._patches = [
            mock.patch.dict(sys.modules, {"torch": fake_torch(), "huggingface_hub": fake_hub(cls.world["real_zip"])}),
            mock.patch.dict(os.environ, {"PATH": f"{cls.world['bin']}:{os.environ['PATH']}"}),
            mock.patch("ocsr_bench.metrics.Scorer.from_config", side_effect=stub_scorer),
        ]
        for patch in cls._patches:
            patch.start()
        logging_off = __import__("logging").disable
        logging_off(50)
        cls.cfg = make_config(cls.tmp, cls.world)
        cls.res = run_benchmark(cls.cfg, scorer=stub_scorer())

    @classmethod
    def tearDownClass(cls):
        for patch in reversed(cls._patches):
            patch.stop()
        __import__("logging").disable(0)
        cls._tmp.cleanup()

    def em(self, tool, dataset, category="all"):
        return self.res.metrics[tool][dataset]["categories"][category]["exact_match"]

    def test_metrics_for_both_tools_on_all_datasets(self):
        self.assertAlmostEqual(self.em("biominer_molglyph", "clef"), 2 / 3)
        self.assertAlmostEqual(self.em("biominer_molglyph", "uob"), 1 / 2)
        self.assertAlmostEqual(self.em("molnextr", "clef"), 2 / 3)
        self.assertAlmostEqual(self.em("molnextr", "uob"), 1 / 2)
        self.assertEqual(self.res.metrics["molnextr"]["uob"]["categories"]["all"]["n_tool_errors"], 1)

    def test_dataset_layouts_were_understood(self):
        info = lambda ds: self.res.metrics["molnextr"][ds]["dataset_info"]
        self.assertEqual(info("clef")["layout"], "reference")           # file_path relative to the archive root
        self.assertEqual(info("uob")["n_unresolved_rows"], 0)           # relative to the CSV folder, colliding names
        self.assertEqual(info("biovista_ocsr")["layout"], "composite")  # BioVista's pdf_name + index convention

    def test_confidence_and_failure_types(self):
        p = self.res.predictions
        glyph, nexus = p[p.tool == "biominer_molglyph"], p[p.tool == "molnextr"]
        self.assertTrue(glyph.confidence.isna().all())                  # MolGlyph reports none
        self.assertAlmostEqual(nexus.confidence.dropna().iloc[0], 0.8)  # mean of the per-atom confidences
        kind = lambda tool, ds, gt: p[(p.tool == tool) & (p.dataset == ds) & (p.gt_smiles == gt)].failure_type.iloc[0]
        self.assertEqual(kind("biominer_molglyph", "uob", "C[C@H](N)O"), "stereo_only")
        self.assertEqual(kind("molnextr", "uob", "CCCC"), "tool_error")
        self.assertEqual(kind("molnextr", "clef", "CCC"), "correct")

    def test_raw_outputs_keep_everything_the_tools_returned(self):
        raw = (self.res.out_dir / "raw" / "molnextr__clef.jsonl").read_text().splitlines()
        extra = json.loads(raw[0])["extra"]
        self.assertEqual(extra["confidence_source"], "per-atom/bond mean")
        self.assertIn("atom_sets", extra["output_keys"])

    def test_output_files_and_pairwise_table(self):
        for name in ("predictions.csv", "comparison.csv", "pairwise_comparison.csv", "failure_summary.csv",
                     "metrics.json", "summary.md", "run_metadata.json"):
            self.assertTrue((self.res.out_dir / name).is_file(), name)
        self.assertEqual(len(self.res.pairwise[self.res.pairwise.dataset == "clef"]), 3)   # 3 tool pairs
        metadata = json.loads((self.res.out_dir / "run_metadata.json").read_text())
        self.assertEqual(metadata["tool_resources"]["molnextr"]["worker_torch"], "0.0-fake")
        self.assertIn("biominer_commit", metadata["tool_resources"]["biominer_molglyph"])

    def test_resume_runs_nothing_and_a_changed_config_is_refused(self):
        calls = []
        original = MolNexTRTool.predict_batch
        with mock.patch.object(MolNexTRTool, "predict_batch", lambda self, paths: (calls.append(1), original(self, paths))[1]):
            run_benchmark(self.cfg, scorer=stub_scorer())
        self.assertEqual(calls, [])
        changed = json.loads(json.dumps(self.cfg))
        changed["run"]["batch_size"] = 3
        with self.assertRaises(PipelineError):
            run_benchmark(changed, scorer=stub_scorer())

    def test_resume_survives_moving_the_output_folder_settings(self):
        # data_dir / output_dir are location-only settings: they must not trigger the config-mismatch refusal
        from ocsr_bench.utils import config_hash
        moved = json.loads(json.dumps(self.cfg))
        moved["run"]["data_dir"] = "/somewhere/else"
        self.assertEqual(config_hash(moved), config_hash(self.cfg))

    def test_worker_is_restarted_if_it_dies(self):
        spec = Spec("molnextr", "molnextr", self.cfg["tools"][1]["params"])
        tool = MolNexTRTool(spec, self.cfg["run"], Path(self.cfg["run"]["data_dir"]))
        tool.load()
        try:
            image = next(Path(self.cfg["run"]["data_dir"]).rglob("CLEF/1.png"))
            self.assertEqual(tool.predict_batch([image])[0].smiles, "CCO")
            tool._worker._proc.kill()
            tool._worker._proc.wait()
            self.assertEqual(tool.predict_batch([image])[0].smiles, "CCO")
        finally:
            tool.close()

    def test_worker_start_failure_reports_the_traceback_and_a_hint(self):
        broken = self.tmp / "broken_molnextr"
        (broken / "MolNexTR").mkdir(parents=True, exist_ok=True)
        (broken / "MolNexTR" / "__init__.py").write_text("import package_that_is_missing_xyz\n")
        (broken / "requirements.txt").write_text("")
        params = dict(self.cfg["tools"][1]["params"], repo_path=str(broken))
        tool = MolNexTRTool(Spec("molnextr", "molnextr", params), self.cfg["run"], Path(self.cfg["run"]["data_dir"]))
        with self.assertRaisesRegex(ToolSetupError, "package_that_is_missing_xyz") as caught:
            tool.load()
        self.assertIn("extra_requirements", str(caught.exception))

    def test_worker_finds_the_class_when_the_package_attribute_is_the_submodule(self):
        """MolNexTR/__init__ may bind `molnextr` to the submodule of the same name that holds the class."""
        shadow = self.tmp / "shadow_molnextr"
        (shadow / "MolNexTR").mkdir(parents=True, exist_ok=True)
        (shadow / "MolNexTR" / "__init__.py").write_text("from . import molnextr\n")
        (shadow / "MolNexTR" / "molnextr.py").write_text(
            "from pathlib import Path\n"
            "class molnextr:\n"
            "    def __init__(self, model_path, device=None): pass\n"
            "    def predict_final_results(self, image_path, return_atoms_bonds=False, return_confidence=False):\n"
            "        return {'predicted_smiles': Path(image_path).with_suffix('.pred2').read_text().strip()}\n")
        (shadow / "requirements.txt").write_text("")
        params = dict(self.cfg["tools"][1]["params"], repo_path=str(shadow))
        tool = MolNexTRTool(Spec("molnextr", "molnextr", params), self.cfg["run"], Path(self.cfg["run"]["data_dir"]))
        tool.load()
        try:
            image = next(Path(self.cfg["run"]["data_dir"]).rglob("CLEF/1.png"))
            self.assertEqual(tool.predict_batch([image])[0].smiles, "CCO")
        finally:
            tool.close()

    def test_worker_finds_the_model_class_by_its_method_wherever_it_lives(self):
        """Real layout guess: the class is in another module under another name and the top-level `molnextr` is a module."""
        odd = self.tmp / "odd_layout_molnextr"
        (odd / "MolNexTR").mkdir(parents=True, exist_ok=True)
        (odd / "MolNexTR" / "__init__.py").write_text("from .interface import MolNexTRModel\nfrom . import molnextr\n")
        (odd / "MolNexTR" / "interface.py").write_text(
            "from pathlib import Path\n"
            "class MolNexTRModel:\n"
            "    def __init__(self, model_path, device=None): pass\n"
            "    def predict_final_results(self, image_path, return_atoms_bonds=False, return_confidence=False):\n"
            "        return {'predicted_smiles': Path(image_path).with_suffix('.pred2').read_text().strip()}\n")
        (odd / "MolNexTR" / "molnextr.py").write_text("def get_predictions(*a, **k): ...\n")
        (odd / "requirements.txt").write_text("")
        params = dict(self.cfg["tools"][1]["params"], repo_path=str(odd))
        tool = MolNexTRTool(Spec("molnextr", "molnextr", params), self.cfg["run"], Path(self.cfg["run"]["data_dir"]))
        tool.load()
        try:
            image = next(Path(self.cfg["run"]["data_dir"]).rglob("CLEF/1.png"))
            self.assertEqual(tool.predict_batch([image])[0].smiles, "CCO")
            self.assertEqual(tool.resource_summary()["worker_class"], "MolNexTR.interface.MolNexTRModel")
        finally:
            tool.close()

    def test_worker_without_any_callable_class_lists_what_the_package_exposes(self):
        empty = self.tmp / "no_class_molnextr"
        (empty / "MolNexTR").mkdir(parents=True, exist_ok=True)
        (empty / "MolNexTR" / "__init__.py").write_text("get_predictions = None\nsomething_else = 1\n")
        (empty / "requirements.txt").write_text("")
        params = dict(self.cfg["tools"][1]["params"], repo_path=str(empty))
        tool = MolNexTRTool(Spec("molnextr", "molnextr", params), self.cfg["run"], Path(self.cfg["run"]["data_dir"]))
        with self.assertRaisesRegex(ToolSetupError, "get_predictions"):
            tool.load()

    def test_prefetch_loads_both_models_on_the_cpu_and_leaves_no_worker_running(self):
        from ocsr_bench.tools.molglyph import MolGlyphTool
        glyph = MolGlyphTool(Spec("g", "molglyph", self.cfg["tools"][0]["params"]), self.cfg["run"],
                             Path(self.cfg["run"]["data_dir"]))
        built = []
        original = MolGlyphTool._instantiate
        with mock.patch.object(MolGlyphTool, "_instantiate",
                               lambda self, cls, ckpt, dev: (built.append(dev), original(self, cls, ckpt, dev))[1]):
            glyph.prefetch()
        self.assertEqual(built, ["device(cpu)"])           # the (fake) torch.device("cpu"), not a GPU
        nexus = MolNexTRTool(Spec("m", "molnextr", self.cfg["tools"][1]["params"]), self.cfg["run"],
                             Path(self.cfg["run"]["data_dir"]))
        nexus.prefetch()
        self.assertIsNone(nexus._worker)                   # started once to prove it works, then stopped

    def test_prefetch_reports_a_model_that_cannot_start(self):
        broken = self.tmp / "prefetch_broken_molnextr"
        (broken / "MolNexTR").mkdir(parents=True, exist_ok=True)
        (broken / "MolNexTR" / "__init__.py").write_text("import pystow_missing_xyz\n")
        (broken / "requirements.txt").write_text("")
        params = dict(self.cfg["tools"][1]["params"], repo_path=str(broken))
        tool = MolNexTRTool(Spec("m", "molnextr", params), self.cfg["run"], Path(self.cfg["run"]["data_dir"]))
        with self.assertRaisesRegex(ToolSetupError, "pystow_missing_xyz"):
            tool.prefetch()

    def test_missing_python_executable_explains_how_to_create_it(self):
        params = dict(self.cfg["tools"][1]["params"], python_executable=str(self.tmp / "no_such_python"))
        tool = MolNexTRTool(Spec("molnextr", "molnextr", params), self.cfg["run"], Path(self.cfg["run"]["data_dir"]))
        with self.assertRaisesRegex(ToolSetupError, "setup_envs"):
            tool.load()


class CliTests(unittest.TestCase):
    """The same world driven through `python -m ocsr_bench` commands, with env-var expansion in the YAML."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)
        cls.world = build_world(cls.tmp)
        cfg = make_config(cls.tmp, cls.world)
        cfg["run"].update(output_dir="${OCSR_TEST_OUT}", data_dir="${OCSR_TEST_WORK}/data", name="cli_run")
        cls.config_path = cls.tmp / "config.yaml"
        cls.config_path.write_text(yaml.safe_dump(cfg))
        cls._patches = [
            mock.patch.dict(sys.modules, {"torch": fake_torch(), "huggingface_hub": fake_hub(cls.world["real_zip"])}),
            mock.patch.dict(os.environ, {"PATH": f"{cls.world['bin']}:{os.environ['PATH']}",
                                         "OCSR_TEST_OUT": str(cls.tmp / "out"), "OCSR_TEST_WORK": str(cls.tmp / "work")}),
            mock.patch("ocsr_bench.metrics.Scorer.from_config", side_effect=stub_scorer),
            mock.patch("ocsr_bench.data.RDKitBackend", side_effect=lambda *a, **k: StubBackend()),
        ]
        for patch in cls._patches:
            patch.start()
        __import__("logging").disable(50)

    @classmethod
    def tearDownClass(cls):
        for patch in reversed(cls._patches):
            patch.stop()
        __import__("logging").disable(0)
        cls._tmp.cleanup()

    def invoke(self, *args):
        out = io.StringIO()
        with redirect_stdout(out):
            code = cli.main([*args])
        return code, out.getvalue()

    def test_1_prefetch_then_overview_then_run_then_report(self):
        base = ["--config", str(self.config_path)]
        code, out = self.invoke("prefetch", *base)
        self.assertEqual(code, 0, out)
        self.assertIn("Prefetch complete", out)

        code, out = self.invoke("overview", *base)
        self.assertEqual(code, 0, out)
        self.assertIn("clef", out)
        run_dir = self.tmp / "out" / "cli_run"
        self.assertTrue((run_dir / "dataset_overview.csv").is_file())

        code, out = self.invoke("run", *base)
        self.assertEqual(code, 0, out)
        for name in ("predictions.csv", "pairwise_comparison.csv", "summary.md"):
            self.assertTrue((run_dir / name).is_file(), name)
        self.assertTrue((run_dir / "figures" / "exact_match_by_dataset.png").is_file())
        self.assertTrue((run_dir / "figures" / "failure_types.png").is_file())
        self.assertIn("[ OK ] More than one tool compared", out)

        (run_dir / "figures" / "failure_types.png").unlink()
        code, out = self.invoke("report", *base)                    # rebuilt from the saved files only
        self.assertEqual(code, 0, out)
        self.assertTrue((run_dir / "figures" / "failure_types.png").is_file())

    def test_restricting_tools_and_datasets(self):
        code, out = self.invoke("prefetch", "--config", str(self.config_path), "--only-tools", "molnextr",
                                "--only-datasets", "clef")
        self.assertEqual(code, 0, out)

    def test_clear_errors_return_exit_code_2(self):
        self.assertEqual(self.invoke("run", "--config", str(self.tmp / "missing.yaml"))[0], 2)
        with mock.patch.dict(os.environ):
            os.environ.pop("OCSR_TEST_OUT")
            self.assertEqual(self.invoke("preflight", "--config", str(self.config_path))[0], 2)   # unset ${VAR}
        self.assertEqual(self.invoke("run", "--config", str(self.config_path), "--only-tools", "nope")[0], 2)


class SlurmScriptTests(unittest.TestCase):
    """Static checks of the job scripts (they cannot be run without a cluster)."""

    scripts = Path(__file__).resolve().parents[1] / "slurm"

    def directives(self, name):
        text = (self.scripts / name).read_text()
        return {line.split()[1].split("=")[0]: line for line in text.splitlines() if line.startswith("#SBATCH")}, text

    def test_gpu_scripts_request_one_gpu_on_the_course_gpu_partition_within_its_limits(self):
        for name in ("benchmark.sbatch", "smoke.sbatch"):
            _, text = self.directives(name)
            self.assertIn("--partition=courses-gpu", text, name)
            self.assertIn("--gres=gpu:1", text, name)
            memory = int(re.search(r"--mem=(\d+)G", text).group(1))
            self.assertLessEqual(memory, 12, f"{name}: courses-gpu allows at most 12 GB RAM")
            hours = int(re.search(r"--time=(\d+):", text).group(1))
            self.assertLessEqual(hours, 8, f"{name}: courses-gpu allows at most 8 h")

    def test_cpu_scripts_use_the_course_cpu_partition_and_no_gpu(self):
        for name in ("setup_envs.sbatch", "prefetch.sbatch"):
            _, text = self.directives(name)
            self.assertIn("--partition=courses", text, name)
            self.assertNotIn("courses-gpu", text, name)
            self.assertNotIn("gres", text, name)

    def test_submit_all_passes_partition_and_memory_from_settings(self):
        text = (self.scripts / "submit_all.sh").read_text()
        for needle in ("settings.sh", '--partition="${OCSR_GPU_PARTITION}"', '--mem="${OCSR_GPU_MEM}"',
                       '--partition="${OCSR_CPU_PARTITION}"'):
            self.assertIn(needle, text)

    def test_settings_defaults_match_the_course_partitions_and_can_be_overridden(self):
        import shutil, subprocess
        if shutil.which("bash") is None:
            self.skipTest("bash not available")

        def show(extra_env):
            env = {"PATH": os.environ["PATH"], "USER": "alice", **extra_env}
            out = subprocess.run(["bash", "-c", f". {self.scripts}/settings.sh && echo $OCSR_CPU_PARTITION $OCSR_GPU_PARTITION "
                                  "$OCSR_GPU_MEM $OCSR_WORK_DIR"], capture_output=True, text=True, env=env)
            return out.stdout.split()

        self.assertEqual(show({}), ["courses", "courses-gpu", "12G", "/scratch/alice/ocsr_bench"])
        self.assertEqual(show({"OCSR_COURSE_DIR": "/courses/X.1/students/alice"})[3], "/courses/X.1/students/alice/ocsr_bench")
        self.assertEqual(show({"OCSR_CPU_PARTITION": "short", "OCSR_GPU_PARTITION": "gpu", "OCSR_GPU_MEM": "32G"})[:3],
                         ["short", "gpu", "32G"])

    def test_benchmark_resubmits_itself_before_the_time_limit_with_the_same_partition(self):
        _, text = self.directives("benchmark.sbatch")
        self.assertIn("--signal=B:USR1@", text)
        self.assertIn("trap requeue USR1", text)
        self.assertIn("RESUBMIT_COUNT", text)
        self.assertIn('--partition="${SLURM_JOB_PARTITION}"', text)

    def test_logs_directory_is_created_before_submitting(self):
        self.assertIn("mkdir -p logs", (self.scripts / "submit_all.sh").read_text())

    def test_shell_syntax_is_valid(self):
        import shutil, subprocess
        if shutil.which("bash") is None:
            self.skipTest("bash not available")
        for path in sorted(self.scripts.iterdir()):
            result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, f"{path.name}: {result.stderr}")

    def test_setup_script_repairs_one_missing_module_at_a_time(self):
        import shutil, subprocess
        text = (self.scripts / "setup_envs.sbatch").read_text()
        self.assertIn("pip install pystow", text)
        self.assertIn("installing ${package}", text)
        if shutil.which("bash") is None:
            self.skipTest("bash not available")
        line = next(l for l in text.splitlines() if "No module named" in l)
        expression = line.split('missing="$(echo "${output}" | ')[1].rsplit(')"', 1)[0]
        traceback_text = "Traceback (most recent call last):\n    import pystow\nModuleNotFoundError: No module named 'pystow'"
        result = subprocess.run(["bash", "-c", f'output="$1"; echo "${{output}}" | {expression}', "x", traceback_text],
                                capture_output=True, text=True)
        self.assertEqual(result.stdout.strip(), "pystow", result.stderr)

    def test_submit_wrapper_passes_partitions_from_settings_for_every_job(self):
        text = (self.scripts / "submit.sh").read_text()
        for needle in ("settings.sh", '--partition="${OCSR_GPU_PARTITION}"', '--partition="${OCSR_CPU_PARTITION}"',
                       '--time="${OCSR_GPU_TIME}"', "mkdir -p logs", '"$@"'):
            self.assertIn(needle, text)

    def test_pystow_cache_is_kept_out_of_home(self):
        self.assertIn("PYSTOW_HOME", (self.scripts / "env.sh").read_text())

    def test_no_credentials_or_user_specific_paths_are_hardcoded(self):
        for path in sorted(self.scripts.iterdir()):
            text = path.read_text()
            self.assertNotRegex(text, r"hf_[A-Za-z0-9]{20,}", path.name)
            self.assertNotIn("/home/claude", text, path.name)


if __name__ == "__main__":
    unittest.main()


class PreflightTests(unittest.TestCase):
    def test_remote_checks_can_be_skipped_for_gpu_jobs(self):
        from ocsr_bench import preflight
        checks = []
        with mock.patch.dict(os.environ, {"OCSR_SKIP_REMOTE": "1"}):
            preflight._remote_file_check(checks, "label", "org/repo", "file.bin", "model", None)
        self.assertEqual(checks[0][:2], ("label", "OK"))
        self.assertIn("skipped", checks[0][2])


class RepoHygieneTests(unittest.TestCase):
    root = Path(__file__).resolve().parents[1]

    def test_results_folder_is_not_git_ignored(self):
        lines = [line.strip() for line in (self.root / ".gitignore").read_text().splitlines()]
        self.assertNotIn("results/", lines)
        self.assertNotIn("results", lines)

    def test_readme_does_not_claim_the_code_was_never_run_on_the_cluster(self):
        text = (self.root / "README.md").read_text()
        self.assertNotIn("has **not been run on Explorer", text)
        self.assertIn("Reference run and results", text)
        self.assertIn("Running on another machine", text)

    def test_default_torch_build_matches_the_one_used_for_the_reported_results(self):
        self.assertIn("whl/cu124", (self.root / "slurm" / "env.sh").read_text().split("OCSR_TORCH_INDEX_MAIN:=")[1].split("}")[0])
