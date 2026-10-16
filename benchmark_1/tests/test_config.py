import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ocsr_bench.config import ConfigError, apply_override, expand_env, load_config
from ocsr_bench.utils import config_hash

DEFAULT = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


class ExpandEnvTests(unittest.TestCase):
    def test_variables_and_defaults(self):
        with mock.patch.dict(os.environ, {"FOO": "/scratch/me"}):
            self.assertEqual(expand_env("${FOO}/data"), "/scratch/me/data")
            self.assertEqual(expand_env({"a": ["${FOO}", "${MISSING_XYZ:-fallback}", "${MISSING_XYZ:-}"]}),
                             {"a": ["/scratch/me", "fallback", ""]})
            self.assertEqual(expand_env(5), 5)

    def test_unset_variable_without_default_names_the_variable(self):
        with self.assertRaisesRegex(ConfigError, "NO_SUCH_VAR_XYZ"):
            expand_env("${NO_SUCH_VAR_XYZ}/x")

    def test_empty_variable_uses_the_default(self):
        with mock.patch.dict(os.environ, {"EMPTY_VAR": ""}):
            self.assertEqual(expand_env("${EMPTY_VAR:-dflt}"), "dflt")


class LoadConfigTests(unittest.TestCase):
    def test_default_config_loads_without_any_environment(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith("OCSR_")}
        with mock.patch.dict(os.environ, env, clear=True):
            cfg = load_config(DEFAULT)
        self.assertEqual([d["name"] for d in cfg["datasets"]][:4], ["clef", "uob", "uspto", "jpo"])
        self.assertEqual(cfg["run"]["data_dir"], "ocsr_work/data")
        self.assertEqual(cfg["tools"][1]["params"]["python_executable"], "")

    def test_cluster_environment_is_picked_up(self):
        with mock.patch.dict(os.environ, {"OCSR_WORK_DIR": "/scratch/u/w", "OCSR_RESULTS_DIR": "/home/u/r",
                                          "OCSR_MOLNEXTR_PYTHON": "/scratch/u/w/envs/m/bin/python"}):
            cfg = load_config(DEFAULT)
        self.assertEqual(cfg["run"]["data_dir"], "/scratch/u/w/data")
        self.assertEqual(cfg["run"]["output_dir"], "/home/u/r")
        self.assertEqual(cfg["tools"][1]["params"]["python_executable"], "/scratch/u/w/envs/m/bin/python")

    def test_overrides_by_name_and_filters(self):
        cfg = load_config(DEFAULT, ["run.max_samples=7", "datasets.clef.params.max_samples=5"],
                          only_datasets=["clef", "uob"], only_tools=["molnextr"])
        self.assertEqual(cfg["run"]["max_samples"], 7)
        self.assertEqual(cfg["datasets"][0]["params"]["max_samples"], 5)
        enabled = lambda section: [e["name"] for e in cfg[section] if e.get("enabled", True)]
        self.assertEqual(enabled("datasets"), ["clef", "uob"])
        self.assertEqual(enabled("tools"), ["molnextr"])

    def test_bad_input_is_rejected_with_a_clear_error(self):
        with self.assertRaises(ConfigError):
            apply_override({}, "no_equals_sign")
        for overrides in (["run.batch_size=0"], ["run.device=tpu"], ["run.name=a/b"]):
            with self.assertRaises(ConfigError):
                load_config(DEFAULT, overrides)
        with self.assertRaises(ConfigError):
            load_config(DEFAULT, only_tools=["nope"])
        with self.assertRaises(ConfigError):
            load_config("no/such/file.yaml")

    def test_all_disabled_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.yaml"
            path.write_text("run: {name: x, batch_size: 1, max_samples: null, device: cpu}\nmetrics: {}\n"
                            "datasets: [{name: d, kind: local_dir, enabled: false}]\ntools: [{name: t, kind: osra}]\n")
            with self.assertRaisesRegex(ConfigError, "disabled"):
                load_config(path)


class ConfigHashTests(unittest.TestCase):
    def test_location_settings_do_not_change_the_hash_but_computation_settings_do(self):
        base = {"run": {"name": "x", "seed": 1, "data_dir": "/a", "output_dir": "/b", "resume": True},
                "tools": [{"name": "t", "params": {"python_executable": "/p", "postprocess": True}}]}
        moved = {"run": {"name": "x", "seed": 1, "data_dir": "/other", "output_dir": "/else", "resume": False},
                 "tools": [{"name": "t", "params": {"python_executable": "/q", "postprocess": True}}]}
        changed = {"run": dict(base["run"], seed=2), "tools": base["tools"]}
        self.assertEqual(config_hash(base), config_hash(moved))
        self.assertNotEqual(config_hash(base), config_hash(changed))


if __name__ == "__main__":
    unittest.main()
