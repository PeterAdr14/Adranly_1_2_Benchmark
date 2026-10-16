import json
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from PIL import Image

from ocsr_bench.config import Spec
from ocsr_bench.data import DatasetError, extract_archive, load_dataset_samples, subsample

from .helpers import make_png


def spec(kind="local_dir", name="toy", **params):
    return Spec(name=name, kind=kind, params=params)


class DirectoryDatasetTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "toy_dir"
        self.addCleanup(self._tmp.cleanup)

    def load(self, **params):
        return load_dataset_samples(spec(path=str(self.root), **params), Path(self._tmp.name), seed=1)

    def test_csv_with_filenames(self):
        for stem in "abc":
            make_png(self.root / "images" / f"{stem}.png")
        (self.root / "labels.csv").write_text("image,smiles\na.png,CCO\nb.png,CCN\nc.png,CCC\n")
        loaded = self.load()
        self.assertEqual([s.gt_smiles for s in loaded.samples], ["CCO", "CCN", "CCC"])
        self.assertEqual(loaded.samples[0].sample_id, "images/a")
        self.assertEqual((loaded.info["layout"], loaded.info["columns"][0]), ("reference", "image"))
        self.assertEqual(loaded.info["n_unresolved_rows"], 0)

    def test_tolerant_column_names_and_stem_references(self):
        make_png(self.root / "x" / "m1.png")
        make_png(self.root / "y" / "m2.jpg")
        (self.root / "meta.tsv").write_text("File Name\tSMILES\nm1\tCCO\nm2\tCCN\n")
        loaded = self.load()
        self.assertEqual(len(loaded.samples), 2)
        self.assertEqual(loaded.info["columns"][0], "File Name")

    def test_json_mapping_layout(self):
        make_png(self.root / "p.png")
        make_png(self.root / "q.png")
        (self.root / "gt.json").write_text(json.dumps({"p.png": "CCO", "q.png": "CCN"}))
        self.assertEqual(len(self.load().samples), 2)

    def test_unresolved_rows_and_empty_labels_are_reported(self):
        make_png(self.root / "a.png")
        (self.root / "labels.csv").write_text("image,smiles\na.png,CCO\nmissing.png,CCN\na.png,\n")
        info = self.load().info
        self.assertEqual((info["n_unresolved_rows"], info["n_empty_labels"]), (1, 1))

    def test_ambiguous_label_files_require_explicit_choice(self):
        make_png(self.root / "a.png")
        (self.root / "one.csv").write_text("image,smiles\na.png,CCO\n")
        (self.root / "two.csv").write_text("image,smiles\na.png,CCN\n")
        with self.assertRaisesRegex(DatasetError, "labels_file"):
            self.load()
        self.assertEqual(self.load(labels_file="two.csv").samples[0].gt_smiles, "CCN")

    def test_no_label_file_is_a_clear_error(self):
        make_png(self.root / "a.png")
        with self.assertRaisesRegex(DatasetError, "no label file"):
            self.load()

    def test_split_filter(self):
        for stem in "ab":
            make_png(self.root / f"{stem}.png")
        (self.root / "labels.csv").write_text("image,smiles,split\na.png,CCO,val\nb.png,CCN,test\n")
        loaded = self.load(split_column="split", split_value="test")
        self.assertEqual([s.gt_smiles for s in loaded.samples], ["CCN"])

    def test_subsample_is_seeded_and_order_preserving(self):
        items = list(range(50))
        first, second = subsample(items, 10, seed=3), subsample(items, 10, seed=3)
        self.assertEqual(first, second)
        self.assertEqual(first, sorted(first))
        self.assertNotEqual(first, subsample(items, 10, seed=4))
        self.assertIs(subsample(items, None, seed=3), items)

    def test_cap_uses_smaller_of_global_and_dataset_limit(self):
        for i in range(10):
            make_png(self.root / f"{i}.png")
        rows = "\n".join(f"{i}.png,CC" for i in range(10))
        (self.root / "labels.csv").write_text("image,smiles\n" + rows + "\n")
        loaded = load_dataset_samples(
            spec(path=str(self.root), max_samples=6), Path(self._tmp.name), max_samples=4, seed=0
        )
        self.assertEqual(len(loaded.samples), 4)


class ArchiveTests(unittest.TestCase):
    def test_zip_slip_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "evil.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("../evil.txt", "x")
            with self.assertRaisesRegex(DatasetError, "Unsafe"):
                extract_archive(archive, Path(tmp) / "out")
            self.assertFalse((Path(tmp) / "evil.txt").exists())

    def test_normal_zip_extracts_and_non_archive_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "ok.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("a/b.txt", "hello")
            extract_archive(archive, Path(tmp) / "out")
            self.assertEqual((Path(tmp) / "out" / "a" / "b.txt").read_text(), "hello")
            junk = Path(tmp) / "junk"
            junk.write_text("<html>quota exceeded</html>")
            with self.assertRaisesRegex(DatasetError, "neither a zip"):
                extract_archive(junk, Path(tmp) / "out2")


class _FakeImageFeature:
    pass


class _FakeHFDataset:
    def __init__(self, n):
        self.features = {"image": _FakeImageFeature(), "smiles": "string"}
        self._rows = []
        for i in range(n):
            buffer = Image.new("L", (6, 6), 255)
            self._rows.append({"image": buffer, "smiles": f"C{'C' * i}"})

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, i):
        return self._rows[i]


def _fake_datasets_module():
    module = types.ModuleType("datasets")
    module.Image = _FakeImageFeature
    module.get_dataset_config_names = lambda repo: ["cfgA", "cfgB"]
    module.get_dataset_split_names = lambda repo, config: ["train", "test"]
    module.load_dataset = mock.Mock(side_effect=lambda repo, config, split: _FakeHFDataset(10))
    return module


class HuggingFaceDatasetTests(unittest.TestCase):
    def test_first_config_test_split_and_png_materialization(self):
        fake = _fake_datasets_module()
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(sys.modules, {"datasets": fake}):
            loaded = load_dataset_samples(
                spec("huggingface", "hf_toy", hf_repo="org/toy"), Path(tmp), max_samples=4, seed=5
            )
            fake.load_dataset.assert_called_once_with("org/toy", "cfgA", split="test")
            self.assertEqual(loaded.info["hf_config"], "cfgA")
            self.assertEqual(loaded.info["split"], "test")
            self.assertEqual(len(loaded.samples), 4)
            for sample in loaded.samples:
                self.assertTrue(sample.image_path.is_file())
                self.assertEqual(sample.image_path.suffix, ".png")
                self.assertTrue(sample.gt_smiles.startswith("C"))
            again = load_dataset_samples(
                spec("huggingface", "hf_toy", hf_repo="org/toy"), Path(tmp), max_samples=4, seed=5
            )
            self.assertEqual([s.sample_id for s in loaded.samples], [s.sample_id for s in again.samples])

    def test_missing_columns_give_actionable_error(self):
        fake = _fake_datasets_module()
        bare = _FakeHFDataset(3)
        bare.features = {"foo": "string"}
        fake.load_dataset = mock.Mock(return_value=bare)
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(sys.modules, {"datasets": fake}):
            with self.assertRaisesRegex(DatasetError, "image_column"):
                load_dataset_samples(spec("huggingface", "hf_toy", hf_repo="org/toy"), Path(tmp))


class UnknownKindTests(unittest.TestCase):
    def test_unknown_kind(self):
        with self.assertRaises(DatasetError):
            load_dataset_samples(spec("ftp"), Path("."))


if __name__ == "__main__":
    unittest.main()


class BioVistaAndArchiveLayoutTests(unittest.TestCase):
    """BioVista's own convention and MolScribe-style CSVs inside one shared archive."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name) / "d"

    def load(self, **params):
        return load_dataset_samples(spec(path=str(self.root), **params), Path(self._tmp.name), seed=1)

    def test_pdf_name_index_convention(self):
        make_png(self.root / "ocsr" / "data" / "40_6s8a_image_merge_bbox_3-0.png")
        make_png(self.root / "ocsr" / "data" / "41_6s8a_image_merge_bbox_3-1.png")
        (self.root / "ocsr" / "gd.csv").write_text("pdf_name,index,smiles\n40_6s8a,0,CCO\n41_6s8a,1,CCN\n42_x,2,CC\n")
        loaded = self.load()
        self.assertEqual(loaded.info["layout"], "composite")
        self.assertEqual((len(loaded.samples), loaded.info["n_unresolved_rows"]), (2, 1))

    def test_labels_match_and_label_folder_relative_paths_with_colliding_names(self):
        for folder, smiles in (("CLEF", "CCO"), ("UOB", "CCN")):
            make_png(self.root / "real" / folder / "1.png")
            (self.root / "real" / f"{folder}.csv").write_text(f"image_id,file_path,SMILES\nx,{folder}/1.png,{smiles}\n")
        clef = self.load(labels_match="clef")
        uob = self.load(labels_match="uob")
        self.assertEqual((clef.samples[0].gt_smiles, uob.samples[0].gt_smiles), ("CCO", "CCN"))
        self.assertIn("CLEF", clef.samples[0].image_path.as_posix())
        self.assertIn("UOB", uob.samples[0].image_path.as_posix())          # same file name, different folder
        with self.assertRaisesRegex(DatasetError, "matching 'nothing'"):
            self.load(labels_match="nothing")


class SampleCacheAndScanTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.data_dir = Path(self._tmp.name)
        self.root = self.data_dir / "toy_dir"
        for stem in "abc":
            make_png(self.root / "images" / f"{stem}.png")
        (self.root / "labels.csv").write_text("image,smiles\na.png,CCO\nb.png,CCN\nc.png,CCC\n")

    def load(self, **kwargs):
        return load_dataset_samples(spec(path=str(self.root)), self.data_dir, seed=1, **kwargs)

    def test_second_load_comes_from_the_cache_without_scanning_the_folder(self):
        first = self.load()
        self.assertTrue(list((self.data_dir / "_sample_cache").glob("toy_*.json")))
        with mock.patch("ocsr_bench.data.ImageIndex", side_effect=AssertionError("the folder was scanned again")):
            second = self.load()
        self.assertEqual([(s.sample_id, s.image_path, s.gt_smiles) for s in first.samples],
                         [(s.sample_id, s.image_path, s.gt_smiles) for s in second.samples])
        self.assertEqual(first.info["layout"], second.info["layout"])

    def test_cache_is_keyed_by_cap_and_seed(self):
        self.load()
        with mock.patch("ocsr_bench.data.ImageIndex", side_effect=AssertionError("rescanned")):
            with self.assertRaises(AssertionError):
                self.load(max_samples=2)       # a different cap must not reuse the cached list

    def test_cache_is_ignored_when_its_images_disappear(self):
        self.load()
        (self.root / "images" / "a.png").unlink()
        (self.root / "labels.csv").write_text("image,smiles\nb.png,CCN\nc.png,CCC\n")
        self.assertEqual(len(self.load().samples), 2)

    def test_cache_can_be_disabled(self):
        self.load()
        with mock.patch.dict("os.environ", {"OCSR_NO_SAMPLE_CACHE": "1"}), \
                mock.patch("ocsr_bench.data.ImageIndex", side_effect=AssertionError("rescanned")):
            with self.assertRaises(AssertionError):
                self.load()

    def test_walk_skips_macosx_junk_and_is_sorted(self):
        make_png(self.root / "__MACOSX" / "images" / "ghost.png")
        from ocsr_bench.data import ImageIndex
        names = [p.name for p in ImageIndex(self.root).paths()]
        self.assertEqual(names, ["a.png", "b.png", "c.png"])
