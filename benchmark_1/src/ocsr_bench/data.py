"""Dataset acquisition and loading: every dataset becomes a list of (image file, ground-truth SMILES)."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import shutil
import tarfile
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from .chem import RDKitBackend, is_markush, molblock_to_smiles
from .config import Spec, get_specs
from .utils import resolve_dir

logger = logging.getLogger("ocsr_bench")

IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif", ".webp"})
LABEL_EXTS = frozenset({".csv", ".tsv", ".json", ".jsonl"})
IMAGE_COLUMN_ALIASES = ("image", "file_path", "image_path", "image_file", "img", "file_name", "filename", "file",
                        "path", "image_id", "name", "id")
SMILES_COLUMN_ALIASES = ("smiles", "gt_smiles", "isomeric_smiles", "canonical_smiles", "ground_truth",
                         "label", "labels", "text")
HF_LABEL_ALIASES = ("smiles", "SMILES", "mol", "molfile", "mol_block", "label", "text")
LABEL_FILE_HINTS = ("ocsr", "label", "smiles", "gt", "ground", "annot")


class DatasetError(RuntimeError):
    """Raised when a dataset cannot be downloaded, parsed or matched to images."""


@dataclass(frozen=True)
class Sample:
    """One benchmark item: an image on disk and its ground-truth SMILES."""
    dataset: str
    sample_id: str
    image_path: Path
    gt_smiles: str


@dataclass
class LoadedDataset:
    name: str
    samples: List[Sample]
    info: Dict[str, Any] = field(default_factory=dict)


def _walk_files(root: Path) -> List[Path]:
    """All files under `root`, sorted. Uses os.walk, which takes file types from the directory listing and so avoids
    one stat call per file (the cost that makes Path.rglob + is_file crawl on a slow shared filesystem)."""
    found: List[Path] = []
    for directory, subdirs, names in os.walk(root):
        subdirs.sort()
        found.extend(Path(directory) / name for name in sorted(names))
    return found


def subsample(items: List[Any], cap: Optional[int], seed: int) -> List[Any]:
    """Seeded, order-preserving random subset of `items` (no-op if cap is None)."""
    if cap is None or cap >= len(items):
        return items
    chosen = sorted(random.Random(seed).sample(range(len(items)), cap))
    return [items[i] for i in chosen]


# ---- archive download / extraction ------------------------------------------------------------
def _download_gdrive(file_id: str, dest: Path) -> None:
    try:
        import gdown
    except ImportError as exc:
        raise DatasetError("gdown is required for 'gdrive_zip' datasets (pip install gdown).") from exc
    result = gdown.download(id=file_id, output=str(dest), quiet=False)
    if result is None or not dest.exists():
        raise DatasetError(
            f"Could not download Google Drive file '{file_id}'. This is usually Drive's download quota or a "
            "permissions issue. Download the archive manually, extract it, and switch this dataset to "
            "kind 'local_dir' with params['path'] pointing at the extracted folder."
        )


def extract_archive(archive: Path, dest: Path) -> None:
    """Extract a zip or tar archive into `dest` with path-traversal protection."""
    dest.mkdir(parents=True, exist_ok=True)
    dest_resolved = dest.resolve()
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as handle:
            for member in handle.infolist():
                if not (dest / member.filename).resolve().is_relative_to(dest_resolved):
                    raise DatasetError(f"Unsafe path in archive: {member.filename}")
            handle.extractall(dest)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as handle:
            for member in handle.getmembers():
                if not (dest / member.name).resolve().is_relative_to(dest_resolved):
                    raise DatasetError(f"Unsafe path in archive: {member.name}")
            if hasattr(tarfile, "data_filter"):
                handle.extractall(dest, filter="data")
            else:  # Python < 3.12 without the extraction-filter backport
                handle.extractall(dest)
    else:
        raise DatasetError(
            f"{archive.name} is neither a zip nor a tar archive. If it is small, it is probably an HTML "
            "error page from Google Drive (quota exceeded)."
        )


# ---- label tables and image matching ----------------------------------------------------------
def _json_to_frame(obj: Any) -> pd.DataFrame:
    if isinstance(obj, list):
        return pd.DataFrame(obj)
    if isinstance(obj, dict):
        for key in ("data", "annotations", "samples", "items", "images", "records"):
            if isinstance(obj.get(key), list):
                return pd.DataFrame(obj[key])
        values = list(obj.values())
        if values and all(isinstance(v, str) for v in values):
            return pd.DataFrame({"image": list(obj.keys()), "smiles": values})
        if values and all(isinstance(v, dict) for v in values):
            return pd.DataFrame([{"image": k, **v} for k, v in obj.items()])
    raise DatasetError("Unsupported JSON label structure.")


def read_label_table(path: Path) -> pd.DataFrame:
    """Read a CSV/TSV/JSON/JSONL label file into a string-typed DataFrame."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            frame = pd.DataFrame([json.loads(line) for line in handle if line.strip()])
    elif suffix == ".json":
        with path.open(encoding="utf-8") as handle:
            frame = _json_to_frame(json.load(handle))
    else:
        raise DatasetError(f"Unsupported label file type: {path.name}")
    return frame.fillna("").astype(str)


def find_columns(frame: pd.DataFrame, image_col: Optional[str] = None,
                 smiles_col: Optional[str] = None) -> Tuple[str, str]:
    """Pick the image-reference and SMILES columns (explicit names win over aliases)."""
    lookup = {str(c).strip().lower().replace(" ", "_").replace("-", "_"): c for c in frame.columns}

    def pick(explicit, aliases, what, exclude=None):
        if explicit:
            if explicit not in frame.columns:
                raise DatasetError(f"Column '{explicit}' not found. Available: {list(frame.columns)}")
            return explicit
        for alias in aliases:
            if alias in lookup and lookup[alias] != exclude:
                return lookup[alias]
        raise DatasetError(f"No {what} column found among columns {list(frame.columns)}.")

    smiles = pick(smiles_col, SMILES_COLUMN_ALIASES, "SMILES")
    image = pick(image_col, IMAGE_COLUMN_ALIASES, "image-reference", exclude=smiles)
    return image, smiles


class ImageIndex:
    """Resolves the image references found in a label table to files on disk."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._by_rel: Dict[str, Path] = {}
        self._by_name: Dict[str, List[Path]] = defaultdict(list)
        self._by_stem: Dict[str, List[Path]] = defaultdict(list)
        for path in _walk_files(root):
            if path.suffix.lower() in IMAGE_EXTS and "__MACOSX" not in path.parts:
                self._by_rel[path.relative_to(root).as_posix()] = path
                self._by_name[path.name].append(path)
                self._by_stem[path.stem].append(path)

    def __len__(self) -> int:
        return len(self._by_rel)

    def paths(self) -> List[Path]:
        return list(self._by_rel.values())

    def resolve(self, reference: str, hints: Sequence[Path] = ()) -> Optional[Path]:
        """Find the file a label row points to: exact relative path, then relative to each hint
        folder (e.g. the label file's folder), then a *unique* file name or stem."""
        ref = str(reference).strip().replace("\\", "/")
        if ref.startswith("./"):
            ref = ref[2:]
        if ref in self._by_rel:
            return self._by_rel[ref]
        for hint in hints:
            try:
                prefix = hint.relative_to(self.root).as_posix()
            except ValueError:
                continue
            key = ref if prefix in ("", ".") else f"{prefix}/{ref}"
            if key in self._by_rel:
                return self._by_rel[key]
        name = ref.rsplit("/", 1)[-1]
        for table, key in ((self._by_name, name), (self._by_stem, name), (self._by_stem, Path(name).stem)):
            matches = table.get(key, [])
            if len(matches) == 1:
                return matches[0]
        return None


def _layout_summary(root: Path, max_entries: int = 12) -> str:
    """One-line overview of an extracted archive, used in error messages."""
    parts = []
    for child in sorted(root.iterdir())[:max_entries]:
        if child.is_dir():
            n_images = sum(1 for p in child.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
            parts.append(f"{child.name}/ ({n_images} images)")
        else:
            parts.append(child.name)
    return ", ".join(parts) or "(empty)"


def _biovista_key(path: Path) -> Optional[Tuple[str, int]]:
    """Parse '<pdb_idx>_<pdb_name>_..._<page>-<index>.png' the way BioMiner's own OCSR script does."""
    items = path.name.split(".")[0].split("_")
    try:
        return f"{items[0]}_{items[1]}", int(items[-1].split("-")[1])
    except (IndexError, ValueError):
        return None


def _composite_columns(frame: pd.DataFrame, smiles_col: Optional[str] = None):
    """(name, index, smiles) columns if the table is keyed like BioVista's OCSR files, else None."""
    lookup = {str(c).strip().lower().replace(" ", "_").replace("-", "_"): c for c in frame.columns}
    if "pdf_name" not in lookup or "index" not in lookup:
        return None
    smiles = smiles_col if smiles_col in frame.columns else next(
        (lookup[a] for a in SMILES_COLUMN_ALIASES if a in lookup), None)
    return (lookup["pdf_name"], lookup["index"], smiles) if smiles else None


def detect_label_layout(frame: pd.DataFrame, params: Dict[str, Any]):
    """('composite', (name, index, smiles)) or ('reference', (image, smiles)); raises DatasetError otherwise."""
    if not params.get("image_column"):
        composite = _composite_columns(frame, params.get("smiles_column"))
        if composite:
            return "composite", composite
    return "reference", find_columns(frame, params.get("image_column"), params.get("smiles_column"))


def _resolve_labels_file(root: Path, params: Dict[str, Any]) -> Path:
    explicit = params.get("labels_file")
    if explicit:
        path = Path(str(explicit))
        path = path if path.is_absolute() else root / path
        if not path.is_file():
            raise DatasetError(f"labels_file not found: {path}")
        return path
    candidates = [p for p in _walk_files(root) if p.suffix.lower() in LABEL_EXTS and "__MACOSX" not in p.parts]
    match = str(params.get("labels_match") or "").lower()
    if match:  # several datasets share one archive: pick this dataset's label file by name
        candidates = [p for p in candidates if match in p.relative_to(root).as_posix().lower()]
    usable, columns_seen = [], {}
    for path in candidates:
        try:
            table = read_label_table(path)
            columns_seen[path.relative_to(root).as_posix()] = [str(c) for c in table.columns][:10]
            detect_label_layout(table, params)
            usable.append(path)
        except (DatasetError, ValueError, OSError):
            continue
    if len(usable) > 1:
        hinted = [p for p in usable if any(h in p.name.lower() for h in LABEL_FILE_HINTS)]
        usable = hinted if len(hinted) == 1 else usable
    if len(usable) == 1:
        return usable[0]
    shown = [p.relative_to(root).as_posix() for p in (usable or candidates)][:15]
    problem = "several possible label files" if usable else "no label file with image + SMILES columns"
    raise DatasetError(
        f"Found {problem} under {root}" + (f" matching '{match}'" if match else "") + f": {shown}. "
        f"Columns seen: {dict(list(columns_seen.items())[:8])}. Archive contents: {_layout_summary(root)}. "
        "Set the dataset's params 'labels_file' / 'labels_match' (and 'image_column' / 'smiles_column' if needed) in CONFIG."
    )


def _samples_from_directory(spec: Spec, root: Path, cap: Optional[int], seed: int):
    params = spec.params
    labels_path = _resolve_labels_file(root, params)
    frame = read_label_table(labels_path)
    layout, columns = detect_label_layout(frame, params)
    split_col, split_val = params.get("split_column"), params.get("split_value")
    if split_col:
        if split_col not in frame.columns:
            raise DatasetError(f"split_column '{split_col}' not found in {list(frame.columns)}")
        frame = frame[frame[split_col] == str(split_val)]

    index = ImageIndex(root)
    if len(index) == 0:
        raise DatasetError(f"No image files found under {root}.")

    # Each label row becomes (reference text, image path or None, ground-truth SMILES).
    if layout == "composite":  # BioVista convention: rows keyed by (pdf_name, index)
        name_col, index_col, smiles_col = columns
        by_key: Dict[Tuple[str, int], Path] = {}
        for path in index.paths():
            key = _biovista_key(path)
            if key is not None:
                by_key.setdefault(key, path)
        rows = []
        for name, idx, gt in zip(frame[name_col], frame[index_col], frame[smiles_col]):
            try:
                key: Optional[Tuple[str, int]] = (str(name).strip(), int(float(str(idx))))
            except ValueError:
                key = None
            rows.append((f"{name}/{idx}", by_key.get(key) if key else None, str(gt).strip()))
    else:  # label rows reference image files by path, file name or stem
        image_col, smiles_col = columns
        rows = [(str(ref), index.resolve(ref, hints=[labels_path.parent]), str(gt).strip())
                for ref, gt in zip(frame[image_col], frame[smiles_col])]

    samples: List[Sample] = []
    seen_ids: Dict[str, int] = {}
    unresolved: List[str] = []
    n_empty = 0
    for row_number, (reference, image_path, gt) in enumerate(rows):
        if not gt:
            n_empty += 1
            continue
        if image_path is None:
            unresolved.append(reference)
            continue
        sample_id = image_path.relative_to(root).with_suffix("").as_posix()
        if sample_id in seen_ids:  # same image listed twice: keep both, disambiguate
            sample_id = f"{sample_id}#{row_number}"
        seen_ids[sample_id] = row_number
        samples.append(Sample(spec.name, sample_id, image_path, gt))
    if unresolved:
        logger.warning("'%s': %d label rows had no matching image (e.g. %s)", spec.name, len(unresolved), unresolved[:3])
    if not samples:
        raise DatasetError(f"'{spec.name}': none of the {len(frame)} label rows matched an image in {root} "
                           f"(layout={layout}, columns={columns}). Row examples: {[r[0] for r in rows[:3]]}; "
                           f"image examples: {[p.relative_to(root).as_posix() for p in index.paths()[:3]]}")
    info = {"labels_file": labels_path.relative_to(root).as_posix(), "layout": layout, "columns": list(columns),
            "n_label_rows": int(len(frame)), "n_images_found": len(index),
            "n_unresolved_rows": len(unresolved), "n_empty_labels": n_empty}
    return subsample(samples, cap, seed), info


# ---- the three dataset kinds ------------------------------------------------------------------
def _load_gdrive_zip(spec: Spec, data_dir: Path, cap, seed):
    file_id = spec.params.get("gdrive_file_id")
    if not file_id:
        raise DatasetError(f"Dataset '{spec.name}': params['gdrive_file_id'] is required.")
    root = data_dir / spec.name
    extracted, marker = root / "extracted", root / ".extracted_ok"
    if not marker.exists():
        root.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(extracted, ignore_errors=True)  # clear any partial earlier extraction
        archive = root / "archive.download"
        logger.info("Downloading '%s' from Google Drive (id=%s)...", spec.name, file_id)
        _download_gdrive(file_id, archive)
        extract_archive(archive, extracted)
        archive.unlink()
        marker.write_text(file_id, encoding="utf-8")
    return _samples_from_directory(spec, extracted, cap, seed)


def _load_local_dir(spec: Spec, data_dir: Path, cap, seed):
    raw_path = spec.params.get("path")
    if not raw_path:
        raise DatasetError(f"Dataset '{spec.name}': params['path'] is required for kind 'local_dir'.")
    path = Path(str(raw_path)).expanduser()
    path = path if path.is_absolute() else (data_dir / path)
    if not path.is_dir():
        raise DatasetError(f"Dataset '{spec.name}': directory not found: {path}")
    return _samples_from_directory(spec, path, cap, seed)


def _load_hf_archive(spec: Spec, data_dir: Path, cap, seed):
    """Download one archive file from a Hugging Face repo (shared by several datasets), extract it once."""
    p = spec.params
    repo, filename = p.get("hf_repo"), p.get("hf_filename")
    if not repo or not filename:
        raise DatasetError(f"Dataset '{spec.name}': params 'hf_repo' and 'hf_filename' are required.")
    folder = data_dir / "_archives" / str(p.get("extract_name") or Path(filename).stem)
    extracted, marker = folder / "extracted", folder / ".extracted_ok"
    if not marker.exists():
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise DatasetError("huggingface_hub is required for 'hf_archive' datasets.") from exc
        logger.info("Downloading %s from Hugging Face repo %s ...", filename, repo)
        try:
            archive = Path(hf_hub_download(repo_id=repo, filename=filename, repo_type=p.get("repo_type", "dataset"),
                                           token=os.environ.get("HF_TOKEN") or None))
        except Exception as exc:
            raise DatasetError(f"Could not download '{filename}' from '{repo}': {type(exc).__name__}: {exc}") from exc
        shutil.rmtree(extracted, ignore_errors=True)  # clear any partial earlier extraction
        extract_archive(archive, extracted)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(f"{repo}/{filename}", encoding="utf-8")
    return _samples_from_directory(spec, extracted, cap, seed)


def _label_to_smiles(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    if "M  END" in text:  # MDL molfile -> SMILES
        return molblock_to_smiles(text) or ""
    return text.strip()


def _load_huggingface(spec: Spec, data_dir: Path, cap, seed):
    try:
        import datasets as hf
    except ImportError as exc:
        raise DatasetError("The 'datasets' package is required (pip install datasets).") from exc
    params = spec.params
    repo = params.get("hf_repo")
    if not repo:
        raise DatasetError(f"Dataset '{spec.name}': params['hf_repo'] is required.")
    config = params.get("hf_config")
    if config is None:
        names = list(hf.get_dataset_config_names(repo))
        if names:
            config = names[0]
            if len(names) > 1:
                logger.warning("'%s' has several configs %s; using '%s'. Set params['hf_config'] to choose.",
                               repo, names, config)
    split = params.get("split")
    if split is None:
        splits = list(hf.get_dataset_split_names(repo, config))
        split = "test" if "test" in splits else splits[0]
        logger.info("'%s': using split '%s' (available: %s)", repo, split, splits)

    dataset = hf.load_dataset(repo, config, split=split)
    features = dataset.features
    image_col = params.get("image_column") or next(
        (n for n, f in features.items() if isinstance(f, hf.Image)), None)
    label_col = params.get("label_column") or next((a for a in HF_LABEL_ALIASES if a in features), None)
    if image_col is None or label_col is None:
        raise DatasetError(f"Could not infer columns for '{repo}' (features: {list(features)}). "
                           "Set params['image_column'] and params['label_column'].")

    indices = subsample(list(range(len(dataset))), cap, seed)
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{config or 'default'}_{split}")
    image_dir = data_dir / spec.name / "images" / safe
    image_dir.mkdir(parents=True, exist_ok=True)
    samples: List[Sample] = []
    n_bad_label = 0
    for i in indices:
        row = dataset[int(i)]
        gt = _label_to_smiles(row[label_col])
        if not gt:
            n_bad_label += 1
            continue
        target = image_dir / f"{int(i):06d}.png"
        if not target.exists():
            row[image_col].convert("RGB").save(target)
        samples.append(Sample(spec.name, f"{safe}/{int(i):06d}", target, gt))
    info = {"hf_repo": repo, "hf_config": config, "split": split, "image_column": image_col,
            "label_column": label_col, "n_rows_in_split": len(dataset), "n_skipped_unusable_label": n_bad_label}
    return samples, info


_CACHE_VERSION = 1


def _cache_file(spec: Spec, data_dir: Path, cap: Optional[int], seed: int) -> Path:
    key = hashlib.sha256(json.dumps({"v": _CACHE_VERSION, "name": spec.name, "kind": spec.kind, "params": spec.params,
                                     "cap": cap, "seed": seed}, sort_keys=True, default=str).encode()).hexdigest()[:16]
    return data_dir / "_sample_cache" / f"{spec.name}_{key}.json"


def _read_cached_samples(path: Path, data_dir: Path):
    """The cached (samples, info) or None. Ignored if disabled, unreadable, or its images are gone."""
    if os.environ.get("OCSR_NO_SAMPLE_CACHE") == "1" or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        samples = [Sample(payload["dataset"], row["id"], data_dir / row["image"], row["gt"]) for row in payload["samples"]]
        info = payload["info"]
    except (OSError, ValueError, KeyError):
        return None
    if not samples or not samples[0].image_path.is_file() or not samples[-1].image_path.is_file():
        return None
    return samples, info


def _write_cached_samples(path: Path, data_dir: Path, name: str, samples: List[Sample], info: Dict[str, Any]) -> None:
    def where(image: Path) -> str:  # relative to data_dir when possible, else absolute
        try:
            return image.relative_to(data_dir).as_posix()
        except ValueError:
            return image.as_posix()

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [{"id": s.sample_id, "image": where(s.image_path), "gt": s.gt_smiles} for s in samples]
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"dataset": name, "info": info, "samples": rows}), encoding="utf-8")
        temporary.replace(path)
    except (OSError, TypeError) as exc:  # a cache that cannot be written only costs time
        logger.warning("Could not write the sample cache %s: %s", path, exc)


def load_dataset_samples(spec: Spec, data_dir: Path, max_samples: Optional[int] = None,
                         seed: int = 42) -> LoadedDataset:
    """Materialize `spec` under `data_dir` and return its samples.

    The global `max_samples` and the dataset's own params['max_samples'] both cap the sample count
    (the smaller wins). Subsampling is seeded, hence reproducible.

    The resulting sample list is cached under `<data_dir>/_sample_cache/`, so later runs (notably the GPU job) skip
    the folder scans. Delete that folder, or set OCSR_NO_SAMPLE_CACHE=1, to force a rescan.
    """
    loaders = {"gdrive_zip": _load_gdrive_zip, "hf_archive": _load_hf_archive,
               "local_dir": _load_local_dir, "huggingface": _load_huggingface}
    if spec.kind not in loaders:
        raise DatasetError(f"Unknown dataset kind '{spec.kind}'. Known kinds: {sorted(loaders)}")
    caps = [c for c in (spec.params.get("max_samples"), max_samples) if c is not None]
    cap = min(caps) if caps else None
    data_dir = Path(data_dir)
    cache = _cache_file(spec, data_dir, cap, seed)
    cached = _read_cached_samples(cache, data_dir)
    if cached is not None:
        samples, info = cached
        logger.info("Dataset '%s': %d samples (from the sample cache)", spec.name, len(samples))
        return LoadedDataset(spec.name, samples, info)
    samples, info = loaders[spec.kind](spec, data_dir, cap, seed)
    if not samples:
        raise DatasetError(f"Dataset '{spec.name}' produced no samples.")
    info.update(kind=spec.kind, n_samples=len(samples), max_samples=cap)
    _write_cached_samples(cache, data_dir, spec.name, samples, info)
    logger.info("Dataset '%s': %d samples", spec.name, len(samples))
    return LoadedDataset(spec.name, samples, info)


def dataset_overview(cfg: Dict[str, Any]) -> pd.DataFrame:
    """Download / load every enabled dataset and summarise it (size, ground-truth validity, categories)."""
    backend = RDKitBackend(cfg["metrics"]["fingerprint_radius"], cfg["metrics"]["fingerprint_bits"])
    run, data_dir, rows = cfg["run"], resolve_dir(cfg["run"]["data_dir"]), []
    for spec in get_specs(cfg, "datasets"):
        loaded = load_dataset_samples(spec, data_dir, run["max_samples"], run["seed"])
        infos = [backend.analyze(s.gt_smiles) for s in loaded.samples]
        valid = [(s, i) for s, i in zip(loaded.samples, infos) if i is not None]
        categories = {"markush": 0, "chiral": 0, "plain": 0}
        for s, i in valid:
            categories["markush" if is_markush(s.gt_smiles) else ("chiral" if i.has_stereo else "plain")] += 1
        rows.append({"dataset": spec.name, "source": spec.kind, "n_samples": len(loaded.samples),
                     "gt_parseable": f"{len(valid) / len(loaded.samples):.1%}", **categories,
                     "labels_file": loaded.info.get("labels_file", loaded.info.get("hf_repo", "")),
                     "example": f"{loaded.samples[0].sample_id}: {loaded.samples[0].gt_smiles[:40]}"})
    return pd.DataFrame(rows)
