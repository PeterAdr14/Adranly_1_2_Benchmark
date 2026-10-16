"""BioMiner's MolGlyph OCSR model, run through BioMiner's own model code and post-processing."""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import logging
import os
import sys
import types
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..utils import resolve_device
from .base import OCSRTool, RawPrediction, ToolSetupError, register_tool
from .common import HF_TOKEN_ENV, clone_repo, git_head

logger = logging.getLogger("ocsr_bench")

MOLGLYPH_DEFAULTS: Dict[str, Any] = {
    "checkpoint_repo": "jiaxianustc/MolGlyph",
    "checkpoint_filename": "molglyph_large.pt",
    "checkpoint_path": None,
    "biominer_repo_url": "https://github.com/jiaxianyan/BioMiner.git",
    "biominer_ref": None,
    "biominer_repo_path": None,
    "postprocess": True,
    "allow_unsafe_torch_load": False,
}
_MODEL_PKG = "_biominer_molscribe"  # private package name for BioMiner's vendored MolScribe/MolGlyph code
_UTILS_MOD = "_biominer_molglyph_utils"


@contextlib.contextmanager
def _torch_load_trusted():
    """Temporarily make torch.load default to weights_only=False (only used if explicitly allowed)."""
    import torch
    original = torch.load

    def patched(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)

    torch.load = patched
    try:
        yield
    finally:
        torch.load = original


def _is_weights_only_error(exc: Exception) -> bool:
    return "weights_only" in str(exc) or "Unpickl" in type(exc).__name__


@register_tool("molglyph")
class MolGlyphTool(OCSRTool):
    """BioMiner's MolGlyph OCSR model, run through BioMiner's own code and post-processing."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.params = {**MOLGLYPH_DEFAULTS, **self.spec.params}
        self._model: Any = None
        self._device = "cpu"
        self._repo: Optional[Path] = None
        self._translator: Any = None
        self._get_refactor: Optional[Callable] = None

    # ---- weights ----------------------------------------------------------------------------
    def _fetch_checkpoint(self) -> Path:
        local = self.params["checkpoint_path"]
        if local:
            path = Path(str(local)).expanduser()
            if not path.is_file():
                raise ToolSetupError(f"checkpoint_path does not exist: {path}")
            return path
        try:
            from huggingface_hub import HfApi, hf_hub_download
        except ImportError as exc:
            raise ToolSetupError("huggingface_hub is required to download the checkpoint.") from exc
        repo, filename = self.params["checkpoint_repo"], self.params["checkpoint_filename"]
        token = os.environ.get(HF_TOKEN_ENV) or None
        try:
            return Path(hf_hub_download(repo_id=repo, filename=filename, token=token))
        except Exception as exc:  # the hub raises several distinct, version-dependent types
            hint = (f"Could not fetch '{filename}' from '{repo}' ({type(exc).__name__}: {exc}). MolGlyph is a "
                    f"gated repository: accept its terms on huggingface.co/{repo} while logged in as the account "
                    f"that owns the token, then provide a read token as the {HF_TOKEN_ENV} secret.")
            try:
                hint += f" Files in the repo: {HfApi().list_repo_files(repo, token=token)}."
            except Exception:  # listing is best effort only
                pass
            raise ToolSetupError(hint) from exc

    # ---- BioMiner code ----------------------------------------------------------------------
    def _ensure_repo(self) -> Path:
        given = self.params["biominer_repo_path"]
        if given:
            path = Path(str(given)).expanduser().resolve()
            if not path.is_dir():
                raise ToolSetupError(f"biominer_repo_path is not a directory: {path}")
            return path
        return clone_repo(self.params["biominer_repo_url"], self.data_dir / "_third_party" / "BioMiner",
                          self.params["biominer_ref"])

    def load_code(self) -> Tuple[Any, Any]:
        """Clone BioMiner if needed and import its MolGlyph class and post-processing helpers."""
        repo = self._repo = self._ensure_repo()
        pkg_dir = repo / "BioMiner" / "MolScribe" / "molscribe"
        utils_file = repo / "BioMiner" / "commons" / "molglyph_utils.py"
        for needed in (pkg_dir / "interface_molglyph.py", utils_file):
            if not needed.is_file():
                raise ToolSetupError(f"Expected file missing in the BioMiner checkout: {needed.relative_to(repo)}. "
                                     "The repository layout may have changed; pin biominer_ref to a known commit.")
        # Register BioMiner's vendored package under a private name WITHOUT running its __init__ (which also
        # imports the unrelated MolScribe model); only interface_molglyph and its relative imports are loaded.
        if _MODEL_PKG not in sys.modules:
            package = types.ModuleType(_MODEL_PKG)
            package.__path__ = [str(pkg_dir)]
            package.__file__ = str(pkg_dir / "__init__.py")
            sys.modules[_MODEL_PKG] = package
        try:
            model_module = importlib.import_module(f"{_MODEL_PKG}.interface_molglyph")
        except Exception as exc:
            for key in [k for k in sys.modules if k == _MODEL_PKG or k.startswith(_MODEL_PKG + ".")]:
                del sys.modules[key]
            raise ToolSetupError(
                f"Importing BioMiner's MolGlyph code failed: {type(exc).__name__}: {exc}. BioMiner pins "
                "albumentations==2.0.5 and timm==1.0.13; install the missing/old package in the main environment (BioMiner also needs OpenNMT-py==2.2.0; see requirements.txt and slurm/setup_envs.sbatch)."
            ) from exc
        spec = importlib.util.spec_from_file_location(_UTILS_MOD, utils_file)
        utils_module = importlib.util.module_from_spec(spec)
        sys.modules[_UTILS_MOD] = utils_module
        try:
            spec.loader.exec_module(utils_module)
        except Exception as exc:
            sys.modules.pop(_UTILS_MOD, None)
            raise ToolSetupError(f"Importing BioMiner's molglyph_utils failed: {type(exc).__name__}: {exc}") from exc
        return model_module.MolGlyph, utils_module

    def load(self) -> None:
        self._device = resolve_device(self.run["device"])
        checkpoint = self._fetch_checkpoint()
        molglyph_cls, utils = self.load_code()
        self._translator, self._get_refactor = utils.Translator, utils.get_refactor
        import torch
        device = torch.device(self._device)
        logger.info("Loading MolGlyph from %s on %s (BioMiner commit %s)", checkpoint.name, self._device,
                    git_head(self._repo) if self._repo else None)
        self._model = self._instantiate(molglyph_cls, checkpoint, device)

    def _instantiate(self, molglyph_cls: Any, checkpoint: Path, device: Any) -> Any:
        """Build the model, handling torch's weights_only refusal as configured."""
        try:
            return molglyph_cls(str(checkpoint), device)
        except Exception as exc:
            if _is_weights_only_error(exc) and self.params["allow_unsafe_torch_load"]:
                logger.warning("torch.load refused the checkpoint; retrying with weights_only=False "
                               "(allow_unsafe_torch_load=True). Only do this for a checkpoint you trust.")
                with _torch_load_trusted():
                    return molglyph_cls(str(checkpoint), device)
            hint = (" torch.load refused the checkpoint; if you trust the source, set "
                    "'allow_unsafe_torch_load': True in the tool params." if _is_weights_only_error(exc) else "")
            raise ToolSetupError(f"MolGlyph could not load the checkpoint: {type(exc).__name__}: {exc}.{hint}") from exc

    # ---- inference --------------------------------------------------------------------------
    def _finalize(self, raw: Any) -> RawPrediction:
        """Turn one raw model output (a SMILES string) into the SMILES that BioMiner itself would use."""
        raw_text = raw if isinstance(raw, str) else str(raw)
        extra: Dict[str, Any] = {"raw_output": raw_text}
        if not self.params["postprocess"]:
            return RawPrediction(smiles=raw_text, extra=extra)
        try:
            final = self._get_refactor(self._translator.refactor(raw_text))
        except Exception as exc:  # recorded, scored as an invalid prediction
            extra["postprocess_error"] = f"{type(exc).__name__}: {exc}"
            return RawPrediction(smiles=None, extra=extra)
        return RawPrediction(smiles=final if isinstance(final, str) or final is None else str(final), extra=extra)

    def predict_batch(self, image_paths: Sequence[Path]) -> List[RawPrediction]:
        if self._model is None:
            raise ToolSetupError("predict_batch() called before load().")
        outputs = self._model.predict_image_files([str(p) for p in image_paths],
                                                  return_atoms_bonds=False, return_confidence=False)
        return [self._finalize(o) for o in outputs]

    def prefetch(self) -> None:
        """Download the checkpoint, clone BioMiner, check that its code imports and that the checkpoint loads (CPU)."""
        checkpoint = self._fetch_checkpoint()
        molglyph_cls, _ = self.load_code()
        import torch
        model = self._instantiate(molglyph_cls, checkpoint, torch.device("cpu"))
        del model

    def resource_summary(self) -> Dict[str, Any]:
        summary: Dict[str, Any] = {"biominer_commit": git_head(self._repo) if self._repo else None,
                                   "postprocess": self.params["postprocess"]}
        try:
            import torch
            if torch.cuda.is_available():
                summary["peak_gpu_memory_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        except ImportError:
            pass
        return summary

    def close(self) -> None:
        self._model = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
