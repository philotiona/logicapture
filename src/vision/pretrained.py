"""Pinned, optional local pretrained vision models. No hosted inference.

Weights are fetched only by an explicit ``download_model`` call/CLI. Normal
encoding uses the local cache with networking disabled. Labels, source paths,
and episode concepts never enter the image feature cache.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Sequence
from zipfile import BadZipFile

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
MODEL_CACHE = ROOT / ".cache" / "huggingface"
FEATURE_CACHE = ROOT / "data" / "feature_cache"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    repository: str
    revision: str
    license: str


MODELS = {
    "siglip": ModelSpec("SigLIP 2 Base / 224", "google/siglip2-base-patch16-224",
                        "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2", "Apache-2.0"),
    "qwen": ModelSpec("Qwen3 VL / 2B Instruct", "Qwen/Qwen3-VL-2B-Instruct",
                      "89644892e4d85e24eaac8bacfd4f463576704203", "Apache-2.0"),
    "qwen8b": ModelSpec("Qwen3 VL / 8B Instruct / local NF4", "Qwen/Qwen3-VL-8B-Instruct",
                        "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b", "Apache-2.0"),
}


class LocalModelUnavailable(RuntimeError):
    pass


def model_directory(name: str, cache_dir: Path = MODEL_CACHE) -> Path | None:
    spec = MODELS[name]
    folder = Path(cache_dir) / ("models--" + spec.repository.replace("/", "--")) / "snapshots" / spec.revision
    if not all((folder / f).is_file() for f in ("config.json", "preprocessor_config.json")): return None
    if (folder/"model.safetensors").is_file(): return folder
    try:
        index = json.loads((folder/"model.safetensors.index.json").read_text(encoding="utf-8"))
        shards = set(index["weight_map"].values())
        if shards and all(Path(name).name == name and (folder/name).is_file() for name in shards): return folder
    except (OSError, KeyError, TypeError, ValueError): pass
    return None


def download_model(name: str, cache_dir: Path = MODEL_CACHE) -> Path:
    """Download official weights at a fixed revision. Never execute remote code."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise LocalModelUnavailable("Install requirements-general.txt first.") from error
    spec = MODELS[name]
    folder = snapshot_download(
        spec.repository, revision=spec.revision, cache_dir=str(cache_dir),
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "README.md", "LICENSE*"],
    )
    return Path(folder)


def rgb_image(image) -> Image.Image:
    if isinstance(image, Image.Image):
        image = ImageOps.exif_transpose(image)
    else:
        value = np.asarray(image)
        if value.dtype != np.uint8:
            raise ValueError("Image arrays must contain uint8 pixels.")
        if value.ndim not in (2, 3) or (value.ndim == 3 and value.shape[2] not in (1, 3, 4)):
            raise ValueError("Images require grayscale, RGB, or RGBA channels.")
        if value.ndim == 3 and value.shape[2] == 1: value = value[:, :, 0]
        image = Image.fromarray(value)
    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
        rgba = image.convert("RGBA")
        image = Image.alpha_composite(Image.new("RGBA", image.size, "white"), rgba)
    return image.convert("RGB")


def image_digest(image) -> str:
    image = rgb_image(image)
    return sha256(f"RGB:{image.width}:{image.height}:".encode() + image.tobytes()).hexdigest()


class SiglipEncoder:
    """Frozen semantic global and dense features; no class vocabulary.

    Inference requires optional PyTorch/Transformers. Importing this module
    requires neither. Batch features can subsequently be used on CPU.
    """
    def __init__(self, *, device="auto", cache_dir=MODEL_CACHE,
                 feature_cache=FEATURE_CACHE, batch_size=8):
        if device not in ("auto", "cpu", "cuda"):
            raise ValueError("Device must be auto, cpu, or cuda.")
        self.device_request = device
        self.cache_dir = Path(cache_dir)
        self.feature_cache = Path(feature_cache) if feature_cache is not None else None
        self.batch_size = int(batch_size)
        if not 1 <= self.batch_size <= 64:
            raise ValueError("batch_size must be between 1 and 64.")
        self._model = self._processor = None
        spec = MODELS["siglip"]
        self.encoder_id = f"{spec.repository}@{spec.revision}:vision224:v2"

    def _load(self):
        if self._model is not None:
            return
        folder = model_directory("siglip", self.cache_dir)
        if folder is None:
            raise LocalModelUnavailable("SigLIP weights missing. Run python -m src.vision.pretrained --download siglip.")
        try:
            import torch
            os.environ.setdefault("USE_TF", "0")
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as error:
            raise LocalModelUnavailable("Install requirements-general.txt first.") from error
        device = "cuda" if self.device_request == "auto" and torch.cuda.is_available() else self.device_request
        if device == "auto": device = "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise LocalModelUnavailable("CUDA is not available. Select CPU or install the appropriate PyTorch build.")
        self.device = device
        self._processor = AutoImageProcessor.from_pretrained(folder, local_files_only=True, use_fast=False)
        model = AutoModel.from_pretrained(folder, local_files_only=True, trust_remote_code=False,
                                          dtype=torch.float16 if device == "cuda" else torch.float32)
        self._model = model.vision_model.eval().to(device)
        # The language tower is unnecessary for vocabulary-free episode inference.
        del model

    def unload(self):
        self._model = self._processor = None
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except ImportError:
            pass

    def _cache_path(self, digest):
        key = sha256((self.encoder_id + ":" + digest).encode()).hexdigest()
        return self.feature_cache / f"{key}.npz" if self.feature_cache else None

    def encode(self, images: Sequence, image_ids=None):
        from src.reasoning.features import FeatureBatch
        images = [rgb_image(image) for image in images]
        if not images: raise ValueError("At least one image is required.")
        ids = tuple(image_ids) if image_ids is not None else tuple(f"image_{i}" for i in range(len(images)))
        if len(ids) != len(images): raise ValueError("Image ID count does not match images.")
        vectors, patches, missing = {}, {}, []
        digests = [image_digest(image) for image in images]
        for i, digest in enumerate(digests):
            path = self._cache_path(digest)
            try:
                if path is None or not path.exists(): raise FileNotFoundError
                with np.load(path, allow_pickle=False) as saved:
                    if str(saved["encoder_id"]) != self.encoder_id: raise ValueError("Cache version mismatch")
                    vector, patch = saved["global"].astype(np.float32), saved["patches"].astype(np.float32)
                    if vector.shape != (768,) or patch.shape != (196, 768): raise ValueError("Cache shape mismatch")
                    if not np.isfinite(vector).all() or not np.isfinite(patch).all(): raise ValueError("Nonfinite cache")
                    vectors[i], patches[i] = vector, patch
            except (OSError, ValueError, KeyError, EOFError, BadZipFile):
                missing.append(i)
        if missing:
            self._load()
            import torch
            for start in range(0, len(missing), self.batch_size):
                indices = missing[start:start+self.batch_size]
                inputs = self._processor(images=[images[i] for i in indices], return_tensors="pt")
                pixels = inputs["pixel_values"].to(self.device, dtype=next(self._model.parameters()).dtype)
                with torch.inference_mode():
                    output = self._model(pixel_values=pixels, return_dict=True)
                global_values = output.pooler_output.float().cpu().numpy()
                patch_values = output.last_hidden_state.float().cpu().numpy()
                for j, i in enumerate(indices):
                    vectors[i], patches[i] = global_values[j], patch_values[j]
                    path = self._cache_path(digests[i])
                    if path:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
                            np.savez_compressed(stream, **{
                                "global": vectors[i], "patches": patches[i], "encoder_id": self.encoder_id})
                            temporary = Path(stream.name)
                        temporary.replace(path)
        return FeatureBatch(ids, np.stack([vectors[i] for i in range(len(images))]),
                            tuple(patches[i] for i in range(len(images))), self.encoder_id,
                            metadata={"revision": MODELS["siglip"].revision, "pixel_hashes": digests,
                                      "preprocessing": "official SigLIP2 fixed-resolution 224", "local_only": True})


class QwenLocalBackend:
    """Small local multi-image instruction model for proposing/checking concepts.

    Returned text is data. It is never evaluated as Python or a symbolic program.
    No automatic download, external request, tools, or filesystem access by model.
    """
    def __init__(self, *, device="auto", cache_dir=MODEL_CACHE, max_pixels=786432, model="qwen"):
        if model not in ("qwen", "qwen8b"): raise ValueError("Unsupported local language model.")
        self.model_key = model
        self.device_request, self.cache_dir, self.max_pixels = device, Path(cache_dir), int(max_pixels)
        if model == "qwen8b": self.max_pixels = min(self.max_pixels, 524288)
        self._model = self._processor = None
        self.model_id = MODELS[model].repository + "@" + MODELS[model].revision + (":nf4-bf16-compact-v2" if model == "qwen8b" else ":bf16-v1")

    def _load(self):
        if self._model is not None: return
        folder = model_directory(self.model_key, self.cache_dir)
        if folder is None:
            raise LocalModelUnavailable(f"Qwen weights missing. Run python -m src.vision.pretrained --download {self.model_key}.")
        try:
            import torch
            os.environ.setdefault("USE_TF", "0")
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, BitsAndBytesConfig
        except ImportError as error:
            raise LocalModelUnavailable("Install requirements-general.txt first.") from error
        self.device = "cuda" if self.device_request == "auto" and torch.cuda.is_available() else self.device_request
        if self.device == "auto": self.device = "cpu"
        if self.device not in ("cuda", "cpu"): raise ValueError("Invalid model device.")
        if self.device == "cuda" and not torch.cuda.is_available(): raise LocalModelUnavailable("CUDA unavailable.")
        self._processor = AutoProcessor.from_pretrained(folder, local_files_only=True,
                                                         min_pixels=65536, max_pixels=self.max_pixels)
        options = {}
        if self.model_key == "qwen8b":
            if self.device != "cuda":
                raise LocalModelUnavailable("The 8B NF4 profile requires CUDA. Use the 2B profile for CPU.")
            options = {"quantization_config": BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_enable_fp32_cpu_offload=True),
                "device_map": "auto", "max_memory": {0: "6200MiB", "cpu": "5GiB"}}
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            folder, local_files_only=True, trust_remote_code=False,
            dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            attn_implementation="sdpa", **options).eval()
        self._model = model if options else model.to(self.device)

    def complete(self, prompt: str, images: Sequence, *, max_new_tokens=384) -> str:
        self._load()
        import torch
        images = [rgb_image(image) for image in images]
        messages = [{"role": "user", "content": [
            *[{"type": "image"} for _ in images], {"type": "text", "text": prompt}]}]
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(text=[text], images=images or None, padding=True, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            generated = self._model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                              temperature=None, top_p=None, top_k=None)
        return self._processor.batch_decode(generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]

    def unload(self):
        self._model = self._processor = None
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except ImportError: pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", choices=tuple(MODELS), action="append", default=[])
    args = parser.parse_args()
    for name in args.download:
        print(f"Fetching {MODELS[name].repository} @ {MODELS[name].revision}", flush=True)
        print(download_model(name), flush=True)
    if not args.download:
        print(json.dumps({name: {"model": spec.repository, "revision": spec.revision,
                                 "downloaded": model_directory(name) is not None}
                          for name, spec in MODELS.items()}, indent=2))


if __name__ == "__main__": main()
