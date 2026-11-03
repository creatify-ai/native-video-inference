"""Pins, workspace layout and helpers shared by every preparation step. No model imports."""
import hashlib
import json
from pathlib import Path
import shutil
import uuid

REFERENCE_COMMIT = "be6d6d0a69d11a0a399c462ed92606692e5f8697"
MODEL_REVISION = "49b093a39c19169a8a81721dcc541e07e945232f"
FILES = {
    "transformer": "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
    "text_encoder": "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    "video_vae": "vae/ltx-2.5-video-vae-bf16.safetensors",
    "audio_vae": "vae/ltx-2.5-audio-vae-bf16.safetensors",
    "spatial_upsampler": "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
}
# Capture order; the encoder runs for I2V only. Each name is a work and profile directory.
COMPONENTS = ("pipeline", "forward", "text", "upsampler", "encoder", "audio", "video")
# `profile.json` field consumed by the native runtime for each component.
PROFILE_KEYS = {"pipeline": "gold", "forward": "forward", "text": "text", "upsampler": "upsampler",
                "encoder": "encoder", "audio": "audio", "video": "video"}


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.partial")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(16 << 20):
            digest.update(block)
    return digest.hexdigest()


class Workspace:
    """`job.json` plus, per component, `<component>/run` (one capture) and `<component>/blobs`."""

    def __init__(self, job_path):
        self.job_path = Path(job_path)
        self.root = self.job_path.parent
        self.job = json.loads(self.job_path.read_text())
        self.request = self.job["identity"]["request"]

    def run(self, component):
        return self.root / component / "run"

    def blobs(self, component):
        return self.root / component / "blobs"

    def fresh(self, component):
        root = self.run(component)
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)
        return root

    def weight(self, kind):
        return str(Path(self.job["weights"]) / FILES[kind])

    def image(self):
        return self.root / "inputs" / self.request["image_file"]

    def manifest(self, component):
        return json.loads((self.run(component) / "manifest.json").read_text())

    def load(self, component, manifest, name):
        """A saved tensor of an earlier capture, contiguous on the GPU."""
        import torch
        entry = manifest["tensors"][name]
        path = self.blobs(component) / (entry["sha256"] + ".raw")
        data = bytearray(path.read_bytes())
        if (entry["file"] != str(path) or len(data) != entry["bytes"]
                or hashlib.sha256(data).hexdigest() != entry["sha256"]):
            raise RuntimeError(f"{component} tensor {name} is missing or corrupt")
        dtype = getattr(torch, entry["dtype"].removeprefix("torch."))
        return torch.frombuffer(data, dtype=dtype).clone().reshape(entry["shape"]).cuda()
