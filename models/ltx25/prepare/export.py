"""Turn the captures in a workspace into a native profile: programs, CUBINs and model constants only.

Every recorded activation loses its bytes; manifests keep dtypes, shapes and hashes. Paths inside
the profile are absolute and point into `destination`, so export it where inference will read it.
"""
import hashlib
import json
from pathlib import Path
import shutil

from .common import COMPONENTS, PROFILE_KEYS, write_json
from .recorder import immutable


def export(workspace, destination, weights):
    destination = Path(destination)
    if (destination / "profile.json").exists():
        raise FileExistsError(f"{destination} already holds a profile")
    if destination.exists():
        shutil.rmtree(destination)  # an interrupted export
    components = [c for c in COMPONENTS if (workspace.run(c) / "manifest.json").exists()]
    if not {"pipeline", "forward", "text", "upsampler", "audio", "video"} <= set(components):
        raise RuntimeError(f"incomplete captures: {components}")
    target_root = {component: destination / component / "run" for component in components}
    config = {"inference": True, "profile_manifests": {}, "weights": str(weights),
              "output": str(destination / "outputs")}
    retained = removed = 0
    for component in components:
        source, target = workspace.run(component), target_root[component]
        source_blobs, target_blobs = workspace.blobs(component), target.parent / "blobs"
        target.mkdir(parents=True)
        target_blobs.mkdir()

        def rewrite(value):
            if isinstance(value, str):
                for old, new in ((str(source), str(target)), (str(source_blobs), str(target_blobs))):
                    if value == old or value.startswith(old + "/"):
                        return new + value[len(old):]
                return value
            if isinstance(value, list):
                return [rewrite(item) for item in value]
            if isinstance(value, dict):
                return {key: rewrite(item) for key, item in value.items()}
            return value

        manifest = json.loads((source / "manifest.json").read_text())
        for name, entry in manifest["tensors"].items():
            if entry["file"] is None:
                continue
            if not immutable(name, weights=component in ("audio", "video")):
                entry["file"] = None  # a saved input of a later capture
                removed += 1
                continue
            blob = source_blobs / (entry["sha256"] + ".raw")
            if entry["file"] != str(blob) or blob.stat().st_size != entry["bytes"]:
                raise RuntimeError(f"{component}: constant {name} is not content-addressed")
            destination_blob = target_blobs / blob.name
            if not destination_blob.exists():
                shutil.copyfile(blob, destination_blob)
            entry["file"] = str(destination_blob)
            retained += 1
        for program in source.glob("*.graph.json"):
            write_json(target / program.name, rewrite(json.loads(program.read_text())))
        if (source / "triton").exists():
            shutil.copytree(source / "triton", target / "triton")
        manifest["identity"].update(runtime_profile=True, source_gold=str(target_root["pipeline"]))
        write_json(target / "manifest.json", manifest)
        key = PROFILE_KEYS[component]
        config[key] = str(target)
        config["profile_manifests"][key] = hashlib.sha256((target / "manifest.json").read_bytes()).hexdigest()
    write_json(destination / "profile.json", config)
    request = workspace.request
    example = {"prompt": request["prompt"], "seed": request["seed"]}
    if "encoder" in components:
        shutil.copyfile(workspace.image(), destination / request["image_file"])
        example["image"] = request["image_file"]
    write_json(destination / "request.json", example)
    reference = workspace.run("pipeline") / "reference.mp4"
    if reference.exists():
        shutil.copyfile(reference, destination / "reference.mp4")
    return {"profile": str(destination / "profile.json"), "components": components,
            "constants_retained": retained, "activations_removed": removed}
