"""Command line: `run` prepares a whole profile; `capture` is one step, in its own process."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .common import COMPONENTS, FILES, MODEL_REVISION, REFERENCE_COMMIT, Workspace, file_sha256, write_json


def request_identity(path, image):
    """The capture request; its geometry, frame rate and branch become the profile's."""
    request = json.loads(Path(path).read_text())
    mode = request.pop("mode")
    width, height, frames, fps = (int(request[key]) for key in ("width", "height", "num_frames", "frame_rate"))
    if width <= 0 or height <= 0 or width % 64 or height % 64:
        raise SystemExit("width and height must be positive multiples of 64")
    if frames < 9 or (frames - 1) % 8 or fps <= 0:
        raise SystemExit("num_frames must be 8k+1 (at least 9) and frame_rate positive")
    result = {"prompt": str(request["prompt"]), "seed": int(request["seed"]), "width": width, "height": height,
              "num_frames": frames, "frame_rate": fps}
    if mode == "i2v":
        if image is None or Path(image).read_bytes()[:3] != b"\xff\xd8\xff":
            raise SystemExit("I2V preparation needs a JPEG conditioning image")
        result.update(image_strength=float(request.get("image_strength", 0.8)), image_file="image.jpg",
                      image_sha256=file_sha256(image))
    elif mode != "t2v":
        raise SystemExit("mode must be t2v or i2v")
    return result


def weight_identity(weights):
    """Full-file SHA-256 of each checkpoint file; the runtime admits only these bytes."""
    path = Path(weights) / "weights.json"
    if path.exists():
        return json.loads(path.read_text())
    identity = {kind: {"file": relative, "bytes": (Path(weights) / relative).stat().st_size,
                       "sha256": file_sha256(Path(weights) / relative)} for kind, relative in FILES.items()}
    write_json(path, identity)
    return identity


def run(args):
    started = time.monotonic()
    work, image = Path(args.work), args.image and Path(args.image)
    source = json.loads(Path(args.source_identity).read_text())
    if source["commit"] != REFERENCE_COMMIT:
        raise SystemExit(f"reference source {source['commit']} is not the pinned {REFERENCE_COMMIT}")
    request = request_identity(args.request, image)
    if work.exists():
        shutil.rmtree(work)
    (work / "inputs").mkdir(parents=True)
    if image:
        shutil.copyfile(image, work / "inputs" / request["image_file"])
    identity = {"commit": source["commit"], "tree_sha256": source["tree_sha256"], "model_revision": MODEL_REVISION,
                "weights": weight_identity(args.weights), "request": request}
    job = work / "job.json"
    write_json(job, {"weights": str(Path(args.weights).resolve()), "identity": identity})
    package = Path(__file__).resolve().parent
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [str(package.parent)] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]))
    reports = {}
    for component in COMPONENTS:
        if component == "encoder" and "image_file" not in request:
            continue
        print(json.dumps({"event": "capture-started", "component": component}), flush=True)
        subprocess.run([sys.executable, "-m", package.name, "capture", component, "--job", str(job)],
                       env=environment, check=True)
        reports[component] = json.loads((Workspace(job).run(component) / "report.json").read_text())
    from .export import export
    exported = export(Workspace(job), args.profile, Path(args.weights).resolve())
    if not args.keep_work:
        shutil.rmtree(work)
    result = {"passed": True, **exported, "captures": reports, "seconds": round(time.monotonic() - started, 1)}
    write_json(Path(args.profile) / "prepare.json", result)
    print(json.dumps(result), flush=True)


def capture(args):
    workspace = Workspace(args.job)
    started = time.monotonic()
    if args.component == "pipeline":
        from .pipeline import capture as step
    else:
        from .components import CAPTURES
        step = CAPTURES[args.component]
    report = step(workspace) | {"seconds": round(time.monotonic() - started, 1)}
    write_json(workspace.run(args.component) / "report.json", report)


def main():
    parser = argparse.ArgumentParser(prog="python -m prepare", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    whole = commands.add_parser("run", help="capture every component and export a profile")
    whole.add_argument("--work", required=True, help="scratch directory for the captures")
    whole.add_argument("--profile", required=True, help="profile directory to create")
    whole.add_argument("--weights", required=True, help="directory with the pinned checkpoint files")
    whole.add_argument("--source-identity", required=True, help="JSON written by source.py")
    whole.add_argument("--request", required=True,
                       help="JSON: mode (t2v|i2v), width, height, num_frames, frame_rate, prompt, seed")
    whole.add_argument("--image", help="I2V conditioning JPEG")
    whole.add_argument("--keep-work", action="store_true", help="keep the captures after export")
    one = commands.add_parser("capture", help="one capture step")
    one.add_argument("component", choices=COMPONENTS)
    one.add_argument("--job", required=True)
    args = parser.parse_args()
    run(args) if args.command == "run" else capture(args)


if __name__ == "__main__":
    main()
