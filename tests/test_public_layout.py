"""Repository contracts that need no GPU or model weights."""

import ast
import json
from pathlib import Path
import re
import sys
import tomllib
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
H3 = "models/minimax_h3_official"
MEMBERS = {
    "core_infra",
    f"{H3}/pipeline", f"{H3}/text_encoder", f"{H3}/transformer", f"{H3}/transformer/router",
    f"{H3}/video_vae", f"{H3}/video_vae/router", f"{H3}/audio_vae",
    "models/wan2.1",
    "models/ltx25",
}
VENDORED = {"vendor/native-tokenizers"}
ML_FRAMEWORKS = {"torch", "triton", "diffusers", "transformers", "peft"}
# The core runtime is also copied into flat serving images under these module names.
FLAT_CORE_ALIASES = {
    "startup_ownership": "core_infra/runtime/__init__.py",
    "startup_files": "core_infra/runtime/files.py",
    "startup_components": "core_infra/runtime/components.py",
    "declared_startup": "core_infra/runtime/declared_startup.py",
    "native_startup": "core_infra/runtime/native_startup.py",
    "startup_spec": "core_infra/runtime/startup_spec.py",
    "startup_gate": "core_infra/runtime/startup_gate.py",
}


def test_workspace_is_self_contained():
    workspace = tomllib.loads((ROOT / "Cargo.toml").read_text())["workspace"]
    assert set(workspace["members"]) == MEMBERS
    assert set(workspace.get("exclude", ())) == VENDORED
    names = set()
    for member in MEMBERS | VENDORED:
        path = ROOT / member
        manifest = tomllib.loads((path / "Cargo.toml").read_text())
        names.add(manifest["package"]["name"])
        for section in ("dependencies", "build-dependencies", "dev-dependencies"):
            for dependency in manifest.get(section, {}).values():
                if isinstance(dependency, dict) and "path" in dependency:
                    resolved = (path / dependency["path"]).resolve(strict=True)
                    assert str(resolved.relative_to(ROOT)) in MEMBERS | VENDORED, (member, dependency)
    lock = tomllib.loads((ROOT / "Cargo.lock").read_text())
    assert {p["name"] for p in lock["package"] if not p.get("source")} == names


def offline(path):
    """One-time checkpoint preparation and kernel export may use PyTorch; inference may not."""
    return any("prepare" in part or part.endswith("_export.py") for part in path.relative_to(ROOT).parts)


def test_inference_imports_need_no_ml_framework():
    allowed = sys.stdlib_module_names | {"core_infra", "models", "numpy"}
    for directory in ("core_infra", "models"):
        for path in (ROOT / directory).rglob("*.py"):
            if offline(path) or "tests" in path.relative_to(ROOT).parts:
                continue
            for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    modules = [node.module]
                else:
                    continue
                for module in modules:
                    name = module.split(".")[0]
                    assert name not in ML_FRAMEWORKS, (path, module)
                    if name in FLAT_CORE_ALIASES:
                        assert (ROOT / FLAT_CORE_ALIASES[name]).is_file(), (path, module)
                        continue
                    sibling = path.parent.joinpath(*module.split("."))
                    assert (name in allowed or sibling.with_suffix(".py").is_file()
                            or (sibling / "__init__.py").is_file()), (path, module)


def test_official_h3_component_setup_is_present():
    for variant in ("FL2VA", "Ref2VA"):
        root = ROOT / H3 / variant
        index = json.loads((root / "model_index.json").read_text())
        for component in ("transformer", "text_encoder", "video_vae", "audio_vae"):
            assert component in index and (root / component / "config.json").is_file()


def test_document_links_resolve_inside_the_repository():
    for path in ROOT.rglob("*.md"):
        if {"target", ".git", "vendor"} & set(path.relative_to(ROOT).parts):
            continue
        for target in re.findall(r"\]\(([^)\s]+)\)", path.read_text()):
            if ":" in target or target.startswith("#"):
                continue
            destination = (path.parent / unquote(target.split("#")[0])).resolve()
            assert ROOT == destination or ROOT in destination.parents, (path, target)
            assert destination.exists(), (path, target)
