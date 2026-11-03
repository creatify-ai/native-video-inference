"""Materialize the pinned LTX reference source with `git archive` and record its tree identity.

    python source.py --repo CHECKOUT --commit SHA --dest DIR --identity OUT.json [--expect TREE_SHA256]

Only `pyproject.toml` and `packages/` are archived, never the working tree. The tree SHA-256
covers every archived file's path and content (not mtimes); every prepared profile carries it,
and the native runtime requires all components of one profile to agree on it.
Standalone and stdlib-only: the image build runs it before the reference environment exists.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile


def tree_identity(root, commit):
    root = Path(root)
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            files[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {"commit": commit, "tree_sha256": hashlib.sha256(encoded).hexdigest(), "files": files}


def archive(repo, commit, dest):
    data = subprocess.run(["git", "-C", str(repo), "archive", "--format=tar", commit, "pyproject.toml", "packages"],
                          check=True, stdout=subprocess.PIPE).stdout
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(data)) as bundle:
        for member in bundle:
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or not (member.isfile() or member.isdir() or member.issym()):
                raise ValueError(f"unsafe reference archive member: {member.name}")
        bundle.extractall(dest, filter="data")
    return tree_identity(dest, commit)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--repo", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--dest", required=True)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--expect", default="")
    args = parser.parse_args()
    identity = archive(args.repo, args.commit, args.dest)
    if args.expect and identity["tree_sha256"] != args.expect:
        raise SystemExit(f"reference tree {identity['tree_sha256']} differs from the pinned {args.expect}")
    Path(args.identity).write_text(json.dumps(identity, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"commit": args.commit, "tree_sha256": identity["tree_sha256"], "files": len(identity["files"])}))


if __name__ == "__main__":
    main()
