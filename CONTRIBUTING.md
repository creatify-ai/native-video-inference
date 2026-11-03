# Contribute

Start with the [architecture](docs/architecture.md) and [build guide](docs/build.md).
Use Python 3.12 and the Rust version in `rust-toolchain.toml`.

## Check a change

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-dev.txt
python tools/check_public.py
```

These checks need no GPU, model weights, Modal account, or ML framework.
They cover the core CPU tests, Python runtime, examples, links, and package boundaries.
The first Cargo run can download public registry dependencies.

On Linux, also check all native targets:

```sh
H3_CHECK_ONLY=1 WAN_CHECK_ONLY=1 LTX25_CHECK_ONLY=1 NATIVE_CORE_CHECK_ONLY=1 \
  cargo check --locked --workspace --all-targets --all-features
```

This command checks types. It does not build runnable GPU programs.

## Keep changes reviewable

- Keep core independent of model names and model shapes.
- Keep each model's graph, kernels and preparation inside its own `models/<model>/` directory.
- Preserve the official H3 configuration files in `models/minimax_h3_official/{FL2VA,Ref2VA}/`.
- Keep weights, generated tensors, credentials, and run logs outside Git.
- Keep formatting changes separate from numerical changes.
- Explain behavior changes and record the checks you ran.

Changes to arithmetic, operation order, tensor layouts, loading, or startup need GPU validation.
Use the same fixed requests and compare complete output hashes.
Record the GPU, SDKs, model artifacts, source identity, and timing definition.
CPU tests cannot establish numerical equality or startup performance.

## Report a problem

Include the command, platform, toolchain, component, and relevant error output.
For inference, include artifact identities and request geometry.
Remove credentials, private media, and tensor payloads before sharing a report.

Contributions are accepted under the repository [license](LICENSE).
