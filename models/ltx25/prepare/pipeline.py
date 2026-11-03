"""Trace one complete reference request: the pinned two-stage `DistilledPipeline`, through MP4.

Every stage, transformer forward, block, x0, noiser, prompt, upsampler and RNG boundary is
hashed; the native runtime checks a profile's structure against these hashes. The first block
of each stage is traced into the block program shared by all 48 blocks. Only model constants
and the inputs of the later component captures are written to disk.
"""
import collections
import inspect

from .recorder import Recorder, immutable

INPUTS = ("forward.0.0.args.", "forward.0.0.kwargs.", "forward.1.0.args.", "forward.1.0.kwargs.",
          "upsampler.0.args.", "stage.1.out.")


def capture(workspace):
    root = workspace.fresh("pipeline")
    from .determinism import begin
    policy, kernels = begin(root)
    import torch
    from .tracing import GraphTrace
    from ltx_core.components.diffusion_steps import EulerAncestralDiffusionStep
    from ltx_core.components.noisers import GaussianNoiser
    from ltx_core.model.transformer.model import LTXModel, X0Model
    from ltx_core.model.transformer.transformer import BasicAVTransformerBlock
    from ltx_core.model.video_vae import get_video_chunks_number
    from ltx_pipelines.distilled import DistilledPipeline
    from ltx_pipelines.utils import blocks
    from ltx_pipelines.utils.media_io import encode_video
    from ltx_pipelines.utils.model_paths import ModelPaths
    from ltx_pipelines.utils.types import ImageConditioningInput

    request = workspace.request
    torch.manual_seed(request["seed"])
    identity = workspace.job["identity"] | {"policy": policy, "gpu": torch.cuda.get_device_name(0),
                                            "gpu_count": torch.cuda.device_count()}
    writer = Recorder(root, workspace.blobs("pipeline"), lambda name: immutable(name) or name.startswith(INPUTS))
    counters = collections.Counter()
    context = {"stage": -1, "forward": -1}
    fixtures = []

    def wrap(cls, method, label):
        original = getattr(cls, method)

        def wrapped(self, *args, **kwargs):
            name = f"{label}.{counters[label]}"
            counters[label] += 1
            writer.tree(name + ".args", args)
            writer.tree(name + ".kwargs", kwargs)
            result = original(self, *args, **kwargs)
            writer.tree(name + ".out", result)
            return result
        setattr(cls, method, wrapped)

    original_stage = blocks.DiffusionStage.__call__

    def stage(self, *args, **kwargs):
        context["stage"] += 1
        context["forward"] = -1
        name = f"stage.{context['stage']}"
        writer.tree(name + ".args", args)
        writer.tree(name + ".kwargs", kwargs)
        output = original_stage(self, *args, **kwargs)
        writer.tree(name + ".out", output)
        return output
    blocks.DiffusionStage.__call__ = stage

    original_forward = LTXModel.forward

    def forward(self, *args, **kwargs):
        context["forward"] += 1
        name = f"forward.{context['stage']}.{context['forward']}"
        for index, block in enumerate(self.transformer_blocks):
            block._prepare_index = index
        writer.tree(name + ".args", args)
        writer.tree(name + ".kwargs", kwargs)
        output = original_forward(self, *args, **kwargs)
        writer.tree(name + ".out", output)
        return output
    LTXModel.forward = forward

    original_block = BasicAVTransformerBlock.forward

    def block(self, *args, **kwargs):
        index = self._prepare_index
        name = f"block.{context['stage']}.{context['forward']}.{index}"
        writer.tree(name + ".args", args)
        writer.tree(name + ".kwargs", kwargs)
        if index == 0 and context["forward"] == 0:
            # The native backbone checks each checkpoint block against these first-block hashes.
            for key, value in self.state_dict().items():
                writer.tensor(name + ".weights." + key, value)
            with GraphTrace(writer, name, self, args, kwargs) as graph:
                output = original_block(self, *args, **kwargs)
            graph.finish(output)
        else:
            output = original_block(self, *args, **kwargs)
        writer.tree(name + ".out", output)
        return output
    BasicAVTransformerBlock.forward = block

    original_randn = torch.randn

    def randn(*args, **kwargs):
        generator = kwargs.get("generator")
        if generator is None and str(kwargs.get("device", "")).startswith("cuda"):
            generator = torch.cuda.default_generators[torch.cuda.current_device()]
        cuda = generator is not None and str(generator.device).startswith("cuda")
        seed, offset = (generator.initial_seed(), generator.get_offset()) if cuda else (None, None)
        value = original_randn(*args, **kwargs)
        if value.device.type != "cuda":
            return value
        name = f"rng.{counters['rng']}"
        counters["rng"] += 1
        writer.tensor(name + ".out", value)
        writer.metadata[name] = {"seed": seed, "offset_before": offset,
                                 "offset_after": generator.get_offset() if seed is not None else None,
                                 "device": str(value.device)}
        fixtures.append({"kind": "rng", "name": name})
        return value
    torch.randn = randn

    original_step = EulerAncestralDiffusionStep.step
    step_signature = inspect.signature(original_step)

    def step(self, *args, **kwargs):
        bound = step_signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        name = f"op.ancestral.{counters['step']}"
        counters["step"] += 1
        for key, value in bound.arguments.items():
            if key != "self":
                writer.tree(name + "." + key, value)
        output = original_step(self, *args, **kwargs)
        writer.tensor(name + ".output", output)
        fixtures.append({"kind": "ancestral", "name": name})
        return output
    EulerAncestralDiffusionStep.step = step

    wrap(GaussianNoiser, "__call__", "noiser")
    wrap(blocks.PromptEncoder, "__call__", "prompt")
    wrap(blocks.VideoUpsampler, "__call__", "upsampler")
    wrap(X0Model, "forward", "x0")
    pipeline = DistilledPipeline(
        model_paths=ModelPaths.from_split(
            transformer_path=workspace.weight("transformer"), text_encoder_path=workspace.weight("text_encoder"),
            video_vae_path=workspace.weight("video_vae"), audio_vae_path=workspace.weight("audio_vae")),
        spatial_upsampler_path=workspace.weight("spatial_upsampler"), loras=[], device=torch.device("cuda"))
    images = []
    if request.get("image_file"):
        images = [ImageConditioningInput(str(workspace.image()), 0, request["image_strength"])]
    with torch.inference_mode():
        output = pipeline(**{key: request[key] for key in ("prompt", "seed", "height", "width", "num_frames",
                                                            "frame_rate")}, images=images)
        # Decoding happens while the video is encoded; the decoder's RNG draws are recorded here.
        encode_video(output.video, fps=request["frame_rate"], audio=output.audio,
                     output_path=str(root / "reference.mp4"),
                     video_chunks_number=get_video_chunks_number(output.num_frames, output.tiling_config))
    torch.cuda.synchronize()
    writer.metadata["fixtures"] = fixtures
    manifest = writer.close(identity)
    kernels.close()
    return {"tensors": len(manifest["tensors"]), "fixtures": len(fixtures),
            "peak_memory_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 1)}
