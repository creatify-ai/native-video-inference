"""Component programs traced from the pipeline capture's saved inputs, one fresh process each:
transformer input/output heads, Gemma and its connectors, the spatial upsampler, the I2V image
encoder, and the audio and video decoders."""
from contextlib import contextmanager
import dataclasses
import importlib

from .recorder import Recorder, immutable


def _start(workspace, component, capture):
    root = workspace.fresh(component)
    from .determinism import begin
    policy, kernels = begin(root)
    source = workspace.manifest("pipeline")
    identity = source["identity"] | {"policy": policy, "source_gold": str(workspace.run("pipeline")),
                                     "capture": capture}
    return root, kernels, source, identity


def _finish(writer, kernels, identity, graphs, **report):
    writer.metadata.update(graphs=graphs, fixtures=[])
    manifest = writer.close(identity)
    kernels.close()
    return {"graphs": graphs, "tensors": len(manifest["tensors"]), **report}


def _traced(writer, trace, name, module, method, args, kwargs, graphs):
    """Trace `method(*args, **kwargs)` as program `name`; `module` owns its weights and buffers."""
    writer.tree(name + ".args", args)
    writer.tree(name + ".kwargs", kwargs)
    for key, value in module.named_parameters():
        writer.tensor(name + ".weights." + key, value)
    for key, value in module.named_buffers():
        writer.tensor(name + ".buffers." + key, value)
    with trace(writer, name, module, args, kwargs) as graph:
        output = method(*args, **kwargs)
    graph.finish(output)
    writer.tree(name + ".out", output)
    graphs.append(name + ".graph.json")
    return output


def forward(workspace):
    """Transformer input preparation (`prep.N`) and output heads (`head.N`) of each stage's first
    forward, executed by the complete transformer from the pipeline's saved forward inputs."""
    root, kernels, source, identity = _start(workspace, "forward", "forward-components-v2")
    import torch
    from .tracing import ComponentTrace
    from ltx_core.model.transformer.modality import Modality
    from ltx_core.model.transformer.rope import generate_freq_grid_np, generate_freq_grid_pytorch
    from ltx_pipelines.utils.blocks import DiffusionStage
    writer = Recorder(root, workspace.blobs("forward"))

    def read(name):
        if name in source["metadata"]:
            return source["metadata"][name]
        value = workspace.load("pipeline", source, name)
        if name.endswith(".args.0.latent") or name.endswith(".kwargs.video.latent"):
            # The patchifier/noiser hands the transformer [B,T,C] video tokens with strides [T*C,1,T].
            return value.transpose(-1, -2).contiguous().transpose(-1, -2)
        return value

    class NonBlockWeights:
        def __init__(self, model):
            self.model = model

        def named_parameters(self):
            return ((key, value) for key, value in self.model.named_parameters()
                    if not key.startswith("transformer_blocks."))

    torch.manual_seed(workspace.request["seed"])
    stage = DiffusionStage.from_checkpoint(workspace.weight("transformer"), dtype=torch.bfloat16,
                                           device=torch.device("cuda"))
    model = stage._build_transformer().velocity_model
    carrier = NonBlockWeights(model)
    graphs = []
    counters = {"prep": 0, "head": 0}

    def component(label, method):
        def call(*args, **kwargs):
            name = f"{label}.{counters[label]}"
            counters[label] += 1
            # Each program owns its dataflow: regenerate cached RoPE grids from CPU constants.
            generate_freq_grid_np.cache_clear()
            generate_freq_grid_pytorch.cache_clear()
            writer.tree(name + ".args", args)
            writer.tree(name + ".kwargs", kwargs)
            for key, value in carrier.named_parameters():
                writer.tensor(name + ".weights." + key, value)
            with ComponentTrace(writer, name, carrier, args, kwargs) as graph:
                output = method(*args, **kwargs)
            graph.finish(output)
            writer.tree(name + ".out", output)
            graphs.append(name + ".graph.json")
            return output
        return call

    model.video_args_preprocessor.prepare = component("prep", model.video_args_preprocessor.prepare)
    model.audio_args_preprocessor.prepare = component("prep", model.audio_args_preprocessor.prepare)
    model._process_output = component("head", model._process_output)
    consistent = True
    with torch.inference_mode():
        for stage_index in (0, 1):
            prefix = f"forward.{stage_index}.0"
            modalities = []
            for index, label in enumerate(("video", "audio")):
                path = prefix + (f".args.{index}" if prefix + f".args.{index}.latent" in source["tensors"]
                                 else ".kwargs." + label)
                modalities.append(Modality(**{field.name: read(path + "." + field.name)
                                              for field in dataclasses.fields(Modality)}))
            perturbations = source["metadata"].get(prefix + ".args.2",
                                                   source["metadata"].get(prefix + ".kwargs.perturbations", "missing"))
            if perturbations is not None:
                raise RuntimeError("only the official no-perturbation branch is supported")
            writer.tree(prefix + ".args", tuple(modalities))
            output = model(*modalities, None)
            writer.tree(prefix + ".out", output)
            # Not needed by inference: whether the re-executed transformer reproduced the pipeline.
            consistent &= all(writer.tensors.get(key, {}).get("sha256") == source["tensors"].get(key, {}).get("sha256")
                              for key in (f"{prefix}.out.{index}" for index in range(len(output))))
            torch.cuda.synchronize()
    return _finish(writer, kernels, identity, graphs, reproduces_pipeline=consistent)


def text(workspace):
    """Gemma layers 0 (sliding) and 5 (full attention) as the two layer programs, its final norm,
    RoPE constants and config, and the embeddings processor with both connectors."""
    root, kernels, source, identity = _start(workspace, "text", "text-components-v1")
    import torch
    from .tracing import ConstantTrace
    from transformers.models.gemma4_unified.modeling_gemma4_unified import (
        Gemma4UnifiedTextDecoderLayer, Gemma4UnifiedTextScaledWordEmbedding)
    from ltx_core.text_encoders.gemma.embeddings_processor import EmbeddingsProcessor
    from ltx_core.text_encoders.gemma.encoders.base_encoder import LTXGemmaTextEncoder
    from ltx_core.text_encoders.gemma.tokenizer import LTXGemmaTokenizer
    from ltx_pipelines.utils.blocks import PromptEncoder
    from ltx_pipelines.utils.model_paths import ModelPaths
    writer = Recorder(root, workspace.blobs("text"))
    graphs = []
    original_layer = Gemma4UnifiedTextDecoderLayer.forward

    def layer(self, *args, **kwargs):
        name = f"gemma.layer.{self.layer_idx}"
        writer.tree(name + ".args", args)
        writer.tree(name + ".kwargs", kwargs)
        if self.layer_idx not in (0, 5):
            output = original_layer(self, *args, **kwargs)
        else:
            for key, value in self.named_parameters():
                writer.tensor(name + ".weights." + key, value)
            for key, value in self.named_buffers():
                writer.tensor(name + ".buffers." + key, value)
            with ConstantTrace(writer, name, self, args, kwargs) as graph:
                output = original_layer(self, *args, **kwargs)
            graph.finish(output)
            graphs.append(name + ".graph.json")
        writer.tree(name + ".out", output)
        return output
    Gemma4UnifiedTextDecoderLayer.forward = layer

    original_embedding = Gemma4UnifiedTextScaledWordEmbedding.forward

    def embedding(self, input_ids):
        writer.tensor("gemma.input_ids", input_ids)
        writer.tensor("gemma.embed_scale", self.embed_scale)
        output = original_embedding(self, input_ids)
        writer.tensor("gemma.embedding.out", output)
        return output
    Gemma4UnifiedTextScaledWordEmbedding.forward = embedding

    original_encode = LTXGemmaTextEncoder.encode

    def encode(self, prompts):
        writer.metadata["gemma.attention_implementation"] = self.model.config.text_config._attn_implementation
        output = original_encode(self, prompts)
        if len(output) != 1:
            raise RuntimeError("only single-prompt requests are supported")
        writer.tree("gemma.hidden_states", output[0][0])
        writer.tensor("gemma.attention_mask", output[0][1])
        return output
    LTXGemmaTextEncoder.encode = encode

    original_tokenize = LTXGemmaTokenizer.tokenize_with_weights

    def tokenize(self, text, *args, **kwargs):
        output = original_tokenize(self, text, *args, **kwargs)
        writer.metadata["tokenizer.output"] = {key: [[int(value) for value in row] for row in rows]
                                               for key, rows in output.items()}
        return output
    LTXGemmaTokenizer.tokenize_with_weights = tokenize

    original_build = PromptEncoder._build_text_encoder

    def build(self):
        encoder = original_build(self)
        model = encoder.model.model.language_model
        writer.metadata["gemma.config"] = model.config.to_dict()
        for key, value in model.rotary_emb.named_buffers():
            writer.tensor("gemma.rope.constants." + key, value)
        writer.metadata["gemma.rope.scaling"] = {kind: getattr(model.rotary_emb, kind + "_attention_scaling")
                                                 for kind in model.unique_layer_types}
        norm, forward = model.norm, model.norm.forward
        norm.forward = lambda *args, **kwargs: _traced(writer, ConstantTrace, "gemma.final_norm", norm, forward,
                                                       args, kwargs, graphs)
        return encoder
    PromptEncoder._build_text_encoder = build

    original_process = EmbeddingsProcessor.process_hidden_states

    def process(self, *args, **kwargs):
        return _traced(writer, ConstantTrace, "processor", self, lambda *a, **k: original_process(self, *a, **k),
                       args, kwargs, graphs)
    EmbeddingsProcessor.process_hidden_states = process

    torch.manual_seed(workspace.request["seed"])
    encoder = PromptEncoder(ModelPaths.from_split(
        transformer_path=workspace.weight("transformer"), text_encoder_path=workspace.weight("text_encoder"),
        video_vae_path=workspace.weight("video_vae"), audio_vae_path=workspace.weight("audio_vae")),
        dtype=torch.bfloat16, device=torch.device("cuda"))
    with torch.inference_mode():
        output = encoder([workspace.request["prompt"]])
    writer.tree("prompt.0.out", output)
    return _finish(writer, kernels, identity, graphs)


def upsampler(workspace):
    """LTX's spatial x2 latent upsampler between the video VAE's channel un-normalize/normalize."""
    root, kernels, source, identity = _start(workspace, "upsampler", "spatial-upsampler-v1")
    import torch
    from .tracing import ConstantTrace
    from ltx_pipelines.utils.blocks import VideoUpsampler
    writer = Recorder(root, workspace.blobs("upsampler"))
    latent = workspace.load("pipeline", source, "upsampler.0.args.0")
    torch.manual_seed(workspace.request["seed"])
    block = VideoUpsampler(workspace.weight("video_vae"), workspace.weight("spatial_upsampler"), torch.bfloat16,
                           torch.device("cuda"))
    encoder = block._encoder_builder.build(device=torch.device("cuda"), dtype=torch.bfloat16).eval()
    spatial = block._upsampler_builder.build(device=torch.device("cuda"), dtype=torch.bfloat16).eval()

    class Spatial(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.statistics = encoder.per_channel_statistics
            self.upsampler = spatial

        def forward(self, value):
            return self.statistics.normalize(self.upsampler(self.statistics.un_normalize(value)))

    model = Spatial().eval()
    graphs = []
    with torch.inference_mode():
        _traced(writer, ConstantTrace, "upsampler", model, model, (latent,), {}, graphs)
    consistent = writer.tensors["upsampler.out"]["sha256"] == source["tensors"]["upsampler.0.out"]["sha256"]
    return _finish(writer, kernels, identity, graphs, reproduces_pipeline=consistent)


def encoder(workspace):
    """I2V image encoding at both stages' resolutions, and the conditioning CRF."""
    root, kernels, source, identity = _start(workspace, "encoder", "image-encoder-v1")
    import torch
    from .tracing import ConstantTrace
    from ltx_pipelines.utils.blocks import ImageConditioner
    from ltx_pipelines.utils.helpers import combined_image_conditionings
    from ltx_pipelines.utils.types import ImageConditioningInput
    writer = Recorder(root, workspace.blobs("encoder"))
    request = workspace.request
    torch.manual_seed(request["seed"])
    block = ImageConditioner(workspace.weight("video_vae"), torch.bfloat16, torch.device("cuda"))
    images = block.resolve_crf([ImageConditioningInput(str(workspace.image()), 0, request["image_strength"])])
    writer.metadata["image.crf"] = images[0].crf
    model = block._build_encoder()
    original = model.forward
    graphs = []
    stage = {"index": 0}
    model.forward = lambda value: _traced(writer, ConstantTrace, f"encoder.{stage['index']}", model, original,
                                          (value,), {}, graphs)
    with torch.inference_mode():
        for index in (0, 1):
            stage["index"] = index
            scale = 2 if index == 0 else 1
            items = combined_image_conditionings(images, request["height"] // scale, request["width"] // scale,
                                                 model, torch.bfloat16, torch.device("cuda"))
            if len(items) != 1:
                raise RuntimeError("expected one first-frame conditioning")
    return _finish(writer, kernels, identity, graphs)


def audio(workspace):
    """Audio VAE decoder and vocoder from the final audio latent; their weights stay in the profile."""
    root, kernels, source, identity = _start(workspace, "audio", "audio-components-v1")
    import torch
    from .tracing import ConstantTrace
    from ltx_pipelines.utils.blocks import AudioDecoder
    writer = Recorder(root, workspace.blobs("audio"), lambda name: immutable(name, weights=True))
    latent = workspace.load("pipeline", source, "stage.1.out.audio.latent")
    torch.manual_seed(workspace.request["seed"])
    block = AudioDecoder(workspace.weight("audio_vae"), torch.bfloat16, torch.device("cuda"))
    decoder = block._decoder_builder.build(device=torch.device("cuda"), dtype=torch.bfloat16).eval()
    vocoder = block._vocoder_builder.build(device=torch.device("cuda"), dtype=torch.bfloat16).eval()
    graphs = []
    with torch.inference_mode():
        decoded = _traced(writer, ConstantTrace, "audio.decoder", decoder, decoder, (latent,), {}, graphs)
        waveform = _traced(writer, ConstantTrace, "audio.vocoder", vocoder, vocoder, (decoded,), {}, graphs)
        writer.tensor("audio.waveform", waveform.squeeze(0).float())
    writer.metadata["sampling_rate"] = vocoder.output_sampling_rate
    return _finish(writer, kernels, identity, graphs)


def video(workspace):
    """The diffusion video decoder as one program whose repeated blocks are nested component
    programs. Its noise uses the pipeline's recorded decoder RNG offset; the native runtime
    substitutes the request seed."""
    root, kernels, source, identity = _start(workspace, "video", "diffusion-video-pipeline-v1")
    import torch
    from torch.utils._python_dispatch import _disable_current_modes
    import natten.backends.fna as fna
    from .tracing import ACTIVE_TRACE, CompactTrace
    from ltx_core.model.video_vae import AUTO_TILING
    from ltx_core.model.video_vae.transformer.attention import NattenAttention
    from ltx_core.types import VideoPixelShape
    from ltx_pipelines.utils.blocks import VideoDecoder
    from ltx_pipelines.utils.helpers import ensure_tiling_config, tiling_scale_factors_for_vae
    writer = Recorder(root, workspace.blobs("video"), lambda name: immutable(name, weights=True))
    latent = workspace.load("pipeline", source, "stage.1.out.video.latent")
    request = workspace.request
    vae_path = workspace.weight("video_vae")
    block = VideoDecoder(vae_path, torch.bfloat16, torch.device("cuda"))
    tiling = ensure_tiling_config(
        AUTO_TILING, scale_factors=tiling_scale_factors_for_vae(vae_path), vae_checkpoint_path=vae_path,
        video_shape=VideoPixelShape(batch=1, frames=request["num_frames"], height=request["height"],
                                    width=request["width"], fps=request["frame_rate"]),
        diffvae_optimization=block.diffvae_optimization, device=torch.device("cuda"), keyframes=False)
    writer.metadata["tiling"] = dataclasses.asdict(tiling) if dataclasses.is_dataclass(tiling) else repr(tiling)
    decoder = block._prepared_builder().build(device=torch.device("cuda"), dtype=torch.bfloat16).eval()
    # The pipeline's decoder noise: 5-D, 3-channel draws seeded by the request seed.
    draws = [f["name"] for f in source["metadata"]["fixtures"] if f["kind"] == "rng"
             and source["metadata"][f["name"]]["seed"] == request["seed"]
             and len(source["tensors"][f["name"] + ".out"]["shape"]) == 5
             and source["tensors"][f["name"] + ".out"]["shape"][1] == 3]
    if not draws:
        raise RuntimeError("the pipeline capture recorded no decoder noise")
    generator = torch.Generator(device="cuda").manual_seed(request["seed"])
    generator.set_offset(source["metadata"][draws[0]]["offset_before"])
    original_randn = torch.randn

    def randn(*args, **kwargs):
        trace = ACTIVE_TRACE.get()
        before = generator.get_offset()
        if trace is None:
            return original_randn(*args, **kwargs)
        if kwargs.get("generator") is not generator:
            raise RuntimeError("unexpected video decoder random generator")
        with _disable_current_modes():
            output = original_randn(*args, **kwargs)
        trace.nodes.append(dict(op="ltx25.randn", args=[], kwargs=dict(
            seed=request["seed"], offset_before=before, offset_after=generator.get_offset()),
            out=trace.outputs(output)))
        return output
    torch.randn = randn

    # Run the original Python bodies of the packaged custom ops, so their kernels are traced.
    modules = [importlib.import_module("ltx_core.model.video_vae.transformer." + name)
               for name in ("det_attn_rope", "swiglu", "chunked.attn", "chunked.mlp")]

    @contextmanager
    def expose_bodies():
        saved = []
        for module in modules:
            for key, value in list(vars(module).items()):
                if hasattr(value, "_init_fn") and getattr(value, "_namespace", "") == "ltx_core":
                    saved.append((module, key, value))
                    setattr(module, key, value._init_fn)
        try:
            yield
        finally:
            for module, key, value in saved:
                setattr(module, key, value)

    tiles = {}
    original_config = fna.check_cutlass_fna_forward_config

    def config(*args, **kwargs):
        tiles["value"] = original_config(*args, **kwargs)
        return tiles["value"]
    fna.check_cutlass_fna_forward_config = config
    original_na = NattenAttention.__call__

    def na(self, attn, q, k, v):
        trace = ACTIVE_TRACE.get()
        if trace is None:
            return original_na(self, attn, q, k, v)
        if q.dtype != v.dtype:
            q = q.to(v.dtype)
        if k.dtype != v.dtype:
            k = k.to(v.dtype)
        args = trace.encode((q, k, v))
        tiles.clear()
        with _disable_current_modes():
            output = original_na(self, attn, q, k, v)
        if "value" not in tiles:
            raise RuntimeError(f"unexpected NATTEN backend {self._backend}: expected cutlass-fna")
        trace.nodes.append(dict(op="ltx25.natten3d", args=args, kwargs=dict(
            kernel_size=list(attn.kernel_size), q_tile=list(tiles["value"][0]), kv_tile=list(tiles["value"][1]),
            scale=1.0, backend=self._backend), out=trace.outputs(output)))
        return output
    NattenAttention.__call__ = na

    graphs, templates = [], {}

    def call_values(prefix, module, args, kwargs):
        values = {}

        def visit(path, value):
            if isinstance(value, torch.Tensor):
                values[path] = value
            elif isinstance(value, (tuple, list)):
                for index, item in enumerate(value):
                    visit(path + "." + str(index), item)
            elif isinstance(value, dict):
                for key, item in value.items():
                    visit(path + "." + str(key), item)
        visit(prefix + ".args", args)
        visit(prefix + ".kwargs", kwargs)
        values.update((prefix + ".weights." + key, value) for key, value in module.named_parameters())
        values.update((prefix + ".buffers." + key, value) for key, value in module.named_buffers())
        return values

    def tensor_inputs(value):
        if isinstance(value, dict):
            if "tensor" in value:
                yield value["tensor"]
            else:
                for child in value.values():
                    yield from tensor_inputs(child)
        elif isinstance(value, list):
            for child in value:
                yield from tensor_inputs(child)

    def structure(value):
        if isinstance(value, torch.Tensor):
            return ("tensor", tuple(value.shape), tuple(value.stride()), str(value.dtype))
        if isinstance(value, (tuple, list)):
            return tuple(structure(item) for item in value)
        if isinstance(value, dict):
            return tuple((key, structure(item)) for key, item in value.items())
        return value

    def hierarchical(name, module, method, args, kwargs):
        """One program per distinct block signature; every call becomes an `ltx25.component` node."""
        outer = ACTIVE_TRACE.get()
        signature = (name.rsplit(".", 1)[0], repr(structure((args, kwargs))),
                     repr([(tuple(v.shape), tuple(v.stride()), str(v.dtype)) for v in module.parameters()]))
        template = templates.get(signature)
        key = template[0] if template else f"video.component.{len(templates)}"
        values = call_values(key, module, args, kwargs)
        # Encode incoming identities before the component can mutate any storage.
        bindings = {field: outer.encode(value) for field, value in values.items()}
        token = ACTIVE_TRACE.set(None)
        try:
            with _disable_current_modes():
                if template is None:
                    for field, value in values.items():
                        writer.tensor(field, value)
                    with expose_bodies(), CompactTrace(writer, key, module, args, kwargs) as inner:
                        output = method(*args, **kwargs)
                    program = inner.finish(output)
                    templates[signature] = (key, program)
                    graphs.append(key + ".graph.json")
                else:
                    output = method(*args, **kwargs)
                    program = template[1]
        finally:
            ACTIVE_TRACE.reset(token)
        used = set()
        for node in program["nodes"]:
            used.update(tensor_inputs(node["args"]))
            used.update(tensor_inputs(node["kwargs"]))
        used.intersection_update(program["externals"])
        names = sorted(used.intersection(bindings))
        constants = {}
        for field in sorted(used.difference(bindings)):
            if not field.startswith(key + ".constant."):
                raise RuntimeError("unbound video component input " + field)
            constants[field] = dict(entry=writer.tensors[field], layout=program["externals"][field])
        outer.nodes.append(dict(op="ltx25.component", args=[bindings[field] for field in names],
                                kwargs=dict(program=str(root / (key + ".graph.json")), names=names,
                                            constants=constants), out=outer.outputs(output)))
        return output

    def instrument(name, module, method_name):
        method = getattr(module, method_name)

        def call(*args, **kwargs):
            if ACTIVE_TRACE.get() is None:
                return method(*args, **kwargs)
            return hierarchical(name, module, method, args, kwargs)
        setattr(module, method_name, call)

    for stage, stage_blocks in enumerate(decoder.det_stages):
        for index, module in enumerate(stage_blocks):
            instrument(f"video.det.{stage}.{index}", module, "forward")
    for index, module in enumerate(decoder.diff_blocks):
        instrument(f"video.diff.{index}", module, "forward_x_ctx")
    with torch.inference_mode():
        name = "video.pipeline"
        for key, value in call_values(name, decoder, (latent,), {}).items():
            writer.tensor(key, value)
        with CompactTrace(writer, name, decoder, (latent,), {}) as graph:
            outputs = tuple(decoder.decode_video(latent, tiling, generator))
        graph.finish(outputs)
        graphs.append(name + ".graph.json")
        for index, pixels in enumerate(outputs):
            writer.tensor(f"video.pixels.{index}", pixels)
    return _finish(writer, kernels, identity, graphs, decoder_noise_draws=len(draws))


CAPTURES = {"forward": forward, "text": text, "upsampler": upsampler, "encoder": encoder,
            "audio": audio, "video": video}
