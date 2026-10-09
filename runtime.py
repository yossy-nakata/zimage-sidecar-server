"""Single-GPU inference runtime: Z-Image-Turbo BF16 + canonical Sidecar + INT8 text encoder.

Optional fixed LoRA support for transformer-side LoRAs (e.g. anatomy / body fixes).
The LoRA is loaded once at startup from config.json and remains active for all requests.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import logging
import math
import re
import time
import warnings
from pathlib import Path

import torch
from safetensors.torch import load_file

from sidecar import IdentitySidecar

_BNB_CAST_NOTICE = r"^MatMul8bitLt: inputs will be cast from torch\..* to float16 during quantization$"


class _BnbCastLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return re.fullmatch(_BNB_CAST_NOTICE, record.getMessage()) is None


logging.getLogger('bitsandbytes.autograd._functions').addFilter(_BnbCastLogFilter())
warnings.filterwarnings(
    'ignore',
    message=_BNB_CAST_NOTICE,
    category=UserWarning,
    module=r'^bitsandbytes\.autograd\._functions$',
)


def _require_file(path: str | Path, label: str) -> Path:
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(f'{label} missing: {file}')
    return file


def _validate_request(prompt: str, width: int, height: int, steps: int, seed: int, scale: float) -> None:
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 5000:
        raise ValueError('prompt must be nonempty and at most 5000 characters')
    if any(isinstance(v, bool) or not isinstance(v, int) for v in (width, height)):
        raise ValueError('width/height must be integers')
    if not (512 <= width <= 1024 and 512 <= height <= 1024 and width % 16 == 0 and height % 16 == 0):
        raise ValueError('width/height must be 512..1024 and divisible by 16')
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 20:
        raise ValueError('steps must be an integer from 1 to 20')
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError('seed must be a nonnegative integer below 2**63')
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or not 0 <= scale <= 2:
        raise ValueError('sidecar_scale must be finite and in [0,2]')


def _gpu_memory(label: str) -> None:
    free, total = torch.cuda.mem_get_info(0)
    mib = 1024 ** 2
    print(
        f'[vram] {label}: free={free / mib:.0f} MiB / total={total / mib:.0f} MiB; '
        f'allocated={torch.cuda.memory_allocated(0) / mib:.0f} MiB; '
        f'reserved={torch.cuda.memory_reserved(0) / mib:.0f} MiB',
        flush=True,
    )


def _convert_original_zimage_lora(raw: dict, transformer) -> dict:
    """Convert original Z-Image LoRA names to Diffusers transformer module names.

    Only A/B pairs plus matching alpha are accepted. Fused QKV B weights are
    split against the actual Diffusers model dimensions; no guessed shapes.
    """
    from collections import defaultdict

    modules = dict(transformer.named_modules())
    pairs = defaultdict(dict)

    def normalized_key(key):
        for prefix in ('transformer.', 'model.diffusion_model.', 'diffusion_model.'):
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
        key = key.replace('.lora_A.default.weight', '.lora_A.weight')
        key = key.replace('.lora_B.default.weight', '.lora_B.weight')
        key = key.replace('.attention.to.q.', '.attention.to_q.')
        key = key.replace('.attention.to.k.', '.attention.to_k.')
        key = key.replace('.attention.to.v.', '.attention.to_v.')
        key = key.replace('.adaLN.modulation.', '.adaLN_modulation.')
        key = key.replace('.feed.forward.', '.feed_forward.')
        return key

    suffixes = {'.lora_A.weight': 'A', '.lora_B.weight': 'B',
                '.lora_down.weight': 'A', '.lora_up.weight': 'B',
                '.lora.down.weight': 'A', '.lora.up.weight': 'B',
                '.alpha': 'alpha'}
    for key, value in raw.items():
        k = normalized_key(key)
        match = next(((s, t) for s, t in suffixes.items() if k.endswith(s)), None)
        if not match:
            raise ValueError(f'unsupported LoRA tensor key: {key}')
        suffix, field = match
        target = k[:-len(suffix)]
        if field in pairs[target]:
            raise ValueError(f'duplicate LoRA tensor: {target}.{field}')
        pairs[target][field] = value

    # Normalize known original-module aliases *after* parsing so duplicate
    # out/to_out and qkv/to_qkv can be identified rather than overwritten.
    canonical = defaultdict(dict)
    for target, values in pairs.items():
        dest = target
        if dest.endswith('.attention.out'):
            dest = dest[:-len('.attention.out')] + '.attention.to_out.0'
        elif dest.endswith('.attention.to_out'):
            dest += '.0'
        if dest in canonical:
            # Some files store the alpha under the Diffusers path and its A/B
            # tensors under the original path. Merge only complementary keys;
            # never silently override a conflicting tensor.
            old = canonical[dest]
            if any(f in old and not torch.equal(old[f], t) for f, t in values.items()):
                raise ValueError(f'conflicting LoRA aliases: {target} -> {dest}')
            old.update(values)
        else:
            canonical[dest] = values

    # A combined QKV LoRA can be split into individual projections. If all three
    # individual LoRAs also exist, accept the duplicate fused form only when
    # no other tensors need it (common in non-diffusers exports).
    for target in list(canonical):
        if not target.endswith('.attention.qkv'):
            continue
        stem = target[:-3]  # e.g. layers.0.attention.
        dests = [stem + x for x in ('to_q', 'to_k', 'to_v')]
        vals = canonical.pop(target)
        if all(k in canonical for k in dests):
            continue  # redundant original fused representation
        if any(k in canonical for k in dests):
            raise ValueError(f'partially duplicated fused QKV: {target}')
        if 'A' not in vals or 'B' not in vals:
            raise ValueError(f'fused QKV A/B missing: {target}')
        a, b = vals['A'], vals['B']
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
            raise ValueError(f'fused QKV rank mismatch: {target}')
        if any(k not in modules for k in dests):
            raise ValueError(f'Z-Image QKV targets missing: {dests}')
        outdims = [int(modules[k].weight.shape[0]) for k in dests]
        if b.shape[0] != sum(outdims):
            raise ValueError(f'fused QKV output size mismatch at {target}: {b.shape[0]} vs {outdims}')
        for k, part in zip(dests, torch.split(b, outdims, dim=0)):
            canonical[k] = {'A': a, 'B': part, **({'alpha': vals['alpha']} if 'alpha' in vals else {})}

    if not canonical:
        raise ValueError('LoRA contains no compatible transformer targets')
    converted = {}
    for target, vals in canonical.items():
        if set(vals) not in ({'A', 'B'}, {'A', 'B', 'alpha'}):
            raise ValueError(f'LoRA A/B weights missing at {target}: {set(vals)}')
        if target not in modules or not hasattr(modules[target], 'weight'):
            raise ValueError(f'LoRA target not in Diffusers Z-Image transformer: {target}')
        a, b = vals['A'], vals['B']
        weight = modules[target].weight
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1] or \
                a.shape[1] != weight.shape[1] or b.shape[0] != weight.shape[0]:
            raise ValueError(f'LoRA shape mismatch at {target}: A{tuple(a.shape)} B{tuple(b.shape)} '
                             f'base{tuple(weight.shape)}')
        if 'alpha' in vals:
            alpha = vals['alpha']
            if alpha.numel() != 1 or not math.isfinite(float(alpha.item())) or float(alpha.item()) <= 0:
                raise ValueError(f'invalid alpha at {target}')
            # Diffusers / PEFT expect alpha/rank. Fold this into A/B while
            # preserving precision and keeping the PEFT configured scale=1.
            factor = float(alpha.item()) / float(a.shape[0])
            root = math.sqrt(factor)
            a = (a.float() * root).to(a.dtype)
            b = (b.float() * root).to(b.dtype)
        converted[f'transformer.{target}.lora_A.weight'] = a
        converted[f'transformer.{target}.lora_B.weight'] = b
    return converted


class SidecarRuntime:
    def __init__(self, config: dict, clean_snapshot: Path):
        from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, ZImagePipeline, ZImageTransformer2DModel
        from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

        try:
            bnb_version = importlib.metadata.version('bitsandbytes')
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError('bitsandbytes is required for INT8 text encoding; install via uv in the test environment') from exc
        import bitsandbytes as bnb

        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required; refusing CPU fallback for INT8 server')
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError('BF16 CUDA support required by the existing image generator')
        clean_snapshot = Path(clean_snapshot)
        if clean_snapshot.name != config['base_revision']:
            raise ValueError('Clean Turbo snapshot revision mismatch')
        for subfolder in ('text_encoder', 'tokenizer', 'transformer', 'vae', 'scheduler'):
            if not (clean_snapshot / subfolder).is_dir():
                raise FileNotFoundError(f'Clean Turbo component missing: {subfolder}')
        _require_file(config['identity_path'], 'canonical identity')
        _require_file(config['sidecar_path'], 'Sidecar weights')

        self.config = config
        self.snapshot = clean_snapshot
        self.device = torch.device('cuda:0')
        self.dtype = torch.bfloat16
        self.sidecar = None
        self.pipe = None
        self.tokenizer = None
        self.text_encoder = None
        self.active_lora_info = None
        self.transformer_precision = config.get('transformer_precision', 'bf16')
        if self.transformer_precision not in ('bf16', 'int8'):
            raise ValueError("transformer_precision must be 'bf16' or 'int8'")

        print(f'[runtime] GPU={torch.cuda.get_device_name(0)}; bitsandbytes={bnb_version}', flush=True)
        _gpu_memory('before models')
        print('[text] loading INT8 text encoder on GPU (FP16 unquantized layers)', flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(clean_snapshot / 'tokenizer', local_files_only=True)
        self.text_encoder = AutoModel.from_pretrained(
            clean_snapshot / 'text_encoder',
            dtype=torch.float16,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            device_map=0,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        self.text_encoder.eval()
        self.text_encoder.requires_grad_(False)

        quantized_linears = sum(isinstance(m, bnb.nn.Linear8bitLt) for m in self.text_encoder.modules())
        if quantized_linears == 0:
            raise RuntimeError('Text encoder loaded but has zero bitsandbytes Linear8bitLt modules; not INT8')
        misplaced = [name for name, p in self.text_encoder.named_parameters() if p.device.type != 'cuda' or p.device.index not in (None, 0)]
        if misplaced:
            raise RuntimeError(f'Text encoder not fully on cuda:0, first misplaced weights: {misplaced[:3]}')
        print(f'[text] confirmed INT8 Linear layers={quantized_linears}; device=cuda:0', flush=True)
        _gpu_memory('after text encoder')

        print('[identity] verifying canonical embedding', flush=True)
        identity_values = load_file(str(config['identity_path']), device='cpu')
        if 'embedding' not in identity_values:
            raise KeyError("identity file lacks 'embedding'")
        identity = identity_values['embedding'].float().contiguous()
        if tuple(identity.shape) != (1, 512):
            raise ValueError(f'expected identity [1,512], got {tuple(identity.shape)}')
        if not torch.isfinite(identity).all():
            raise ValueError('identity contains NaN/Inf')
        if abs(float(identity.norm(dim=-1).item()) - 1.0) > 1e-4:
            raise ValueError('identity is not unit-normalized')
        fingerprint = hashlib.sha256(identity.numpy().tobytes()).hexdigest()
        if fingerprint != config['identity_fingerprint_sha256']:
            raise ValueError('identity fingerprint mismatch')
        self.identity = identity.to(self.device, dtype=torch.float32)

        transformer_cfg = json.loads((clean_snapshot / 'transformer' / 'config.json').read_text(encoding='utf-8'))
        dim = transformer_cfg['dim']
        n_layers = transformer_cfg['n_layers']
        layers = config['injection_layers_0based']
        layer_scales = config['layer_scales']
        if not isinstance(dim, int) or not isinstance(n_layers, int) or n_layers < 1:
            raise ValueError('invalid clean Turbo transformer architecture config')
        if not isinstance(layers, list) or not layers or len(layers) != len(layer_scales):
            raise ValueError('invalid Sidecar layers/scales')
        if len(set(layers)) != len(layers) or any(not isinstance(x, int) or not 0 <= x < n_layers for x in layers):
            raise ValueError('Sidecar injection layers invalid for this transformer')
        if any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in layer_scales):
            raise ValueError('invalid Sidecar layer scale')
        if config['sidecar_rank'] != 512 or config['identity_tokens'] != 8:
            raise ValueError('Sidecar rank/tokens mismatch from validated 512/8 architecture')

        self.sidecar = IdentitySidecar(dim, config['sidecar_rank'], config['identity_tokens'], layers, layer_scales).to(device=self.device, dtype=torch.float32)
        self.sidecar.load_state_dict(load_file(str(config['sidecar_path']), device='cpu'), strict=True)
        self.sidecar.eval()
        self.sidecar.requires_grad_(False)

        if self.transformer_precision == 'bf16':
            # Preserve the already validated BF16 loading path without modification.
            print('[model] loading clean Z-Image-Turbo BF16 transformer on GPU', flush=True)
            transformer = ZImageTransformer2DModel.from_pretrained(
                clean_snapshot / 'transformer',
                dtype=self.dtype,
                local_files_only=True,
                low_cpu_mem_usage=True,
            )
            transformer.requires_grad_(False)
            transformer.to(self.device)
            transformer.eval()
        else:
            # Ported from zimage-sidecar_code_validate_sidecar8_turbo.py.
            # Diffusers and Transformers expose different BitsAndBytesConfig classes.
            from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig

            print('[model] loading clean Z-Image-Turbo INT8 transformer on GPU', flush=True)
            transformer = ZImageTransformer2DModel.from_pretrained(
                clean_snapshot,
                subfolder='transformer',
                quantization_config=DiffusersBitsAndBytesConfig(load_in_8bit=True),
                dtype=self.dtype,
                device_map='cuda',
                local_files_only=True,
            )
            transformer.requires_grad_(False)
            transformer.eval()
            # No .to(self.device) on the INT8 model: quantized weights are placed by from_pretrained().
            count = sum(isinstance(m, bnb.nn.Linear8bitLt) for m in transformer.modules())
            if count == 0:
                raise RuntimeError('Turbo INT8 requested but no Linear8bitLt layers were loaded')
            misplaced = [name for name, p in transformer.named_parameters()
                         if p.device.type != 'cuda' or p.device.index not in (None, 0)]
            if misplaced:
                raise RuntimeError(f'Turbo INT8 parameters are not all on cuda:0: {misplaced[:3]}')
            print(f'[model] confirmed Turbo INT8 Linear layers={count}; device=cuda:0', flush=True)
            _gpu_memory('after INT8 transformer')

        vae = AutoencoderKL.from_pretrained(
            clean_snapshot / 'vae',
            dtype=self.dtype,
            local_files_only=True,
            low_cpu_mem_usage=True,
        ).to(self.device)
        vae.eval()
        vae.requires_grad_(False)
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(clean_snapshot / 'scheduler', local_files_only=True)
        self.pipe = ZImagePipeline(scheduler=scheduler, vae=vae, text_encoder=None, tokenizer=None, transformer=transformer)

        self._load_optional_lora(transformer)
        self.sidecar.attach(transformer)
        _gpu_memory('models ready (idle)')
        msg = f'[ready] models loaded; transformer={self.transformer_precision}; no silent INT8 fallback'
        if self.active_lora_info:
            msg += f"; lora={self.active_lora_info['name']} scale={self.active_lora_info['scale']}"
        print(msg, flush=True)

    def _load_optional_lora(self, transformer) -> None:
        lora = self.config.get('lora')
        if not lora:
            return
        if not isinstance(lora, dict):
            raise ValueError('config.lora must be an object')
        if not lora.get('enabled', False):
            return

        path = _require_file(lora.get('path', ''), 'LoRA weights')
        adapter_name = str(lora.get('adapter_name', 'bodyfix')).strip() or 'bodyfix'
        scale = lora.get('scale', 0.3)
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or not 0.0 <= scale <= 2.0:
            raise ValueError('config.lora.scale must be finite and between 0.0 and 2.0')
        scale = float(scale)
        print(f'[lora] loading path={path} adapter={adapter_name} scale={scale}', flush=True)

        from safetensors import safe_open
        from peft.tuners.tuners_utils import BaseTunerLayer
        with safe_open(str(path), framework='pt', device='cpu') as file:
            has_alpha = any(k.endswith('.alpha') for k in file.keys())

        expected_modules = None
        if has_alpha:
            # Original Z-Image and Diffusers use different module paths and
            # fused QKV vs separate Q/K/V. Convert before touching the model;
            # a bad checkpoint must never leave a partially installed adapter.
            raw = load_file(str(path), device='cpu')
            weights = _convert_original_zimage_lora(raw, transformer)
            del raw
            expected_modules = {k.removeprefix('transformer.').removesuffix('.lora_A.weight')
                                for k in weights if k.endswith('.lora_A.weight')}
            print(f'[lora] remapped targets={len(expected_modules)}', flush=True)
            transformer.load_lora_adapter(weights, adapter_name=adapter_name, prefix='transformer')
            transformer.set_adapters(adapter_name, weights=[scale])
            del weights
        else:
            # Preserve the proven path for ordinary LoRAs without per-layer alpha.
            self.pipe.load_lora_weights(str(path), adapter_name=adapter_name)
            self.pipe.set_adapters(adapter_name, adapter_weights=[scale])

        active = {name: mod for name, mod in transformer.named_modules()
                  if isinstance(mod, BaseTunerLayer) and adapter_name in getattr(mod, 'lora_A', {})}
        if not active:
            raise RuntimeError(f'LoRA {adapter_name!r} has zero loaded Transformer layers')
        if expected_modules is not None and set(active) != expected_modules:
            missing = sorted(expected_modules - set(active))
            extra = sorted(set(active) - expected_modules)
            raise RuntimeError(f'LoRA target mismatch: missing={missing[:5]}, extra={extra[:5]}')
        if any(adapter_name not in getattr(m, 'active_adapters', []) for m in active.values()):
            raise RuntimeError('LoRA injected but not activated')
        if any(not math.isclose(float(m.scaling[adapter_name]),
                                float(m.lora_alpha[adapter_name]) / float(m.r[adapter_name]) * scale,
                                rel_tol=1e-5, abs_tol=1e-6) for m in active.values()):
            raise RuntimeError('LoRA adapter scale was not applied')
        print(f'[lora] verified active Transformer layers={len(active)}', flush=True)
        self.active_lora_info = {'name': adapter_name, 'path': str(path), 'scale': scale}
        _gpu_memory('after lora')
        print('[lora] active', flush=True)

    @torch.no_grad()
    def encode_prompt(self, prompt: str) -> torch.Tensor:
        t0 = time.perf_counter()
        text = self.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )
        toks = self.tokenizer([text], padding='max_length', max_length=512, truncation=True, return_tensors='pt')
        ids = toks.input_ids.to(self.device)
        mask = toks.attention_mask.to(self.device).bool()
        hidden = self.text_encoder(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states[-2]
        emb = hidden[0][mask[0]].detach().to(self.device, dtype=self.dtype).contiguous()
        if emb.ndim != 2 or emb.shape[1] != 2560 or emb.shape[0] < 1:
            raise ValueError(f'unexpected text embedding shape: {tuple(emb.shape)}')
        if not torch.isfinite(emb).all():
            raise ValueError('text encoder produced NaN/Inf embeddings')
        torch.cuda.synchronize(self.device)
        print(f'[timing] text_encoder={time.perf_counter() - t0:.2f}s, tokens={emb.shape[0]}', flush=True)
        return emb

    @torch.no_grad()
    def generate(self, *, prompt: str, width: int, height: int, steps: int, seed: int, scale: float) -> bytes:
        _validate_request(prompt, width, height, steps, seed, scale)
        torch.cuda.reset_peak_memory_stats(self.device)
        try:
            text_emb = self.encode_prompt(prompt)
            self.sidecar.set_context(self.identity, (height // 16) * (width // 16), scale)
            generator = torch.Generator(device=self.device).manual_seed(seed)
            t0 = time.perf_counter()
            image = self.pipe(
                prompt=None,
                prompt_embeds=[text_emb],
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=0.0,
                generator=generator,
            ).images[0]
            torch.cuda.synchronize(self.device)
            print(f'[timing] diffusion_and_vae={time.perf_counter() - t0:.2f}s', flush=True)
            output = io.BytesIO()
            image.save(output, format='PNG')
            png = output.getvalue()
            if not png.startswith(b'\x89PNG\r\n\x1a\n'):
                raise RuntimeError('VAE/image encoding did not produce a valid PNG signature')
            print(f'[image] bytes={len(png)}, peak_reserved={torch.cuda.max_memory_reserved(0) / 1024**2:.0f} MiB', flush=True)
            return png
        except torch.cuda.OutOfMemoryError as exc:
            _gpu_memory('CUDA OOM')
            raise RuntimeError('CUDA OOM during INT8 inference; generation failed (no hidden CPU fallback). Check VRAM and restart if necessary') from exc
        finally:
            if self.sidecar is not None:
                self.sidecar.set_context(self.identity, 0, 0.0)
