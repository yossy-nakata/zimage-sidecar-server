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
        msg = '[ready] models loaded; no silent INT8 fallback'
        if self.active_lora_info:
            msg += f"; lora={self.active_lora_info['name']} scale={self.active_lora_info['scale']}"
        print(msg, flush=True)

    def _load_optional_lora(self, transformer) -> None:
        lora = self.config.get('lora')
        if not lora:
            return
        if not isinstance(lora, dict):
            raise ValueError('config.lora must be an object')
        enabled = bool(lora.get('enabled', False))
        if not enabled:
            return
        path = _require_file(lora.get('path', ''), 'LoRA weights')
        adapter_name = str(lora.get('adapter_name', 'bodyfix')).strip() or 'bodyfix'
        scale = lora.get('scale', 0.3)
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or not (0.0 <= float(scale) <= 2.0):
            raise ValueError('config.lora.scale must be finite and between 0.0 and 2.0')
        scale = float(scale)
        print(f'[lora] loading transformer LoRA: path={path} adapter={adapter_name} scale={scale}', flush=True)

        last_exc = None
        loaded = False
        # Preferred: pipeline-level loading, then set active adapter scale.
        if hasattr(self.pipe, 'load_lora_weights'):
            try:
                self.pipe.load_lora_weights(str(path), adapter_name=adapter_name)
                if hasattr(self.pipe, 'set_adapters'):
                    self.pipe.set_adapters(adapter_name, adapter_weights=[scale])
                loaded = True
            except Exception as exc:
                last_exc = exc
                print(f'[lora] pipeline.load_lora_weights failed: {exc!r}', flush=True)

        # Fallback: transformer-level adapter loading, if exposed by this diffusers build.
        if (not loaded) and hasattr(transformer, 'load_lora_adapter'):
            try:
                transformer.load_lora_adapter(str(path), adapter_name=adapter_name)
                if hasattr(transformer, 'set_adapters'):
                    transformer.set_adapters(adapter_name, adapter_weights=[scale])
                loaded = True
            except Exception as exc:
                last_exc = exc
                print(f'[lora] transformer.load_lora_adapter failed: {exc!r}', flush=True)

        if not loaded:
            raise RuntimeError(
                'Failed to load LoRA with this diffusers build. '\
                'Tried pipeline.load_lora_weights and transformer.load_lora_adapter.'
            ) from last_exc

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
