"""One-GPU inference runtime for the validated canonical-512 Z-Image Turbo Sidecar."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from sidecar import IdentitySidecar


class SidecarRuntime:
    def __init__(self, config: dict, clean_snapshot: Path):
        from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, ZImagePipeline, ZImageTransformer2DModel
        from transformers import AutoModel, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required')
        if clean_snapshot.name != config['base_revision']:
            raise ValueError('Clean Turbo revision mismatch')
        self.config = config
        self.snapshot = clean_snapshot
        self.device = torch.device('cuda')
        self.dtype = torch.bfloat16
        self.sidecar = None
        self.pipe = None
        self.tokenizer = None
        self.text_encoder = None

        # CPU-resident text encoder: never unload the GPU generation stack per request.
        # This uses bf16 like the reference, but CPU/GPU kernels are not guaranteed pixel-identical.
        print('[text] loading tokenizer and CPU-resident text encoder', flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(clean_snapshot / 'tokenizer', local_files_only=True)
        self.text_encoder = AutoModel.from_pretrained(
            clean_snapshot / 'text_encoder',
            torch_dtype=self.dtype,
            local_files_only=True,
            low_cpu_mem_usage=True,
        ).to('cpu')
        self.text_encoder.eval()
        self.text_encoder.requires_grad_(False)

        print('[identity] verifying canonical embedding', flush=True)
        identity_values = load_file(config['identity_path'], device='cpu')
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
            raise ValueError('identity embedding fingerprint mismatch')
        self.identity = identity.to(self.device, dtype=torch.float32)

        transformer_cfg = json.loads((clean_snapshot / 'transformer' / 'config.json').read_text(encoding='utf-8'))
        dim = transformer_cfg['dim']
        layers = config['injection_layers_0based']
        if any(x < 0 or x >= transformer_cfg['n_layers'] for x in layers):
            raise ValueError('invalid sidecar injection layers')
        self.sidecar = IdentitySidecar(
            dim, config['sidecar_rank'], config['identity_tokens'],
            layers, config['layer_scales'],
        ).to(device=self.device, dtype=torch.float32)
        self.sidecar.load_state_dict(load_file(config['sidecar_path'], device='cpu'), strict=True)
        self.sidecar.eval()
        self.sidecar.requires_grad_(False)

        print('[model] loading clean Z-Image-Turbo transformer on GPU', flush=True)
        transformer = ZImageTransformer2DModel.from_pretrained(
            clean_snapshot / 'transformer',
            torch_dtype=self.dtype,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        transformer.requires_grad_(False)
        transformer.to(self.device)
        transformer.eval()
        self.sidecar.attach(transformer)

        vae = AutoencoderKL.from_pretrained(
            clean_snapshot / 'vae',
            torch_dtype=self.dtype,
            local_files_only=True,
            low_cpu_mem_usage=True,
        ).to(self.device)
        vae.eval()
        vae.requires_grad_(False)
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            clean_snapshot / 'scheduler', local_files_only=True,
        )
        self.pipe = ZImagePipeline(
            scheduler=scheduler, vae=vae, text_encoder=None, tokenizer=None, transformer=transformer,
        )
        print('[ready] models loaded', flush=True)

    @torch.no_grad()
    def encode_prompt(self, prompt: str) -> torch.Tensor:
        # Exactly the reference tokenizer/template/hidden-state selection.
        text = self.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )
        toks = self.tokenizer(
            [text], padding='max_length', max_length=512, truncation=True, return_tensors='pt',
        )
        mask = toks.attention_mask.bool()
        hidden = self.text_encoder(
            input_ids=toks.input_ids,
            attention_mask=mask,
            output_hidden_states=True,
        ).hidden_states[-2]
        emb = hidden[0][mask[0]].detach().to(self.device, dtype=self.dtype).contiguous()
        if emb.ndim != 2 or emb.shape[1] != 2560 or emb.shape[0] < 1:
            raise ValueError(f'unexpected text embedding shape: {tuple(emb.shape)}')
        return emb

    @torch.no_grad()
    def generate(self, *, prompt: str, width: int, height: int, steps: int, seed: int, scale: float) -> bytes:
        # API server serializes calls to this function. The Sidecar context is mutable.
        text_emb = self.encode_prompt(prompt)
        self.sidecar.set_context(self.identity, (height // 16) * (width // 16), scale)
        generator = torch.Generator(device='cuda').manual_seed(seed)
        image = self.pipe(
            prompt=None,
            prompt_embeds=[text_emb],
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=0.0,
            generator=generator,
        ).images[0]
        output = io.BytesIO()
        image.save(output, format='PNG')
        return output.getvalue()
