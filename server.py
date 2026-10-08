#!/usr/bin/env python3
"""Minimal authenticated OpenAI-style image-generation API (one RTX 3090 GPU)."""
from __future__ import annotations

import argparse
import base64
import hmac
import json
import math
import os
import secrets
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def parse_generation_request(obj: object, config: dict) -> dict:
    if not isinstance(obj, dict):
        raise ValueError('request must be a JSON object')
    model = obj.get('model', config['model_name'])
    if model != config['model_name']:
        raise ValueError('unknown model')
    prompt = obj.get('prompt')
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 5000:
        raise ValueError('prompt must be a nonempty string of at most 5000 characters')
    if obj.get('n', 1) != 1 or isinstance(obj.get('n', 1), bool):
        raise ValueError('only n=1 is supported')
    if obj.get('response_format', 'b64_json') != 'b64_json':
        raise ValueError('only response_format=b64_json is supported')
    defaults = config['defaults']
    size = obj.get('size', f"{defaults['width']}x{defaults['height']}")
    if not isinstance(size, str) or size.count('x') != 1:
        raise ValueError('size must look like 1024x1024')
    parts = size.split('x')
    if not all(p.isdigit() and p for p in parts):
        raise ValueError('size must look like 1024x1024')
    width, height = map(int, parts)
    if not (512 <= width <= 1024 and 512 <= height <= 1024 and width % 16 == 0 and height % 16 == 0):
        raise ValueError('size must be 512..1024 pixels on each side and divisible by 16')
    steps = obj.get('steps', defaults['steps'])
    if isinstance(steps, bool) or not isinstance(steps, int) or not (1 <= steps <= 20):
        raise ValueError('steps must be an integer between 1 and 20')
    scale = obj.get('sidecar_scale', defaults['sidecar_scale'])
    if isinstance(scale, bool) or not isinstance(scale, (float, int)) or not math.isfinite(scale) or not (0.0 <= scale <= 2.0):
        raise ValueError('sidecar_scale must be finite and between 0.0 and 2.0')
    seed = obj.get('seed')
    if seed is None:
        seed = secrets.randbits(32)
    elif isinstance(seed, bool) or not isinstance(seed, int) or not (0 <= seed < 2**63):
        raise ValueError('seed must be a nonnegative integer below 2^63')
    # Refuse unsupported settings instead of silently producing a different result.
    unsupported = set(obj) - {'model', 'prompt', 'n', 'response_format', 'size', 'steps', 'sidecar_scale', 'seed'}
    if unsupported:
        raise ValueError('unsupported fields: ' + ', '.join(sorted(unsupported)))
    return {'prompt': prompt, 'width': width, 'height': height,
            'steps': steps, 'seed': seed, 'scale': float(scale)}


def build_handler(runtime, config: dict, token: str):
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, status: int, payload: dict) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(data)

        def authorized(self) -> bool:
            supplied = self.headers.get('Authorization', '')
            if not hmac.compare_digest(supplied, 'Bearer ' + token):
                self.send_json(401, {'error': {'message': 'unauthorized', 'type': 'authentication_error'}})
                return False
            return True

        def do_GET(self) -> None:
            if not self.authorized():
                return
            if self.path == '/health':
                self.send_json(200, {'status': 'ready', 'model': config['model_name'],
                                     'gpu': 'cuda', 'busy': lock.locked()})
            elif self.path == '/v1/models':
                self.send_json(200, {'object': 'list', 'data': [{
                    'id': config['model_name'], 'object': 'model', 'owned_by': 'local',
                }]})
            else:
                self.send_json(404, {'error': {'message': 'not found'}})

        def do_POST(self) -> None:
            if not self.authorized():
                return
            if self.path != '/v1/images/generations':
                self.send_json(404, {'error': {'message': 'not found'}})
                return
            try:
                length_str = self.headers.get('Content-Length', '')
                if not length_str.isdecimal() or not (1 <= int(length_str) <= 16384):
                    raise ValueError('Content-Length must be 1..16384 bytes')
                body = self.rfile.read(int(length_str))
                data = json.loads(body.decode('utf-8'))
                request = parse_generation_request(data, config)
            except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
                self.send_json(400, {'error': {'message': str(exc), 'type': 'invalid_request_error'}})
                return
            if not lock.acquire(blocking=False):
                self.send_json(429, {'error': {'message': 'GPU busy; retry later', 'type': 'rate_limit_error'}})
                return
            try:
                png = runtime.generate(**request)
                self.send_json(200, {
                    'created': int(time.time()),
                    'data': [{'b64_json': base64.b64encode(png).decode('ascii')}],
                    'seed': request['seed'],
                })
            except Exception:
                self.log_error('generation failed: %s', sys.exc_info()[1])
                import traceback
                traceback.print_exc()
                self.send_json(500, {'error': {'message': 'generation failed; check server logs',
                                               'type': 'server_error'}})
            finally:
                lock.release()

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.json'))
    parser.add_argument('--snapshot', type=Path, default=Path(os.environ.get('SIDECAR_CLEAN_SNAPSHOT', '')))
    args = parser.parse_args()
    token = os.environ.get('SIDECAR_API_KEY', '')
    if not token:
        parser.error('SIDECAR_API_KEY is required')
    if not os.environ.get('SIDECAR_CLEAN_SNAPSHOT') and str(args.snapshot) in ('', '.'):
        parser.error('--snapshot or SIDECAR_CLEAN_SNAPSHOT is required')
    config = json.loads(args.config.read_text(encoding='utf-8'))
    from runtime import SidecarRuntime
    runtime = SidecarRuntime(config, args.snapshot)
    port = int(os.environ.get('PORT', '8000'))
    httpd = ThreadingHTTPServer(('0.0.0.0', port), build_handler(runtime, config, token))
    print(f'[http] listening on 0.0.0.0:{port}', flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        runtime.sidecar.detach()


if __name__ == '__main__':
    main()
