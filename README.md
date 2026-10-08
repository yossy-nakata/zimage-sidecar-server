# Z-Image Canonical 512-d Sidecar Image Server

RTX 3090 24GB、既存の Python/uv/CUDA Docker 環境用。**Docker イメージを新しく build しません。** GitHub に置くコード、R2 に置く推論重み、Hugging Face の Clean Z-Image-Turbo を使います。

## 同梱ファイル

- `bootstrap.sh` — 既存 Docker 内でソース・推論重み・Clean Turbo を取得して起動
- `server.py` — Python 標準ライブラリのHTTPサーバー、Bearer認証、生成API
- `runtime.py` — モデル常駐・prompt encode・生成
- `sidecar.py` — 提供された比較用 standalone の IdentityProjector / IdentityAttentionResidual / IdentitySidecar を維持
- `config.json` — 検証済みの固定パラメータ、Sidecar step 6000

## 準備

1. この5ファイルを **自分のGitHubリポジトリ** に配置する（公開コード可、秘密鍵・重みは含めない）。
2. `bootstrap.sh` の `https://github.com/yossy-nakata/zimage-sidecar-server.git` の **`yossy-nakata` を実際のGitHub owner名に置換**する。リポジトリ名が異なるならそこも合わせて変更する。未作成のリポジトリは作成する必要がある。
3. R2の次の2ファイルを確認する（存在はまだ実機で未検証）。
   - `r2:nana-storage/zimage-sidecar/identity/character-v3.safetensors`
   - `r2:nana-storage/zimage-sidecar/runs/sidecar-n40-turbo-mix-v1/sidecar-step-006000.safetensors`
4. 既存Dockerには `git`, `rclone`, `/usr/local/bin/python`, `torch`, `safetensors`, `diffusers`, `transformers`, `huggingface_hub` が必要。**起動スクリプトでパッケージのインストールはしない。**
5. Podの機密環境変数に `HF_TOKEN`, `R2_KEY`, `R2_SECRET`, `R2_URL`, `SIDECAR_API_KEY` を設定する。APIキーは長いランダム文字列にする。公開ポートを割り当てるならTLS対応の入口を利用する。

## 自動起動

Pod の start command は、GitHub に push した後、例えば次の形にする（実際の owner に置換）。

```bash
curl -fsSL https://raw.githubusercontent.com/yossy-nakata/zimage-sidecar-server/main/bootstrap.sh | bash
```

既存コンテナ内に `/workspace/sidecar-server/server.py` が存在する場合、bootstrapはローカルコードをそのまま使うので、GitHub未公開の動作確認も可能です。

サーバーは `0.0.0.0:8000` でリクエストを待つ。ポートを変更する場合は `PORT` 環境変数を使う。

## 手動での動作確認

サーバー起動後、コンテナ内から：

```bash
curl -fsS -H "Authorization: Bearer $SIDECAR_API_KEY" http://127.0.0.1:8000/health
```

画像生成（返却値は Base64 JSON）：

```bash
curl -fsS http://127.0.0.1:8000/v1/images/generations \
  -H "Authorization: Bearer $SIDECAR_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"zimage-sidecar-n40-step6000","prompt":"realistic photo, front portrait, natural light","size":"1024x1024","steps":9,"seed":12664542,"sidecar_scale":0.8}' \
  -o /workspace/result.json
```

画像の抽出（標準ライブラリ）：

```bash
python - <<'PY'
import json, base64
from pathlib import Path
obj = json.loads(Path('/workspace/result.json').read_text())
Path('/workspace/result.png').write_bytes(base64.b64decode(obj['data'][0]['b64_json']))
print('saved: /workspace/result.png')
PY
```

## 実装上の注意

- **起動時1回だけロード**：Clean Turbo transformer + VAE + Sidecar step 6000 はGPU常駐、Text EncoderはCPU常駐。
- Text Encoderは比較スクリプトと同じ chat template / `enable_thinking=True` / `hidden_states[-2]` / 512トークン制限。CPU bf16のため**生成画像のpixel完全一致は未保証**。CPU上での実行時間や対応演算は現地GPU/CPUで検証が必要。
- `guidance_scale=0.0`、Sidecar rank=512、tokens=8、layers=5/15、layer scales=0.25/1.5 を維持。
- 同時生成は1件。生成中に追加リクエストが来るとHTTP `429`。プロンプトやサイズが不正なら `400`。認証失敗は `401`。
- 提供元スクリプトにない編集API、negative prompt、バッチ、複数人物対応は入れていない。
- `/health` と `/v1/models` も **Bearer認証必須**。
- これは **API/構文・契約テストまで** の初期パッケージ。Z-Image-TurboとSidecar実モデルを使った3090上の生成確認は未実施。初回実機検証でAPIへの出力とVRAM使用量、text encoding時間を測定する。
- `SIDECAR_API_KEY` をログや GitHub に出さないこと。APIはPodのTLS入口側から公開すること。
