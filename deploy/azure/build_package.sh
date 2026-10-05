#!/usr/bin/env bash
# Build the torch-free Azure App Service package: build/azure/ and build/blurring-system-azure.zip.
# Needs: artifacts/model.onnx (`uv run blurring export-onnx`) and Node for the UI build.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$ROOT/build/azure"

[ -f "$ROOT/artifacts/model.onnx" ] || { echo "artifacts/model.onnx missing: run 'uv run blurring export-onnx'" >&2; exit 1; }

(cd "$ROOT/web" && npm ci --no-audit --no-fund && npm run build)

rm -rf "$OUT" && mkdir -p "$OUT/src" "$OUT/configs" "$OUT/artifacts" "$OUT/web"
rsync -a --exclude "__pycache__" --exclude "*.md" "$ROOT/src/blurring" "$OUT/src/"
cp "$ROOT/configs/azure.yaml" "$OUT/configs/"
cp "$ROOT/artifacts/model.onnx" "$ROOT/artifacts/model.json" "$OUT/artifacts/"
cp -R "$ROOT/web/dist" "$OUT/web/dist"
cp "$ROOT/deploy/azure/requirements.txt" "$OUT/requirements.txt"
# metrics shown in the UI's model panel (optional)
[ -f "$ROOT/artifacts/metrics_test.json" ] && cp "$ROOT/artifacts/metrics_test.json" "$OUT/artifacts/"
[ -f "$ROOT/artifacts/train_history.json" ] && cp "$ROOT/artifacts/train_history.json" "$OUT/artifacts/"

(cd "$OUT" && rm -f ../blurring-system-azure.zip && zip -qr ../blurring-system-azure.zip .)
du -sh "$OUT" "$ROOT/build/blurring-system-azure.zip"
