#!/usr/bin/env bash
# Deploy the torch-free package to Azure App Service on a Free (F1) Linux plan.
# Creates the resource group / plan / web app if missing, then zip-deploys build/blurring-system-azure.zip.
# Usage: SUBSCRIPTION=<id> ./deploy/azure/deploy.sh      (other variables below are optional overrides)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

SUBSCRIPTION="${SUBSCRIPTION:?set SUBSCRIPTION to the Azure subscription id}"
LOCATION="${LOCATION:-centralindia}"
RESOURCE_GROUP="${RESOURCE_GROUP:-rg-blurring-system}"
PLAN="${PLAN:-plan-blurring-system}"
PLAN_RESOURCE_GROUP="${PLAN_RESOURCE_GROUP:-$RESOURCE_GROUP}"   # point at an existing plan's RG to reuse it
APP="${APP:-blurring-system}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
ZIP="$ROOT/build/blurring-system-azure.zip"

[ -f "$ZIP" ] || { echo "$ZIP missing: run deploy/azure/build_package.sh" >&2; exit 1; }
az account set --subscription "$SUBSCRIPTION"

az group show -n "$RESOURCE_GROUP" >/dev/null 2>&1 || az group create -n "$RESOURCE_GROUP" -l "$LOCATION" -o none
az appservice plan show -g "$PLAN_RESOURCE_GROUP" -n "$PLAN" >/dev/null 2>&1 ||
  az appservice plan create -g "$PLAN_RESOURCE_GROUP" -n "$PLAN" -l "$LOCATION" --sku F1 --is-linux -o none
PLAN_ID="$(az appservice plan show -g "$PLAN_RESOURCE_GROUP" -n "$PLAN" --query id -o tsv)"

az webapp show -g "$RESOURCE_GROUP" -n "$APP" >/dev/null 2>&1 ||
  az webapp create -g "$RESOURCE_GROUP" -n "$APP" -p "$PLAN_ID" --runtime "PYTHON:$PYTHON_VERSION" -o none

az webapp config set -g "$RESOURCE_GROUP" -n "$APP" -o none \
  --startup-file "python -m uvicorn blurring.serving.asgi:app --app-dir src --host 0.0.0.0 --port 8000"
az webapp config appsettings set -g "$RESOURCE_GROUP" -n "$APP" -o none --settings \
  SCM_DO_BUILD_DURING_DEPLOYMENT=true \
  BLURRING_CONFIG=configs/azure.yaml \
  WEBSITES_PORT=8000 \
  MALLOC_ARENA_MAX=2 \
  OMP_NUM_THREADS=1

az webapp deploy -g "$RESOURCE_GROUP" -n "$APP" --src-path "$ZIP" --type zip --async false -o none
echo "Deployed: https://$(az webapp show -g "$RESOURCE_GROUP" -n "$APP" --query defaultHostName -o tsv)"
