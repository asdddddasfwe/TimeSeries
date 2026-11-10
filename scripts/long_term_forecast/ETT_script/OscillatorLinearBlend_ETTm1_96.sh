#!/usr/bin/env bash
set -euo pipefail

# Activate TSLib first. This evaluates separately fitted components; no retraining.
artifact=${ARTIFACT:?Set ARTIFACT to the packaged candidate directory}
run_tag=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)_$$}
split=${SPLIT:-val}

python -u scripts/evaluate_frozen_timemix.py \
  --artifact "$artifact" \
  --split "$split" \
  --device "${DEVICE:-cpu}" \
  --output "tmp/lag_study_timemix_blend_${split}_${run_tag}"
