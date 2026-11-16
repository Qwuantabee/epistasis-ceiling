#!/bin/bash
# Full ESM-2 re-run for reviewer F7. Long job -> systemd-run --user, per rails.
# Writes both the json artefact and a log, because a --collect unit that dies
# silently leaves no evidence of why.
set -uo pipefail
LOG=/home/ubuntu/biotech-portfolio/epistasis-ceiling/results/esm_full.log
cd /home/ubuntu/biotech-portfolio/epistasis-ceiling || exit 1
{
  echo "start $(date -Iseconds)"
  /home/ubuntu/.openclaw/workspace/venv/bin/python \
    scripts/esm_context.py \
    --index /tmp/nd2/data/DMS_substitutions.csv \
    --max-assays 24 \
    --max-per-assay 1500 \
    --out /home/ubuntu/biotech-portfolio/epistasis-ceiling/results/esm_context.json
  echo "exit=$? at $(date -Iseconds)"
} >> "$LOG" 2>&1
