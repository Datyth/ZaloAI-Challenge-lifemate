#!/usr/bin/env bash
set -euo pipefail
pid="${1:-}"
for ((attempt=1; attempt<=120; attempt++)); do
  if curl -fsS --max-time 3 http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q 'lifemate_sft'; then
    echo "S1 LoRA server ready"
    exit 0
  fi
  if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
    echo "S1 LoRA server process $pid exited before readiness" >&2
    exit 1
  fi
  sleep 5
done
echo "S1 LoRA server did not become ready within 10 minutes" >&2
exit 1
