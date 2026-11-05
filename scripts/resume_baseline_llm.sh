#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

model_dir="models/qwen3-4b-instruct-2507"
revision="cdbee75f17c01a7cc42f958dc650907174af0554"

resume_shard() {
  local shard_name="$1" expected_size="$2" expected_sha="$3" partial="$4"
  local target="$model_dir/$shard_name"
  local source="https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/resolve/$revision/$shard_name"
  if [[ -f "$target" ]]; then
    printf '%s  %s\n' "$expected_sha" "$target" | sha256sum --check --status
    echo "verified existing $shard_name"
    return
  fi
  if [[ ! -f "$partial" ]]; then
    echo "Missing saved partial: $partial" >&2
    exit 1
  fi
  for attempt in {1..10}; do
    local current_size
    current_size="$(stat -c '%s' "$partial")"
    if (( current_size >= expected_size )); then
      break
    fi
    echo "resuming $shard_name from $current_size / $expected_size bytes (attempt $attempt)"
    curl --silent --show-error --fail --location --continue-at - --output "$partial" --retry 3 --retry-delay 5 "$source" || true
  done
  if [[ "$(stat -c '%s' "$partial")" != "$expected_size" ]]; then
    echo "Incomplete $shard_name after retries" >&2
    exit 1
  fi
  printf '%s  %s\n' "$expected_sha" "$partial" | sha256sum --check
  mv "$partial" "$target"
  echo "verified $target"
}

resume_shard "model-00001-of-00003.safetensors" 3957900840 \
  "75311d91bb08cf0b882913da464a1e722a31fb44db35208663487efb7a3d8ed6" \
  "$model_dir/.cache/huggingface/download/Y6g195DXbsCiN_Ka1NwH6jyLkvc=.75311d91bb08cf0b882913da464a1e722a31fb44db35208663487efb7a3d8ed6.6fd3b91f.incomplete"

resume_shard "model-00002-of-00003.safetensors" 3987450520 \
  "0b48adbb1f60e901153d91907ba11ce63bd4b8b584482e730f48808d055dfba1" \
  "$model_dir/.cache/huggingface/download/D5-QTaCbl9VUFv9Wr0C28eFewRc=.0b48adbb1f60e901153d91907ba11ce63bd4b8b584482e730f48808d055dfba1.c9f35849.incomplete"

# Remove only orphaned partials created by the interrupted download attempts.
find "$model_dir/.cache/huggingface/download" -maxdepth 1 -type f -name '*.incomplete' -delete
