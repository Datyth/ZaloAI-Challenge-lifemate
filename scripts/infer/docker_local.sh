#!/usr/bin/env bash
# The image's prediction code (lifemate.docker_predict) outside Docker, with the settings docker/flags.sh exports, against the
# local server of scripts/infer/serve.sh: the same code path as /code/predict.sh, for checks on a machine without Docker.
#   bash scripts/infer/serve.sh                                   # terminal 1
#   bash scripts/infer/docker_local.sh public_test outputs/docker_local [--limit 20]
# Like /data at scoring, the --data folder holds exactly one split (a link to data/train+public_test/<split>).
set -euo pipefail
cd "$(dirname "$0")/../.."
source configs/release.sh
split=$1; out=$2; shift 2
stage=$(mktemp -d)
ln -s "$PWD/data/train+public_test/$split" "$stage/$split"
export LIFEMATE_SERVED_MODEL=$LORA_NAME LIFEMATE_ASR_DIR=outputs/release/$RELEASE_NAME/$ASR_DIRNAME LIFEMATE_ASR_THREADS=8 \
       LIFEMATE_DEADLINE=27 LIFEMATE_ORCHESTRATION=$AGENT_ORCHESTRATION LIFEMATE_RETRY_CAP=$AGENT_RETRY_CAP \
       LIFEMATE_MAX_TURNS=$AGENT_MAX_TURNS LIFEMATE_ENSEMBLE=$ENSEMBLE
PYTHONPATH=src .venv/bin/python -m lifemate.docker_predict --data "$stage" --out "$out" "$@"
rm -rf "$stage"
