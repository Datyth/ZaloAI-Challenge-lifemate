#!/usr/bin/env bash
# Build, check and export the submission image (guide sections 4, 10, 11 and 8). Needs Docker access (sudo or the docker group)
# and an NVIDIA container runtime. Run from anywhere:
#   TEAM=<team> bash docker/build.sh build      # image zac2026_lifemate_<team>:v1
#   TEAM=<team> bash docker/build.sh test       # predict.sh on data/train+public_test/public_test, no network -> result_docker/
#   TEAM=<team> bash docker/build.sh export     # zac2026_<team>.tar.gz + .md5 (upload to Google Drive, "anyone with the link")
# `build` streams only the code, the base LLM and outputs/release/$RELEASE_NAME, with symlinks resolved (tar -h), so the
# image holds real weights even when models/qwen3.5-4b is a link and data/ or other outputs never enter the context.
set -euo pipefail
cd "$(dirname "$0")/.."
source configs/release.sh
TEAM=${TEAM:?set TEAM=<team name>}
IMAGE=zac2026_lifemate_${TEAM,,}:v1
DOCKER=${DOCKER:-docker}
case "${1:-}" in
  build)
    for f in models/qwen3.5-4b/config.json "outputs/release/$RELEASE_NAME/$ASR_DIRNAME/tokens.txt" \
             "outputs/release/$RELEASE_NAME/$LORA_DIRNAME/adapter_config.json"; do
      [ -e "$f" ] || { echo "missing $f (scripts/setup/get_models.sh, scripts/setup/get_checkpoint.sh)" >&2; exit 1; }
    done
    tar -ch --exclude=__pycache__ --exclude='*.pyc' .dockerignore configs docker scripts src synth pyproject.toml README.md \
        models/qwen3.5-4b "outputs/release/$RELEASE_NAME" \
      | $DOCKER build --platform linux/amd64 -f docker/Dockerfile --build-arg RELEASE="outputs/release/$RELEASE_NAME" -t "$IMAGE" -
    ;;
  test)
    split=${DATA:-$PWD/data/train+public_test/public_test}    # /data holds exactly one split, as at scoring
    mkdir -p result_docker
    $DOCKER run --rm --gpus '"device=0"' --network none -v "$(realpath "$split")":/data/$(basename "$split"):ro \
      -v "$PWD/result_docker":/result "$IMAGE" bash /code/predict.sh
    echo "same calls as the local run? .venv/bin/python scripts/infer/compare_submissions.py result_docker/submission.jsonl result/<file>"
    ;;
  export)
    $DOCKER save "$IMAGE" | gzip > "zac2026_${TEAM}.tar.gz"
    md5sum "zac2026_${TEAM}.tar.gz" | tee "zac2026_${TEAM}.tar.gz.md5"
    ;;
  *) echo "usage: TEAM=<team> bash docker/build.sh build|test|export" >&2; exit 2 ;;
esac
