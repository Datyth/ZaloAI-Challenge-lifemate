#!/usr/bin/env bash
# Publish outputs/release/$RELEASE_NAME (built by scripts/train/build_release.sh) to the Hugging Face model repo that
# training_code/README.md points to. The guide forbids changing it after the Docker deadline, so upload the final folder once.
#   hf auth login && bash scripts/setup/upload_release.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
source configs/release.sh
HF="${HF:-uvx --from huggingface_hub[cli]==0.36.0 hf}"
DEST=outputs/release/$RELEASE_NAME
(cd "$DEST" && sha256sum --quiet -c SHA256SUMS)
$HF repo create "$RELEASE_HF_REPO" --repo-type model ${PRIVATE:+--private} -y 2>/dev/null || true
$HF upload "$RELEASE_HF_REPO" "$DEST" . --repo-type model --commit-message "LifeMate release $RELEASE_NAME"
