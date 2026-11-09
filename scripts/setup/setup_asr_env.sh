#!/usr/bin/env bash
# Isolated icefall/k2 environment for Gipformer fine-tuning (torch 2.8 cu128 runs on Blackwell sm_120).
set -euo pipefail
cd "$(dirname "$0")/../.."
uv venv .venv-asr --python 3.12 -q
uv pip install --python .venv-asr/bin/python torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv-asr/bin/python "https://huggingface.co/csukuangfj2/k2/resolve/main/ubuntu-cuda/1.24.4.dev20260625/k2-1.24.4.dev20260625+cuda12.8.torch2.8.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl" "https://huggingface.co/csukuangfj/kaldifeat/resolve/main/cuda/1.25.5.dev20241029/linux/kaldifeat-1.25.5.dev20250807+cuda12.8.torch2.8.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
uv pip install --python .venv-asr/bin/python lhotse sentencepiece onnx onnxruntime "numpy<2.3" soundfile
mkdir -p models/.research
[ -d models/.research/icefall ] || git clone --depth 1 https://github.com/k2-fsa/icefall.git models/.research/icefall
.venv-asr/bin/python -c "import torch, k2, kaldifeat; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability(0), k2.__version__ if hasattr(k2,'__version__') else 'k2 ok')"
echo asr env ready
