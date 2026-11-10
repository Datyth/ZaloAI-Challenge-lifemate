# LifeMate private-test image, laid out as the organizers' guide asks (Docker_Guide_ZAC2026_LifeMate.pdf):
#   /code/predict.sh  /code/flags.sh  /code/README.md  /code/requirements.txt  /code/src/lifemate (inference code)
#   /code/models/{llm_base,llm-lora-p1,llm-lora-e1,llm-lora-synth,asr-gipformer}  /code/predict_notebook.ipynb  /code/start_jupyter.sh
#   /code/training_code
#   /code/configs/release.sh (model and agent settings shared with local runs; sourced by flags.sh)
# Build with docker/build.sh after scripts/setup/get_models.sh and get_checkpoint.sh: it streams a context with the symlinks
# resolved (models/qwen3.5-4b is often a link) holding only the code, the base LLM and the release checkpoints, so the image
# always matches the checkpoints scored locally.
# Scoring command (guide, section 10):
#   docker run --rm --gpus '"device=0"' --network none -v /path/to/data:/data:ro -v /path/to/result:/result zac2026_lifemate_<team>:v1 bash /code/predict.sh
# Base image: vLLM 0.32.0 (the version all runs were made with; it loads Qwen3.5 and runs on A30 and RTX 5090), CUDA <= 13.2.
FROM vllm/vllm-openai:v0.32.0@sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34
ENTRYPOINT []
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PIP_NO_CACHE_DIR=1 VLLM_NO_USAGE_STATS=1 TOKENIZERS_PARALLELISM=false \
    PYTHONHASHSEED=0 PYTHONPATH=/code/src TZ=Asia/Ho_Chi_Minh
WORKDIR /code
COPY docker/requirements.txt /code/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /code/requirements.txt
# weights first, so a code change does not invalidate their layers
# base LLM (weights used at inference with the LoRA adapters) and outputs/release/$RELEASE_NAME of configs/release.sh
ARG RELEASE=outputs/release/lifemate-final
COPY models/qwen3.5-4b/ /code/models/llm_base/
COPY ${RELEASE}/ /code/models/
COPY pyproject.toml README.md /code/training_code/
COPY configs/ /code/training_code/configs/
COPY scripts/ /code/training_code/scripts/
COPY synth/ /code/training_code/synth/
COPY src/ /code/training_code/src/
COPY configs/release.sh /code/configs/release.sh
COPY docker/README.md docker/predict.sh docker/flags.sh docker/start_jupyter.sh docker/predict_notebook.ipynb /code/
COPY src/lifemate /code/src/lifemate
RUN chmod +x /code/predict.sh /code/start_jupyter.sh && python3 -c "import sherpa_onnx, soundfile, jupyterlab, lifemate.docker_predict"
CMD ["bash", "/code/predict.sh"]
