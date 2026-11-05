# Vietnamese ASR benchmark on TRAIN

The benchmark used the existing LifeMate `benchmark-asr` interface and the frozen 200 TRAIN validation IDs in `outputs/splits/validation_ids.json`. It compared the preserved Qwen3-ASR-0.6B run with two downloaded checkpoints: [ChunkFormer-RNNT-Large-Vie](https://huggingface.co/khanhld/chunkformer-rnnt-large-vie) at revision `9863592e00d6c02634c4ce2db8a4f3414f1ba7d6`, and [Gipformer 1.5](https://huggingface.co/g-group-ai-lab/gipformer1.5-68M-rnnt) at revision `dd9227dcd8705c13f33bdbe59728d546ab94480f`. Checkpoint hashes and licenses are recorded in each local `models/<model>/lifemate_manifest.json`. No model weights were changed.

## Measured result

All three runs contain the same 200 sample IDs in the same order. Both new runs completed 200/200 inferences with zero failures on physical GPU 2. WER and CER are corpus error rates after the project's lowercase Unicode normalization; numbers are not verbalized. A critical slot is counted only when a gold tool argument value appears verbatim in the reference transcript. This covers 64 eligible values, so slot accuracy is a limited lexical proxy rather than downstream task accuracy. Latency excludes model load and includes the first inference warmup. The Qwen run was measured earlier, making latency comparisons indicative.

| Model | WER | CER | Critical slots | Mean / P95 inference latency |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-ASR-0.6B | 33.89% | 23.24% | 27/64 (42.19%) | 1.105 / 1.721 s |
| ChunkFormer-RNNT-Large-Vie | 22.15% | 16.02% | 43/64 (67.19%) | 0.237 / 0.399 s |
| Gipformer 1.5 | **17.60%** | **12.51%** | **45/64 (70.31%)** | 0.279 / 0.427 s |

Gipformer 1.5 performs best by WER, CER, and this critical-slot measure. ChunkFormer had the lowest measured mean latency among the three. Gipformer has an accompanying PyTorch `model.pt` and [research training code](https://github.com/ggroup-ai-lab/gipformer), so it is a technical candidate for fine-tuning; the benchmark does not establish whether fine-tuning improves LifeMate tool calls. No fine-tuning was run.

The 20 clips shorter than four seconds had WER 36.5% for Qwen, 19.4% for ChunkFormer, and 18.0% for Gipformer. Their slot accuracies were 1/4, 3/4, and 3/4 respectively. The TRAIN records contain no regional accent, noise, or truncated-utterance labels, so those groups cannot be measured honestly from this split. No clip in the frozen 200 is shorter than two seconds.

## Artifacts and rerun

- Qwen: `outputs/runs/baseline_asr_200/`
- ChunkFormer: `outputs/runs/asr_compare_final_20261105_chunkformer/`
- Gipformer: `outputs/runs/asr_compare_final_20261105_gipformer/`
- Aligned comparison: `outputs/reports/asr_compare_final_20261105/`

Each new model directory contains `asr_outputs.jsonl` (raw transcripts), `per_sample.jsonl` (reference, transcript, counts, latency, eligible slots and slices), `errors.jsonl` (empty because inference had no failures), `metrics.json`, and `run_config.json`. The comparison directory contains `comparison.json`, `per_sample_comparison.jsonl`, and `report.md`.

To run again in the existing Conda environment, choose an unused prefix:

```bash
cd /home/datpt/projects/ZaloAI-Challenge-lifemate
source /home/shared/miniconda3/etc/profile.d/conda.sh
conda activate datpt_lifemate
nohup bash scripts/run_asr_comparison.sh cuda:2 asr_compare_manual_01 > outputs/asr_compare_manual_01.log 2>&1 &
echo "ASR benchmark PID: $!"
```

The script selects physical GPU 2, runs ChunkFormer and Gipformer sequentially, and compares both with the preserved Qwen run. It will not overwrite an incomplete run directory. The model files are already present locally under `models/`. The active environment has `chunkformer==1.2.2`, `sherpa-onnx==2.0.1+cuda12.cudnn9.onnxruntime1.28.2`, `jiwer==4.0.0`, and `colorama==0.4.6` installed.
