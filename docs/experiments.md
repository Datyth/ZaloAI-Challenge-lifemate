# Nhật ký thí nghiệm và quyết định (04–10/11/2026)

Gộp từ các báo cáo cũ (orchestration, ASR benchmark, SFT, Gipformer SFT, cải tiến 07/11, nghiên cứu 08/11). Phân tích lỗi chi
tiết của bản tốt nhất ở [error_analysis.md](error_analysis.md); luật thi ở [competition.md](competition.md).

**Cách đo.**
- Val là 200 câu TRAIN cố định (`configs/splits/validation_ids.json`, `scripts/data/make_validation_split.py`: 150 ngẫu nhiên +
  50 khó, seed 42). Các câu này và mọi câu dùng chung audio lịch sử với chúng bị loại khỏi mọi dữ liệu train.
- Từ 07/11, điểm val là điểm local của `analysis/score.py` (hash là cổng, rồi tính F1 của call), thang 0–100. Trước đó các báo
  cáo dùng số câu khớp state và "exact-call F1", không so trực tiếp được với điểm này.
- Chênh lệch dưới khoảng 1,5 điểm trên val là nhiễu (xem mục Tính tất định).

## Bản 08/11 (public 58,02); bản release ở mục 7

Gipformer ASR v2 (CPU, modified beam search, beam 4, chữ thường) → Qwen3.5-4B + LoRA `q35_e1` trong vLLM → vòng agent E1,
`--dedup-calls`, tối đa 10 lượt. Val: **70,91** với ASR v2, **82,33** với transcript chuẩn. Cấu hình nằm ở `configs/release.sh`.

| Mốc | Val | Public |
|---|---:|---:|
| S1 (Qwen3-4B LoRA) + Gipformer gốc | 36,4 | 30,77 |
| + viết thường transcript | 44,0 | 38,80 |
| S2 + ASR v2 | 58,7 | 47,4 |
| S1c + ASR v2 + dedup | 60,0 | 48,18 |
| **q35_e1 (Qwen3.5-4B) + ASR v2 + dedup** | **70,91** | **58,02** |
| q35 + dữ liệu app tổng hợp | 69,13 | 56,69 |

## 1. Orchestration (04–05/11, Qwen3-ASR-0.6B + Qwen3-4B-Instruct-2507, chưa SFT)

Các chế độ E0, E2, E3, E6, O1–O5 và S2 dưới đây đã gỡ khỏi `src/lifemate/agent.py` ngày 10/11 (còn trong lịch sử git); code
hiện giữ E1, E1D và E1P.

- **E0–E3** trên 200 câu val, cùng checkpoint. Kết quả: khớp state E0 49, **E1 52**, E2 52 (4 câu lỗi), E3 51. F1 E0 0,143,
  **E1 0,240**.
  - E1 thêm lịch sử có cấu trúc, chỉ đọc: lượt, call, kết quả, cờ verified, ID đã thấy. Câu rỗng ở nhóm lịch sử từ 12 lượt giảm
    từ 33/34 xuống 1/34, và câu "kể" tool call bằng chữ giảm từ 73 xuống 6.
  - **Chọn E1** làm mặc định.
- **E4–E6.**
  - E4 (transcript chuẩn) được 88/200, cho thấy ASR là nút cổ chai.
  - E6 (kiểm tra đọc → ghi) +0. Không dùng.
- **O1–O5** (sửa giao thức, sổ ghi call, grounding x-ref theo collection, prompt tổng quát) với transcript chuẩn.
  - O0/E1 89/200, O5 90/200. Trên holdout 50 câu thì bằng nhau, và p95 tăng 9,8 → 19,1 s.
  - Giữ E1. Các mode này còn lại trong `lifemate.agent` để so sánh.

## 2. ASR

- **Benchmark 200 câu val** (WER sau chuẩn hoá chữ thường):

  | Model | WER |
  |---|---:|
  | Qwen3-ASR-0.6B | 33,89% |
  | ChunkFormer-RNNT-Large-Vie | 22,15% |
  | **Gipformer 1.5 (68M RNN-T)** | **17,60%** |

  Chọn Gipformer (sherpa-onnx, có code train icefall).
- **Gipformer SFT run_01 hỏng**: WER 99%. Nguyên nhân là 48,8% token BPE của target là `<unk>`: bộ BPE gốc viết hoa, không dấu
  câu, còn target đưa vào là chữ thường có dấu câu.
- **run_02** (target đã sửa): WER 14,6%, khớp state 102 so với 93. Bị loại vì số câu lỗi tăng từ 7 lên 11.
- **Viết thường transcript trước agent**: +7,6 val, +8,0 public. LoRA được train trên `audio_text` viết thường.
- **ASR v2** (`scripts/train/train_gipformer.py`, 07/11). Đây là ASR đang dùng (`gipformer-lm-v2-01`).
  - Target viết hoa, bỏ dấu câu. Đoạn chữ số được thay bằng cách đọc mà Gipformer gốc nghe được (`scripts/train/asr_targets.py`).
  - Train toàn bộ model: lr 2e-5, tích luỹ 4 bước, 2 epoch, 3.644 audio.
  - WER 17,05% → **10,74%**. "dời" 2/14 → 13/14, SJC 7/18 → 18/18.
  - Bản 4 epoch, lr 3e-5 (v2-02) được 10,98%, không tốt hơn.
- **Hotword** (sherpa-onnx biasing):
  - +2,2 val với ASR gốc;
  - với ASR v2 thì chèn sai nhiều (WER 11,09%), nên **không dùng**.
- **ITN** (số đọc → chữ số): +0,6, trong mức nhiễu. Không dùng; code đã gỡ khỏi repo ngày 10/11 (còn trong lịch sử git).
- **ASR trên CPU** (8 luồng): 0,07 s mỗi file, transcript trùng bản GPU 150/150. Docker chạy ASR trên CPU, GPU dành cho vLLM.

## 3. SFT LLM

- **S1** (v1): QLoRA Qwen3-4B, mỗi quyết định là một mẫu, max_len 8192. Cách này bỏ 308/3.589 quyết định dài.
  - Transcript chuẩn 71,4 (model gốc 28,6).
  - Gipformer thô 36,4.
- **S2** (`scripts/data/prepare_agent_sft.py`, orchestration S2):
  - mỗi câu có hai bản, transcript chuẩn và Gipformer viết thường, tổng 2.396 quỹ đạo;
  - cả quỹ đạo là một mẫu, max_len 16384, Unsloth QLoRA r16, lr 1e-4, 2 epoch;
  - val 58,7 với ASR v2; với Gipformer gốc +6,9 so với S1.
- **S1c**: học tiếp từ S1, prompt E1, cùng dữ liệu hai bản. Val 60,0. Với `--dedup-calls` (không chạy lại call giống hệt; dừng
  sau 3 lần lặp), public 48,18.
- **q35_e1** (08/11), bản đang dùng.
  - Qwen3.5-4B (`--language-model-only`, non-thinking, `LIFEMATE_MERGE_SYSTEM=1`) QLoRA trên đúng dữ liệu S1c (`data_v2_e1`).
  - 2 epoch, 600 bước, 3 giờ 15 phút.
  - Zero-shot Qwen3.5 đã được 40,0 so với 28,6 của Qwen3-4B.
  - Val 70,91 (S1c 59,71): 46 câu tốt hơn, 27 kém hơn. Public 58,02.
  - Epoch 2 có lợi: checkpoint 300 được 63,2, bản cuối 70,9.

## 4. App chưa thấy (LAO: giấu 6 app train khỏi dữ liệu SFT)

- **Qwen3-4B `lao_base`**: app quen 67,7, app lạ 49,8, chênh 17,9 điểm. Model gốc làm hai nhóm như nhau (27,7 / 29,9).
- **Đổi tên namespace app khi train** (`--rename-apps`): −1,1 trên app lạ. Bỏ.
- **Qwen3.5 `q35_lao_base`**: 76,4 / **65,3**, chênh còn 11,1. Qwen3.5 thu hẹp phần lớn khoảng cách.
- **`--ground-args`** (ID không x-ref phải có căn cứ; bỏ tham số tùy chọn chỉ chép từ context):
  - +1,2 trên app lạ, 0 trên val;
  - public 48,15 so với 48,18;
  - không bật trong release; code đã gỡ ngày 10/11 (còn trong lịch sử git).
- **Luật "số không căn cứ thì hỏi lại"**: +0,2. Không dùng, vì nhiều giá trị đúng là kết quả tính
  toán chứ không chép nguyên văn.
- **Dữ liệu app tổng hợp** (`synth/`, `scripts/data/build_synth_split.py`; 27 app, 164 task, không trùng miền với app chỉ có ở
  public):
  - public 56,69 so với 58,02;
  - làm lệch cân bằng hỏi lại / làm luôn: tỷ lệ hỏi 12% → 8%;
  - không dùng. Nếu làm lại, quy ước hỏi lại và đọc trước khi ghi phải lấy từ thống kê nhãn train.

## 5. Tính tất định và thời gian

- Temperature 0, seed 42. Hai lần chạy đơn luồng trên 50 câu val: 0/50 dòng khác khi bật prefix caching, 2/50 khi tắt. Giữ bật.
- Chạy nhiều shard song song trên một server làm kết quả dao động. Chạy lại public từng cho 55/634 dòng khác.
- Thời gian đơn luồng trên RTX PRO 4000: trung bình 3,2 s, p95 6,0 s, lâu nhất 10,4 s mỗi câu, chưa tính ASR. Chưa câu nào
  chạm 16k token; phần lớn prompt là schema tool.
- `VLLM_BATCH_INVARIANT` cần compute capability ≥ 9.0, mà A30 là 8.0.

## 6. Hướng còn mở (theo error_analysis.md mục 9)

1. **ASR theo ngữ cảnh** (11,4 điểm val): giải mã lại với hotword lấy từ danh bạ, lịch sử và kết quả tool; fine-tune bằng audio
   TTS cho tên riêng và số.
2. **Suy luận ngắn trước call ghi** cho điều kiện, tính toán và ngày giờ (7,1 điểm).
3. **Bản cuối train trên toàn TRAIN** (thêm 200 câu val), thử 3 epoch bằng giao thức LAO trước.
4. **Không làm**: thêm luật hậu xử lý, hotword toàn cục với ASR v2, đổi tên app, thinking mode khi chưa có speculative decoding,
   train ít epoch hơn.

## 7. Từ 09/11 đến bản release (public 61,26)

Báo cáo chi tiết: [training_audit_2026-11-09.md](training_audit_2026-11-09.md),
[research_bottlenecks_2026-11-09.md](research_bottlenecks_2026-11-09.md).

- **Prompt E1D, E1P** (`src/lifemate/agent.py`). E1 không cho thứ trong tuần: trên public 5/13 câu "thứ X tuần sau" ra sai ngày.
  E1D thêm bảng 15 ngày có tên thứ (model gốc với transcript chuẩn: val 40,0 → 45,1). E1P thêm tên tiếng Việt và mô tả của
  từng app (lấy từ catalog) cùng 4 gợi ý chung khi tìm kiếm ra rỗng, viết bằng app của TRAIN.
- **Dữ liệu recovery** (`scripts/data/inject_recovery.py`): nhãn TRAIN gần như không có lần tìm ra rỗng, nên model chưa từng
  học bước sửa. 92 quỹ đạo chèn một lần tìm sai (danh xưng, tên nghe sai, từ khoá quá dài) đã chạy trong ToolEnv và ra rỗng,
  không tính loss, rồi tới lần tìm đúng.
- **Loss có trọng số** (`--fuzzy-weight 0.3`): trường chữ tự do (x-match fuzzy) chiếm 4–6% token được train nhưng 42–66% loss
  trên câu chưa thấy (`analysis/token_loss.py`).
- **Retry cap** (`--retry-cap 2`): sau recovery, model có lúc tìm mãi (một câu đặt vé: 10 lời gọi); chặn tool đọc đã ra rỗng 2 lần.
- **p1** = toàn bộ task + E1P + recovery x2 + trọng số 0,3: val 69,55 (q35_e1: 70,91), public 57,65 (58,02).
- **ASR v3**: nhãn số viết theo cách đọc (giữ 368/368 câu có số, v2 bỏ 112), augmentation tốc độ / nhiễu / SpecAugment, 3 epoch:
  WER 8,66 → 7,74, val end to end 70,91 → 71,87. **ASR v4** thêm 1.202 câu TTS (VieNeu, giọng nhân bản từ audio TRAIN, tên và số
  mới): nghe đúng thêm cụm từ khoá (155 → 159/221) nhưng end to end 68,64, nên bản cuối **không dùng TTS**.
- **Bản cuối train trên toàn bộ dữ liệu** (kể cả 200 câu val): ASR final (5.036 recording) và adapter p1, e1, synth
  (`scripts/data/build_final_data.sh`, `scripts/train/train_final.sh`). Không còn val cho các model này.
  - **p1 + ASR final: public 61,26** (+3,24 so với 58,02). Đây là bản release.
  - Bỏ phiếu theo từng bước của p1, e1, synth trên một ToolEnv (`src/lifemate/ensemble.py`; trên val với các bản cũ: 70,91 →
    72,71): 90/634 câu có hash khác bản 61,26; file `result/submission_ens3full_asrfinal_20261110.jsonl`, chưa có điểm public.
- **Đường chạy của Docker** (`scripts/infer/docker_local.sh`, chính `lifemate.docker_predict`) trên 634 câu public: 618/634
  câu cùng trạng thái cuối với file 61,26 (xem bảng thời gian dưới).

### Thời gian mỗi mẫu

| 634 câu public, một luồng, một RTX PRO 4000 Blackwell (24 GB) | Giá trị |
| --- | ---: |
| Nạp model (vLLM khởi động tới sẵn sàng) | khoảng 45 s |
| Thời gian trung bình mỗi câu, gồm ASR | 2,88 s |
| Câu lâu nhất | 8,71 s |
| Số câu quá 30 s | 0 |
| Cùng trạng thái cuối với file 61,26 | 618 / 634 |

Giải mã greedy trong vLLM không trùng từng bit giữa các lượt: cách gộp batch (file 61,26 chạy 4 luồng song song) và trạng thái prefix
cache làm vài lựa chọn sát nút đổi đi. Hai lượt chạy một luồng trên cùng 100 câu public trùng 97 câu; ở 3 câu còn lại, model nhận
prompt y hệt nhưng chọn số tiền khác ở một lượt. Khi kết quả chạy lại khác, luật Track 1 lấy trung bình 5 lần chạy của BTC.
