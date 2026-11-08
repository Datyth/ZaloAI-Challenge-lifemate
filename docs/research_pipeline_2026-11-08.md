# LifeMate: đánh giá lại bài toán, đo pipeline, nghiên cứu và đề xuất (08/11/2026)

Tiếp nối [phân tích lỗi](error_analysis_2026-11-07.md) và [cải tiến 07/11](improvements_2026-11-07.md). Bản nộp tốt nhất hiện tại: **48,18** public (S1c + ASR v2 + `--dedup-calls`). Bản có luật `--ground-args` được 48,15, nghĩa là luật không có tác dụng trên public.

## 1. Bài toán, nhìn lại

### 1.1 Luật thi ảnh hưởng tới thiết kế

| Luật (Rule tab, cập nhật 03/11) | Hệ quả cho pipeline |
|---|---|
| Hash state là cổng, sau đó tính F1 của call; mọi call đã gọi đều bị chấm, kể cả call bị từ chối | Mỗi lệnh đọc thừa vẫn trừ precision. Không được "thử rồi làm lại". |
| Mỗi mẫu đúng **một** ToolEnv; cấm clone env, cấm `get_state` | Không thể chạy nhiều quỹ đạo rồi chọn. Muốn ensemble thì phải bỏ phiếu **trước** khi gọi tool, ở từng bước. |
| Giải mã tất định (greedy / nhiệt độ 0); nếu chạy lại ra khác thì lấy trung bình 5 lần chạy Docker | Self-consistency bằng sampling là rủi ro. Ensemble hợp lệ là nhiều prompt hoặc nhiều nguồn đầu vào, mỗi cái đều greedy. |
| ≤ 30 s mỗi mẫu trên 1×A30 (Docker guide lại ghi RTX 5090); LLM < 5B, tổng < 6B, ngữ cảnh ≤ 16.384 | Xem đo ở mục 2.1: hiện còn dư nhiều thời gian. |
| Model và dữ liệu phải ra trước 26/10/2026; **được dùng LLM/TTS ngoài để sinh dữ liệu tổng hợp trong lúc phát triển** | Mở đường cho dữ liệu app tổng hợp và audio TTS (mục 4). |
| Docker và mã train nộp trước 23:59 ngày 16/11; private test phát hành 20/11 | **Repo LifeMate chưa có Docker.** Đây là việc bắt buộc, phải làm trong tuần này. |

### 1.2 Lệch phân bố giữa các split: app chưa thấy là rủi ro số một

| | Train | Public |
|---|---:|---:|
| Số mẫu | 1.545 | 634 |
| App cài trung bình mỗi mẫu | 9,2 | 8,5 |
| Mẫu có lịch sử hội thoại | 66% | 66% |
| Mẫu có cài ít nhất một app không có trong train | 0 | **410 (65%)** |

- 25 app có ở cả hai split, 11 app chỉ có ở public, 21 app chỉ có ở train. Mỗi split có tập app riêng, nên test private gần như chắc chắn cũng có app chưa thấy.
- Nhãn train: 7% câu rỗng, 14% câu `sys.ask_missing_info`, trung bình 2,29 call và 1,12 lệnh ghi mỗi câu.

## 2. Đo pipeline hiện tại (S1c + ASR v2 + dedup)

### 2.1 Ngân sách token và thời gian

Đo từ log, cùng cấu hình bản 48,18. Chạy đơn luồng: 50 câu val, không chạy gì khác trên GPU (RTX PRO 4000 Blackwell).

| Chỉ số | Val 200 | Public 634 |
|---|---:|---:|
| Prompt lượt đầu (token), trung bình / p95 / max | 5.618 / 8.388 / 10.071 | 5.125 / 7.791 / 9.292 |
| Prompt lớn nhất mỗi mẫu, max | 10.605 | 10.077 |
| Lượt LLM mỗi mẫu, trung bình | 3,61 | 3,56 |
| Token sinh mỗi lượt, trung bình | 26 | 26 |

- **Thời gian đơn luồng:** trung bình 3,2 s mỗi mẫu, p95 6,0 s, lâu nhất 10,4 s, chưa tính ASR (khoảng 0,15–0,5 s mỗi audio).
- **Nghĩa là còn khoảng 20 s dư mỗi mẫu** so với giới hạn 30 s. Giải mã chủ yếu bị giới hạn bởi băng thông bộ nhớ, và A30 (933 GB/s) không chậm hơn card đo (672 GB/s). Phần dư này có thể dùng cho base model mạnh hơn, few-shot, bước kiểm tra hoặc ensemble.
- **Chưa có mẫu nào chạm giới hạn 16k ngữ cảnh.** Phần lớn prompt là schema tool.

### 2.2 Tính tất định

- Chạy cùng 50 câu hai lần liên tiếp, tuần tự, nhiệt độ 0, seed 42: **5/50 câu ra khác nhau**, điểm 47,1 so với 45,8.
- Trước đây, khi chạy lại cả public, 55/634 dòng khác nhau.
- Nghi phạm chính: prefix caching của vLLM (lần chạy sau dùng KV đã cache, tính theo chunk khác), cộng với kernel LoRA.
- `VLLM_BATCH_INVARIANT` của vLLM cần GPU compute capability ≥ 9.0 theo tài liệu hiện hành, trong khi A30 là 8.0, nên không trông vào được.
- **Việc cần làm:** thử tắt prefix caching (`--no-enable-prefix-caching`) và xử lý tuần tự trong Docker, rồi đo lại.
- **Hệ quả đánh giá:** chênh lệch dưới khoảng 1,5 điểm trên val 200 câu là nhiễu.

### 2.3 Bộ chấm local có lạc quan không?

- Chấm lại theo cách hiểu chặt nhất của luật (fuzzy > 0,6 thay cho ≥ 0,5; đối số không fuzzy phải giống hệt sau NFC): val giảm 59,71 → 59,21, oracle giảm 71,44 → 71,15.
- Kết luận: bộ chấm local lệch không quá 0,5 điểm, không phải nguyên nhân khoảng cách val–public.

### 2.4 Điểm mất ở đâu

| Nguồn mất điểm | Ước lượng | Cách đo |
|---|---:|---|
| Agent, kể cả khi transcript chuẩn và app quen | **27,5** | S1c transcript chuẩn đạt 72,45 trên val (lần chạy mới hôm nay) |
| ASR, trên app quen | **12,7** | 72,45 → 59,71 (ASR v2) |
| App chưa thấy | **khoảng 18 trên nhóm này**, tức khoảng 9 điểm public vì khoảng một nửa câu public gọi app mới | Thí nghiệm LAO: 67,7 → 49,8 |

**ASR làm hỏng gì.** Có 39 câu val đúng với transcript chuẩn nhưng sai với ASR v2:

| Nhóm | Số câu | Ví dụ |
|---|---:|---|
| Từ ý định hoặc nội dung | ~16 | "đá bóng"→"đa vọng", "họp"→"hộp", "hủy hẹn"→"ủy viên", "chút tiền"→"chục ngàn" (biến câu cần hỏi lại thành câu tự chuyển 10.000) |
| Tên riêng, tên nước ngoài, thương hiệu | ~13 | Peter Hansen→"minh sang", Nathan Reed, Kowalski, Maple Leaf, Northwind, omeprazole |
| Lập luận mong manh, ASR chỉ đọc số bằng chữ mà agent tính sai | ~5 | "đón trước giờ hẹn 45 phút" bị cộng thay vì trừ; câu điều kiện "trên 122 triệu thì… chưa tới thì…" làm cả hai nhánh |
| Chỉ sai ở lượt lịch sử | ~5 | |

**Nhóm lỗi của agent (đã đo hôm 07/11):**

- Đúng tool nhưng sai tham số: 17,5 điểm trên val, 19,8 trên app lạ. Nhiều nhất là thời gian tính lệch theo một mốc và tên liên hệ.
- Tự làm trong khi lẽ ra phải hỏi lại: 6 điểm. Model bịa giá trị bắt buộc.
- Hỏi lại trong khi lẽ ra phải làm: 6 điểm trên app lạ, 1,6 trên app quen.
- Thừa hoặc thiếu lệnh đọc: 6 điểm.

### 2.5 Các bước tiền xử lý và hậu xử lý đã thử

| Bước | Tác dụng đo được | Giữ? |
|---|---|---|
| Viết thường transcript | +7,6 val, +8,0 public | Có |
| ASR v2 (fine-tune Gipformer trên audio train) | WER 17,05% → 10,74%; +12 val | Có |
| ITN (đổi số đọc thành chữ số) | +0,6 val | Không |
| Hotword toàn cục | +2,2 với ASR gốc; chèn sai nhiều với ASR v2 | Không |
| `--dedup-calls` | chặn vòng lặp trên app lạ; góp phần lên 48,18 | Có |
| `--ground-args` | +1,2 trên app lạ của LAO, 0 trên public (48,15) | Không cần |
| Luật "số không có căn cứ thì hỏi" | +0,2 | Không |
| Đổi tên app khi train (rename aug) | −1,1 trên app lạ | Không |

Bài học: **luật hậu xử lý gần như đã cạn**. Mỗi luật chỉ sửa vài câu, nên phần còn lại cần mô hình tốt hơn và dữ liệu tốt hơn.

## 3. Nghiên cứu: các hướng phù hợp với bài toán

1. **Base model mạnh hơn trong giới hạn 5B.** Qwen3.5-4B (Qwen, 27/02/2026, Apache-2.0) được train RL quy mô lớn cho agent. Model card ghi TAU2-Bench 79,9 và BFCL-V4 50,3 ở chế độ thinking. Tổng 4,66B tham số kể cả vision; khi chỉ dùng phần ngôn ngữ (`--language-model-only`) thì nhỏ hơn. ASR + LLM khoảng 4,73B, dưới 6B. Kiến trúc lai Gated DeltaNet và attention (3:1), vLLM 0.32 chạy được (DataMate đang dùng Qwen3.5-9B). Không có model Qwen mới hơn ở cỡ nhỏ (Qwen3.6 và 3.8 chỉ có 27B trở lên). Gemma-4-E4B có tổng tham số vượt 5B.
2. **Sinh dữ liệu cho tool chưa thấy.**
   - ToolACE (ICLR 2025) tự tiến hoá một kho 26k API.
   - APIGen-MT (Salesforce, NeurIPS 2025) dựng "blueprint" đã kiểm định rồi mô phỏng hội thoại, ra xLAM-2 1B–70B.
   - SynthTools (2025) sinh 74k tool trong 6,8k môi trường mô phỏng, có bước kiểm định; train Qwen3 trên đó cải thiện cả với API thật.
   - VULCAN (ICLR 2026) dựng môi trường tool có thể chạy và tất định từ schema.
   - AWM (Snowflake 2026) sinh 1.000 môi trường có SQL phía sau cho RL.
   - Điểm chung: **đa dạng môi trường quan trọng hơn số câu mỗi môi trường**, và mọi quỹ đạo đều được kiểm chứng bằng cách chạy thật. Bài toán của ta rất hợp vì app của zaic_tools chỉ là các class CRUD nhỏ (12–106 dòng).
3. **Tập trung vào mô tả hơn tên.** Hammer (ICLR 2025) che tên function và tham số khi train, kèm dữ liệu "không có tool phù hợp". Thử đổi tên của ta (rename aug) không giúp, có lẽ vì app mới khác ở cấu trúc tham số chứ không chỉ ở tên.
4. **Dữ liệu cho quyết định hỏi lại.** AskToAct xoá bớt tham số chủ chốt khỏi câu lệnh để tạo nhãn "phải hỏi". ClarifyBench và When2Call đo đúng lỗi "tự làm thay vì hỏi" của ta. Có thể làm từ chính dữ liệu train: xoá số tiền hoặc giờ khỏi câu rồi đổi nhãn thành `sys.ask_missing_info`.
5. **Fine-tune bằng phần thưởng từ môi trường.**
   - Rejection-sampling SFT, như cách làm ở DataMate.
   - GRPO với phần thưởng nhị phân. Tool-N1 (ICLR 2026) cho thấy phần thưởng nhị phân ổn định hơn phần thưởng chia nhỏ. Một bài năm 2026 báo phần thưởng dày theo từng lượt có thể làm hại tới 14 điểm.
   - Môi trường của ta có phần thưởng chính xác (hash + F1) trên train, nên rất hợp.
   - Lưu ý: SFT chỉ trên quỹ đạo on-policy thành công có thể làm model kém đi, nên trộn với nhãn gốc.
6. **ASR cho miền và tên mới.**
   - Dùng TTS để sinh audio cho văn bản miền mới rồi fine-tune RNN-T (Deng et al., 2021; D'Alterio et al.). Audio TTS tốt hơn thích nghi chỉ bằng văn bản.
   - Biasing theo từ khoá cho transducer (Idiap 2024): thực thể +32% tương đối.
   - TTS tiếng Việt mở: VieNeu-TTS 0,3B/0,5B, nhân bản giọng từ 3–5 s audio mẫu.
   - Sửa lỗi ASR bằng LLM trên N-best (HyPoradise) mạnh nhất khi được fine-tune.
7. **Giọng nói tới tool call.** VoiceAgentBench (2025) cho thấy pipeline ASR rồi LLM vẫn hơn speech-LM end-to-end, và các chuỗi call phụ thuộc nhau là khó nhất. Điều này ủng hộ kiến trúc nối tầng hiện tại. Thứ cần nâng là độ bền của LLM trước lỗi ASR, chứ không phải thay bằng mô hình audio.

## 4. Thí nghiệm hôm nay

### 4.1 Qwen3.5-4B zero-shot, chưa LoRA, transcript chuẩn

| Tập | Qwen3-4B-2507 gốc | **Qwen3.5-4B gốc** |
|---|---:|---:|
| Val 200 (app quen) | 28,6 | **40,0** |
| App lạ LAO (182 câu) | 29,9 | **38,8** |

- Qwen3.5 không cần train đã hơn gần 9–11 điểm trên cả hai nhóm.
- Ở chế độ non-thinking, model vẫn tự viết suy luận ngắn trước tool call (khoảng 100 token mỗi lượt), nên mỗi lượt chậm hơn. Sau SFT (nội dung trống như nhãn) thì sẽ không còn.
- Cần `LIFEMATE_MERGE_SYSTEM=1`: template Qwen3.5 chỉ nhận một system message ở đầu, nên phần "prior context" được gộp vào system đầu tiên.
- Thinking mode chưa đo. Với tốc độ khoảng 40–70 token/s, thinking 500 token mỗi lượt sẽ vượt 30 s, trừ khi dùng MTP speculative decoding.

### 4.2 Qwen3.5-4B + SFT trên dữ liệu S1c (kết quả sáng 08/11)

- **Train:** `outputs/sft/q35_e1_01`, Qwen3.5-4B QLoRA trên đúng dữ liệu của S1c (`data_v2_e1`: transcript chuẩn + Gipformer, prompt E1, 200 câu val bị loại), 2 epoch, lr 1e-4, 600 bước, 3 giờ 15 phút, đỉnh 6,3 GiB GPU. Script `scripts/train_agent_sft_q35.py` (Unsloth FastModel, kiểm tra 0 chỗ lệch giữa text train và prompt suy luận).
- **Val 200, ASR v2, dedup:**

  | Model | Điểm | Khớp hash | F1 trung bình |
  |---|---:|---:|---:|
  | S1c (Qwen3-4B) | 59,71 | 137/200 | 0,706 |
  | **q35_e1 (Qwen3.5-4B)** | **70,91** | **155/200** | **0,782** |

  - Tăng 11,2 điểm: 46 câu tốt hơn, 27 câu kém hơn.
- **Public, không dùng nhãn** (`analysis/unseen_apps.py`):

  | Chỉ số | S1c | q35_e1 |
  |---|---:|---:|
  | Câu gọi app lạ có call bị chặn vì sai schema | 9% | 2% |
  | Câu gọi app lạ kết thúc bằng lỗi tool | 4% | 2% |
  | Câu gọi app quen có call bị chặn | 6% | 3% |

  - Trên val, câu có call bị chặn luôn được 0 điểm, nên mức giảm này là tín hiệu tốt cho nhóm app lạ.
  - Tỷ lệ hỏi lại tăng nhẹ: 10% lên 13% (app quen), 4% lên 6% (app lạ).
- **File public:** `result/submission_q35e1_asrv2_dedup_20261108.jsonl`. Kiểm tra:
  - 634 dòng, 0 lỗi schema;
  - chạy lại toàn bộ call trong ToolEnv cho hash trùng 634/634;
  - 3 câu dừng ở giới hạn 10 lượt, giữ nguyên các call đã chạy.
- **Đường Docker** (`lifemate.docker_predict`, ASR CPU từ audio): 12/12 câu public cho hash trùng với lượt chạy CLI, 0 lỗi, mỗi câu 2–10 s (đo khi GPU đang chạy song song việc khác).

### 4.3 Bộ sinh dữ liệu app tổng hợp (bản thử)

- **14 app mới:** khách sạn, phòng khám, thú cưng, giặt là, salon, sạc xe điện, Wi-Fi, vườn thông minh, sổ chi tiêu, sổ liên lạc, chung cư, sân thể thao, bảo hiểm, ổ đám mây.
- **102 task, 204 quỹ đạo** (chữ gốc và bản giả ASR). Các task có:
  - ID lồng nhau không có x-ref (`items[].service_id`);
  - tham số tùy chọn không được tự điền;
  - câu cần hỏi thiếu và câu ngoài phạm vi;
  - câu điều kiện;
  - phối hợp với contacts, sms, reminders, calendar.
- **Cố ý không làm miền nào trùng với 11 app chỉ có ở public**, để public vẫn là phép đo tổng quát hoá trung thực.
- **Đợt 2** (cùng đêm) thêm 13 app nữa: đặt bàn, bảo dưỡng xe, giúp việc, phạt nguội, thời gian dùng máy của con, album ảnh, từ thiện, xem phim, sổ sức khoẻ, dịch vụ công, điều khiển ô tô, nhật ký em bé, mục tiêu tiết kiệm. Tổng **27 app, 124 tool, 164 task**, cả 164 đều chạy lại đúng. Đợt này cũng tránh miền của 6 app bị giấu trong thí nghiệm LAO.
- **Dữ liệu:**
  - `outputs/sft/data_synth_v1`: 328 quỹ đạo, cho model cuối;
  - `outputs/sft/data_lao_synth`: `data_lao_base` cộng dữ liệu tổng hợp lặp 2 lần, không app gây nhiễu nào thuộc nhóm bị giấu, dùng cho thí nghiệm đo.
- **Cách dựng:**
  - `synth/catalog.py` và `synth/apps/synth_apps.py`;
  - task trong `synth/tasks/*.py`;
  - `scripts/build_synth_split.py` dựng split `synth` đúng cấu trúc kit. App gây nhiễu và hồ sơ người dùng lấy từ một mẫu train ngẫu nhiên qua `ToolEnv.get_state()`. Nhãn được chạy lại bằng Session gốc: 102/102 hợp lệ;
  - `scripts/prepare_agent_sft_v2.py --split synth --variants gold,pseudo` tạo dòng SFT.

## 5. Đề xuất, xếp theo lợi ích so với chi phí

| # | Hướng | Lợi ích kỳ vọng | Chi phí | Trạng thái |
|---|---|---|---|---|
| 1 | **Đổi base sang Qwen3.5-4B** + SFT dữ liệu hiện có | Lớn, nhất là trên app lạ (zero-shot +9) | 3 h train | Xong: val 70,91 (S1c 59,71); file public sáng 08/11 |
| 2 | **Dữ liệu app tổng hợp** (bộ sinh ở 4.3) | Nhắm thẳng vào khoảng 18 điểm mất trên app lạ | Viết app/task: 0,5–1 ngày; train 3 h | Đã có bản thử; đo bằng giao thức LAO (lao_base + synth, so với 49,8) |
| 3 | **Dữ liệu "phải hỏi lại"** (xoá tham số kiểu AskToAct trên train) và dữ liệu "lập luận thời gian/điều kiện" | 3–6 điểm ở nhóm hỏi/làm và tham số thời gian | 0,5 ngày | Chưa làm |
| 4 | **ASR:** TTS cho task tổng hợp và tên riêng (VieNeu-TTS) để fine-tune Gipformer v3; giải mã lại lần hai với hotword lấy từ kết quả tool (tên trong danh bạ, tên món, tên file) | Một phần của 12,7 điểm ASR; khoảng 1/3 lỗi ASR là tên riêng | 1 ngày | Chưa làm |
| 5 | **Docker + tính tất định** (tắt prefix cache, chạy tuần tự, đo trên GPU thật) | Bắt buộc; giảm phương sai của trung bình 5 lần chạy | 1 ngày | Phải xong trước 15/11 |
| 6 | Ensemble tất định (2–3 prompt hoặc nguồn transcript, bỏ phiếu từng bước trước khi gọi tool) | Nhỏ–vừa | 0,5 ngày, thời gian chạy ×2–3 | Chỉ làm nếu còn thời gian |
| 7 | GRPO trên môi trường train | Có thể lớn | Nhiều ngày, rủi ro cao | Để sau hạn nộp |

**Không đề xuất:**
- thêm luật hậu xử lý (đã cạn);
- hotword toàn cục với ASR v2;
- đổi tên app;
- thinking mode khi chưa có speculative decoding.

### Thiết kế Docker (theo hướng dẫn của BTC, bản PDF trong `docs/portal_2026-11-07/`)

- **Môi trường:** base image CUDA ≤ 13.2. Ảnh chụp `nvidia-smi` trong hướng dẫn là A30 24 GB. Không có mạng khi chấm; loopback vẫn dùng được, nên vẫn chạy được vLLM dạng server cục bộ.
- **ASR chạy trên CPU:** Gipformer v2 bằng sherpa-onnx trên CPU 8 luồng mất 0,07 s mỗi file (RTF 0,009), và ra đúng y hệt bản GPU (150/150 transcript trùng). GPU dành riêng cho vLLM, tránh xung đột CUDA giữa ONNX và vLLM.
- **`predict.sh`:**
  - khởi động `vllm serve` (base model và LoRA), chờ sẵn sàng;
  - xử lý từng mẫu tuần tự: ASR rồi agent, ghi `time` tính từ trước khi đọc audio;
  - zaic_tools import từ `/data/<split>`, không dùng bản copy trong image;
  - ghi `/result/submission.jsonl`.
- **Kèm theo:** `predict_notebook.ipynb` (seed, nạp model, đọc test, dự đoán), `start_jupyter.sh`, `training_code/` gồm README và link dữ liệu train/synthetic trên Hugging Face. Base model dùng để train không đưa vào image.

### Lịch đề xuất tới hạn 16/11

| Ngày | Việc |
|---|---|
| 08/11 | Nộp public Qwen3.5 SFT. Hàng đợi `outputs/logs/queue_q35_next.sh`: đo tính tất định (prefix cache bật/tắt); train Qwen3.5 + dữ liệu tổng hợp (`data_q35_synth`, 3052 dòng) rồi chạy val và public; đo hướng 2 bằng LAO trên Qwen3.5. |
| 09–10/11 | Mở rộng dữ liệu tổng hợp lên 40–60 app nếu hướng 2 có tác dụng. Làm hướng 3. Dựng Docker. |
| 11–12/11 | Train bản cuối (Qwen3.5 + toàn bộ dữ liệu). Làm ASR v3 nếu kịp. |
| 13–15/11 | Chạy Docker 2 lần so khớp, đo thời gian, nộp public. |
| 16/11 | Đóng băng và nộp. |

## Nguồn

- [Qwen3.5-4B model card](https://huggingface.co/Qwen/Qwen3.5-4B), [tóm tắt bảng điểm small models](https://rits.shanghai.nyu.edu/ai/qwen-3-5-small-models-9b-parameters-that-beat-120b/)
- [ToolACE](https://arxiv.org/abs/2409.00920), [APIGen-MT](https://arxiv.org/abs/2504.03601), [Hammer](https://arxiv.org/html/2410.04587v2)
- [SynthTools](https://arxiv.org/abs/2511.09572v1), [VULCAN](https://iclr.cc/virtual/2026/10019750), [Agent World Model](https://www.snowflake.com/en/blog/engineering/agent-world-model-for-agentic-reinforment-learning/)
- [Nemotron-Research-Tool-N1](https://arxiv.org/pdf/2505.00024), [Exploring Expert Failures (RFT)](https://arxiv.org/pdf/2504.13145), [MT-GRPO / phần thưởng theo lượt](https://arxiv.org/abs/2604.02869)
- [ClarifyBench](https://arxiv.org/html/2511.08798v1), [AskToAct](https://ar5iv.labs.arxiv.org/html/2503.01940)
- [HyPoradise](https://arxiv.org/pdf/2309.15701), [TTS cho thích nghi ASR (D'Alterio et al.)](https://cdn.amazon.science/fa/d7/44451af94472b4dd7c6840197ca0/can-unpaired-textual-data-replace-synthetic-speech-in-asr-model-adaptation.pdf), [Deng et al. 2021](https://www.isca-archive.org/interspeech_2021/deng21_interspeech.pdf), [keyword biasing cho transducer (Idiap)](https://arxiv.org/html/2409.13514v2), [VieNeu-TTS](https://pypi.org/project/vieneu/1.1.2/)
- [VoiceAgentBench](https://arxiv.org/html/2510.07978v1), [vLLM batch invariance](https://docs.vllm.ai/en/latest/features/batch_invariance/)
