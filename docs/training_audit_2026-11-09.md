# Rà soát log train, loss theo token và kết quả trên public (09/11)

Đối tượng: q35e1, model đang tốt nhất (public 58,02; val ASR 70,91; val gold 82,33). Câu hỏi đặt ra: lỗi nằm ở đâu trong quá trình train, và khi đánh giá thì model sai như thế nào.

## Tóm tắt

| # | Phát hiện | Bằng chứng | Đã làm |
|---|---|---|---|
| 1 | **11% dữ liệu TRAIN chưa từng được train.** `prepare_agent_sft_v2.py` tách 10% task ra `dev.jsonl`, nhưng `train_agent_sft_q35.py` chỉ đọc `train.jsonl` và cũng không dùng `dev.jsonl` để đánh giá | manifest data_v2_e1: train 1.198 task, dev 147 task (294 dòng) | Lần train A dùng đủ 1.345 task |
| 2 | **Phần lớn loss rơi vào chữ tự do, không nằm ở chỗ quyết định điểm** | Trên task chưa thấy, token `x-match: fuzzy` (body, content, title…) chỉ chiếm 4–6% số token được train nhưng gánh 41–66% loss. Các trường này bị loại khỏi state hash và chỉ cần trùng token ≥ 0,5. Trong khi đó 54/70 lượt sai đầu tiên (dev ASR) nằm ở chọn tool, ID, nguồn tiền, dừng hay gọi tiếp | Thêm `--fuzzy-weight` vào trainer; lần train B dùng 0,3 |
| 3 | **Model tính sai thứ trong tuần** | Public: 5/13 câu "thứ X tuần sau" và 1/7 câu "thứ X này" ra ngày không khớp quy ước nào. Prompt E1 không có thứ của `current_time`. Val chỉ có 4 câu dạng này nên val không đo được lỗi | Thêm orchestration **E1D** = E1 + bảng `date_reference` (đã có từ S2). Lần train A dùng E1D |
| 4 | **Không biết xoay xở khi tìm ra rỗng** | TRAIN gần như không có ví dụ tìm kiếm trả về rỗng. Val ASR: 15 câu gặp kết quả rỗng chỉ được 20 điểm, so với 71 trung bình. Public: 75/634 câu gặp kết quả rỗng (52 câu là app lạ); 37 câu sau đó gọi lại đúng tool đó, 13 câu dừng luôn, chỉ 5 câu hỏi lại người dùng | `scripts/inject_recovery.py` tạo 77 quỹ đạo phục hồi thật từ ToolEnv. Lần train B dùng bộ này ×2 |
| 5 | Trên app lạ ở public còn có: đoán mã thay vì tra cứu, bỏ dở chuỗi mua hàng, bịa tên từ ASR méo | Mục 4 | Chưa xử lý, xem mục 6 |

Những phần **không có lỗi**:
- **Chat template** khớp với lúc chạy thật: mọi lượt assistant đều có khối `<think>` rỗng, và đoạn đứng trước mỗi lượt giống hệt prompt mà vLLM nhận.
- **LoRA** phủ cả các lớp linear-attention của Qwen3.5 (`in_proj_qkv/z/a/b`, `out_proj`) lẫn full-attention và MLP.
- Không có dòng nào bị cắt vì vượt 16k token.
- Grad norm không có đỉnh bất thường.

## 1. Log train của q35e1

- 600 bước (2 epoch × 300), batch hiệu dụng 8 quỹ đạo, lr 1e-4 với cosine, peak GPU 6,3 GiB, mất 3 giờ 15 phút.
- Chỉ **1,6% token** được tính loss: 223k trên 13,7 triệu, khoảng 93 token mỗi quỹ đạo. Phần còn lại là prompt, schema và kết quả tool.
- Loss epoch 1 dao động 0,03–0,09. Vừa sang epoch 2 (bước 305) loss tụt ngay xuống 0,01–0,02, và kết thúc ở 0,004. Đây là dấu hiệu học thuộc: model đã thấy đúng các quỹ đạo này. Bản 3 epoch giảm tiếp còn 0,003 nhưng val tụt 4 điểm. Vì vậy muốn tiến thêm cần **dữ liệu mới** hoặc **tập trung đúng token**, không phải train thêm epoch.
- Grad norm giảm từ 2,8 xuống khoảng 0,1–0,7. Không có đỉnh nào cho thấy dữ liệu hỏng.

## 2. Loss theo từng loại token

Script `analysis/token_loss.py` lấy model bf16 + LoRA (đúng như vLLM phục vụ), chạy teacher-forcing trên quỹ đạo gold, rồi gắn mỗi token được train vào một nhóm: quyết định gọi/dừng (token đầu của lượt), tên tool, tên tham số, giá trị (ID lấy từ kết quả, ngày giờ, số, chép từ lời nói, lấy từ context, khác), và cú pháp. Báo cáo sinh bằng `analysis/token_loss_report.py`.

Dữ liệu đo:
- **dev**: 147 task q35e1 chưa thấy; đo 192 dòng gồm cả bản gold và bản ASR Gipformer gốc.
- **val_asr**: 64 dòng val, transcript ASR v2 như lúc chạy thật.

| Tập | Token tự do (fuzzy) chiếm | …và gánh bao nhiêu loss | Lượt sai đầu tiên: chữ tự do / còn lại |
|---|---|---|---|
| dev, gold | 6,3% | 66,4% | 17 / 24 |
| dev, ASR | 6,3% | 48,0% | 16 / 54 |
| val, ASR v2 | 4,4% | 41,5% | 7 / 18 |

- Gần như toàn bộ loss ở chữ tự do đến từ `email.body`, `chat_ott.content`, `tasks.title`, `calendar.title` và `reminders.content`. Trong zaic_tools (`utils.FREE_TEXT_KEYS`), các trường này **bị bỏ khỏi state hash** và chỉ cần trùng token ≥ 0,5 khi tính F1.
- Các lượt sai thật sự ảnh hưởng điểm có dạng:
  - chọn nhầm tool (`notes.list` thay vì `notes.create`, `banking` thay vì `contacts`, `alarm` thay vì `reminders`);
  - dừng sớm (`<|im_end|>` trong khi còn `reminders.create`);
  - chọn nhầm ID (`c_00` thay vì `c_10`, `acc_01` thay vì `acc_02`);
  - chọn nhầm nguồn tiền (`bank` thay vì `ewallet`);
  - hỏi lại sai tool (`sms.send` thay vì `phone.call`).
- ASR chủ yếu làm hỏng **việc chọn tool**. Trên dev, bản ASR có NLL ở token tên tool cao gấp 4,6 lần bản gold (0,042 so với 0,009) và 12 lỗi so với 2. Giá trị số và ngày giờ gần như không lỗi khi được teacher-forcing.

Kết luận: với loss phẳng như hiện tại, gradient chủ yếu dùng để học thuộc cách viết email và tin nhắn. Hướng sửa là **giảm trọng số token chữ tự do** (thêm `--fuzzy-weight w` vào `train_agent_sft_q35.py`).
- Loss vẫn chuẩn hoá theo tổng số token được train, nên các token còn lại giữ nguyên độ lớn gradient.
- Đã kiểm tra: với w = 1,0, loss sau 5 bước là 0,3369, khớp bản cũ (0,3368).

## 3. Thứ trong tuần

`analysis/weekday_dates.py` dò các cụm "thứ X tuần sau/tới/này" và "chủ nhật này", rồi so ngày trong lời gọi tool với `current_time`.

- **TRAIN gold** có 16 câu dạng này, theo quy ước tuần lịch Thứ Hai → Chủ Nhật: "tuần sau" là tuần kế tiếp (8 câu chỉ khớp quy ước này, 1 câu ngoại lệ).
- **Public (q35e1)**: trong các câu "tuần sau" có ngày, 5/13 không khớp quy ước nào.
  - "thứ bảy tuần sau" từ Thứ Hai 05/10 cho ra 11 và 12/10, đúng ra là 17/10.
  - "thứ năm tuần sau" từ Thứ Sáu 25/09 cho ra 02/10 (là thứ Sáu), đúng ra là 01/10.
  - Với câu "thứ X này" có 1/7 sai: "thứ bảy này" từ Thứ Ba 03/11 cho ra 06/11 (thứ Sáu).
- **Val** chỉ có 4 câu dạng này, nên val không thấy được lỗi.

Nguyên nhân: context E1 chỉ có `"current_time": "2026-10-05T09:00:00+07:00"`, model 4B phải tự nhẩm ra đó là thứ mấy. Bảng `date_reference` (15 ngày kèm tên thứ, khoảng this_week và next_week) đã có trong `lifemate.context` từ bản S2 nhưng bị bỏ khi chuyển sang E1.

Đã thêm orchestration `E1D`: prompt E1 thêm một dòng hướng dẫn, và context có `date_reference`. Tùy chọn này có cả ở cli (`--orchestration E1D`), ở `prepare_agent_sft_v2.py --orchestration E1D`, và trong docker qua biến `LIFEMATE_ORCHESTRATION`. Dữ liệu E1D giống E1 từng dòng, chỉ khác system prompt và context.

## 4. Đọc kết quả trên public (không dùng nhãn)

`analysis/run_dump.py` in mỗi câu kèm transcript, app (đánh dấu app lạ), từng lời gọi và kết quả. `analysis/run_patterns.py` đếm các mẫu lỗi trên cả 634 câu; với val thì có kèm điểm.

| Mẫu | Public | Trong đó app lạ | Val (điểm TB) |
|---|---|---|---|
| Gặp kết quả rỗng | 75 (11,8%) | 52 | 15 (20 điểm) |
| … sau đó gọi lại đúng tool đó | 37 | 27 | 6 |
| … sau đó dừng luôn | 13 | 10 | 2 |
| … sau đó hỏi người dùng | 5 | 1 | 3 |
| Thêm vào giỏ nhưng không checkout | 19 | 19 | 0 |
| Mã không tồn tại ("không tìm thấy mã") | 5 | 5 | 0 |
| Kết thúc mà chỉ có lệnh đọc | 68 (10,7%) | 42 | 14 |

Ví dụ đọc tay:
- **Đoán mã thay vì tra cứu** (`stocks`): "thép đại hưng" → `get_quote` lần lượt THDA, STP, VHM, VST, đều không tìm thấy rồi lặp lại. Mã trong app là mã giả (ZDH, ZMX…) nên đoán kiểu này luôn trượt. Lẽ ra phải gọi `get_portfolio` hoặc tool tìm kiếm trước. Có 3 câu như vậy trong mẫu đã đọc.
- **Tìm sai cách rồi bỏ dở**:
  - `event_tickets.search {"query": "thứ ba tuần sau"}` (đưa ngày vào ô từ khoá) ra rỗng rồi dừng.
  - "Gly Water Farra" (ASR méo) ra rỗng rồi dừng.
  - "Charlton" ra rỗng, sau đó gọi `calendar.list_events` 7 lần, mỗi lần một ngày, tới khi chạm giới hạn bước.
- **Ngày sai**: tìm tàu "thứ bảy tuần sau" với ngày 12 rồi 11/10, ra rỗng rồi dừng.
- **Bỏ dở chuỗi mua hàng**: "đặt đi em một cái compact k2, thanh toán khi nhận hàng" → chỉ gọi `ecommerce.add_to_cart`, không có `checkout`. Có 19 câu như vậy, đều là app lạ.
- **Tự điền giá trị phải tính từ dữ liệu**: "giao trước bữa tối với gia đình một tiếng" → `grocery.checkout` với giờ 19:00 mà không đọc lịch.
- **Bịa tên từ ASR méo**:
  - "lotus liễu giai" → "Lotus Land";
  - "notevel" → "Notebook Dell";
  - "chars" → "Charlton".
  
  Đây đúng là thói quen học từ 12,5% bản sao ASR có nhãn mâu thuẫn: nhãn đòi một tên không có trong transcript.
- Các câu dạng "à thôi khỏi" trả về rỗng đúng. Nhưng có câu bỏ sót hành động trên app quen: hỏi giá xăng ở cây xăng hôm qua, `fuel_price` đã cài mà không gọi.

## 5. Bộ chèn lỗi phục hồi (đề xuất 1a, bản đầu)

Thăm dò ToolEnv: `contacts.search` khớp chuỗi con, không phân biệt hoa thường.
- Thêm danh xưng ("anh Dũng" khi danh bạ lưu "Dũng") luôn ra rỗng (131/131).
- Tìm bằng tên cuối, hoặc query rỗng (liệt kê tất cả), luôn có người cần tìm.

`scripts/inject_recovery.py` tạo ba loại quỹ đạo, mỗi cái đều chạy thật trong ToolEnv:
- `honorific` (23 dòng): danh xưng chính người nói dùng cộng tên → rỗng → tìm lại bằng tên.
- `asr_name` (23 dòng): bản ASR nghe sai tên ("xếp dũng", "bên sale" thay cho Ben Taylor, "đan" thay cho Lan) → rỗng → tìm bằng tên cuối nếu ra đúng người, nếu không thì liệt kê tất cả.
- `long_query` (31 dòng): query thêm một từ nói kèm ("podcast Nhà giả kim") → rỗng → query gold.

Cách ghép vào dữ liệu train:
- Lượt sai được giữ trong ngữ cảnh với `"train": false`, và trainer không tính loss cho lượt đó. Như vậy model học cách xử lý sau khi gặp lỗi, chứ không học cách gây ra lỗi.
- Các kết quả tool phía sau được chạy lại trong ToolEnv và phải khớp kết quả gốc.
- Đã kiểm tra trên một dòng thật: lượt "xếp dũng" không có nhãn, lượt "dũng" có nhãn, nội dung SMS mang trọng số 0,3.

## 6. Đang chạy và cách đọc kết quả

Đã thống nhất chạy **một bản duy nhất** gộp mọi thay đổi, không chạy các bản thử riêng lẻ. Lý do: các bản A (E1D + đủ task) và B (A + phục hồi + trọng số) đều là tập con của bản này, và chạy song song làm mọi thứ chậm gấp đôi.

| Bản | Nội dung | Trạng thái |
|---|---|---|
| q35e1 | bản đang tốt nhất | public 58,02, val ASR 70,91 |
| **q35_p1** | đủ 1.345 task + prompt **E1P** (bảng thứ, tên app, quy tắc phục hồi) + 77 quỹ đạo phục hồi ×2 + trọng số chữ tự do 0,3 + LoRA trên base bf16 | train từ 02:52, xong khoảng 07:00; val và public xong khoảng 07:45 → `result/submission_q35p1_asrv2_dedup_20261109.jsonl` |
| A, B, đo prompt kèm adapter | | đã dừng; dữ liệu giữ lại để làm ablation nếu cần |

Hàng đợi: `outputs/logs/queue_p1.sh` (tmux `jobs:p1`), log train ở `outputs/logs/train_q35_p1.log`.

Cách quyết định:
- Val ≥ 70 (q35e1 trừ nhiễu khoảng 1 điểm): nộp file p1.
- Xem thêm nhóm "gặp kết quả rỗng" trên val (15 câu, hiện 20 điểm), và trên public xem số câu gặp rỗng rồi dừng hoặc lặp lại, số câu sai ngày (`analysis/run_patterns.py`, `analysis/weekday_dates.py pub_q35p1_asrv2_dedup`).
- Nếu val tụt rõ: chạy lại bỏ trọng số chữ tự do (thay đổi có rủi ro cao nhất), giữ các phần còn lại.

**Cấu hình GPU.** Đo 4 bước, 32 dòng:

| Cấu hình | Thời gian | Bộ nhớ đỉnh |
|---|---|---|
| 4-bit + checkpointing Unsloth (cũ) | 109 s | 6,2 GB |
| bf16 + checkpointing Unsloth | 112 s | 11,1 GB |
| bf16 + checkpointing thường | 100 s | 12,4 GB |
| bf16, không checkpointing | hết bộ nhớ | — |

GPU vốn đã chạy 100% tính toán; dùng ít bộ nhớ không có nghĩa là đang bỏ phí. Kernel linear-attention (`fla`) đã là bản nhanh, chỉ `causal_conv1d` chạy bản dự phòng (phần tính toán nhỏ). Bản p1 dùng bf16 + checkpointing thường:
- nhanh hơn khoảng 8%;
- adapter được train trên đúng trọng số bf16 mà vLLM phục vụ, thay vì bản 4-bit;
- dòng dài nhất (khoảng 11k token) cần 12,8 GB.

## 7. Đánh giá prompt

Bố cục một prompt (trung vị khoảng 5.800 token, tối đa khoảng 10.000):
1. Schema tool của các app đã cài, khoảng 72% số token.
2. Khối hướng dẫn tiếng Anh của template Qwen.
3. 13 dòng quy tắc tiếng Việt (E1).
4. Hướng dẫn tiếng Anh kèm history dạng JSON.
5. Lượt user: `{"request": transcript, "context": {...}}`.

Các quy tắc E1 **khớp với nhãn TRAIN**:
- "Dời lịch": 27/28 câu đúng mẫu `list_events` rồi `update_event`.
- "Nhắn [người]" khi có SMS: 77 câu gọi `sms.send`. Các câu còn lại là nhánh điều kiện, hỏi lại, hoặc người dùng nói rõ "Zalo" (5 câu, đều gọi `chat_ott`).

Prompt không có quy tắc nào sai. Vấn đề là thiếu thông tin:
- **Không có thứ trong tuần.** Model gốc chưa fine-tune, val 200 câu với transcript chuẩn: **E1 40,0 → E1D 45,1** (+5,1; khớp hash 117 → 128).
- **Không có tên tiếng Việt và chức năng của app.** Catalog có `name_vi` và `description` ("Đi chợ online: Mua thực phẩm, siêu thị online"), nhưng prompt chỉ đưa id tiếng Anh. Người dùng lại gọi app bằng tên Việt ("app đi chợ", "bên mua sắm"), nên với app lạ đây là đầu mối chính.
- **Không có cách xử lý khi tìm ra rỗng**, cũng không dặn "lấy mã từ tool đọc, không đoán".

Prompt **E1P** = E1D + `context.apps` (tên Việt và chức năng của mỗi app đã cài, lấy từ catalog) + 4 quy tắc chung:
- chọn app theo tên;
- tìm ra rỗng thì rút gọn từ khoá, bỏ trống query, hoặc dùng tool danh sách; không đưa ngày vào ô từ khoá; vẫn không có thì hỏi;
- không đoán mã;
- đọc dữ liệu trước khi tính giá trị.

Ví dụ trong quy tắc chỉ lấy từ app TRAIN. Quy tắc giỏ hàng/checkout bị bỏ, vì nó chỉ đến từ app public.

Model gốc với E1P trên 68 câu đầu đạt 36,6, so với E1D 37,3: ngang nhau, không hại. Val không có app lạ, nên phần lợi của E1P (tên app) chỉ đo được trên public.

## 8. Ý của Đạt

- **Thay dữ liệu sạch bằng bản ASR để model chịu nhiễu.** Đã có sẵn: mỗi task có một bản gold và một bản Gipformer gốc.
  - Đo trên dev, nhóm yếu nhất là task đọc+ghi khi đầu vào là ASR: 19/60 quỹ đạo đúng hết, so với 38/60 với transcript chuẩn. Vậy chịu nhiễu ở task nhiều bước đúng là điểm yếu.
  - Nhưng thêm bản sao của cùng task là thêm lượt học thuộc, mà bản 3 epoch đã tụt 4 điểm. Hướng đúng là **nhiễu khác nhau ở mỗi epoch** (epoch 2 dùng một bản ASR khác) hoặc nhiễu từ TTS→ASR trên câu mới. Xem đề xuất 2.
- **Lọc nhãn mâu thuẫn.**
  - Tham số không phải chữ tự do bị ASR làm mất chỉ có 56/993 (54/1.345 dòng, 4%): `contacts.search` 19, `gold_price.type` 10, query podcast/news/library/maps 17.
  - Một phần học được, ví dụ "ét giê xê" → `sjc`, nên che đi là mất tín hiệu.
  - Phần danh bạ đã có bộ phục hồi lo; phần chữ tự do (chiếm phần lớn con số 12,5% trước đây) đã có trọng số 0,3.
  - Không che thêm.
- **Augment nhóm hay sai.**
  - Chọn nhóm bằng val sẽ làm val mất tác dụng đo, nên mình chọn bằng dev.
  - Trên dev, các nhóm thiểu số đều ổn (`empty` 9/9, `ask_only` 1/1); nhóm yếu là đọc+ghi khi đầu vào là ASR, vốn đã chiếm 51,5% dữ liệu.
  - Upsample theo nhóm không giải quyết được chuyện đó, nên không làm.

Chưa làm:
- Đoán mã (`stocks`) và bỏ dở chuỗi mua hàng là lỗi của app lạ. Muốn sửa cần dữ liệu có luồng "tra cứu rồi mới ghi" và "giỏ hàng rồi checkout", tức synth v2 lấy khung từ TRAIN. Lần synth v1 đã kéo tỷ lệ hỏi lại xuống, nên lần này phải đo bằng LAO trước khi gộp.
- Phân tích loss trên val mới chạy 64/200 dòng, vì chạy song song làm train chậm gấp đôi. Có thể chạy tiếp khi GPU rảnh.
- Model gốc với E1P mới đo 68/200 câu val; phép đo bị dừng để dồn GPU cho p1.
