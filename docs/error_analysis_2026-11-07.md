# Phân tích lỗi LifeMate: Gipformer + Qwen3-4B LoRA S1 (07/11/2026)

## Thiết lập

- **Tập đánh giá:** 200 câu validation cố định của TRAIN (`scripts/make_validation_split.py`, seed 42). Các câu này không nằm trong dữ liệu SFT của LoRA S1 (`outputs/sft/qwen3_4b_e1_qlora_01/best_adapter`, chính là checkpoint-898).
- **Điểm local** (`analysis/score.py`) mô phỏng luật chấm mô tả trong `zaic_tools/README.md`:
  - hash state cuối là cổng (state đã bỏ các trường text tự do và các app thông tin);
  - khớp hash thì tính F1 của tool call theo cặp một-một: cùng tên, tham số không fuzzy bằng nhau sau chuẩn hoá, tham số `x-match: fuzzy` có token F1 ≥ 0.5;
  - câu có gold rỗng thì dự đoán phải rỗng;
  - điểm là trung bình × 100.
  
  Kiểm tra: chấm chính gold call được 100. Replay gold cho ra đúng `tool_results` của nhãn (300/300 câu).
- **Cấu hình chạy:** vLLM 0.32, orchestration E1, parser hermes, temperature 0, `max_turns` 10.
- **Phần cứng:** một RTX PRO 4000 24 GB, chạy song song 4–6 shard. Vì vậy deadline được nới lên 600 giây để không có câu nào bị timeout, và **độ trễ đo ở đây không đại diện cho A30**.

## Kết quả tổng

| Cấu hình | Điểm | Khớp state | Đúng hoàn toàn |
|---|---:|---:|---:|
| LoRA S1 + transcript chuẩn (oracle) | **71.4** | 159/200 | 126 |
| LoRA S1 + Gipformer (cấu hình đã nộp, public 30.77) | 36.4 | 96/200 | 60 |
| LoRA S1 + Gipformer **viết thường** | **44.0** | 111/200 | 73 |
| LoRA S1 + Gipformer viết thường + số chuyển thành chữ số | 44.6 | 110/200 | 72 |
| Model gốc + transcript chuẩn | 28.6 | 87/200 | 30 |
| Model gốc + Gipformer | 18.0 | 65/200 | 23 |

Đọc bảng:

- **LoRA S1 rất mạnh khi transcript đúng:** từ 28.6 lên 71.4. So với model gốc, LoRA sửa được 51 câu và chỉ làm hỏng 1 câu.
- **ASR làm mất 35 điểm** (71.4 xuống 36.4). Trong đó **7.5 điểm chỉ do viết hoa**: Gipformer trả chữ in hoa không dấu câu, còn LoRA được train trên `audio_text` viết thường có dấu câu. Chỉ cần `.lower()` là lấy lại được 7.5 điểm.
- **Chuyển số sang chữ số** (`src/lifemate/itn.py`) chỉ thêm khoảng 0.7 điểm, nằm trong mức dao động. Khi đã viết thường, LLM tự đọc được số viết bằng chữ.
- **Phần còn lại, 27.5 điểm, là do nhận dạng sai từ thật sự.** Có 61 câu oracle đúng nhưng ASR viết thường sai, và chỉ 8 câu theo chiều ngược lại.
- **Cấu hình đã nộp được 36.4 điểm local, trên public là 30.77.** Mức chênh hợp lý nếu các câu dùng app mới khó hơn (xem mục 3).

## 1. ASR (Gipformer 1.5)

### 1.1 Mức lỗi

- **WER trên toàn bộ audio train** (5.036 file, cả lượt hiện tại lẫn lịch sử): 16.6%, CER 11.9%. Lượt hiện tại 16.9%, lịch sử 16.4%.
- **Một phần WER là vô hại**, như các nhóm sau:

| Loại khác biệt | Ví dụ (ref → asr) | Ảnh hưởng |
|---|---|---|
| Phương ngữ được chuẩn hoá | hông → không (99), ni → nay (82), tui → tôi (28), mô → mua (21) | thường vô hại |
| Vị trí dấu thanh | hoà → hòa (75), khoá → khóa (62), huỷ → hủy (41) | vô hại |
| Số đọc thành chữ | 9 → chín, 15 → mười lăm, ron95 → ron chín lăm, vtv1 → vtv một | 12% số lỗi; LLM đọc được khi đã viết thường |

- **Các lỗi chí mạng** đánh vào động từ ý định, thương hiệu và tên riêng:

| Loại | Ví dụ (ref → asr) | Hậu quả |
|---|---|---|
| Động từ ý định | **dời** → giờ / rời / về / giời (36+), **nhắn** → nhấn (18), **nhắc** → nhất / dắt / trách / giá trị (15+), **họp** → học (20), **đặt** báo thức → tập báo / đọc báo, **tắt** đèn → phát điện | chọn sai tool (update thành create, bỏ mất reminder) |
| Thương hiệu, app | **zalo** → gia lô / bị mất (19), **sjc** → sc / sdg / syc / fjc (37), **sms** bị mất (7) | "gia lô" bị coi là tên người (contacts.search "Gia Lô"), không gửi Zalo |
| Tên người | Sếp Dũng → XẾP RŨ, Minh → MÌNH / RAMIN, Peter Hansen → MITER HAN SANG | tìm sai liên hệ hoặc không tìm thấy, `contacts.search.query` lệch |
| Cụm thời gian | nửa tiếng → nửa chuyến, ba giờ → hai giờ, trước lúc xe → trước úp xe | sai `time`, `start`, `schedule_time` |

- Các giá trị chuỗi trong gold call xuất hiện nguyên văn trong `audio_text` chỉ còn giữ được **74,8%** (450/602) sau ASR. Theo trường: `contacts.search.query` giữ 84%, `reminders.create.content` 71%, `gold_price.type` 38%, `maps.search_place.query` 42%.

### 1.2 ASR làm hỏng những câu nào

Điểm theo đặc trưng của câu, trên 200 câu validation:

| Đặc trưng | n | oracle | ASR thô | ASR viết thường |
|---|---:|---:|---:|---:|
| Tool ghi có tham số giờ/ngày | 41 | 68.3 | **7.5** | 22.9 |
| Có số tiền (`amount`) | 19 | 57.5 | 12.3 | 37.5 |
| Text gold có chữ số | 25 | 94.0 | 27.6 | 46.6 |
| Nhắc Zalo/SMS/SJC | 26 | 88.6 | 33.1 | 52.2 |
| Có `contacts.search` | 31 | 77.4 | 28.4 | 41.3 |
| Lịch sử từ 3 lượt user | 68 | 64.2 | 29.1 | 38.6 |
| Tất cả | 200 | 71.4 | 36.4 | 44.0 |

Theo WER của lượt hiện tại, so ASR viết thường với oracle:

| WER | n | oracle | ASR viết thường |
|---|---:|---:|---:|
| 0 | 20 | 62.5 | 57.5 |
| < 10% | 30 | 60.3 | 54.0 |
| 10–25% | 93 | 77.5 | 44.7 |
| 25–50% | 53 | 70.9 | 33.3 |

Cụm câu có giờ/ngày mất nặng nhất, nhưng phần lớn **không phải do LLM tính giờ sai** mà do động từ ý định bị nghe nhầm ("nhắc tôi" thành "trách tôi", "dời" thành "giờ"). Khi đó model chọn sai tool hoặc bỏ mất hành động, nên tham số giờ không bao giờ đúng. Ví dụ:

- `train_0026`: "**dời** lịch cà phê … muộn hơn **một tiếng**" bị nghe thành "**về** lịch cà phê … ba mũi ho một **chuyến**". Kết quả là `create_event` thay vì `update_event`.
- `train_0255`, `0279`, `0362`: "vàng **SJC** trên 122 triệu thì nhắc …, chưa tới thì chuyển …". Với ASR in hoa, model làm **cả hai nhánh** (thêm reminder); viết thường thì lấy lại được.
- `train_0034`, `0907`: "nhắn qua **Zalo**" thành "NHẮN QUA **GIA LÔ**", và model gọi `contacts.search("Gia Lô")` ba lần.

## 2. Lỗi của agent (transcript chuẩn): 74 câu sai, mất 28.6 điểm

| Nhóm lỗi | Câu | Điểm mất | Mô tả |
|---|---:|---:|---|
| Sai tham số của tool ghi | 13 | 6.5 | giờ/ngày lệch, id thực thể sai |
| Không gọi `ask_missing_info` | 11 | 5.2 | tự điền giá trị thay vì hỏi |
| Tool ghi thừa | 8 | 4.0 | `banking.open_savings`, `video_meeting.join`… |
| Read call lệch | 13 | 3.4 | thiếu/sai tham số đọc, thiếu calculator/exchange_rate |
| Trường text, sai tool, làm khi ngoài phạm vi… | 29 | 9.5 | |

Các mẫu lặp lại, có thể sửa bằng dữ liệu hoặc prompt:

1. **"Trả bằng ví, thiếu thì lấy ngân hàng"** (6 câu): model gọi `bill_payment.pay(source=ewallet)` trước, bị `done=false`, rồi mới trả bằng bank. Gold xem số dư rồi trả bằng bank luôn, nên call thừa làm mất F1. Ngoài ra, cụm "không đủ thì **lấy tiết kiệm**" bị hiểu thành mở sổ tiết kiệm (`banking.open_savings`, 3 câu 0 điểm và 2 câu lặp tới giới hạn 10 lượt), và tài khoản nguồn chọn sai (acc_01 thay vì acc_02).
2. **Không hỏi thông tin còn thiếu** (11 câu): tự đặt `volume=20`, `steps=7000`, `start=10:00`, ngày bay, `note_id`, `subscription_id`. Quy ước của nhãn là hỏi khi người dùng không nói rõ. Phần lớn rơi vào câu có lịch sử dài (9–12 lượt), đúng nhóm mà dữ liệu SFT bỏ đi vì quá 8192 token.
3. **Ngày tương đối lệch một ngày**: "thứ ba tuần sau", "thứ sáu", "ba giờ chiều" (hôm nay hay mai), "trước hạn chót thứ tư ba tiếng".
4. **Quy ước của read call**:
   - `contacts.search` giữ nguyên danh xưng như lời nói ("Vợ Hằng", "Thằng Út", "Anh Hùng");
   - truyền filter khi người dùng nói rõ (`tv.guide(channel=VTV6)`, `smart_home.get_status(room=...)`, `weather.get_current(location=...)`);
   - tính toán bằng `calculator.calculate` hoặc `exchange_rate.get_rate` thay vì tính nhẩm.
   
   Hash vẫn khớp nhưng F1 bị trừ.
5. **Số tiền nói tắt**: "hai trăm" (ví) phải là 200.000 thì model ghi 200. Số điện thoại đọc từng chữ số bị thiếu một số.
6. **Làm khi ngoài phạm vi** (3 câu): "đặt suất khám mắt bên phòng khám" mà không có app phòng khám, model lại tạo sự kiện lịch.

Theo độ dài lịch sử (oracle): 0 lượt 75.7, 1–2 lượt 74.7, 3–5 lượt 70.0, **6+ lượt 58.4**.

Độ dài prompt:

- Lượt đầu: p50 khoảng 5,2k token, p90 7,5k, max 10k. Riêng tool schema đã chiếm trung bình 4k token.
- Các lượt sau dài thêm do kèm kết quả tool, lên tới 10,7k.
- Có 67 câu train vượt 8192 token ngay ở lượt đầu.
- SFT với `max_len` 8192 đã bỏ 308/3589 lượt quyết định, toàn là câu dài.

## 3. Public test: app mới và hành vi của model

- Public có **11 app không có trong train**: ecommerce, event_tickets, grocery, habit_tracker, pantry, parking, public_transit, recipes, stocks, train_bus_tickets, translator.
  - 410/634 câu có cài ít nhất một app trong số đó.
  - Khi chạy S1 + Gipformer trên public, model **gọi tool của app mới ở 268/634 câu (42%)**, nhiều nhất là `grocery.search`, `train_bus_tickets.*`, `recipes.search`, `parking.*`, `event_tickets.*`.
  - LoRA chưa từng thấy các app này, nên phần điểm của chúng chỉ dựa vào khả năng đọc schema của model.
- **So sánh hành vi:**

| | Rỗng | Có ghi | Hỏi thiếu |
|---|---:|---:|---:|
| Public (S1 + Gipformer) | 9,1% | 70,8% | **5,8%** |
| Val (cùng cấu hình) | 9,0% | 74,0% | 5,5% |
| Nhãn train | 7,2% | 75,9% | **14,0%** |

  Model hỏi thông tin thiếu ít hơn nhãn rất nhiều, khớp với nhóm lỗi số 2 ở trên.
- 9/634 câu public dừng vì vượt 10 lượt LLM. Lỗi bị validation chặn chủ yếu là id chưa được tra cứu (`contact_id`, `scene_id`, `order_id`).
- Private test có thể còn nhiều app mới hơn, nên **khả năng tổng quát sang app chưa thấy** là rủi ro lớn nhất khi chỉ tối ưu trên val.

## 4. Đề xuất, xếp theo lợi ích trên chi phí

1. **Viết thường transcript ASR trước khi đưa vào agent.** Val tăng 36.4 lên 44.0, không tốn gì. Áp dụng cho cả lượt hiện tại và lịch sử, và giữ đúng cách chuẩn hoá này ở mọi bước train sau.
2. **SFT v2 trên transcript ASR.**
   - Chạy Gipformer trên toàn bộ audio train (đã có cache tại `outputs/asr/gipformer_train`), lấy transcript viết thường làm `request` và lịch sử, giữ nhãn gold. Trộn thêm một phần transcript chuẩn.
   - Model sẽ học sửa lỗi kiểu "giờ lịch" thành dời, "gia lô" thành Zalo, "syc" thành SJC, nhắm vào phần 27.5 điểm còn lại.
   - Dùng `max_len` 16384 để không bỏ câu lịch sử dài.
3. **ASR.**
   - Fine-tune lại Gipformer với cách chuẩn hoá đúng: chữ hoa, bỏ dấu câu, **đổi số thành chữ thay vì bỏ hẳn câu có số**. Chỉ nhận checkpoint khi qua ngưỡng WER và ngưỡng điểm agent.
   - Hotword biasing bằng beam search của sherpa-onnx (Zalo, SMS, SJC, động từ ý định, tên lấy từ lịch sử).
   - Đưa nhiều phương án nhận dạng (N-best) cho LLM.
4. **Quy ước của agent**, sửa qua dữ liệu và prompt:
   - đưa bảng ngày (hôm nay, mai, thứ trong tuần này và tuần sau) vào context để sửa lỗi ngày;
   - xem số dư trước khi chọn nguồn thanh toán, và hiểu "lấy tiết kiệm" là trừ từ tài khoản tiết kiệm, không phải mở sổ mới;
   - không tự điền giá trị người dùng không nói, mà dùng `sys.ask_missing_info`;
   - giữ danh xưng trong `contacts.search`;
   - tính toán bằng tool.
5. **Tổng quát sang app mới.** Tạo thêm một tập validation giữ lại vài app, loại các app này khỏi SFT để đo đúng tình huống public/private. Trong SFT nên trộn hoặc xáo schema để model đọc schema thay vì nhớ app.
6. **Giới hạn lượt.** Nâng `max_turns` lên 15–20 nhưng giữ deadline 30 giây.

## Tái lập

```bash
.venv/bin/python analysis/asr_cache.py --split train --output-dir outputs/asr/gipformer_train --workers 10    # cache ASR, khoảng 5 phút trên CPU
bash scripts/serve_local.sh                                                                                 # vLLM: model gốc + lifemate_sft
DEADLINE=600 bash analysis/run_shards.sh v200_sft_asr outputs/splits/validation_ids.json 4 asr sft       # chạy 200 câu val theo 4 shard
DEADLINE=600 bash analysis/run_shards.sh v200_sft_oracle outputs/splits/validation_ids.json 4 oracle sft
.venv/bin/python analysis/score.py outputs/runs/v200_sft_asr outputs/runs/v200_sft_oracle --write
.venv/bin/python analysis/errors.py --asr v200_sft_asr --oracle v200_sft_oracle                          # bảng lỗi và từng case
.venv/bin/python analysis/asr_errors.py                                                                     # lỗi ASR theo từ và theo slot
```

Biến thể viết thường hoặc ITN chỉ cần đổi `CACHE=` sang `outputs/asr/gipformer_lower_train/...` hoặc `gipformer_itn_train/...`. Bảng chi tiết nằm ở `outputs/analysis/*_errors.json`, từng câu sai ở `outputs/analysis/*_cases.jsonl`.
