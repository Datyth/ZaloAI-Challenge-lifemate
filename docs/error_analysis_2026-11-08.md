# LifeMate: đánh giá lỗi tổng thể của bản tốt nhất (08/11/2026)

Bản được đánh giá: **q35_e1** = Qwen3.5-4B + LoRA trên dữ liệu S1c, ASR Gipformer v2 (chữ thường), dedup. Public **58,02**.

Nguồn số liệu:
- val 200 câu TRAIN không dùng để train, chạy hai lần: transcript chuẩn (`v200_q35e1_oracle_dedup`) và ASR v2 (`v200_q35e1_asrv2_dedup`);
- log public (`pub_q35e1_asrv2_dedup_*`), không dùng nhãn;
- log huấn luyện (`outputs/sft/q35_e1_01`).

Công cụ:
- `analysis/errors.py`: phân loại lỗi;
- `analysis/error_matrix.py` (mới): các ma trận lỗi;
- `analysis/val_clusters_q35e1.json` (mới): nhãn cụm cho từng câu sai, gán tay sau khi đọc từng câu.

## Tóm tắt

1. **Val mất 29,1 điểm:**
   - **11,4** do ASR: câu đúng khi có transcript chuẩn nhưng sai khi dùng ASR;
   - **17,7** do agent: sai ngay cả với transcript chuẩn.
2. **Public thấp hơn val 12,9 điểm.** Nguyên nhân chính:
   - **app lạ:** 65% câu public cài app không có trong train;
   - **nhiều từ lạ hơn:** 37% transcript public có âm tiết chưa từng gặp trong train, val là 26%;
   - model **đoán mã định danh trên app lạ** rồi lặp lại cùng một call.
3. **Cụm lỗi agent lớn nhất (5,1 điểm)** là điều kiện kèm tính toán trên kết quả tool:
   - "ví không đủ thì…";
   - "không đủ cả hai khoản thì…";
   - cộng trừ phút, tính tiền theo giờ.
4. **Lỗi ASR chia gần đều ba nhóm:**
   - từ ý định hoặc nội dung: 3,9;
   - tên riêng, tên nước ngoài: 3,8;
   - số và chữ số: 2,5.
5. **Dữ liệu app tổng hợp (public 56,69) làm hỏng cân bằng "hỏi lại / làm luôn".**
   - Tỷ lệ hỏi giảm 12% → 8%.
   - Số câu lẽ ra phải hỏi mà lại tự làm tăng gấp đôi.
   - Gọi lệnh đọc nhiều hơn.
   - Quy ước trong task tôi viết lệch với nhãn của BTC.
6. **Log huấn luyện:**
   - không đo eval loss (tập dev có sẵn nhưng không dùng);
   - loss train giảm 0,045 → 0,004 trong epoch 2, tức model gần như học thuộc;
   - **dù vậy, epoch 2 vẫn có lợi** (mục 8.2): app lạ tăng 60,4 → 65,3 ở đoạn cuối, val ASR tăng 63,2 → 70,9 từ 1 lên 2 epoch.
7. **App lạ trên Qwen3.5 (mục 8.1):**
   - chênh giữa app quen và app lạ giảm từ 17,9 (Qwen3-4B) xuống **11,1**;
   - khoảng 45% câu public chạm app lạ, nên phần này giải thích khoảng 5 trong 12,9 điểm chênh;
   - bớt 21% dữ liệu train làm app quen mất 6,2 điểm, nên thêm dữ liệu (200 câu val) đáng làm.

## 1. Ngân sách điểm

| Mức | Điểm | Mất so với mức trên |
|---|---:|---:|
| Trần (mọi câu đúng) | 100 | |
| Val, transcript chuẩn | 82,33 | 17,67 (agent) |
| Val, ASR v2 | 70,91 | 11,41 (ASR) |
| Public, ASR v2 | 58,02 | 12,9 (phân bố public khác val) |

Thời gian mỗi câu khi chạy 3 shard song song: trung bình 4,6 s, p95 8,8 s, max 11,0 s.

## 2. Ma trận kết quả: transcript chuẩn (hàng) × ASR (cột)

Mỗi ô: số câu (số điểm bản ASR mất ở ô đó).

| | ASR: đúng hoàn toàn | ASR: hash đúng, F1 < 1 | ASR: hash sai |
|---|---|---|---|
| chuẩn: đúng hoàn toàn | 122 (0) | 12 (1,8) | 18 (9,0) |
| chuẩn: hash đúng, F1 < 1 | 2 (0) | 17 (4,8) | 6 (3,0) |
| chuẩn: hash sai | 2 (0) | 0 (0) | 21 (10,5) |

- **ASR biến câu đúng thành câu sai** ở 30 câu, mất 10,8 điểm. Phần lớn là hỏng hẳn hash (18 câu).
- **Hai bên cùng sai hash:** 21 câu, 10,5 điểm. Đây là lỗi agent thuần.
- **"Hash đúng, F1 < 1":** 25 câu khi có transcript chuẩn. Ở nhóm này trạng thái cuối đúng nhưng thừa hoặc thiếu lệnh đọc, nên mất một phần F1.
- **Lật ngược:** có 4 câu sai với transcript chuẩn nhưng đúng với ASR. Đây là dấu hiệu quyết định của model dễ lật khi đầu vào đổi nhẹ.

## 3. Ma trận hành động: loại hành động gold × loại model làm

| Gold \ dự đoán | n | ghi | chỉ đọc | hỏi lại | hỏi + ghi | không làm gì |
|---|---:|---:|---:|---:|---:|---:|
| **Transcript chuẩn** | | | | | | |
| ghi | 152 | 151 | 1 | 0 | 0 | 0 |
| chỉ đọc | 14 | 1 | 12 | 1 | 0 | 0 |
| hỏi lại | 16 | 1 | 0 | 15 | 0 | 0 |
| hỏi + ghi | 8 | 1 | 0 | 0 | 7 | 0 |
| không làm gì | 10 | 1 | 0 | 0 | 0 | 9 |
| **ASR v2** | | | | | | |
| ghi | 152 | 146 | 3 | 2 | 1 | 0 |
| chỉ đọc | 14 | 2 | 11 | 1 | 0 | 0 |
| hỏi lại | 16 | 4 | 0 | 12 | 0 | 0 |
| hỏi + ghi | 8 | 1 | 0 | 1 | 6 | 0 |
| không làm gì | 10 | 1 | 0 | 0 | 0 | 9 |

- **Transcript chuẩn:** model chọn đúng **loại** hành động gần như mọi câu (194/200). Lỗi agent nằm ở **chi tiết** (tham số, nhánh điều kiện, lệnh đọc), không phải ở việc hiểu sai yêu cầu.
- **ASR:** lệch rõ nhất ở chỗ lẽ ra phải hỏi lại mà lại tự làm (4/16). Ví dụ "chút tiền" bị nghe thành "chục ngàn", nên model chuyển luôn 10.000 đ.

**Call ghi gold (232 call):**

| Kết quả | Transcript chuẩn | ASR |
|---|---:|---:|
| khớp hoàn toàn | 199 (86%) | 185 (80%) |
| đúng tool, sai tham số | 27 (12%) | 30 (13%) |
| thiếu call | 6 (3%) | 17 (7%) |

- **Nhầm tool hay gặp nhất:** `ewallet.transfer` → `ewallet.topup_mobile`, 4 lần, kể cả với transcript chuẩn.
- **Nhãn train cũng lẫn hai tool này:**
  - "chuyển" đi với `ewallet.topup_mobile` 8 lần;
  - "nạp" đi với `ewallet.transfer` 7 lần.

**Tham số sai, phân theo kiểu trong schema** (số lần lệch):

| Kiểu | Transcript chuẩn | ASR | Ví dụ |
|---|---:|---:|---|
| ngày giờ | 12 | 15 | `calendar.create_event.end/start`, `reminders.create.time`, `tasks.add.due` |
| chuỗi so khớp chính xác | 9 | 21 | `contacts.search.query`, `phone.call.phone_number`, `news.search.query` |
| văn bản so khớp mờ | 10 | 11 | `title`, `content` của sms, ghi chú, công việc |
| ID | 6 | 5 | `banking.transfer.from_account_id`, `medication.log_taken.schedule_id` |
| số | 5 | 5 | `banking.transfer.amount`, `delay_minutes`, `minutes` |
| enum | 4 | 3 | `bill_payment.pay.source` |

Tham số kiểu chuỗi khớp chính xác là chỗ ASR làm hại nhiều nhất (9 lên 21). Ví dụ: tên người cần tìm, số điện thoại, từ khoá tìm kiếm.

## 4. Lỗi agent khi có transcript chuẩn (17,7 điểm), theo cụm

Gán tay sau khi đọc từng câu; tổng khớp 17,67.

| Cụm | Câu | Điểm mất | Ví dụ |
|---|---:|---:|---|
| **Điều kiện, tính toán trên kết quả tool** | 11 | **5,05** | "lấy từ TK thanh toán, không đủ cả hai khoản thì lấy tiết kiệm": chọn sai tài khoản ở khoản đầu (3 câu cùng mẫu); "ví không đủ thì trừ ngân hàng": trả cả hai nguồn hoặc chọn sai; 9h30–10h20 tính ra 40 phút (đúng là 50); "đợi tới giờ báo thức đi ngủ" không tính `delay_minutes` |
| Quy ước tham số tùy chọn | 10 | 3,17 | Thêm `end` cho lịch dù người dùng không nói thời lượng; thêm `time` khi ghi đã uống thuốc; thêm `label`, `room`, `date` cho lệnh đọc. Nhãn train không đều: `calendar.create_event` có `end` 27/39, `medication.log_taken` có `time` 8/10 |
| Lệnh đọc (thừa, thiếu, sai từ khoá) | 9 | 2,09 | Thừa `notes.list`, `email.list`; thiếu `calculator.calculate`, `sms.read_recent`; `news.search` "giá xăng" thay vì "xăng" (so khớp chính xác nên mất trọn câu) |
| Ngày giờ tương đối | 4 | 2,00 | "mười giờ thứ năm", "trước bữa bay một ngày", "hôm trước ngày hết hạn giữ": lệch 1 ngày |
| Xác định thực thể (liên hệ, ID) | 5 | 2,00 | "Hương" khi danh bạ có hai Hương và lịch sử dài; chọn nhầm lịch uống thuốc |
| Chọn giữa các tool giống nhau | 3 | 1,50 | Chuyển ví với nạp điện thoại, chuyển ví với trả QR, ví với ngân hàng |
| Hỏi lại hay làm luôn | 4 | 1,50 | "volume để mức chị hay coi" (phải hỏi); "cô muốn đi trước nửa tiếng" (chỉ đọc, model lại sửa lịch) |
| Trường văn bản tự do | 2 | 0,38 | Tiêu đề ghi chú, tiêu đề công việc |

**Nhận xét:**
- Cụm đầu là chỗ model nhỏ thường yếu: phải so sánh số trong kết quả tool rồi mới chọn nhánh.
- Hiện model gọi tool ngay mà không có bước suy luận nào.
- Ở bài DataMate, "ghi chú trước SQL" đã giúp đúng loại lỗi này. Ở đây có thể làm tương tự: dạy model viết một dòng tính toán ngắn trước call ghi.

## 5. Lỗi do ASR (11,4 điểm), theo cụm

| Cụm | Câu | Điểm mất | Ví dụ (chuẩn → ASR) |
|---|---:|---:|---|
| Từ ý định hoặc nội dung | 10 | **3,93** | "chuyển tiền mạng" → "truyền mạng" (thành trả hoá đơn); "khoan trả" → "hoàn trả" (trả cả hoá đơn > 500k); "tiền điện" → "tiền" (trả mọi hoá đơn); "gọi báo" → "và báo"; "mang laptop" → "nâng lớp đóp"; "liền" → "vì" (thêm hẹn giờ 60 phút) |
| Tên riêng, tên nước ngoài | 12 | **3,76** | Peter Hansen → "minh sang"; Nathan Reed → "nat timest"; Northwind → "note wink"; Toronto → "tô lento"; Sophie Martin → "sạo phim marketin"; Nhà giả kim → "nhà giả kêu"; Brightpath, Skyline Park |
| Số và chữ số | 5 | 2,50 | "131 triệu" → "121 triệu" (rẽ sai nhánh); "dưới 124 triệu" mất chữ "dưới"; "chút tiền" → "chục ngàn"; số điện thoại đảo chữ số hoặc thêm thừa chữ "không" |
| Không có lỗi chữ thật | 3 | 0,64 | Transcript chỉ khác chữ hoa và dấu câu, model vẫn đổi hành vi (thêm `banking.get_balance`) |

**Nhận xét:**
- Theo mức lỗi chữ (WER) của câu hiện tại, điểm theo ASR giảm dần:

  | WER câu hiện tại | Điểm ASR | Điểm transcript chuẩn |
  |---|---:|---:|
  | dưới 10% | 72,5 | 81,2 |
  | 10–25% | 69,1 | 88,9 |
  | 25–50% | 50,1 | 80,2 |

- Phần lớn lỗi ASR gây hại nằm ở **từ mang nghĩa quyết định**, như động từ, tên, số, chứ không rải đều.
- Tên riêng xuất hiện **trong ngữ cảnh**: danh bạ, lịch sử, kết quả tool, user profile. Đây là dữ liệu sinh trước hoặc trong lúc agent chạy, nên có thể dùng để sửa ASR theo từng câu.

## 6. Public: vì sao thấp hơn val 12,9 điểm (không dùng nhãn)

| Chỉ số | Val | Public |
|---|---:|---:|
| Câu cài ít nhất một app không có trong train | 0% | 65% |
| Transcript có âm tiết chưa gặp trong train | 26% | 37% |
| Tỷ lệ âm tiết lạ | 1,7% | 2,4% |
| Số lượt nói trước trung bình | 2,81 | 2,05 |
| Prompt lớn nhất mỗi câu (token, trung bình) | 6.033 | 5.531 |

- Public có lịch sử ngắn hơn, nên lẽ ra dễ hơn ở mặt này. Phần chênh đến từ **app lạ** và **từ vựng lạ**: các miền mới kéo theo tên món, tên công ty, cụm từ nước ngoài.
- **Log public cho thấy kiểu lỗi đặc trưng của app lạ: đoán mã thay vì tra cứu.**

  | Câu | Model gọi | Kết quả |
  |---|---|---|
  | "thép Đại Hưng" | `stocks.get_quote` với THDA, STP, VHM, VST | đều báo không tìm thấy, rồi lặp VHM tới khi bị chặn |
  | chuyến bay | `flights.search` với mã sân bay đoán (RVG, RV221) | không ra chuyến nào |

  - 5 câu bị dừng vì lặp call, 3 câu chạm giới hạn 10 lượt, 26 đề xuất bị chặn.
  - Sau khi đổi sang Qwen3.5, tỷ lệ call bị chặn ở nhóm app lạ đã giảm từ 9% xuống 2%.
- **Số đo trực tiếp cho app lạ (mục 8.1):**
  - Với Qwen3.5, app lạ mất 11,1 điểm (65,3 so với 76,4); Qwen3-4B mất 17,9.
  - 45% câu public chạm app lạ, nên app lạ giải thích khoảng 5 điểm của mức chênh.

## 7. Log huấn luyện

| | q35_e1_01 (bản đang dùng) | q35_synth_01 |
|---|---|---|
| Dữ liệu | 2.396 quỹ đạo (1.198 câu × transcript chuẩn và Gipformer) | thêm 328 quỹ đạo tổng hợp × 2 |
| Bước, thời gian | 600 bước, 3 giờ 15 phút | 764 bước, 4 giờ 8 phút |
| Loss theo bước | 0,120 (0–50) → 0,045 (150–200) → 0,021 (300) → 0,004 (600) | tương tự, 0,016 ở cuối |
| Token được học | 223k trên 13,7M token prompt (1,6%) | |
| Eval loss | không đo | không đo |
| Val ASR | 70,91 | 69,13 |
| Public | 58,02 | 56,69 |

**Nhận xét:**
- **Gần như học thuộc ở epoch 2, nhưng không có hại.**
  - Loss chỉ còn khoảng 1/10 so với cuối epoch 1.
  - Đã lo rằng model học thuộc quy ước của các app đã thấy và kém đi trên app lạ.
  - Mục 8.2 cho thấy ngược lại: app lạ tăng mạnh nhất ở đoạn cuối.
- **Không có eval loss.** Tập dev 147 câu có sẵn nhưng trainer không đánh giá, nên không thấy được điểm bắt đầu overfit. Lần train tới cần bật đánh giá dev mỗi 50 bước.
- **Vì sao dữ liệu tổng hợp làm giảm điểm.** So trên val với q35e1:
  - hỏi lại giảm 12% → 8%;
  - "lẽ ra phải hỏi lại mà tự làm" tăng 5 → 10 câu (mất 2,0 → 4,5 điểm);
  - lỗi lệnh đọc tăng 9 → 14 câu;
  - chọn nhầm tool ghi giảm 8 → 4 câu, nhưng không bù được phần trên.

  Kết luận: task tổng hợp tôi viết có quy ước hỏi lại và đọc dữ liệu khác nhãn BTC, nên kéo lệch hành vi trên cả app quen. Nếu dùng lại dữ liệu tổng hợp, quy ước phải lấy từ thống kê nhãn train (tỷ lệ hỏi, mẫu đọc trước khi ghi), không theo trực giác.

## 8. App lạ và số epoch (đo xong 12h58 08/11)

### 8.1 App lạ trên Qwen3.5

Giao thức LAO như ngày 07/11:
- giấu 6 app (bike_rental, fitness, flights, food_delivery, package_tracking, shopping_list) khỏi dữ liệu train;
- train 2 epoch, lr 1e-4;
- đánh giá với transcript chuẩn trên 182 câu dùng app bị giấu và 186 câu val còn lại.

| Model | App quen (186) | App lạ (182) | Chênh |
|---|---:|---:|---:|
| Qwen3-4B `lao_base` | 67,7 | 49,8 | 17,9 |
| Qwen3.5-4B `q35_lao_base` | 76,4 | **65,3** | **11,1** |

- **Qwen3.5 cải thiện mạnh trên app lạ:** +15,5 điểm, so với +8,7 trên app quen.
  - Trên app lạ, số câu 0 điểm giảm từ 84 xuống 52.
  - Không còn câu nào dừng vì lỗi tool; Qwen3-4B có 13 câu.
  - Nhầm loại hành động (hỏi lại thay vì làm, hoặc ngược lại) giảm từ 26 xuống 8 câu.
- **Theo từng app bị giấu, Qwen3 → Qwen3.5:**

  | App | Điểm |
  |---|---|
  | package_tracking | 48,7 → 80,7 |
  | flights | 51,3 → 66,3 |
  | bike_rental | 56,4 → 66,7 |
  | fitness | 61,3 → 67,5 |
  | food_delivery | 32,6 → 56,0 |
  | shopping_list | 57,5 → 56,3 |

  food_delivery vẫn yếu nhất. Nhóm câu đặt món theo số người trong nhóm chat lại thuộc cụm "điều kiện kèm tính toán" ở mục 4.
- **Trên 14 câu val dùng app bị giấu** (ít mẫu, chỉ để tham khảo): `q35e1` đã thấy các app này lúc train và được 78,6; `q35_lao_base` chưa thấy và được 57,1.
- **Ước lượng ảnh hưởng lên public:**
  - 45% câu public (285/634) có call do q35e1 dự đoán chạm vào app không có trong train.
  - Nếu mỗi câu như vậy mất khoảng 11 điểm, app lạ giải thích khoảng **5 điểm** trong 12,9 điểm chênh public/val.
  - Phần còn lại đến từ ASR trên từ vựng lạ và các khác biệt phân bố khác.
  - Đây là ước lượng thô: số 11 đo với transcript chuẩn, còn với ASR thì có thể lớn hơn.
- **Ít dữ liệu cũng làm mất điểm trên app quen.**
  - `q35_lao_base` train trên 941 câu, ít hơn 21% so với q35e1: bỏ câu dùng app bị giấu và câu có lịch sử nhắc tới chúng.
  - Trên cùng 186 câu app quen, nó được 76,4, còn q35e1 được 82,6.
  - Mức chênh 6,2 điểm gồm cả tác động của việc thiếu dữ liệu lẫn của 6 app lạ vẫn được cài trong môi trường. Dù vậy, nó cho thấy model còn "đói" dữ liệu, nên bản cuối train thêm 200 câu val (+17%) đáng làm.

### 8.2 Checkpoint giữa chừng

| Model | Checkpoint | Epoch | LR lúc lưu | App lạ (182) | App quen (186) | Val ASR (200) |
|---|---|---:|---:|---:|---:|---:|
| `q35_lao_base` | 150 | 0,64 | 7,9e-5 | 59,1 | 72,7 | |
| `q35_lao_base` | 300 | 1,27 | 3,1e-5 | 60,4 | 75,8 | |
| `q35_lao_base` | cuối (472) | 2,0 | 0 | **65,3** | **76,4** | |
| `q35_e1_01` | 300 | 1,0 | 5,2e-5 | | | 63,2 |
| `q35_e1_01` | cuối (600) | 2,0 | 0 | | | **70,9** |

- **Epoch 2 không có hại, kể cả trên app lạ.** Điểm tăng đều theo số bước.
  - Đoạn cuối (1,27 → 2 epoch, LR giảm về 0) cho app lạ +4,9 điểm, nhiều hơn hẳn app quen (+0,6).
  - Loss train xuống 0,004 không phải dấu hiệu overfit có hại ở quy mô này.
- **Giới hạn của phép đo:**
  - Checkpoint giữa chừng chưa được giảm LR về 0, nên chưa đo được "1 epoch với LR giảm hết".
  - Cũng chưa đo được 3 epoch.
  - Kết luận chắc chắn chỉ có một: không có lý do để train ngắn hơn. Train dài hơn (3 epoch) có thể có lợi, đáng thử một lần.

## 9. Kết luận: nên đánh vào đâu

Ước lượng điểm có thể lấy lại trên val. Trên public, phần ASR và app lạ có thể lớn hơn.

| # | Hướng | Cụm lỗi nhắm tới | Điểm val đang mất | Cách làm |
|---|---|---|---:|---|
| 1 | **ASR theo ngữ cảnh** | tên riêng, từ ý định, số | 11,4 (public nhiều hơn) | Giải mã lại với hotword lấy từ ngữ cảnh của chính câu đó (danh bạ, lịch sử, tên app, tên trong kết quả tool); fine-tune thêm Gipformer bằng audio TTS cho tên riêng và số |
| 2 | **Suy luận ngắn trước call ghi** | điều kiện, tính toán, ngày giờ | 7,1 | Dữ liệu SFT có thêm một dòng tính toán trước call ghi. Dòng này do LLM lớn sinh ngoại tuyến từ nhãn gold rồi kiểm lại bằng replay; luật cho phép dùng LLM ngoài khi phát triển |
| 3 | **Bản cuối train trên toàn bộ TRAIN, thử 3 epoch** | tất cả, gồm app lạ | 6,2 điểm mất khi bớt 21% dữ liệu (mục 8.1) | Thêm 200 câu val (+17% dữ liệu). So 2 với 3 epoch bằng LAO trước, rồi train bản cuối với cấu hình thắng |
| 4 | **Hành vi trên app lạ** | đoán mã, cân bằng hỏi/làm | khoảng 5 trên public, 11,1 trên nhóm app lạ | Qwen3.5 đã thu hẹp từ 17,9 xuống 11,1. Hướng 2 nhắm đúng phần còn lại (food_delivery theo số người, tính toán). Không giảm epoch |
| 5 | Quy ước tham số tùy chọn và lệnh đọc | quy ước nhãn | 5,3 | Khó: nhãn không đều. Chỉ sửa nơi có quy tắc rõ trong nhãn, không thêm luật hậu xử lý |

Không làm tiếp:
- dữ liệu tổng hợp theo cách cũ (đã thấy hại);
- thêm luật hậu xử lý (đã cạn từ 11-07);
- train ít epoch hơn hoặc chọn checkpoint sớm (mục 8.2).
