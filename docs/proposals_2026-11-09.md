# LifeMate: ba đề xuất lớn tới hạn 16/11 (09/11/2026)

Bản này thay bản đề xuất nhỏ viết lúc 23h ngày 08/11. Mình đọc lại các nguồn sau và đo thêm:
- log train LLM (`outputs/logs/train_q35_e1_01.log`, `train_q35_e1_3ep.log`);
- log train ASR (`outputs/asr_sft/gip_v2_0*/training.jsonl`, `outputs/logs/asr_train_v2_01.log`);
- các ma trận lỗi trong `docs/error_analysis_2026-11-08.md`;
- log tool của val, LAO và public (`tool_events.jsonl`);
- catalog app của train và public, audio TRAIN.

Bản tốt nhất hiện tại: **q35e1**. Val 70,91, public 58,02.

## Tóm tắt

| # | Đề xuất | Khoảng điểm nhắm tới | Bằng chứng mạnh nhất | Ước lượng | Công sức, GPU |
|---|---|---|---|---|---|
| **1** | **Dạy agent tự xoay xở khi tool trả rỗng và khi gặp app lạ.** Chèn lỗi rồi hướng sửa vào quỹ đạo TRAIN; thêm app tổng hợp có schema đa dạng | Public thấp hơn val 12,9 điểm; val ASR mất 5,6 điểm ở các câu gặp kết quả rỗng | Nhãn TRAIN chỉ có 25/3.541 kết quả rỗng và 0 lỗi, nên model chưa từng thấy cách xoay xở. Câu gặp kết quả rỗng: val ASR được 19,7 điểm so với 74,8. 10% câu public gặp kết quả rỗng, ở app lạ là 14% | +2 đến +4 public | 2 ngày; 2 lần train |
| **2** | **ASR v3 theo hướng dữ liệu:** sửa nhãn số, augmentation, TTS cho tên, số và từ lạ, train trên toàn bộ audio | Val mất 11,4 điểm do ASR; public còn nhiều từ lạ hơn | 30% câu có số bị bỏ khỏi dữ liệu train ASR; ít nhất 7 nhãn số sai; không có augmentation; dev loss đứng từ epoch 1; mọi ASR lớn hơn đều kém xa | +1 đến +3 val, public nhiều hơn | 1,5 ngày; khoảng 4 giờ GPU |
| **3** | **Ghi chú tính toán trước call ghi**, chưng cất từ model thầy và có kiểm chứng | Khoảng 10,5 điểm agent mất trên val khi transcript chuẩn | Model chọn đúng loại hành động 194/200 nhưng sai ở bước tính; hiện mỗi lượt chỉ sinh 26 token, không có bước suy luận | +2 đến +4 val | 1 ngày; khoảng 6 giờ GPU |

Cả ba đề xuất là thay đổi về **dữ liệu và cách train**, không phải luật hậu xử lý. Luật hậu xử lý đã cạn: mỗi luật chỉ được 0 đến +0,6 điểm. Cả ba gộp vào **một lần train cuối ngày 12/11** (mục 5).

## 1. Đọc lại log và ma trận lỗi: các phát hiện

### 1.1 Điểm mất ở đâu

| Mức | Điểm | Mất | Nguyên nhân chính |
|---|---:|---:|---|
| Val, transcript chuẩn | 82,33 | 17,7 | Agent sai chi tiết: tính trên kết quả tool, quy ước tham số, lệnh đọc |
| Val, ASR v2 | 70,91 | 11,4 | ASR làm mất thông tin: từ ý định 3,9, tên 3,8, số 2,5 |
| Public | 58,02 | **12,9** | App lạ, tool trả rỗng, từ lạ |

Khoản lớn nhất là khoảng chênh val/public. Private test nhiều khả năng cũng có app mới, nên khả năng tổng quát hoá quyết định điểm chung cuộc nhiều nhất.

### 1.2 Log train LLM: model đang học thuộc, thiếu dữ liệu chứ không thiếu số bước

| Đoạn | Loss | Grad norm trung bình |
|---|---:|---:|
| Epoch 0–0,17 | 0,117 | 1,21 |
| Epoch 0,85–1,0 | 0,036 | 0,49 |
| **Epoch 1,02–1,17** | **0,018** | 0,23 |
| Epoch 1,85–2,0 | 0,009 | 0,35 |
| **Epoch 2,02–2,25** (bản 3 epoch) | **0,003** | 0,18 |

- **Mỗi lần sang epoch mới, loss tụt một nửa ngay lập tức.** Đó là dấu hiệu model nhớ từng câu đã gặp, không phải học thêm quy luật.
- Epoch 2 vẫn có lợi: val 63,2 lên 70,9. Epoch 3 thì hại: val tụt 4,1 điểm.
- Bớt 21% dữ liệu (thí nghiệm LAO) làm app quen mất 6,2 điểm.
- **Kết luận: muốn tiến thêm thì cần thêm task mới, không cần thêm epoch.** Hai đề xuất 1 và 3 đều tạo thêm tín hiệu học mới.
- Chỉ 1,6% token được tính loss (223k trên 13,7M). Không có câu nào bị cắt ở 16k token. Không đo dev loss.

### 1.3 Model chưa từng thấy tool trả rỗng hay báo lỗi

| Tập | Câu có kết quả tool rỗng hoặc lỗi | Điểm trung bình nhóm đó | Nhóm còn lại | Điểm mất ở nhóm đó |
|---|---:|---:|---:|---:|
| Nhãn TRAIN | 25/1.545 câu (25/3.541 kết quả, 0 lỗi) | | | |
| Val, transcript chuẩn | 4 (2%) | 25,0 | 83,5 | 1,5 |
| **Val, ASR v2** | **14 (7%)** | **19,7** | 74,8 | **5,6** |
| LAO, app bị giấu | 3 (2%) | 13,3 | 66,1 | 1,4 |
| **Public** | **63 (10%)**: app lạ 41/285 (14%), app quen 22/349 (6%) | không có nhãn | | |

**Val ASR: các kiểu gặp kết quả rỗng.**
- `contacts.search` gặp 5 lần:
  - truy vấn có kèm danh xưng: "Sếp Vũ", "Cô Thảo";
  - tên bị ASR nghe sai: "Thầy Minh Sang" (đúng ra là Peter Hansen), "Ông Rớ".
- Tìm sách bằng tên tiếng Anh: "Around the World in Eighty Days".
- Tìm podcast bằng chữ bị nghe sai: "Toronto" thành "tô lento".

Sau khi tìm không ra, model hoặc **vẫn ghi luôn** với giá trị đoán, hoặc **dừng ở lệnh đọc**.

**Public: các câu gặp kết quả rỗng kết thúc ra sao** (63 câu):

| Kết thúc | Số câu |
|---|---:|
| Chỉ có lệnh đọc | 23 |
| Vẫn ghi | 22 |
| Hỏi lại | 10 |
| Hết lượt | 3 |
| Lặp call | 3 |
| Lỗi tool | 2 |

**Public, nhóm app lạ: model chọn hành động khác hẳn val** (không dùng nhãn, chỉ so phân bố):

| Loại hành động | Val gold | Val dự đoán | Public, có app lạ | Public, chỉ app quen |
|---|---:|---:|---:|---:|
| Ghi | 76% | 77% | 77% | 64% |
| Chỉ đọc | 7% | 7% | **17%** | 8% |
| Hỏi lại, có hoặc không kèm ghi | 12% | 12% | **6%** | 16% |
| Không làm gì | 5% | 4% | 0% | 11% |

Với app lạ, model **dừng ở lệnh đọc nhiều gấp 2,4 lần** và **hỏi lại ít đi một nửa**. Đọc từng câu thấy các kiểu sau:
- **Đặt tham số sai chỗ:** `event_tickets.search {"query": "thứ ba tuần sau"}`, tức là đưa ngày vào ô từ khoá.
- **Tìm rồi bỏ dở:** "xem chuyến tàu nào tới Huế trước 11 giờ thì đặt hai vé rồi tạo lịch", model chỉ gọi `search` hai lần rồi dừng.
- **Đoán mã:** đoán mã cổ phiếu "ZTS", "ZTP".

**Giao thức LAO không bắt được kiểu lỗi này:** app bị giấu chỉ có 2% câu gặp kết quả rỗng. Vì vậy đề xuất 1 phải đo trên val ASR và trên phân bố hành động của public.

### 1.4 App: train so với public

| | Train | Public |
|---|---:|---:|
| Số app | 46 | 36 |
| App chỉ có ở public | | 11 (34 tool: 17 đọc, 17 ghi) |
| Câu có cài ít nhất một app lạ | | 410/634 |
| Câu có gọi app lạ | | 285/634 |

App chung giữa hai tập có bộ tool giống hệt nhau.

### 1.5 Log train ASR: nhãn số lỗi và thiếu dữ liệu

1. **Nhãn số chép từ chính lỗi của ASR gốc.**
   - `scripts/asr_targets.py` lấy đoạn số trong transcript của Gipformer gốc thay vì tự đọc số từ transcript chuẩn.
   - Log ghi `skipped_train=112 skipped_dev=23`: **112/368 câu có số (30%) chưa từng được train**.
   - Ít nhất 7 câu được train với số sai:
     - "30 phần trăm" thành "NĂM MƯƠI PHẦN TRĂM";
     - "131 triệu" thành "MỘT TRĂM NĂM MƯƠI MỐT TRIỆU";
     - "500 nghìn" thành "NGHÌN NĂM TRĂM NGÀN", lặp ở 3 câu;
     - 2 số điện thoại bị mất chữ số.
2. **Thiếu dữ liệu.**
   - Dev loss của v2-01 giảm tới hết epoch 1 (1,561 ở bước 3.600), sau đó gần như đứng (1,506 ở cuối).
   - v2-02 (lr cao hơn, 4 epoch) đạt 1,510 rồi tăng lại.
3. **Không có augmentation:** không SpecAugment, không đổi tốc độ, không thêm nhiễu.
4. **ASR lớn hơn đều kém xa** (WER trên val): Gipformer v2 8,7; Gipformer gốc 15,2; ChunkFormer 22,2; PhoWhisper-medium 24,9; whisper-large-v3-turbo 27,7; Qwen3-ASR-0.6B 33,9. Vì vậy nên giữ Gipformer và cho nó dữ liệu tốt hơn.
5. **Audio TRAIN có dấu hiệu là giọng TTS cộng nhiễu.**
   - Đỉnh đã được chuẩn hoá quanh −16 dBFS.
   - Mức nhiễu nền trải rộng: p25 −87, trung vị −67, p95 −36 dBFS.
   - 1/4 số file có nền gần như im lặng hoàn toàn, điều hiếm gặp ở thu âm thật.
   - Nếu đúng vậy, dữ liệu TTS cộng nhiễu sẽ khá khớp với audio thi.

## 2. Đề xuất 1: dạy agent tự xoay xở và dùng app lạ

### Ý tưởng

Quỹ đạo gold của BTC luôn "sạch": tìm là thấy, gọi là đúng. Model học thuộc các quỹ đạo đó nên không biết làm gì khi kết quả không như mong đợi. Tình huống này lại hay gặp nhất đúng ở hai chỗ đang mất điểm: ASR nghe sai tên, và app lạ có schema chưa quen.

### Làm gì

**1a. Chèn lỗi rồi hướng sửa vào quỹ đạo TRAIN** (tự động, giữ nguyên nhãn).

- Với mỗi lệnh đọc trong nhãn có tham số chữ (`query`, tên, mã, ngày), tạo một bản có **lần thử đầu sai**. Có 5 kiểu:

  | Kiểu lỗi | Ví dụ |
  |---|---|
  | Kèm danh xưng | "Sếp Vũ" |
  | Tên bị ASR làm hỏng | Lấy từ chính lỗi của Gipformer trên TRAIN |
  | Truy vấn quá dài | Thêm từ thừa vào `query` |
  | Sai ô | Đưa ngày vào `query` |
  | Mã đoán | Mã không tồn tại |

- Chạy lần thử sai đó trong ToolEnv thật để lấy kết quả rỗng hoặc lỗi thật.
- **Bước sửa** viết theo luật, không cần model thầy:
  - bỏ danh xưng hoặc rút gọn truy vấn;
  - chuyển giá trị sang đúng ô theo schema;
  - gọi tool liệt kê hoặc tìm kiếm của cùng app để lấy mã;
  - nếu dữ liệu thật sự không có thì `sys.ask_missing_info`, đúng quy ước của nhãn.
- Sau bước sửa là phần còn lại của quỹ đạo gold.
- **Không tính loss ở lượt sai được chèn vào**, chỉ học bước sửa và phần sau. Như vậy model không học cách gây lỗi.
- Quy mô: khoảng 1 bản cho mỗi 2 task có lệnh đọc phù hợp, tức khoảng 600–900 quỹ đạo mới.

**1b. App tổng hợp v2 có schema đa dạng.**
- Lấy cấu trúc từ task TRAIN thật: chuỗi đọc rồi điều kiện rồi ghi, tỷ lệ hỏi lại, mẫu lệnh đọc. Đặt lên app mới có schema khác: lọc bằng ô riêng, ngày tách khỏi giờ, mã lồng nhau.
- Model thầy chỉ viết câu nói tiếng Việt. Call được sinh bằng code và chạy kiểm chứng.
- Không dùng miền nào của 11 app chỉ có ở public.
- Mỗi task 1 bản, khoảng 1.000 quỹ đạo.
- Khác synth v1: quy ước lấy từ nhãn thật thay vì do mình tự viết. Đây chính là lý do synth v1 làm tỷ lệ hỏi lại giảm 12% → 8%.

### Đo và ngưỡng giữ

- **1a:**
  - val ASR toàn bộ, so với 70,91;
  - riêng 14 câu gặp kết quả rỗng, so với 19,7;
  - public không dùng nhãn: tỷ lệ "chỉ đọc" ở app lạ (17%) và số câu gặp kết quả rỗng mà kết thúc không sửa (49/63).
  - Giữ nếu val không giảm và hai chỉ số public cải thiện.
- **1b:** giao thức LAO, so với app lạ 65,3 và app quen 76,4. Giữ nếu app lạ +3 trở lên và app quen không giảm quá 1.

### Ước lượng

- Câu public gặp kết quả rỗng: nếu điểm nhóm này giống val (khoảng 20), chúng mất khoảng 5–6 điểm public. Sửa được 40% thì được **+2 public**.
- 1b nhắm vào 11 điểm chênh app lạ trên 45% câu public. Lấy lại 1/4 thì được thêm **+1 đến +1,5**.

### Chi phí, rủi ro

- 1a: 1 ngày, 1 lần train (khoảng 3,5 giờ).
- 1b: 1,5 ngày, 1 lần train LAO (khoảng 2,5 giờ).
- Rủi ro của 1a thấp: nhãn giữ nguyên, chỉ thêm tình huống mới.
- Rủi ro của 1b trung bình, nhưng có LAO chặn trước khi đưa vào bản cuối.

## 3. Đề xuất 2: ASR v3 theo hướng dữ liệu

### Làm gì

- **A. Sửa nhãn số.**
  - Đọc số từ transcript chuẩn bằng `_spell` (`scripts/prepare_agent_sft_v2.py`), sinh các cách đọc khác nhau:
    - "bốn mươi lăm" hay "bốn lăm";
    - "một trăm linh một" hay "một lẻ một";
    - số điện thoại đọc từng chữ số.
  - Chọn cách đọc gần transcript của Gipformer gốc nhất.
  - Kết quả: lấy lại 112 câu bị bỏ và sửa các nhãn sai.
- **B. Augmentation.** SpecAugment, đổi tốc độ 0,9/1,0/1,1, trộn nhiễu theo đúng phân bố nhiễu nền của audio TRAIN.
- **C. TTS cho tên, số và từ lạ.**
  - Dùng TTS tiếng Việt mở có nhân bản giọng (VieNeu-TTS), giọng mẫu lấy từ audio TRAIN.
  - Nội dung: 5–10 nghìn câu theo khuôn câu yêu cầu của TRAIN, thay thực thể bằng:
    - tên Việt kèm danh xưng;
    - tên và tựa đề nước ngoài, thương hiệu;
    - số tiền, số điện thoại, giờ;
    - cụm điều kiện ("dưới", "trên", "không đủ thì").
  - Không lấy gì từ public.
  - Luật cho phép dùng TTS để sinh dữ liệu khi phát triển.
- **D. Bản cuối:** thêm 762 audio val (+18%).
- **Phần lợi thêm:** cho ASR v2 chạy trên audio TTS sẽ ra bản transcript có noise giống lúc chạy thật, trên câu mà ASR chưa từng nghe.
  - Hiện noise trong dữ liệu train LLM nặng gấp 1,7 lần lúc chạy thật (WER 14,5 so với 8,7).
  - Còn ASR v2 chạy trên chính audio TRAIN thì không dùng được làm noise vì nó đã học thuộc (WER 4,6).
  - Dữ liệu này cũng dùng được cho đề xuất 1.

### Đo và ngưỡng giữ

- Trên 762 audio val, dùng `analysis/hotword_eval.py`:
  - WER, hiện 8,66;
  - số mẫu có tham số nói nguyên văn bị hỏng, hiện 13/115;
  - tỷ lệ nhận đúng tên.
- Sau đó chạy cả pipeline trên val.
- Giữ nếu WER và số tham số hỏng cùng giảm.

### Ước lượng

- Bước A–B: +0,5 đến +1,5 val.
- Bước C: tên mất 3,8 điểm val và public còn nhiều từ lạ hơn (37% câu có âm tiết chưa gặp, val 26%). Ước lượng thêm +1 đến +2 nhưng chưa chắc, phụ thuộc chất lượng TTS.

### Chi phí, rủi ro

- A–B: nửa ngày công, khoảng 1 giờ GPU.
- C: 1 ngày công, sinh TTS khoảng 1–2 giờ GPU, train khoảng 1–2 giờ.
- Rủi ro: giọng TTS lệch audio thật. Cách giảm: giữ audio thật tối thiểu 50% và luôn đo trên audio thật.

## 4. Đề xuất 3: ghi chú tính toán trước call ghi

### Bằng chứng

- **Ma trận hành động:** với transcript chuẩn, model chọn đúng loại hành động ở 194/200 câu.
- **Ma trận tham số:** lệch nhiều nhất ở ngày giờ (12) và văn bản (10).
- **Các cụm lỗi cần một bước tính:**

  | Cụm | Điểm mất |
  |---|---:|
  | Điều kiện, tính trên kết quả tool ("ví không đủ thì…", 9h30–10h20 tính ra 40 phút) | 5,05 |
  | Ngày giờ tương đối | 2,00 |
  | Chọn thực thể | 2,00 |
  | Hỏi lại hay làm luôn | 1,50 |
  | **Cộng** | **10,55** |

- Hiện mỗi lượt chỉ sinh khoảng 26 token, đi thẳng vào call.
- Thinking mode quá chậm: 500 token mỗi lượt nhân 3,6 lượt sẽ vượt 30 s.
- Còn dư khoảng 20 s mỗi mẫu, đủ cho ghi chú 30–60 token.
- Ở DataMate, ghi chú trước SQL đã giúp đúng kiểu lỗi này.

### Làm gì

- Model thầy chạy local (Qwen3.8-27B-int4, đã có sẵn) đọc quỹ đạo gold và viết một dòng trước mỗi call ghi. Ví dụ: "Ví còn 85.000đ < 120.000đ nên trừ ngân hàng."
- **Kiểm chứng tự động:**
  - mọi số trong ghi chú phải có trong yêu cầu hoặc kết quả tool, hoặc tính ra đúng giá trị tham số gold;
  - ghi chú không được nhắc tới thứ agent không thấy được;
  - ghi chú nào không qua thì bỏ.
- SFT theo công thức q35e1. Agent đã giữ content đi kèm tool call (`src/lifemate/agent.py:378`), nên không phải đổi vòng lặp.

### Đo và ước lượng

- Đo val với transcript chuẩn và với ASR, xem riêng 11 câu thuộc cụm điều kiện; đo thêm LAO.
- Giữ nếu val tăng ít nhất 1,5 điểm (lớn hơn nhiễu) và ma trận hành động không lệch.
- Sửa được 1/3 số lỗi thì được +3 với transcript chuẩn, khoảng +2 với ASR.

### Chi phí, rủi ro

- 1 ngày công, khoảng 3 giờ để model thầy sinh ghi chú, cộng 3,5 giờ train.
- Rủi ro: model viết ghi chú nghe hợp lý nhưng sai. Bước kiểm chứng giảm được rủi ro này lúc train.

## 5. Lần train cuối và lịch

**Bản cuối:**
- Dữ liệu: toàn bộ 1.545 task TRAIN, cộng các phần đã qua ngưỡng ở đề xuất 1 và 3.
- Bản ASR thứ hai của mỗi task lấy từ TTS rồi ASR v2 (đề xuất 2), hoặc cross-fit.
- Bỏ các bản ASR mâu thuẫn với nhãn (12,5%).
- 2 epoch, đo dev loss mỗi 50 bước.
- Dùng ASR v3 nếu đề xuất 2 qua ngưỡng.

| Ngày | GPU | Việc khác (mình tự chạy, bạn theo dõi qua remote) |
|---|---|---|
| **09/11** | Train ASR v3 bước A–B (khoảng 1 giờ), đo WER; train q35 + dữ liệu 1a (khoảng 3,5 giờ); chạy val ASR và file public | Viết bộ sinh 1a, sửa nhãn số ASR, cài TTS và thử nhân bản giọng |
| 10/11 | Model thầy sinh ghi chú (khoảng 3 giờ); train q35 + ghi chú (đề xuất 3); sinh audio TTS | Viết bộ sinh 1b |
| 11/11 | Train ASR v3 bước C; train LAO + 1b; đo | Gộp các phần thắng |
| **12/11** | **Train bản cuối** (khoảng 4 giờ), chạy public | Dựng Docker |
| 13–15/11 | Chạy Docker 2 lần so khớp, đo thời gian, nộp | Tắt prefix cache để kết quả tất định |
| 16/11 | Đóng băng, nộp | |

**Đã thử và loại, không làm lại:**
- train 3 epoch;
- gợi ý từ vựng cho ASR theo ngữ cảnh;
- ASR lớn hơn;
- đổi tên app khi train;
- synth v1 viết tay;
- luật hậu xử lý;
- hotword toàn cục.
