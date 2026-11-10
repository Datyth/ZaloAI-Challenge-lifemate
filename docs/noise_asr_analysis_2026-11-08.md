# Noise ASR, ý tưởng noise augment và phần điểm ASR làm mất (08/11/2026, tối)

Báo cáo này đánh giá ý tưởng noise augment của Đạt (`adapter_one_mixed`, public 45,05) trên pipeline hiện tại: q35e1, public 58,02, val 70,91 với ASR và 82,33 với transcript chuẩn. Báo cáo cũng tách 11,4 điểm val mà ASR đang làm mất thành phần LLM còn sửa được và phần chỉ ASR mới sửa được.

Số liệu tính trên 200 ID val cố định, chưa bao giờ dùng để train. Script đo nằm trong repo:

| Script | Dùng để làm gì |
|---|---|
| `analysis/noise_analysis.py` | Đo noise ASR ở dữ liệu train và lúc chạy |
| `analysis/hotword_eval.py` | So các bản ASR có gợi ý từ vựng với ASR v2 thường |
| `analysis/asr_cache.py --context-hotwords` | Tạo transcript có gợi ý từ vựng theo từng mẫu |

## Tóm tắt

1. **Ý tưởng chính của Đạt đã có trong q35e1.**
   - Q35e1 dùng đúng cách chia 1.198 task train và 147 task dev.
   - Mỗi task có 2 bản: transcript chuẩn, và transcript Gipformer gốc cho cả câu hiện tại lẫn history (loại S2).
   - Đạt chọn một biến thể cho mỗi task, nên số lần model gặp noise còn ít hơn.
2. **LLM chịu noise tốt hơn thì cứu được tối đa khoảng 1,9 điểm val.**
   - Trong 30 mẫu đúng với transcript chuẩn nhưng sai với ASR, 21 mẫu có thông tin bị ASR làm mất hẳn: tên (6), từ ý định hoặc nội dung (6), từ điều kiện (5), số (4). Nhóm này chiếm khoảng 9,0 điểm.
   - 9 mẫu còn lại thông tin vẫn còn nguyên nhưng model bị noise làm chệch, chiếm khoảng 1,9 điểm. Đây là phần noise augment có thể sửa.
3. **Noise lúc train nặng gấp khoảng 1,7 lần noise lúc chạy thật.**
   - WER 14,5% so với 8,7%.
   - Tỷ lệ tham số nói nguyên văn bị hỏng: 26% so với 11%.
   - Từ khoá bị hỏng ở 30% số mẫu so với 15%.
   - 12,5% số bản ASR trong dữ liệu train có nhãn đòi một giá trị mà transcript không còn chứa.
   - Dù vậy, khi tham số bị hỏng, model chủ yếu chép đúng chữ ASR chứ không bịa: 6/12 tham số chép theo ASR, 4/12 vẫn ra đúng giá trị, 2/12 ra giá trị khác. Lọc các bản mâu thuẫn là việc đúng, nhưng lợi ích nhỏ.
4. **Phần còn lại phải sửa ở ASR.**
   - Với tên riêng: nhiều tên có sẵn trong kết quả tool ở history trước khi agent chạy.
   - Gợi ý từ vựng cho ASR theo từng mẫu, chỉ lấy từ hồ sơ người dùng và call/kết quả tool trong history, giúp tăng số cụm từ được nhận đúng từ 149/221 lên 171/221 mà không thêm cụm nào bị chèn sai.
   - Nhưng khi chạy hết pipeline, val không đổi thật (các mẫu có transcript đổi cộng lại −0,17 điểm; phần tăng chung +0,3 đến +0,8 là nhiễu của vLLM), vì hầu hết lỗi nằm ở những từ không có trong ngữ cảnh. Dừng hướng này.

## 1. Rà lại các bước của pipeline

**Tính nhất quán giữa train và lúc chạy:**
- Transcript ASR được chuẩn hoá bằng cùng một hàm `lifemate.context.normalize_asr` ở cả dữ liệu train (`scripts/prepare_agent_sft_v2.py`) và lúc chạy (`lifemate.cli`, `lifemate.docker_predict`).
- Prompt (`prompt_for`, `user_message`) và history (`structured_history`) cũng dùng chung code.
- Không tìm thấy chỗ nào lệch.

**Sức khoẻ từng bước** (trích từ `terminations.jsonl` và `tool_events.jsonl` của các lần chạy q35e1):

| Lần chạy | Số mẫu | Số lượt LLM/mẫu | Call bị chặn trước khi gọi env | Kết thúc bất thường |
|---|---|---|---|---|
| val, ASR v2 | 200 | 3,51 | 3 (đều là chặn call lặp) | 1 tool_error |
| val, transcript chuẩn | 200 | 3,46 | 0 | 2 tool_error |
| public, ASR v2 | 634 | 3,49 | 26: 20 call lặp, 4 ID chưa thấy, 2 tool không có trong catalog | 17 (2,7%) |

17 mẫu public kết thúc bất thường gồm:
- 9 tool_error (một lệnh ghi trả về `done: false`).
- 5 duplicate_loop.
- 3 step_limit.

13/17 mẫu này dính app không có trong TRAIN (event_tickets, recipes, train_bus_tickets, ecommerce, pantry, stocks, habit_tracker). Kiểu lỗi chung là tìm kiếm ra kết quả rỗng rồi model tìm lại liên tục. Trên val gần như không có, nên đây là lỗi riêng của app lạ, quy mô khoảng 1–2 điểm public.

Một số liệu phụ: prompt trung bình khoảng 6k token, lớn nhất 10,5k, còn xa giới hạn 16k. Không mẫu val nào chạm 10 lượt.

## 2. Noise trong dữ liệu train so với lúc chạy thật

Cả transcript chuẩn và transcript ASR đều được chuyển về một dạng chung rồi căn theo từng từ. Dạng chung gồm: chữ thường, số đọc thành chữ, bỏ dấu câu, và chuẩn hoá vị trí dấu thanh (để "hoà" và "hòa" là một từ).

Một tham số được tính là **nói nguyên văn** khi giá trị của nó xuất hiện đúng từng chữ trong transcript chuẩn. Nó bị tính là **hỏng** khi không còn lần xuất hiện nào giữ nguyên được qua ASR.

| Tập, ASR | WER câu hiện tại | WER cả history | Mẫu có tham số nói nguyên văn bị hỏng | Mẫu có từ khoá bị hỏng | Tham số nói nguyên văn bị hỏng |
|---|---|---|---|---|---|
| 1.345 ID train, Gipformer gốc (noise mà q35e1 học) | 14,5 | 14,7 | 12,5% | 29,7% | 26,4% |
| 1.345 ID train, ASR v2 trên chính audio nó đã train | 4,6 | 4,2 | 4,7% | 6,0% | 9,8% |
| Val, Gipformer gốc | 15,2 | 15,2 | 16,0% | 30,5% | 27,8% |
| Val, ASR v2 (đầu vào thật của agent) | 8,7 | 8,9 | 6,5% | 15,0% | 11,3% |

Từ khoá ở đây là động từ ý định, từ phủ định và bật/tắt.

**Những từ khoá hay hỏng:**
- Trong noise lúc train: đặt 79, nhắc 70, nhắn 64, dời 55, bật 40, tắt 36, mở 36, hẹn 30.
- Với ASR v2 trên val: "không" 9 mẫu, còn lại mỗi từ 1–4 mẫu.

ASR v2 đã sửa gần hết các động từ ý định. Trong 11 chữ "không" bị mất, 9 chữ là trợ từ hỏi cuối câu (ASR viết thành "hông"), nên vô hại. Chỉ 1 chỗ là mất phủ định thật: "không đủ" ở mẫu 0015.

**Rút ra:**
- Noise mà model học nặng gấp khoảng 1,7 lần lúc chạy, và có những kiểu lỗi mà ASR v2 không còn mắc, như "giời" thay cho "dời".
- Không thể lấy ASR v2 chạy trên chính audio TRAIN làm nguồn noise. Nó đã học thuộc các audio này (52% câu khớp hoàn toàn, WER 4,6%).
- Muốn có noise giống lúc chạy thật thì phải dùng cross-fit: chia TRAIN thành 5 phần, mỗi lần fine-tune ASR trên 4 phần rồi nhận dạng phần còn lại.

## 3. Ước lượng noise trên public, không cần nhãn

Tính trên audio câu hiện tại:

| Tập | Số audio | Mức bất đồng giữa Gipformer gốc và ASR v2 | Tỷ lệ từ ASR v2 không có trong TRAIN |
|---|---|---|---|
| Val | 200 | 12,6% | 1,69% |
| Public | 634 | 11,3% | 2,41% |

Mức bất đồng chung tương đương nhau, nhưng public có nhiều từ không có trong TRAIN hơn khoảng 1,4 lần (tên riêng, app lạ). Phần ASR làm mất vì tên riêng trên public vì vậy nhiều khả năng lớn hơn trên val.

## 4. Tách 11,4 điểm ASR làm mất trên val

Theo mức hỏng của transcript ASR v2 (tách tự động):

| Mức hỏng | Số mẫu | Điểm, transcript chuẩn | Điểm, ASR | Số điểm ASR làm mất | Đúng với transcript chuẩn, sai với ASR |
|---|---|---|---|---|---|
| Tham số nói nguyên văn bị hỏng | 13 | 96,2 | 58,2 | 2,47 | 7 |
| Từ khoá bị hỏng | 30 | 82,0 | 72,0 | 1,51 | 4 |
| Có lỗi từ khác | 124 | 80,9 | 67,7 | 8,18 | 19 |
| Không lỗi | 33 | 82,5 | 87,0 | −0,75 | 0 |

Nhóm "có lỗi từ khác" chiếm phần lớn số điểm. Vì vậy cả 30 mẫu ở cột cuối đã được đọc tay từng mẫu: transcript chuẩn, transcript ASR, gold call và call model dự đoán.

**Thông tin bị ASR làm mất: 21 mẫu, khoảng 9,0 điểm.** LLM không thể suy lại được.

| Loại | Mẫu | Ví dụ |
|---|---|---|
| Tên riêng | 6 | Peter Hansen → "minh sang" (1481); Nathan Reed → "nat timest" (1515); Northwind → "note wink" (1560); Brightpath → "bright off" (1687); Dũng → "vũ" (0236); Nhà giả kim → "nhà giả kêu" (1071) |
| Từ ý định hoặc nội dung | 6 | "chuyển tiền" → "trên truyền" (0015); "gọi báo" → "và báo" (0282); "tiền điện" mất "điện" (0501); "hàng chờ" → "hẹn chờ" (1327); "mang laptop" → "nâng lớp đóp" (0070); "sale" → "sau" (1686) |
| Từ điều kiện hoặc logic | 5 | "dưới 124 triệu" → "giá 124 triệu" (0029); "khoan trả" → "hoàn trả" (0080); "ờ nhầm" → "anh em" (1084); "bị mượn rồi thì giữ cuốn khác" → "vì mình rồi thì dẫn khách" (1219); "liền" → "vì" (1396) |
| Số | 4 | 131 → 121 triệu (0279); hai số điện thoại sai chữ số (0676, 0747); "chút tiền" → "chục ngàn", khiến model chuyển 10.000 thay vì hỏi lại số tiền (0164) |

**Thông tin còn nguyên, model bị noise làm chệch: 9 mẫu, khoảng 1,9 điểm.**
- Gọi thêm lệnh đọc khi câu bị nhiễu, 6 mẫu:
  - `banking.get_balance` thừa ở hai câu "ví không đủ thì lấy ngân hàng" (0216, 0313). Transcript câu hiện tại của hai mẫu này gần như giống hệt bản chuẩn; chỉ history có vài chữ sai.
  - Các lệnh đọc thừa khác: `contacts.search` (1575), `news.search` gọi với chữ vô nghĩa (1733), `food_delivery.order_history` (1379), `medication.list` (1474).
- Lẽ ra phải hỏi lại nhưng lại làm luôn, 2 mẫu: tự bịa 8.000 bước (1728), tự lưu một bài báo (0974).
- Chọn nhầm tool đọc, 1 mẫu: `household_chores.list_chores` thay vì `tasks.list` (1021).

**Model xử lý thế nào khi tham số bị hỏng.** Xét 12 tham số bị hỏng ở các mẫu model làm đúng với transcript chuẩn, rồi xem bản chạy với ASR ra gì:
- 6 tham số model chép đúng chữ ASR.
- 4 tham số vẫn ra đúng giá trị, nhờ lần tìm thứ hai hoặc nhờ kết quả tool.
- 2 tham số ra giá trị khác.

Model trung thành với transcript. Nhãn mâu thuẫn trong dữ liệu train chưa dạy model bịa ra giá trị giống gold.

**Kết hợp hai bản ASR có ích không:**
- Trên val, chỉ 1,7% số từ là Gipformer gốc nhận đúng mà v2 nhận sai; ngược lại là 7,8%.
- Với từ khoá, con số là 18 so với 95.
- Kết hợp hai bản ASR vì vậy gần như không còn dư địa.

ASR v2 có vài chỗ kém hơn bản gốc, như "dưới" ở 0029, và lặp "không" trong số điện thoại ở 0747. Lỗi lặp này hiếm: chỉ 1/1933 audio public.

## 5. Đánh giá ý tưởng noise augment của Đạt

| Thành phần | Q35e1 đã có chưa | Đánh giá |
|---|---|---|
| Trộn bản sạch với bản ASR, cùng nhãn gold | Có: 100% task có bản sạch, 100% có bản S2 | Phần cốt lõi đã nằm trong bản 58 điểm |
| S1: câu hiện tại ASR, history chuẩn | Không | Lúc chạy thật history luôn là ASR, nên S1 không giống đầu vào thật. Bỏ qua |
| Lọc bản noise làm mất tham số hoặc từ phủ định | Không | Đúng hướng. 12,5% bản ASR trong dữ liệu train đang mâu thuẫn với nhãn. Lợi ích chắc nhỏ vì model vốn đã chép đúng chữ ASR |
| Thêm 75 biến thể cho 7 nhóm tool hay sai trên val | Không | Chọn nhóm bằng chính val nên val không còn đo được nhóm đó. Chỉ khoảng 6% dữ liệu, 2/7 tool không xuất hiện ở public. Không làm |
| Nguồn noise là Gipformer gốc | Q35e1 cũng vậy | Noise nặng gấp khoảng 1,7 lần lúc chạy và khác kiểu lỗi. Nên dùng thêm noise cross-fit từ ASR v2 |

Về bằng chứng trong tài liệu của Đạt:
- Bản "noise mới" và "noise cũ" đều đã có noise.
- Trên val chỉ khác nhau 1 mẫu (126 lên 127, +1,1 điểm), trong khi public chênh 5,6 điểm.
- Ở các lần thay đổi trước của mình, val và public đi sát nhau: +7,6 thành +8,0; +11,2 thành +9,8; −1,8 thành −1,3.
- Vì vậy phần lớn mức +5,6 nhiều khả năng không đến từ 75 mẫu augment.
- Về điểm tuyệt đối: 45,05 thấp hơn bản Qwen3-4B tốt nhất của mình (S1c, 48,18).

**Kết luận:** không nên mở một vòng train riêng chỉ cho hướng này, vì trần của nó chỉ khoảng 1,9 điểm val. Khi train bản cuối trên toàn bộ TRAIN, nên gộp vào đó hai việc rẻ:
1. Bỏ các bản ASR có tham số nói nguyên văn bị hỏng, hoặc có từ phủ định bị hỏng.
2. Thêm một bản noise cross-fit từ ASR v2.

Cross-fit tốn khoảng 5 × 27 phút GPU.

## 6. Gợi ý từ vựng cho ASR theo từng mẫu (thử offline)

**Cách lấy cụm từ** (`lifemate.context.hotword_phrases`):
- Nguồn: chỉ những thứ agent đã thấy trước khi nhận dạng, tức hồ sơ người dùng và call cùng kết quả tool trong history. Không đọc file `*_environment`.
- Lấy: các giá trị ngắn có chữ viết hoa, cụm các chữ viết hoa liền nhau nằm trong văn bản dài, và từng tên nước ngoài trong những cụm đó.
- Mỗi mẫu trung bình 6,6 cụm, nhiều nhất 32 cụm.
- Cách gợi ý: hotword theo từng stream của sherpa-onnx, dùng ASR v2, modified beam search, beam 4.

**Kết quả trên 762 audio val:**

| Bản ASR | WER câu hiện tại | WER cả history | Cụm người nói thật sự nói | Trong đó nhận đúng | Cụm bị chèn dù không nói | Transcript thay đổi |
|---|---|---|---|---|---|---|
| ASR v2 | 8,66 | 8,86 | 221 | 149 | 1 | — |
| Gợi ý mức 1.0 | 8,64 | 8,71 | 221 | 164 | 1 | 18 |
| Gợi ý mức 2.0 | 8,64 | 8,61 | 221 | 171 | 1 | 33 |
| Gợi ý mức 3.0 | 8,62 | 8,60 | 221 | 173 | 1 | 35 |

Không thêm cụm nào bị chèn sai. Lần thử hotword cố định trước đây thì có chèn sai.

**Những chỗ được sửa đúng:** Peter Hansen (1368), Grace Nakamura, Sawaco, Kha, Long, Nhi, chợ Bình Tây, Python, Cao Nguyên Coffee, Mr Tanaka, "mạ", "trạm Biển Mỹ Khê".

**Những chỗ hỏng thêm:**
- "giặt" → "giao dịch" (1021).
- Mất "trân châu đường đen" (1575 history) và "mockup" (1687 history).
- Câu hiện tại của 1733 ("confirmờ") và 1728 ("skyli") tệ hơn.

Nguyên nhân hỏng là hotword chỉ khớp được một phần.

**Giới hạn:**
- Phần lớn thay đổi rơi vào audio history, vì tên được nhắc trong kết quả tool ngay sau câu nói đó.
- Ở câu hiện tại, chỉ 2/6 tên bị mất trong mục 4 có sẵn trước khi agent chạy: Peter Hansen và Northwind. Các tên còn lại chỉ xuất hiện sau khi agent gọi tool.

**Trên public:** 95/1933 audio thay đổi, trong đó 29 là câu hiện tại, chạm 69 mẫu. Có cả chỗ tốt lên ("xiaomi", "canh chua", "cung thiếu nhi") lẫn chỗ hỏng thêm ("lịch đó" → "lịch thanh").

**Kết quả chạy hết pipeline (q35e1, val 200, xong 20h27):**

| Bản | Điểm val | Số mẫu đúng trạng thái cuối | Số mẫu có transcript đổi | Thay đổi từ các mẫu đó |
|---|---|---|---|---|
| ASR v2, không gợi ý | 70,91 | 155 | 0 | 0 |
| Gợi ý mức 2.0 | 71,25 | 155 | 27 | −0,17 |
| Gợi ý mức 1.0 | 71,72 | 156 | 16 | −0,17 |

- Mức tăng chung là nhiễu của vLLM, không do gợi ý. Có 5 mẫu (mức 2.0) và 7 mẫu (mức 1.0) giữ nguyên transcript mà vẫn ra call khác. Các mẫu này cộng +0,50 và +0,97 điểm, riêng train_1396 từ 0 lên 1.
- Trong các mẫu có transcript đổi, đúng 1 mẫu sửa được (train_1733, 0,67 lên 1) và 1 mẫu hỏng thêm (train_1021, 0,67 xuống 0, "giặt" thành "giao dịch").
- Trong 32 mẫu mà ASR làm mất điểm (transcript chuẩn hơn ASR ít nhất 0,3), gợi ý chỉ đổi transcript của 7 mẫu. Lỗi của các mẫu còn lại nằm ở những từ không có trong hồ sơ hay history: tên mới (Phạm Hùng), số, từ ý định, từ điều kiện. Ngay cả "Peter Hansen" có trong gợi ý mà ASR vẫn ra "PETER HA SANG".
- **Kết luận:** không nộp `result/submission_q35e1_ctxhw_20261108.jsonl` (file hợp lệ, 634 dòng) và dừng hướng này. Chênh lệch val dưới khoảng 1 điểm giữa hai lần chạy là nằm trong nhiễu của vLLM.

## 7. Thứ tự ưu tiên tiếp theo

1. **ASR là đòn bẩy chính:** khoảng 9 trên 11,4 điểm val, và nhiều hơn trên public.
   - Gợi ý từ vựng theo ngữ cảnh: đã đo, không cải thiện (mục 6).
   - ASR lớn hơn trong giới hạn 6B: đã đo zero-shot trên 762 audio val (`analysis/whisper_cache.py`, `outputs/logs/queue_whisper_zs.sh`, 08/11 23h30). Kết quả kém xa Gipformer:

     | ASR | Tham số | WER câu hiện tại | Mẫu có tham số nói nguyên văn bị hỏng |
     |---|---|---|---|
     | Gipformer v2 (đang dùng) | 68M | 8,66 | 13/115 |
     | Gipformer gốc | 68M | 15,20 | 32/115 |
     | PhoWhisper-medium, zero-shot | 764M | 24,93 | 48/115 |
     | whisper-large-v3-turbo, zero-shot | 809M | 27,71 | 53/115 |

     Lỗi là lỗi nghe thật, không phải do chuẩn hoá ("khoá cửa kho giùm cô" thành "khoác cửa kho dùm khô"). Fine-tune Whisper phải đi từ mức gấp ba lần lỗi của v2, nên khó vượt được Gipformer v2.
   - Thêm dữ liệu TTS chứa tên nước ngoài, số và từ điều kiện để fine-tune ASR (luật cho phép dùng TTS lúc phát triển).
2. **Agent với app lạ:** sửa vòng lặp tìm kiếm ra kết quả rỗng (17 mẫu public).
3. **Bản cuối trên toàn bộ TRAIN:**
   - Giữ 2 epoch: bản 3 epoch tụt xuống val 66,84, so với 70,91.
   - Bỏ các bản ASR mâu thuẫn với nhãn.
   - Thêm noise cross-fit từ ASR v2.
4. **Không làm:** augment theo nhóm chọn bằng val, và biến thể S1.
