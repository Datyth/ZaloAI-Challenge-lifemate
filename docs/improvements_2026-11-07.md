# Cải tiến pipeline LifeMate, đêm 07/11/2026 (đang cập nhật)

Báo cáo này nối tiếp [phân tích lỗi](error_analysis_2026-11-07.md). Cách đánh giá giữ nguyên: 200 câu validation cố định của TRAIN và điểm local `analysis/score.py` (hash là cổng, rồi tính F1 của call).

Riêng vụ viết thường transcript: val tăng 7.6 điểm và public tăng 8.0 điểm (30.77 lên 38.80). Như vậy val phản ánh đúng chiều thay đổi, dù thấp hơn public khoảng 5–6 điểm.

## 1. Tiền xử lý ASR

### 1.1 Hotword (sherpa-onnx, beam search có biasing)

- **Danh sách hotword:** `outputs/asr/hotwords/intent_v2.txt`, 34 cụm gồm động từ ý định (dời, nhắc, nhắn, đặt báo thức, họp, tắt, hủy…), tên app/thương hiệu (Zalo, SMS, SJC) và cụm thời gian. Điểm biasing 4.0.
- **Mức ASR**, đo trên 762 audio của 200 câu val:
  - WER từ 17,05% xuống 16,75%;
  - Zalo giữ được 9/9 (trước 4/9), SJC 15/18 (trước 7/18), "báo thức" 20/25 (trước 14/25), "nhắn" 45/51 (trước 40/51);
  - "dời" vẫn chỉ 4/14.
- **Mức agent** (S1 + viết thường): **44.0 lên 46.2**, sửa 7 câu và không làm hỏng câu nào.

### 1.2 Fine-tune Gipformer (ASR v2)

Lần fine-tune trước của Đạt hỏng vì target bị `<unk>`. Bản v2 (`scripts/train_gipformer_sft_v2.py`, môi trường riêng `.venv-asr`: torch 2.8 cu128, k2, kaldifeat, icefall) làm khác ở các điểm sau:

- **Target:** viết hoa, bỏ dấu câu, đúng bộ BPE gốc của model.
- **Câu có chữ số:** đoạn số trong ref ("45 phút", "123 triệu") được thay bằng cách đọc mà Gipformer gốc nghe được trên chính audio đó (`scripts/asr_targets.py`). Nhờ vậy giữ được 256 câu có số mà bản cũ bỏ đi. Còn 112 câu không căn được thì bỏ.
- **Dấu thanh:** chuẩn hoá theo kiểu của model ("HOÀ" thành "HÒA").
- **Cấu hình train:** toàn bộ model, lr 2e-5, tích luỹ gradient 4 bước, 2 epoch, trên 3.644 audio train (đã loại audio của 200 câu val và các mẫu dùng chung audio với chúng), khoảng 27 phút.
- **Dev loss:** 1.86 xuống 1.51; chọn checkpoint ở bước 7200.

Kết quả mức ASR trên 762 audio của 200 câu val (model chưa thấy audio nào trong số này):

| ASR | WER | dời | nhắc | nhắn | Zalo | SJC | họp | báo thức | đặt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Gipformer gốc | 17,05% | 2/14 | 33/43 | 40/51 | 4/9 | 7/18 | 11/16 | 14/25 | 31/42 |
| Gốc + hotword | 16,75% | 4/14 | 35/43 | 45/51 | 9/9 | 15/18 | 12/16 | 20/25 | 32/42 |
| **ASR v2** | **10,74%** | **13/14** | **43/43** | 48/51 | 9/9 | **18/18** | **16/16** | 24/25 | **42/42** |
| ASR v2 + hotword | 11,09% | 14/14 (+12 chèn sai) | 43/43 | 48/51 | 9/9 (+7 chèn sai) | 18/18 | 16/16 | 25/25 | 42/42 |

Với ASR v2 thì **không dùng hotword**, vì hotword làm chèn sai nhiều. Model nằm ở `models/gipformer-lm-v2-01` (ONNX, cùng định dạng với bản gốc, 69,65M tham số).

Đã thử thêm một bản dài hơn (4 epoch, lr 3e-5, `gipformer-lm-v2-02`). Bản này dừng sớm ở bước 5400 với WER val 10,98%, không tốt hơn v2-01, nên giữ v2-01.

## 2. SFT v2 cho LLM (S2)

Xem `scripts/prepare_agent_sft_v2.py` và `scripts/train_agent_sft_v2.py`.

- **Dữ liệu:** mỗi mẫu train có hai bản, một với transcript chuẩn và một với transcript Gipformer viết thường (cả lượt hiện tại lẫn lịch sử). Tổng 2.396 quỹ đạo, đã loại 200 câu val.
- **Độ dài:** tối đa 16.384 token, không mẫu nào bị bỏ (dài nhất 10,8k).
- **Orchestration S2** (`--orchestration S2`), dùng như nhau khi train và khi suy luận:
  - prompt E1 cộng thêm quy ước rút từ phân tích lỗi (lời nói có thể bị nghe nhầm; trả bằng nguồn A, thiếu thì nguồn B, nên đọc số dư trước; không tự điền giá trị thiếu; giữ danh xưng khi tìm liên hệ);
  - `date_reference`: bảng ngày và thứ của hôm qua đến 13 ngày tới, cùng khoảng tuần này và tuần sau.
- **Cách train:** cả quỹ đạo trong một lượt (v1 tách mỗi quyết định thành một mẫu nên lặp lại prompt dài), loss trên mọi lượt assistant. Đã kiểm tra prefix của từng lượt trùng với text lúc suy luận (0 lần lệch).
- **Cấu hình:** Unsloth QLoRA r16, lr 1e-4 cosine, 2 epoch (600 bước), lưu checkpoint mỗi 150 bước.

## 3. Kết quả agent trên val (cập nhật dần)

| Cấu hình | Điểm val | Ghi chú |
|---|---:|---|
| S1 + Gipformer thô | 36.4 | public 30.77 |
| S1 + Gipformer viết thường | 44.0 | public 38.80 |
| S1 + Gipformer + hotword, viết thường | 46.2 | |
| **S1 + ASR v2 viết thường** (file nộp: `result/submission_s1_asrv2_lower_20261107.jsonl`, 634 dòng, 634/634 hash khớp khi chạy lại, 0 lỗi schema, chưa nộp) | **56.1** | khớp state 130/200; so với 44.0: sửa 32 câu, hỏng 6; khoảng cách tới oracle còn 15.3 (trước 27.5) |
| S2 checkpoint 300 (epoch 1) + ASR v2 | 52.1 | khớp state 123; lr còn cao, chưa hội tụ |
| S2 checkpoint 450 + ASR v2 | 57.8 | khớp state 138 |
| **S2 cuối (epoch 2) + ASR v2** (**public 47.4**, file `result/submission_s2_asrv2_20261107.jsonl`) | **58.7** | khớp state 136, 0 lỗi runtime; câu có ghi, câu không lịch sử và câu lịch sử dài đều tăng so với S1 |

### S2 làm tốt hơn ở đâu

| LLM | Transcript chuẩn | Gipformer gốc viết thường | ASR v2 viết thường |
|---|---:|---:|---:|
| S1 | 71.4 | 44.0 | 56.1 |
| S2 | 70.8 | **50.8** | **58.7** |

- Khi đầu vào sạch, S2 giữ nguyên năng lực (khác biệt nằm trong dao động).
- Khi ASR nhiều lỗi, S2 tốt hơn hẳn: +6.9 điểm với Gipformer gốc, +2.5 với ASR v2. Đây đúng là tác dụng của việc train thêm bản transcript ASR.
- Trên public, so với S1, S2 trả rỗng ít hơn (29 câu so với 39) và hỏi thông tin thiếu nhiều hơn (56 so với 38), gần với phân bố nhãn train hơn (7% rỗng, 14% hỏi thiếu).

## 4. Nhận xét về khoảng cách val và public

| Cấu hình | Val | Public | Chênh |
|---|---:|---:|---:|
| S1 + Gipformer thô | 36.4 | 30.77 | 5.6 |
| S1 + Gipformer viết thường | 44.0 | 38.80 | 5.2 |
| S2 + ASR v2 | 58.7 | 47.4 | 11.3 |

Khoảng cách tăng gấp đôi, nghĩa là phần cải tiến chỉ chuyển sang public được khoảng 60%. Giải thích hợp lý nhất: ASR v2 và S2 đều chỉ học trên dữ liệu và app của train, trong khi khoảng 42% câu public dùng 11 app mới (grocery, train_bus_tickets, recipes, parking, event_tickets…) với từ vựng ASR chưa gặp. Tập val 200 câu chỉ chứa app có trong train, nên không đo được phần này.

## 5. S1c và bản nộp tốt nhất

| Cấu hình | Val | Public |
|---|---:|---:|
| S1c (học tiếp từ S1, prompt E1, transcript chuẩn + ASR, 1 epoch, lr 5e-5) + ASR v2 | 60.0 | |
| **S1c + ASR v2 + `--dedup-calls`** (không chạy lại một call giống hệt; dừng sau 3 lần lặp) | | **48.18** |

File: `result/submission_s1c_asrv2_dedup_20261107.jsonl`.

## 6. Phân tích lỗi tiếp (chiều 07/11)

### 6.1 Public: app mới chiếm nửa số câu và yếu hơn rõ

`analysis/unseen_apps.py` tách public theo app mà agent gọi. Chỉ dùng log chạy và catalog, không dùng nhãn, không đọc `public_test_environment`.

| Nhóm (S1c + ASR v2 + dedup) | Số câu | Hỏi thiếu | Đề xuất bị chặn | Lỗi tool | Kẹt vòng lặp | Chỉ đọc rồi dừng |
|---|---:|---:|---:|---:|---:|---:|
| Chỉ gọi app có trong train | 299 (47%) | 10% | 6% | 2% | 0% | 9% |
| Có gọi app mới | 297 (47%) | **4%** | **9%** | **4%** | **2%** | **13%** |

- Trên val, câu có đề xuất bị chặn đều được 0 điểm (6 câu).
- Ước lượng thô: nếu nhóm app cũ trên public được khoảng 55, nhóm app mới chỉ khoảng 40. Như vậy app mới kéo điểm public xuống khoảng 7 điểm. Đây là ước lượng, cần thí nghiệm ở mục 6.4 để đo.
- ASR v2 không có dấu hiệu phá từ vựng app mới: nó thay 12,5% số từ so với Gipformer gốc trên câu app mới, 10,4% trên câu app cũ, 12,6% trên val.

### 6.2 Val (app đã thấy): còn mất 40 điểm ở đâu

S1c + ASR v2, 200 câu:

| Nhóm lỗi | Câu | Điểm mất |
|---|---:|---:|
| Đúng tool, sai tham số | 35 | 17,5 |
| Nhãn hỏi thông tin thiếu, model lại tự làm | 13 | 6,3 |
| State đúng nhưng thừa/thiếu lệnh đọc | 19 | 5,7 |
| Thiếu hoặc sai lệnh ghi | 9 | 4,5 |
| Thừa lệnh ghi | 5 | 2,5 |
| Model hỏi trong khi nhãn làm | 4 | 2,0 |
| Nhãn rỗng, model lại làm | 3 | 1,5 |

- **Sai tham số:**
  - Nhiều nhất là thời gian (khoảng 19 lần). Chủ yếu là phép tính lệch theo một mốc: "đón trước giờ hẹn 45 phút", "đón sớm nửa tiếng", nhắc trước 15 phút so với giờ đón. Một phần do ASR nghe sai con số.
  - Kế đến là tên trong `contacts.query` (6 lần).
- **Hỏi thiếu:** model bịa giá trị bắt buộc mà người dùng không nói ("chút tiền" thành 10.000, "mức chị hay coi" thành volume 10, "nâng mục tiêu bước" thành 8.000, "mai" mà không có giờ thành 10:00).

### 6.3 Thử luật "số không có căn cứ thì hỏi lại": không dùng

- **Cách làm:** `analysis/ground_numbers.py` đổi một lệnh thành `sys.ask_missing_info` khi một tham số số không có trong transcript (sau ITN), context, lịch sử và kết quả tool trước đó. Chạy offline, chấm lại bằng bộ chấm local.
- **Kết quả:** val 59.96 lên 60.16; 11 câu bị sửa, 3 tốt lên, 2 hỏng.
- **Lý do không dùng:** nhiều giá trị đúng được tính ra chứ không chép nguyên văn (3 mét ra 300 cm, "một trăm rưỡi" ra 150.000, đổi đô ra tiền Việt). Nới luật để chấp nhận phép tính thì số bịa nào cũng thành có căn cứ.

### 6.4 Thí nghiệm "app chưa thấy" (đang chạy)

Val 200 câu chỉ có app đã thấy, nên không đo được phần yếu nhất của public. Thí nghiệm dựng một tập đo riêng:

- **App bị giấu:** 6 app train giống kiểu app mới trên public: food_delivery, flights, bike_rental, package_tracking, fitness, shopping_list.
- **Dữ liệu SFT:** bỏ mọi câu gọi các app này, kể cả trong lịch sử, và xoá schema của chúng khỏi prompt các câu khác (`--exclude-apps … --strip-held`). Còn 1.882 quỹ đạo.
- **Hai adapter**, cùng công thức (từ Qwen3-4B gốc, prompt E1, 2 epoch, lr 1e-4):
  - `lao_base`: dữ liệu thường;
  - `lao_aug`: 50% quỹ đạo được đổi tên namespace của mọi app sang tên khác cùng nghĩa (`--rename-apps 0.5`, ví dụ `smart_home` thành `iot_hub`). Đổi nhất quán ở tên function, mô tả, x-ref và lịch sử, để model học đọc schema thay vì nhớ app. Không dùng tên app nào của public.
- **Đo:**
  - 182 câu train gọi app bị giấu, với transcript chuẩn và với Gipformer gốc viết thường;
  - 186 câu val không chứa app bị giấu, với ASR v2, để kiểm tra phần app đã thấy không bị tụt;
  - model gốc không LoRA trên cùng 182 câu, để biết LoRA giúp hay hại khi gặp app mới.
- **Lệnh chạy:** `outputs/logs/queue_lao.sh`, `outputs/logs/queue_lao2.sh` (tmux `jobs:lao`, `jobs:lao2`).
- **Lệnh chạy:** phần còn lại chạy lại bằng `outputs/logs/queue_lao_rest.sh` (tmux `jobs:lao3`), vì `wait` không đối số trong `queue_lao.sh` chờ cả tiến trình vLLM.

#### Kết quả (tối 07/11)

| Model | App quen, transcript chuẩn (186) | App quen, ASR v2 (186) | App lạ, transcript chuẩn (182) | App lạ, Gipformer gốc viết thường (182) |
|---|---:|---:|---:|---:|
| Qwen3-4B gốc, không LoRA | 27,7 | | 29,9 | 20,4 |
| `lao_base` | **67,7** | 56,5 | **49,8** | 37,8 |
| `lao_aug` (đổi tên app) | | 57,1 | 48,7 | 34,0 |

- **Đổi tên app không giúp.** Bootstrap theo cặp, `lao_aug` − `lao_base`: app lạ transcript chuẩn −1,1 [−6,0; +3,7], app lạ ASR −3,8 [−8,0; +0,3], app quen +0,6. Bỏ hướng này; `outputs/sft/data_v2_e1_rename` không dùng.
- **LoRA vẫn có ích trên app lạ** (+20 điểm so với model gốc). Không nên chuyển sang model gốc khi gặp app mới.
- **Khoảng cách do chưa thấy app là khoảng 18 điểm.** Model gốc làm hai nhóm như nhau (27,7 và 29,9), nên 67,7 so với 49,8 không phải do nhóm câu app lạ khó hơn.
- **Điểm mất trên app lạ so với app quen** (`lao_base`, transcript chuẩn):

| Nhóm lỗi | App lạ | App quen |
|---|---:|---:|
| Đúng tool, sai tham số | 19,8 | 10,2 |
| Thừa lệnh ghi | 6,6 | 4,8 |
| State đúng nhưng thừa/thiếu lệnh đọc | 6,3 | 6,3 |
| Model hỏi trong khi nhãn làm | 6,0 | 1,6 |
| Nhãn hỏi, model lại làm | 5,5 | 5,0 |

- **Hai lỗi cụ thể có luật chung để chặn:**
  - **ID bịa trong mảng:** `food_delivery.place_order` có `items[].item_id` không gắn `x-ref`, nên bộ kiểm tra x-ref không chặn được. Model mới thấy app này thì không gọi `get_menu` mà bịa ID (`bunbo_01`, `item_01`). Có 14/41 ID lồng nhau không có căn cứ trên app lạ, 0/3 trên val S1c, 2/19 trên public S1c.
  - **Tham số tùy chọn chép từ hồ sơ:** model tự thêm `address` bằng địa chỉ nhà trong `user_profile` (22 lệnh trên app lạ, 0 trên val S1c, 7 trên public S1c: `grocery.checkout`, `ecommerce.checkout`…). Nhãn train chỉ làm vậy 2/1.545 lần, cả hai khi người dùng có nói "ở nhà mình", "ở công ty".

### 6.5 Luật `--ground-args`

`src/lifemate/agent.py`, chỉ áp cho tool ghi, không dựa vào tên app nào:

1. **ID phải có căn cứ.** Một tham số dạng `id` hoặc `*_id` mà schema không gắn `x-ref`, ở bất kỳ độ sâu nào (ví dụ `items[].item_id`, `merchant_id`), phải có trong một kết quả tool trước đó, lịch sử, context, hoặc lời người dùng. Nếu không, lệnh bị trả về lỗi "đọc danh sách của app trước rồi dùng đúng ID". Cả 59 ID loại này trong nhãn train đều có căn cứ (kiểm bằng cách chạy lại nhãn trong ToolEnv), nên luật không chặn lệnh đúng nào.
2. **Bỏ tham số tùy chọn chỉ chép từ context.** Một tham số tùy chọn kiểu chuỗi (không enum, không x-ref) có giá trị trùng một giá trị trong context (trừ danh sách app và giờ) bị bỏ, trừ khi người dùng nhắc tới. "Nhắc tới" là lời nói chứa đoạn đầu của giá trị, hoặc chứa "nhà" khi giá trị là địa chỉ nhà, hay "công ty / cơ quan / văn phòng / chỗ làm" khi là địa chỉ chỗ làm.

**Sửa sau lần chạy đầu:**
- Bản đầu coi mọi chuỗi đã thấy (kể cả tên món người dùng nói) là căn cứ cho ID, nên `item_id = "mì xào chay"` vẫn lọt (`public_0342`). Bản sửa chỉ nhận giá trị nằm ở trường `id`/`*_id` của kết quả tool, lịch sử hoặc context, hoặc một mã không có khoảng trắng mà người dùng đọc nguyên văn. Cả 59 ID trong nhãn train vẫn có căn cứ theo định nghĩa này.
- "Nhắc tới nhà/công ty" phải có giới từ chỉ nơi ("về nhà", "ở nhà mình", "tới công ty"); "nhà anh có bốn người" không tính. Luật không đổi lệnh ghi nào của nhãn train (0/1.545 câu).

**Kết quả:**
- Val 200 (S1c + ASR v2 + dedup): 59,71 không luật, 59,96 có luật. Luật không kích hoạt câu nào; 1 câu khác là do dao động khi chạy lại.
- App lạ (`lao_base`, 182 câu, transcript chuẩn): 49,79 lên **51,02**. Luật kích hoạt ở 17 câu: 3 câu tốt lên (`train_0694`, `train_1323`, `train_1379`) và 1 câu xấu đi (`train_0257`, 0,89 xuống 0). Câu `train_0257` xấu đi vì quỹ đạo đã rẽ khác từ lượt 2, trước khi luật kích hoạt: nó gọi `get_menu` sớm, rồi tự thêm tra danh bạ và chuyển ví. Phần lớn 13 câu còn lại vẫn 0 điểm vì lỗi khác (đếm số người trong nhóm chat).
- App quen (`lao_base`, 186 câu): 67,69 so với 67,56; luật không kích hoạt câu nào.
- Public, chỉ đọc log chạy: trên lần chạy 48,18, luật đổi 6/634 câu (2 `item_id` bịa, 4 địa chỉ chép từ hồ sơ trong `grocery.checkout`, `ecommerce.checkout`, `food_delivery.place_order`). `analysis/ground_fires.py` tìm các câu này.

**File nộp:**
- `result/submission_s1c_asrv2_dedup_ground6_20261107.jsonl`: file 48,18 với đúng 6 câu đó chạy lại bằng luật (`analysis/merge_submission.py --override`). Chênh với 48,18 chỉ do luật. 634/634 hash khớp khi chạy lại.
- `result/submission_s1c_asrv2_dedup_ground_20261107.jsonl`: chạy lại toàn bộ public có luật. 55 câu khác file 48,18, chỉ 4 câu do luật, còn lại là dao động GPU khi chạy lại (cùng nhiệt độ 0 và seed).
