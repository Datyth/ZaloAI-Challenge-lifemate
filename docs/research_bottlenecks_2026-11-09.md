# Điểm nghẽn của pipeline LifeMate và hướng đánh tiếp (09/11)

Báo cáo gộp số đo của chính mình (val 200 câu TRAIN, LAO, public) với research các model, repo và paper liên quan, theo cách đã làm ở track 2. Mục tiêu là chọn việc đáng làm trong 6 ngày còn lại trước hạn 16/11.

## 1. Điểm mất nằm ở đâu

Số đo trên val, model q35e1:

| Đầu vào cho LLM | Điểm val | Mất so với mức trên |
|---|---|---|
| Transcript chuẩn (oracle) | 82,33 | 17,7 điểm do **agent** (app quen) |
| ASR v2 | 70,91 | 11,4 điểm do **ASR** |
| **ASR v3** (09/11) | **71,87** | ASR còn mất 10,5 |

Trên public (58,02), khoảng cách thêm chừng 13 điểm so với val. Ba quan sát:

- **ASR trên public không khó hơn val.** Hai bản ASR bất đồng 11,2% số từ trên public, 12,5% trên val. Tỷ lệ từ chưa từng gặp trong TRAIN là 3,3% so với 2,2%.
- **Phần lớn khoảng cách đến từ app lạ.** 45% câu public dùng app không có trong TRAIN. Ở thí nghiệm LAO (giấu 6 app khỏi TRAIN), các câu dùng app bị giấu thấp hơn 11 điểm. Như vậy app lạ chiếm khoảng 5–7 điểm.
- **Phần còn lại** là khác biệt phân bố: câu "thứ X tuần sau", câu nhiều bước, kết quả tìm rỗng.

Tổng thể, **agent mất nhiều điểm hơn (khoảng 18 trên app quen và 6 trên app lạ) so với ASR (khoảng 10,5)**. Nhưng hai phía khác nhau ở mức chắc chắn của cải tiến:

| Thay đổi | Val | Public |
|---|---|---|
| Lowercase transcript ASR | +7,6 | +8,0 |
| Đổi base sang Qwen3.5 | +11,2 | +9,8 |
| Thêm app tổng hợp (synth v1) | −1,8 | −1,3 |
| p1 (prompt, phục hồi, trọng số loss) | −1,4 | −0,4 |
| Bỏ phiếu theo hash (không hợp lệ, xem mục 4) | +2,6 | +0,4 |

**Cải tiến ASR và base model sang public gần như trọn vẹn. Các chỉnh hành vi agent đo trên val sang rất yếu**, vì val không có app lạ. Nên ưu tiên các đòn bẩy ASR, còn thay đổi phía agent phải đo bằng LAO chứ không bằng val.

## 2. ASR: còn đánh mạnh được không

**Đổi sang model lớn hơn: không nên.**
- Trên audio val của mình, các model lớn chạy zero-shot đều kém xa: PhoWhisper-medium 24,9% WER, whisper-large-v3-turbo 27,7%, Qwen3-ASR-0.6B 33,9%. Gipformer gốc chỉ 15,2%.
- Bảng xếp hạng ASR tiếng Việt mở ([Vietnamese Open ASR Leaderboard](https://vietaudio-team-vietnamese-asr-leaderboard.hf.space/)) cũng cho thấy họ Zipformer nhỏ đứng ngang hoặc trên PhoWhisper. [Zipformer-30M-6000h](https://hf-767.pages.dev/hynt/Zipformer-30M-RNNT-6000h) có N-WER trung bình tốt nhất bảng.
- Gipformer (fine-tune từ [VietASR](https://arxiv.org/pdf/2505.21527), Interspeech 2025) tự báo cáo tốt hơn Zipformer-30M trên 11/12 tập ([model card](https://hface.pages.dev/g-group-ai-lab/gipformer-65M-rnnt)).

Vậy backbone hiện tại đã đúng chỗ. Muốn giảm lỗi phải **thêm dữ liệu trong miền**, không phải đổi model.

**Lỗi ASR còn lại chủ yếu ở tên riêng, số và từ chỉ ý định.** Trong 11,4 điểm ASR mất (đo trên v2): tên 3,8, từ ý định 3,9, số 2,5. ASR v3 đã sửa nhãn số và thêm augmentation, nên WER giảm từ 8,66 xuống 7,74 và val tăng +0,9 (đã tách khỏi nhiễu chạy lại).

**Đòn bẩy tiếp theo: dữ liệu TTS cho tên riêng và số (ASR v4).**
- Literature: fine-tune Whisper trên chỉ 10 giờ audio tổng hợp giảm WER từ khoá hiếm từ 30,9% xuống 12,2%. Bổ sung câu tổng hợp vào dữ liệu thật giảm 16% WER trên thực thể có tên ([Interspeech 2023](https://www.isca-archive.org/interspeech_2023/naowarat23_interspeech.pdf)). TTS không tự sửa được lỗi do token hiếm hay cách viết lạ, nên cần đa dạng tên.
- Audio TRAIN của cuộc thi có dấu hiệu là giọng TTS cộng nhiễu, nên audio TTS sẽ khớp miền rất tốt.
- Công cụ: [VieNeu-TTS](https://pypi.org/project/vieneu/) (nhân bản giọng từ 3–5 giây, chạy được trên CPU; cần kiểm giấy phép từng bản).
- Nội dung: câu theo khuôn TRAIN, với tên người, tên địa điểm, tên sản phẩm và con số lấy từ kho tên tự tạo. **Không dùng tên lấy từ public test.**
- Chi phí khoảng nửa ngày: dựng TTS, sinh vài nghìn câu, train 30 phút. Kỳ vọng +1 đến +3 val, và loại cải tiến này sang public đủ.

**Đưa nhiều giả thuyết ASR cho LLM.** Hướng sửa lỗi sinh (GER) và [RECOVER (2026)](https://arxiv.org/pdf/2603.16411) dùng nhiều giả thuyết ASR cùng danh sách thực thể ứng viên, giảm 8–46% (tương đối) WER trên cụm thực thể. Với mình, có thể đưa vào prompt đầu ra của cả ASR v3 lẫn Gipformer gốc. Nhưng LLM phải được train lại với đầu vào dạng đó, và cần nhiễu "cross-fit" (ASR không được train trên chính audio đó), tốn khoảng 5 × 30 phút train ASR. Đây là việc lớn, khó kịp trước 16/11 → **để sau**.

## 3. Pipeline / agent: cải thiện được gì

**Model nói thẳng ra lời gọi tool (speech LLM end-to-end): không khả thi.**
- [Voxtral Mini 3B](https://huggingface.co/papers/2507.13264.md) gọi tool trực tiếp từ giọng nói nhưng không hỗ trợ tiếng Việt.
- Benchmark mới [Audio2Tool (2026)](https://arxiv.org/pdf/2604.22821) cho thấy cả speech LM lẫn chuỗi ASR→LLM đều giảm mạnh ở câu nhiều ý và câu có nhiễu. Không có bằng chứng end-to-end hơn chuỗi ASR→LLM.

**RL (GRPO) trên ToolEnv: không đáng.** Nghiên cứu có kiểm soát [SFT or RL for Tool-Calling Agents (2026)](https://arxiv.org/pdf/2609.17848), với model từ 0,6B đến 32B:
- SFT + LoRA thắng 15/18 thiết lập khi dữ liệu train và test cùng phân bố.
- GRPO chỉ hơn trung bình dưới 1 điểm khi chuyển sang tập dữ liệu khác.

Với 6 ngày và một GPU, không nên làm RL.

**Tổng quát hoá sang app lạ.**
- [Hammer (ICLR 2025)](https://arxiv.org/pdf/2410.04587) che hoặc đổi ngẫu nhiên tên hàm để model dựa vào mô tả, kèm dữ liệu "không tool nào hợp". Nghiên cứu ở trên cũng cho thấy **trộn nhiều nguồn dữ liệu** giúp chuyển giao tốt mà không mất điểm trong phân bố.
- Thử nghiệm của mình lại âm: đổi tên app (LAO −1,1 trên val ASR) và synth v1 (public −1,3). Lý do là dữ liệu tự sinh lệch quy ước nhãn.
- Nếu làm tiếp thì chỉ trộn một ít, giữ quy ước của TRAIN, và **chỉ quyết định dựa trên LAO**.

**Ensemble hợp lệ: bỏ phiếu theo từng bước trên một ToolEnv.**
- Quy chế: *"Mỗi mẫu dùng một ToolEnv riêng và chỉ được tạo một lần … không được tạo lại ToolEnv để thử rồi làm lại"* (README của BTC).
- Bỏ phiếu theo state hash giữa các lần chạy độc lập **không được phép**. File 58,42 tạo theo cách này nên không dùng cho bài nộp cuối.
- Thay vào đó, `src/lifemate/ensemble.py` hỏi cả 3 adapter ở mỗi lượt, rồi chỉ thực thi lời gọi được đa số chọn. Lượt đo trên val đang chạy.

**Hiện trạng phía agent sau p1:**
- Đã sửa: ngày "tuần sau" (sai 5 câu → 1), hỏi lại khi gặp app lạ (17 → 24 câu), vòng lặp (giới hạn thử lại).
- Còn lại là các lỗi suy luận trên app quen: điều kiện và phép tính 5,1 điểm, ngày tương đối 2,0, chọn thực thể 2,0, hỏi hay làm 1,5. Đề xuất "ghi chú tính toán trước lời gọi ghi" (teacher 27B) vẫn đúng hướng, nhưng phải train lại, và lợi ích có thể không sang public.

## 4. Đề xuất cho 6 ngày còn lại

| Thứ tự | Việc | Vì sao | Chi phí | Kỳ vọng |
|---|---|---|---|---|
| 1 | **Nộp q35e1 + ASR v3** (hôm nay) | +0,9 val, đã tách khỏi nhiễu; ASR sang public đủ | đã xong | khoảng 58,9 |
| 2 | **ASR v4: thêm dữ liệu TTS** cho tên riêng, số, từ ý định | lỗi ASR còn lại nằm ở đây; literature và miền audio đều ủng hộ | nửa ngày | +1 đến +3 val |
| 3 | Ensemble theo từng bước (nếu val cho thấy lợi) | hợp lệ; dùng các model đã có | đo hôm nay | chờ số |
| 4 | **Train bản cuối trên toàn bộ TRAIN** (LLM thêm 200 task val, ASR thêm 762 audio val) | trước đây bớt 21% task thì mất 6,2 điểm | khoảng 5 giờ GPU, ngày 12–13/11 | +1 đến +2 |
| 5 | Đóng gói docker, đo độ trễ, khoá code | bắt buộc | 14–15/11 | — |
| — | Không làm: RL, speech LLM end-to-end, đổi backbone ASR, bỏ phiếu theo hash | bằng chứng ở trên | — | — |
