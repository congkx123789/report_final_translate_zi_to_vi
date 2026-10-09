# 🏆 BÁO CÁO ĐỐI CHIẾU CHẤT LƯỢNG DỊCH THUẬT VÀ BLEU SCORE TẠI CHECKPOINT 200,000 STEPS

- **Mô hình A (Đề xuất):** `EvolvingHeadSSMModel` (`checkpoints/ssm_model_step_200000.pt`, 71.97M params)
- **Mô hình B (Baseline):** `StandardTransformerDecoder` (`checkpoints_transformer/transformer_step_200000.pt`, 68.2M params)
- **Phần cứng:** NVIDIA GeForce RTX 5060 Ti
- **Số lượng câu test:** 50 câu song ngữ đa miền
- **Thời gian chạy test:** 2026-09-17 13:16:18

## 1. Bảng Tổng Hợp Chỉ Số Đánh Giá Toàn Tập

| Chỉ số đánh giá | Transformer (Baseline) | Evolving-Head Hybrid SSM (Đề xuất) | Chênh lệch / Ưu thế |
| :--- | :---: | :---: | :---: |
| **Corpus SacreBLEU** | **28.53** | **26.56** | **-1.97 BLEU** |
| **Corpus chrF++** | 47.00 | 45.60 | -1.40 pts |
| **NLTK BLEU-1** | 52.96 | 52.49 | -0.47 pts |
| **NLTK BLEU-2** | 40.57 | 39.04 | -1.53 pts |
| **NLTK BLEU-3** | 31.85 | 29.85 | -2.01 pts |
| **NLTK BLEU-4** | 25.43 | 23.57 | -1.86 pts |
| **ROUGE-L Score** | 62.87 | 62.30 | -0.57 pts |
| **Tốc độ sinh (tokens/s)** | 753.0 tok/s | 149.4 tok/s | **0.20x Nhanh hơn** |
| **Độ trễ (Latency)** | 53.8 ms | 242.7 ms | -188.9 ms nhanh hơn |
| **VRAM tiêu thụ** | 1991.1 MB | 1994.7 MB | **-0.2% Tiết kiệm** |

## 2. Bảng Phân Tích Theo Từng Miền Dữ Liệu

| Miền dữ liệu | Số câu | Transformer BLEU | Hybrid SSM BLEU | Chênh lệch BLEU | Tốc độ SSM | Tăng tốc độ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hội thoại thường nhật** | 10 | 47.32 | 44.01 | **-3.31** | 145.0 tok/s | **0.19x** |
| **Tin tức & Xã hội** | 8 | 47.03 | 45.67 | **-1.36** | 153.9 tok/s | **0.21x** |
| **Tiên hiệp & Tu chân** | 19 | 20.22 | 21.42 | **+1.20** | 152.1 tok/s | **0.19x** |
| **Cổ phong & Kiếm hiệp** | 8 | 20.90 | 18.73 | **-2.17** | 155.5 tok/s | **0.20x** |
| **Đoạn văn dài & Viễn tưởng** | 5 | 21.28 | 15.51 | **-5.77** | 131.2 tok/s | **0.26x** |

## 3. Trích Xuất 10 Cặp Câu Song Ngữ Điển Hình

### Câu #1 (Hội thoại thường nhật)
- 🇨🇳 **Nguyên tác Tiếng Trung:** `今天的会议推迟到下午两点，请大家准时到三楼会议室。`
- 🎯 **Bản dịch chuẩn (Ground Truth):** Cuộc họp hôm nay được hoãn đến hai giờ chiều, xin mọi người đến phòng họp tầng ba đúng giờ.
- 🟢 **Evolving-Head Hybrid SSM:** Cuộc họp hôm nay đã hoãn đến hai giờ chiều, xin mọi người đúng giờ đến phòng họp tầng ba. *(BLEU: 67.4 | 64.8 tok/s)*
- 🔴 **Standard Transformer:** Cuộc họp hôm nay đã hoãn đến hai giờ chiều, xin mọi người đến phòng họp tầng ba. *(BLEU: 73.7 | 114.6 tok/s)*

### Câu #2 (Hội thoại thường nhật)
- 🇨🇳 **Nguyên tác Tiếng Trung:** `这份项目报告我已经发到你的邮箱了，请查收一下。`
- 🎯 **Bản dịch chuẩn (Ground Truth):** Bản báo cáo dự án này tôi đã gửi vào hòm thư của bạn rồi, xin vui lòng kiểm tra.
- 🟢 **Evolving-Head Hybrid SSM:** Báo cáo dự án này ta đã gửi đến hộp thư của ngươi rồi, xin vui lòng kiểm tra xem *(BLEU: 42.7 | 152.9 tok/s)*
- 🔴 **Standard Transformer:** Dự án này báo cáo rằng ta đã gửi đến hộp thư của ngươi, xin vui lòng kiểm tra xem xét. *(BLEU: 29.5 | 834.5 tok/s)*

### Câu #3 (Hội thoại thường nhật)
- 🇨🇳 **Nguyên tác Tiếng Trung:** `晚上一起去吃火锅吧，我知道附近新开了一家不错的店。`
- 🎯 **Bản dịch chuẩn (Ground Truth):** Tối nay cùng đi ăn lẩu nhé, tôi biết gần đây mới mở một quán rất ngon.
- 🟢 **Evolving-Head Hybrid SSM:** Tối hôm đó cùng nhau đi ăn lẩu, ta biết gần đây mới mở một quán bar không tệ. *(BLEU: 37.0 | 152.1 tok/s)*
- 🔴 **Standard Transformer:** Tối nay cùng đi ăn lẩu đi, ta biết gần đó có một cửa hàng tốt đẹp ở khu vực mở. *(BLEU: 25.3 | 830.6 tok/s)*

### Câu #4 (Hội thoại thường nhật)
- 🇨🇳 **Nguyên tác Tiếng Trung:** `周末你有什么打算？要不要一起去电影院看新上映的电影？`
- 🎯 **Bản dịch chuẩn (Ground Truth):** Cuối tuần bạn có dự định gì không? Có muốn cùng nhau đến rạp chiếu phim xem bộ phim mới ra mắt không?
- 🟢 **Evolving-Head Hybrid SSM:** Cuối tuần ngươi có ý định gì? Có muốn đi xem phim mới cùng rạp chiếu phim không? *(BLEU: 14.4 | 153.1 tok/s)*
- 🔴 **Standard Transformer:** Cuối tuần, ngươi có ý định gì không? Có muốn cùng đi xem phim mới ra mắt không? *(BLEU: 45.2 | 840.4 tok/s)*

### Câu #5 (Hội thoại thường nhật)
- 🇨🇳 **Nguyên tác Tiếng Trung:** `外面正在下大雨，出门记得带伞，路上注意安全。`
- 🎯 **Bản dịch chuẩn (Ground Truth):** Bên ngoài đang mưa lớn, ra ngoài nhớ mang ô, trên đường chú ý an toàn.
- 🟢 **Evolving-Head Hybrid SSM:** Bên ngoài đang mưa to, ra ngoài nhớ mang ô, trên đường chú ý an toàn *(BLEU: 79.4 | 153.4 tok/s)*
- 🔴 **Standard Transformer:** Ngoài trời đang mưa lớn, ra ngoài nhớ mang ô, trên đường chú ý đến An Vũ Thiên Vũ. *(BLEU: 61.9 | 826.7 tok/s)*

### Câu #6 (Hội thoại thường nhật)
- 🇨🇳 **Nguyên tác Tiếng Trung:** `您的外卖已经送达，放在门口了，请慢用。`
- 🎯 **Bản dịch chuẩn (Ground Truth):** Đơn hàng của quý khách đã được giao tới, để ở cửa rồi ạ, chúc quý khách ngon miệng.
- 🟢 **Evolving-Head Hybrid SSM:** Người bán ngoài cửa của ngài đã được giao, đặt ở cửa, xin vui lòng dùng chậm *(BLEU: 9.1 | 153.1 tok/s)*
- 🔴 **Standard Transformer:** Người bán của ngài đã được gửi đến, đặt ở cửa, xin vui lòng dùng chậm rãi. *(BLEU: 5.8 | 827.6 tok/s)*

### Câu #7 (Hội thoại thường nhật)
- 🇨🇳 **Nguyên tác Tiếng Trung:** `这件衣服不仅款式好看，而且穿起来特别舒服。`
- 🎯 **Bản dịch chuẩn (Ground Truth):** Bộ quần áo này không chỉ có kiểu dáng đẹp, mà mặc vào còn vô cùng thoải mái.
- 🟢 **Evolving-Head Hybrid SSM:** Bộ quần áo này không chỉ kiểu dáng đẹp mà còn mặc rất thoải mái. *(BLEU: 37.8 | 153.5 tok/s)*
- 🔴 **Standard Transformer:** Bộ quần áo này không chỉ đẹp, mà còn mặc một chiếc áo đặc biệt thoải mái. *(BLEU: 37.5 | 834.1 tok/s)*

### Câu #8 (Hội thoại thường nhật)
- 🇨🇳 **Nguyên tác Tiếng Trung:** `从这里坐地铁到高铁站大概需要四十分钟。`
- 🎯 **Bản dịch chuẩn (Ground Truth):** Từ đây đi tàu điện ngầm đến ga tàu cao tốc mất khoảng bốn mươi phút.
- 🟢 **Evolving-Head Hybrid SSM:** Từ đây đến ga tàu điện ngầm cao khoảng bốn mươi phút cần bốn mươi ngày *(BLEU: 28.8 | 154.3 tok/s)*
- 🔴 **Standard Transformer:** Từ đây đi tàu điện ngầm đến ga tàu cao tốc có thể mất bốn mươi phút. *(BLEU: 72.8 | 833.5 tok/s)*

### Câu #9 (Hội thoại thường nhật)
- 🇨🇳 **Nguyên tác Tiếng Trung:** `请问最近的便利店怎么走？我需要买一瓶水。`
- 🎯 **Bản dịch chuẩn (Ground Truth):** Xin hỏi cửa hàng tiện lợi gần nhất đi đường nào? Tôi cần mua một chai nước.
- 🟢 **Evolving-Head Hybrid SSM:** Xin hỏi cửa hàng tiện lợi gần đây đi như thế nào? Ta cần mua một chai nước. *(BLEU: 56.8 | 156.3 tok/s)*
- 🔴 **Standard Transformer:** Xin hỏi cửa hàng tiện lợi gần đây làm sao để đi? Ta cần mua một chai nước. *(BLEU: 54.7 | 817.5 tok/s)*

### Câu #10 (Hội thoại thường nhật)
- 🇨🇳 **Nguyên tác Tiếng Trung:** `无论遇到什么困难，只要我们齐心协力，就一定能够克服。`
- 🎯 **Bản dịch chuẩn (Ground Truth):** Bất kể gặp phải khó khăn gì, chỉ cần chúng ta đồng tâm hiệp lực thì nhất định có thể vượt qua.
- 🟢 **Evolving-Head Hybrid SSM:** Dù gặp khó khăn gì, chỉ cần chúng ta cùng nhau nỗ lực, nhất định có thể vượt qua được. *(BLEU: 50.4 | 157.1 tok/s)*
- 🔴 **Standard Transformer:** Dù gặp khó khăn gì, chỉ cần chúng ta đồng ý, chắc chắn sẽ có thể vượt qua Vạn Lý Linh. *(BLEU: 44.9 | 837.3 tok/s)*

